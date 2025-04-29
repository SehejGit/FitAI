import mediapipe as mp
import cv2
import numpy as np
import math

# Initialize MediaPipe Pose
mp_pose      = mp.solutions.pose
mp_drawing   = mp.solutions.drawing_utils
pose_tracker = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def analyze_tricep_dips(video_path, output_video_path=None):
    cap        = cv2.VideoCapture(video_path)
    w, h       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps        = cap.get(cv2.CAP_PROP_FPS)
    out_writer = None
    if output_video_path:
        fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    dip_count       = 0
    stage           = None  # "up" or "down"
    good_frames     = 0
    elbow_angles    = []
    alignment_scores = []

    def calc_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ang = abs(np.degrees(
            np.arctan2(c[1]-b[1], c[0]-b[0])
            - np.arctan2(a[1]-b[1], a[0]-b[0])
        ))
        return 360 - ang if ang > 180 else ang

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_tracker.process(img_rgb)
        out = frame.copy()

        if results.pose_landmarks:
            good_frames += 1
            lm = results.pose_landmarks.landmark

            # helper to pixel coords
            def px(idx):
                return (int(lm[idx].x * w), int(lm[idx].y * h))

            # key points (right side)
            shoulder = px(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            elbow    = px(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
            wrist    = px(mp_pose.PoseLandmark.RIGHT_WRIST.value)
            hip      = px(mp_pose.PoseLandmark.RIGHT_HIP.value)
            ankle    = px(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

            # compute angles
            elbow_ang = calc_angle(shoulder, elbow, wrist)
            elbow_angles.append(elbow_ang)

            torso_ang = calc_angle(shoulder, hip, ankle)
            # alignment: closer to 180° is better
            score = max(0, min(1, (torso_ang - 150) / 30))
            alignment_scores.append(score)

            # draw landmarks + metrics
            mp_drawing.draw_landmarks(out, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(out, f"Elbow: {elbow_ang:.1f}°", (elbow[0]-50, elbow[1]+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(out, f"Torso: {score*100:.0f}%", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # rep detection logic
            # Up position → elbow > 160°
            if elbow_ang > 160 and stage == "down":
                stage = "up"
            # Bottom position → elbow < 70°
            elif elbow_ang < 70 and (stage == "up" or stage is None):
                stage = "down"
                dip_count += 1
                print(f"Dip #{dip_count} detected at elbow {elbow_ang:.1f}°")

        # overlay count
        cv2.putText(out, f"Dips: {dip_count}", (10, h-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        if out_writer:
            out_writer.write(out)

    cap.release()
    if out_writer:
        out_writer.release()

    # summary & feedback
    if good_frames < 10:
        return {"dip_count": 0, "error": "Too few detections—check camera angle."}

    bottom_angle = min(elbow_angles) if elbow_angles else 0
    top_angle    = max(elbow_angles) if elbow_angles else 0
    avg_align    = (sum(alignment_scores) / len(alignment_scores)) * 100

    result = {
        "dip_count": dip_count,
        "form_analysis": {
            "elbow_angle_bottom": bottom_angle,
            "elbow_angle_top": top_angle,
            "avg_torso_alignment": avg_align,
            "frames_analyzed": good_frames
        },
        "feedback": []
    }

    if bottom_angle > 80:
        result["feedback"].append("Try dipping deeper—aim for elbow angles closer to 70° or below.")
    if top_angle < 160:
        result["feedback"].append("Fully extend your arms at the top of each dip.")
    if avg_align < 80:
        result["feedback"].append("Keep your torso more vertical—reduce forward lean.")

    if not result["feedback"]:
        result["feedback"].append("Great form on your triceps dips!")

    return result

# Example usage:
res = analyze_tricep_dips("clips/dips.mov", "clips/out_dips.mp4")
print(res)