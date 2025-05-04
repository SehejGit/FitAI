import mediapipe as mp
import cv2
import numpy as np
import math

# Initialize MediaPipe Pose
mp_pose      = mp.solutions.pose
mp_drawing   = mp.solutions.drawing_utils
pose_tracker = mp_pose.Pose(static_image_mode=False,
                            min_detection_confidence=0.5)

def detect_orientation(lm):
    """
    Returns 'front', 'left', or 'right' based on nose vs. shoulder midpoint.
    """
    nose = lm[mp_pose.PoseLandmark.NOSE.value]
    ls   = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs   = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    mid_x = (ls.x + rs.x) / 2
    delta = nose.x - mid_x
    if abs(delta) < 0.05:
        return "front"
    return "right" if delta < 0 else "left"

def analyze_band_tricep_pushdowns(video_path, output_video_path=None):
    cap    = cv2.VideoCapture(video_path)
    w, h   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    rep_count     = 0
    stage         = None  # "flexed" or "extended"
    good_frames   = 0
    elbow_angles  = []
    hip_angles    = []
    neck_angles   = []
    shoulder_heights = []
    orientations  = []

    def calc_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ang = abs(np.degrees(
            np.arctan2(c[1]-b[1], c[0]-b[0]) -
            np.arctan2(a[1]-b[1], a[0]-b[0])
        ))
        return 360 - ang if ang > 180 else ang

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res     = pose_tracker.process(img_rgb)
        out     = frame.copy()

        if res.pose_landmarks:
            good_frames += 1
            lm = res.pose_landmarks.landmark
            orientations.append(detect_orientation(lm))

            # normalized coords
            def nc(idx): return [lm[idx].x, lm[idx].y]
            # pixel coords
            def px(idx): return (int(lm[idx].x*w), int(lm[idx].y*h))

            # keypoints
            s_r, e_r, w_r = nc(mp_pose.PoseLandmark.RIGHT_SHOULDER.value), nc(mp_pose.PoseLandmark.RIGHT_ELBOW.value), nc(mp_pose.PoseLandmark.RIGHT_WRIST.value)
            s_l, e_l, w_l = nc(mp_pose.PoseLandmark.LEFT_SHOULDER.value),  nc(mp_pose.PoseLandmark.LEFT_ELBOW.value),  nc(mp_pose.PoseLandmark.LEFT_WRIST.value)
            hip_r = nc(mp_pose.PoseLandmark.RIGHT_HIP.value)
            hip_l = nc(mp_pose.PoseLandmark.LEFT_HIP.value)
            nose  = nc(mp_pose.PoseLandmark.NOSE.value)
            ear_r = nc(mp_pose.PoseLandmark.RIGHT_EAR.value)
            ear_l = nc(mp_pose.PoseLandmark.LEFT_EAR.value)

            # 1) Elbow angle (pin and track)
            elbow_r = calc_angle(s_r, e_r, w_r)
            elbow_l = calc_angle(s_l, e_l, w_l)
            elbow   = (elbow_r + elbow_l) / 2
            elbow_angles.append(elbow)

            # 2) Hip angle (shoulder→hip→knee) to check torso lean
            hip_ang_r = calc_angle(s_r, hip_r, nc(mp_pose.PoseLandmark.RIGHT_KNEE.value))
            hip_ang_l = calc_angle(s_l, hip_l, nc(mp_pose.PoseLandmark.LEFT_KNEE.value))
            hip_ang   = (hip_ang_r + hip_ang_l) / 2
            hip_angles.append(hip_ang)

            # 3) Neck alignment
            neck_ang = calc_angle(ear_l, nose, ear_r)
            neck_angles.append(neck_ang)

            # 4) Shoulder height diff (shrug cue)
            sh_diff = abs((s_r[1] - s_l[1]) * h)
            shoulder_heights.append(sh_diff)

            # draw landmarks + overlay
            mp_drawing.draw_landmarks(out, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(out, f"Elbow: {elbow:.1f}°", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(out, f"Torso: {hip_ang:.1f}°", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(out, f"Neck: {neck_ang:.1f}°", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            # rep detection based on elbow angle:
            #   flexed when <100°, extended when >160°
            if elbow < 100 and stage == "extended":
                stage = "flexed"
            elif elbow > 160 and (stage == "flexed" or stage is None):
                stage = "extended"
                rep_count += 1
                print(f"Tricep Pushdown #{rep_count}")

        # overlay rep count
        cv2.putText(out, f"Reps: {rep_count}", (10, h-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        if writer:
            writer.write(out)

    cap.release()
    if writer:
        writer.release()

    # too few detections
    if good_frames < 10:
        return {"rep_count": 0,
                "error": "Too few detections—check your angle & lighting."}

    # aggregate form metrics
    fa = {
        "min_elbow_deg"    : min(elbow_angles),
        "max_elbow_deg"    : max(elbow_angles),
        "avg_torso_deg"    : sum(hip_angles)/len(hip_angles),
        "avg_neck_deg"     : sum(neck_angles)/len(neck_angles),
        "avg_shoulder_dy"  : sum(shoulder_heights)/len(shoulder_heights),
        "dominant_view"    : max(set(orientations), key=orientations.count),
        "frames_analyzed"  : good_frames
    }

    # feedback
    feedback = []
    if fa["min_elbow_deg"] > 110:
        feedback.append("Pull all the way down—start with elbows ~90°.")
    if fa["max_elbow_deg"] < 160:
        feedback.append("Fully extend at bottom—aim for >160° elbows.")
    if abs(fa["avg_torso_deg"] - 180) > 10:
        feedback.append("Stand upright—avoid leaning forward/backward.")
    if fa["avg_neck_deg"] < 170:
        feedback.append("Keep head neutral—gaze forward.")
    if fa["avg_shoulder_dy"] > 0.05 * h:
        feedback.append("Don’t shrug—keep shoulders down.")

    # orientation-specific cues
    view = fa["dominant_view"]
    if view == "front":
        feedback.append("Front view: watch for even left/right elbow tracking.")
    elif view == "left":
        feedback.append("Left side: ensure left elbow stays pinned and torso upright.")
    else:
        feedback.append("Right side: ensure right elbow stays pinned and torso upright.")

    if not feedback:
        feedback.append("Excellent band tricep pushdown form!")

    return {
        "rep_count"    : rep_count,
        "form_analysis": fa,
        "feedback"     : feedback
    }

# Example usage:
if __name__ == "__main__":
    res = analyze_band_tricep_pushdowns("video.mp4", "out_tricep.mp4")
    print(res)