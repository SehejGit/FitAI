import mediapipe as mp
import cv2
import numpy as np
import math

# Initialize MediaPipe Pose
mp_pose      = mp.solutions.pose
mp_drawing   = mp.solutions.drawing_utils
pose_tracker = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def analyze_plank_shoulder_taps(video_path, output_video_path=None):
    cap     = cv2.VideoCapture(video_path)
    w, h    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS)
    writer  = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    tap_count        = 0
    # track each side separately
    stage_left       = "away"
    stage_right      = "away"
    good_frames      = 0
    alignment_scores = []

    def calc_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ang = abs(np.degrees(
            np.arctan2(c[1]-b[1], c[0]-b[0])
            - np.arctan2(a[1]-b[1], a[0]-b[0])
        ))
        return 360-ang if ang > 180 else ang

    # normalized tap threshold (tweak as needed)
    tap_thresh = 0.10

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

            # helper to get normalized coords
            def ncoord(idx):
                return np.array([lm[idx].x, lm[idx].y])

            # key points
            l_wrist = ncoord(mp_pose.PoseLandmark.LEFT_WRIST.value)
            r_wrist = ncoord(mp_pose.PoseLandmark.RIGHT_WRIST.value)
            l_shoul = ncoord(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            r_shoul = ncoord(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            hip     = ncoord(mp_pose.PoseLandmark.RIGHT_HIP.value)
            ankle   = ncoord(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

            # compute alignment (shoulder–hip–ankle)
            torso_ang = calc_angle(l_shoul, hip, ankle)
            align_score = max(0, min(1, (torso_ang - 150)/30))
            alignment_scores.append(align_score)

            # distances for taps
            dist_left  = np.linalg.norm(l_wrist - r_shoul)   # left hand to right shoulder
            dist_right = np.linalg.norm(r_wrist - l_shoul)   # right hand to left shoulder

            # left-hand tap logic
            if stage_left == "away" and dist_left < tap_thresh:
                tap_count += 1
                stage_left = "touch"
                print(f"Tap #{tap_count} (left) detected")
            elif stage_left == "touch" and dist_left > tap_thresh * 1.3:
                stage_left = "away"

            # right-hand tap logic
            if stage_right == "away" and dist_right < tap_thresh:
                tap_count += 1
                stage_right = "touch"
                print(f"Tap #{tap_count} (right) detected")
            elif stage_right == "touch" and dist_right > tap_thresh * 1.3:
                stage_right = "away"

            # draw pose + metrics
            mp_drawing.draw_landmarks(out, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(out, f"Taps: {tap_count}", (10, h-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            cv2.putText(out, f"Align: {align_score*100:.0f}%", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        if writer:
            writer.write(out)

    cap.release()
    if writer:
        writer.release()

    # summary & feedback
    if good_frames < 10:
        return {"tap_count": 0, "error": "Few detections—check camera angle."}

    avg_align = sum(alignment_scores)/len(alignment_scores)*100
    result = {
        "tap_count": tap_count,
        "avg_plank_alignment": avg_align,
        "frames_analyzed": good_frames,
        "feedback": []
    }
    if avg_align < 80:
        result["feedback"].append("Keep your hips level and core tight -> reduce sag/pike.")
    if tap_count == 0:
        result["feedback"].append("No taps counted—ensure your wrists reach the shoulder.")
    if not result["feedback"]:
        result["feedback"].append("Great control on your plank taps!")

    return result

# Example usage:
res = analyze_plank_shoulder_taps("clips/taps.mov", "clips/out_plank_taps.mp4")
print(res)