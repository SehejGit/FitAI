import mediapipe as mp
import cv2
import numpy as np
import math

# Initialize MediaPipe Pose
mp_pose      = mp.solutions.pose
mp_drawing   = mp.solutions.drawing_utils
pose_tracker = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)


def detect_orientation(lm):
    """
    Determine whether subject is facing 'front', 'left', or 'right'.
    """
    nose = lm[mp_pose.PoseLandmark.NOSE.value]
    ls   = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs   = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    mid_x = (ls.x + rs.x) / 2
    delta = nose.x - mid_x
    if abs(delta) < 0.05:
        return "front"
    return "right" if delta < 0 else "left"


def analyze_band_pull_aparts(video_path, output_video_path=None):
    cap    = cv2.VideoCapture(video_path)
    w, h   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    rep_count        = 0
    stage            = None  # "together" or "apart"
    good_frames      = 0
    elbow_angles     = []
    wrist_dists      = []
    neck_angles      = []
    shoulder_heights = []
    orientations     = []

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
            # record orientation
            orientations.append(detect_orientation(lm))

            # helpers
            def nc(idx): return [lm[idx].x, lm[idx].y]
            def px(idx): return (int(lm[idx].x * w), int(lm[idx].y * h))

            # key points
            s_r, e_r, w_r = nc(mp_pose.PoseLandmark.RIGHT_SHOULDER.value), nc(mp_pose.PoseLandmark.RIGHT_ELBOW.value), nc(mp_pose.PoseLandmark.RIGHT_WRIST.value)
            s_l, e_l, w_l = nc(mp_pose.PoseLandmark.LEFT_SHOULDER.value),  nc(mp_pose.PoseLandmark.LEFT_ELBOW.value),  nc(mp_pose.PoseLandmark.LEFT_WRIST.value)
            nose, ear_r, ear_l = nc(mp_pose.PoseLandmark.NOSE.value), nc(mp_pose.PoseLandmark.RIGHT_EAR.value), nc(mp_pose.PoseLandmark.LEFT_EAR.value)

            # 1) Elbow angle (shoulder→elbow→wrist)
            elbow_ang_r = calc_angle(s_r, e_r, w_r)
            elbow_ang_l = calc_angle(s_l, e_l, w_l)
            elbow_ang   = (elbow_ang_r + elbow_ang_l) / 2
            elbow_angles.append(elbow_ang)

            # 2) Wrist distance (horizontal band stretch)
            dx = abs((w_r[0] - w_l[0]) * w)
            wrist_dists.append(dx)

            # 3) Neck alignment (ear→nose→ear)
            neck_ang = calc_angle(ear_l, nose, ear_r)
            neck_angles.append(neck_ang)

            # 4) Shoulder height diff (shrug detection)
            shoulder_heights.append(abs((s_r[1] - s_l[1]) * h))

            # draw pose
            mp_drawing.draw_landmarks(out, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # overlays
            cv2.putText(out, f"Elbow: {elbow_ang:.1f}°", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(out, f"Wrist Dist: {dx:.0f}px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(out, f"Neck: {neck_ang:.1f}°", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            # rep detection
            shoulder_width = abs((s_r[0] - s_l[0]) * w)
            together_th = 0.2 * shoulder_width
            apart_th    = 0.6 * shoulder_width
            if dx < together_th and stage == "apart":
                stage = "together"
            elif dx > apart_th and (stage == "together" or stage is None):
                stage = "apart"
                rep_count += 1
                print(f"Band Pull-Apart #{rep_count}")

        # rep count overlay
        cv2.putText(out, f"Reps: {rep_count}", (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        if writer:
            writer.write(out)

    cap.release()
    if writer:
        writer.release()

    # summary
    if good_frames < 10:
        return {"rep_count": 0, "error": "Too few detections—check camera angle and lighting."}

    dominant_view = max(set(orientations), key=orientations.count)
    fa = {
        "min_elbow_deg"   : min(elbow_angles),
        "avg_wrist_px"    : sum(wrist_dists)/len(wrist_dists),
        "avg_neck_deg"    : sum(neck_angles)/len(neck_angles),
        "avg_shoulder_dy" : sum(shoulder_heights)/len(shoulder_heights),
        "dominant_view"   : dominant_view,
        "frames_analyzed" : good_frames
    }
    feedback = []
    # generic feedback
    if fa["min_elbow_deg"] < 165:
        feedback.append("Keep arms almost fully straight—avoid bending elbows.")
    if fa["avg_neck_deg"] < 170:
        feedback.append("Keep your head neutral—gaze forward.")
    if fa["avg_shoulder_dy"] > 0.05 * h:
        feedback.append("Don’t shrug—keep shoulders down.")
    # orientation-specific feedback
    if dominant_view == "front":
        feedback.append("Front view: focus on even retraction on both sides.")
    elif dominant_view == "left":
        feedback.append("Side view (left): ensure left scapula fully retracts and torso stays upright.")
    else:
        feedback.append("Side view (right): ensure right scapula fully retracts and torso stays upright.")
    if not feedback:
        feedback.append("Excellent band pull-apart form!")

    return {"rep_count": rep_count, "form_analysis": fa, "feedback": feedback}

