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
    ls  = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs  = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    mid_x = (ls.x + rs.x) / 2
    delta = nose.x - mid_x
    # threshold of ~5% of frame width
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

    rep_count       = 0
    stage           = None  # "together" or "apart"
    good_frames     = 0
    elbow_angles    = []
    wrist_dists     = []
    neck_angles     = []
    shoulder_heights= []
    orientations    = []

    def calc_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ang = abs(np.degrees(
            np.arctan2(c[1]-b[1], c[0]-b[0]) -
            np.arctan2(a[1]-b[1], a[0]-b[0])
        ))
        return 360 - ang if ang > 180 else ang

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res     = pose_tracker.process(img_rgb)
        out     = frame.copy()

        if res.pose_landmarks:
            good_frames += 1
            lm = res.pose_landmarks.landmark

            # record orientation
            orientations.append(detect_orientation(lm))

            # normalized coords
            def nc(idx): return [lm[idx].x, lm[idx].y]
            # pixel coords
            def px(idx): return (int(lm[idx].x*w), int(lm[idx].y*h))

            # keypoints for both sides
            s_r, e_r, w_r = nc(mp_pose.PoseLandmark.RIGHT_SHOULDER.value), nc(mp_pose.PoseLandmark.RIGHT_ELBOW.value), nc(mp_pose.PoseLandmark.RIGHT_WRIST.value)
            s_l, e_l, w_l = nc(mp_pose.PoseLandmark.LEFT_SHOULDER.value),  nc(mp_pose.PoseLandmark.LEFT_ELBOW.value),  nc(mp_pose.PoseLandmark.LEFT_WRIST.value)
            nose = nc(mp_pose.PoseLandmark.NOSE.value)
            ear_r= nc(mp_pose.PoseLandmark.RIGHT_EAR.value)
            ear_l= nc(mp_pose.PoseLandmark.LEFT_EAR.value)

            # 1) Elbow angle
            elbow_r = calc_angle(s_r, e_r, w_r)
            elbow_l = calc_angle(s_l, e_l, w_l)
            elbow   = (elbow_r + elbow_l) / 2
            elbow_angles.append(elbow)

            # 2) Wrist‐to‐wrist horizontal distance
            dx = abs(w_r[0] - w_l[0]) * w
            wrist_dists.append(dx)

            # 3) Neck alignment
            neck = calc_angle(ear_l, nose, ear_r)
            neck_angles.append(neck)

            # 4) Shoulder height diff
            sh_diff = abs((s_r[1] - s_l[1]) * h)
            shoulder_heights.append(sh_diff)

            # draw landmarks + metrics overlay
            mp_drawing.draw_landmarks(out, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(out, f"Elbow: {elbow:.1f}°",        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(out, f"Wrist Dist: {dx:.0f}px",     (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(out, f"Neck: {neck:.1f}°",           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            # rep detection thresholds
            shoulder_width = abs((s_r[0] - s_l[0]) * w)
            th_close = 0.2 * shoulder_width
            th_far   = 0.6 * shoulder_width

            if dx < th_close and stage == "apart":
                stage = "together"
            elif dx > th_far and (stage == "together" or stage is None):
                stage = "apart"
                rep_count += 1
                print(f"Pull-Apart #{rep_count}")

        # overlay rep count
        cv2.putText(out, f"Reps: {rep_count}", (10, h-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        if writer:
            writer.write(out)

    cap.release()
    if writer:
        writer.release()

    # if too few frames detected
    if good_frames < 10:
        return {"rep_count": 0,
                "error": "Too few detections, check your angle & lighting."}

    # aggregate results
    fa = {
        "min_elbow_deg"    : min(elbow_angles),
        "avg_wrist_px"     : sum(wrist_dists)/len(wrist_dists),
        "avg_neck_deg"     : sum(neck_angles)/len(neck_angles),
        "avg_shoulder_dy"  : sum(shoulder_heights)/len(shoulder_heights),
        "dominant_view"    : max(set(orientations), key=orientations.count),
        "frames_analyzed"  : good_frames
    }

    feedback = []
    # -- generic feedback --
    if fa["min_elbow_deg"] < 165:
        feedback.append("Keep arms close to straight, avoid bending elbows.")
    if fa["avg_neck_deg"] < 170:
        feedback.append("Hold your head neutral, gaze forward.")
    if fa["avg_shoulder_dy"] > 0.05 * h:
        feedback.append("Don’t shrug your shoulders keep shoulders down.")

    # -- orientation-specific feedback --
    view = fa["dominant_view"]
    if view == "front":
        feedback.append("Focus on even retraction on both sides.")
    elif view == "left":
        feedback.append("(Left View): Ensure your left shoulder blade fully retracts and your torso stays upright.")
    else:  # right
        feedback.append("(Right View): Ensure your right shoulder blade fully retracts and your torso stays upright.")

    if not feedback:
        feedback.append("Excellent band pull-apart form!")

    return {
        "rep_count"   : rep_count,
        "form_analysis": fa,
        "feedback"    : feedback
    }

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

def analyze_banded_hip_thrusts(video_path, output_video_path=None):
    cap    = cv2.VideoCapture(video_path)
    w, h   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    rep_count     = 0
    stage         = None  # "down" or "up"
    good_frames   = 0
    hip_angles    = []
    knee_angles   = []
    neck_angles   = []
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
        if not ret: break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res     = pose_tracker.process(img_rgb)
        out     = frame.copy()
        if res.pose_landmarks:
            good_frames += 1
            lm = res.pose_landmarks.landmark
            orientations.append(detect_orientation(lm))

            def nc(idx): return [lm[idx].x, lm[idx].y]

            # Hips: shoulder→hip→knee
            hip_ang = calc_angle(nc(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
                                 nc(mp_pose.PoseLandmark.LEFT_HIP.value),
                                 nc(mp_pose.PoseLandmark.LEFT_KNEE.value))
            hip_angles.append(hip_ang)

            # Knees: hip→knee→ankle
            knee_ang = calc_angle(nc(mp_pose.PoseLandmark.LEFT_HIP.value),
                                  nc(mp_pose.PoseLandmark.LEFT_KNEE.value),
                                  nc(mp_pose.PoseLandmark.LEFT_ANKLE.value))
            knee_angles.append(knee_ang)

            # Neck: ear→nose→ear
            neck_ang = calc_angle(nc(mp_pose.PoseLandmark.LEFT_EAR.value),
                                  nc(mp_pose.PoseLandmark.NOSE.value),
                                  nc(mp_pose.PoseLandmark.RIGHT_EAR.value))
            neck_angles.append(neck_ang)

            mp_drawing.draw_landmarks(out, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(out, f"Hip: {hip_ang:.1f}°",  (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(out, f"Knee: {knee_ang:.1f}°", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # rep detect: bottom when hip < 70°, top when hip > 160°
            if hip_ang < 70 and stage == "up":
                stage = "down"
            elif hip_ang > 160 and (stage == "down" or stage is None):
                stage = "up"
                rep_count += 1
                print(f"Hip Thrust #{rep_count}")

        cv2.putText(out, f"Reps: {rep_count}", (10, h-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        if writer: writer.write(out)

    cap.release()
    if writer: writer.release()

    if good_frames < 10:
        return {"rep_count": 0, "error": "Too few detections."}

    fa = {
        "min_hip_deg"    : min(hip_angles),
        "max_hip_deg"    : max(hip_angles),
        "avg_knee_deg"   : sum(knee_angles)/len(knee_angles),
        "avg_neck_deg"   : sum(neck_angles)/len(neck_angles),
        "dominant_view"  : max(set(orientations), key=orientations.count),
        "frames_analyzed": good_frames
    }
    feedback = []
    if fa["min_hip_deg"] < 70:
        feedback.append("Drive hips higher—aim closer to 180° at the top.")
    if fa["max_hip_deg"] < 160:
        feedback.append("Fully extend hips at top (>160°).")
    if fa["avg_knee_deg"] < 85 or fa["avg_knee_deg"] > 95:
        feedback.append("Keep shins vertical—hip-knee-ankle ≈90°.")
    if fa["avg_neck_deg"] < 170:
        feedback.append("Maintain neutral neck—don’t tuck chin.")

    view = fa["dominant_view"]
    if view == "front":
        feedback.append("Front view: watch even hip drive.")
    elif view == "left":
        feedback.append("Left side: ensure hips fully extend without back arch.")
    else:
        feedback.append("Right side: ensure hips fully extend without back arch.")

    if not feedback:
        feedback.append("Great banded hip thrusts form!")

    return {"rep_count": rep_count, "form_analysis": fa, "feedback": feedback}


def analyze_leg_press(video_path, output_video_path=None):
    cap    = cv2.VideoCapture(video_path)
    w, h   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    rep_count    = 0
    stage        = None  # "down" or "up"
    good_frames  = 0
    knee_angles  = []
    hip_angles   = []
    neck_angles  = []
    orientations = []

    def calc_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ang = abs(np.degrees(
            np.arctan2(c[1]-b[1], c[0]-b[0]) -
            np.arctan2(a[1]-b[1], a[0]-b[0])
        ))
        return 360 - ang if ang > 180 else ang

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res     = pose_tracker.process(img_rgb)
        out     = frame.copy()
        if res.pose_landmarks:
            good_frames += 1
            lm = res.pose_landmarks.landmark
            orientations.append(detect_orientation(lm))
            def nc(idx): return [lm[idx].x, lm[idx].y]

            # Knee: hip→knee→ankle
            knee_ang = calc_angle(nc(mp_pose.PoseLandmark.LEFT_HIP.value),
                                  nc(mp_pose.PoseLandmark.LEFT_KNEE.value),
                                  nc(mp_pose.PoseLandmark.LEFT_ANKLE.value))
            knee_angles.append(knee_ang)

            # Hip: shoulder→hip→knee
            hip_ang = calc_angle(nc(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
                                 nc(mp_pose.PoseLandmark.LEFT_HIP.value),
                                 nc(mp_pose.PoseLandmark.LEFT_KNEE.value))
            hip_angles.append(hip_ang)

            # Neck
            neck_ang = calc_angle(nc(mp_pose.PoseLandmark.LEFT_EAR.value),
                                  nc(mp_pose.PoseLandmark.NOSE.value),
                                  nc(mp_pose.PoseLandmark.RIGHT_EAR.value))
            neck_angles.append(neck_ang)

            mp_drawing.draw_landmarks(out, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(out, f"Knee: {knee_ang:.1f}°", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(out, f"Hip: {hip_ang:.1f}°",   (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # rep detect: down < 90°, up > 160°
            if knee_ang < 90 and stage == "up":
                stage = "down"
            elif knee_ang > 160 and (stage == "down" or stage is None):
                stage = "up"
                rep_count += 1
                print(f"Leg Press #{rep_count}")

        cv2.putText(out, f"Reps: {rep_count}", (10, h-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        if writer: writer.write(out)

    cap.release()
    if writer: writer.release()
    if good_frames < 10:
        return {"rep_count": 0, "error": "Too few detections."}

    fa = {
        "min_knee_deg"   : min(knee_angles),
        "max_knee_deg"   : max(knee_angles),
        "avg_hip_deg"    : sum(hip_angles)/len(hip_angles),
        "avg_neck_deg"   : sum(neck_angles)/len(neck_angles),
        "dominant_view"  : max(set(orientations), key=orientations.count),
        "frames_analyzed": good_frames
    }
    feedback = []
    if fa["min_knee_deg"] > 90:
        feedback.append("Lower carriage more—aim knee angle ≈90° at bottom.")
    if fa["max_knee_deg"] < 160:
        feedback.append("Press fully—aim knee extension >160°.")
    if abs(fa["avg_hip_deg"] - 180) > 10:
        feedback.append("Keep hips pressed—avoid lifting lower back.")
    if fa["avg_neck_deg"] < 170:
        feedback.append("Keep head neutral—don’t tuck chin.")

    view = fa["dominant_view"]
    if view == "front":
        feedback.append("Front view: watch even tracking of both knees.")
    elif view == "left":
        feedback.append("Left side: ensure left knee tracks over toes.")
    else:
        feedback.append("Right side: ensure right knee tracks over toes.")

    if not feedback:
        feedback.append("Great leg press form!")

    return {"rep_count": rep_count, "form_analysis": fa, "feedback": feedback}

def analyze_leg_extensions(video_path, output_video_path=None):
    cap    = cv2.VideoCapture(video_path)
    w, h   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    rep_count    = 0
    stage        = None  # "down" or "up"
    good_frames  = 0
    knee_angles  = []
    neck_angles  = []
    orientations = []

    def calc_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ang = abs(np.degrees(
            np.arctan2(c[1]-b[1], c[0]-b[0]) -
            np.arctan2(a[1]-b[1], a[0]-b[0])
        ))
        return 360 - ang if ang > 180 else ang

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res     = pose_tracker.process(img_rgb)
        out     = frame.copy()
        if res.pose_landmarks:
            good_frames += 1
            lm = res.pose_landmarks.landmark
            orientations.append(detect_orientation(lm))
            def nc(idx): return [lm[idx].x, lm[idx].y]

            # Knee: hip→knee→ankle
            knee_ang = calc_angle(nc(mp_pose.PoseLandmark.LEFT_HIP.value),
                                  nc(mp_pose.PoseLandmark.LEFT_KNEE.value),
                                  nc(mp_pose.PoseLandmark.LEFT_ANKLE.value))
            knee_angles.append(knee_ang)

            # Neck
            neck_ang = calc_angle(nc(mp_pose.PoseLandmark.LEFT_EAR.value),
                                  nc(mp_pose.PoseLandmark.NOSE.value),
                                  nc(mp_pose.PoseLandmark.RIGHT_EAR.value))
            neck_angles.append(neck_ang)

            mp_drawing.draw_landmarks(out, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(out, f"Knee: {knee_ang:.1f}°", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            # rep detect: down <100°, up >160°
            if knee_ang < 100 and stage == "up":
                stage = "down"
            elif knee_ang > 160 and (stage == "down" or stage is None):
                stage = "up"
                rep_count += 1
                print(f"Leg Extension #{rep_count}")

        cv2.putText(out, f"Reps: {rep_count}", (10, h-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        if writer: writer.write(out)

    cap.release()
    if writer: writer.release()
    if good_frames < 10:
        return {"rep_count": 0, "error": "Too few detections."}

    fa = {
        "min_knee_deg"   : min(knee_angles),
        "max_knee_deg"   : max(knee_angles),
        "avg_neck_deg"   : sum(neck_angles)/len(neck_angles),
        "dominant_view"  : max(set(orientations), key=orientations.count),
        "frames_analyzed": good_frames
    }
    feedback = []
    if fa["min_knee_deg"] > 100:
        feedback.append("Lower more—get knee closer to 90° at start.")
    if fa["max_knee_deg"] < 160:
        feedback.append("Extend fully—aim for >160° at top.")
    if fa["avg_neck_deg"] < 170:
        feedback.append("Keep head neutral.")

    view = fa["dominant_view"]
    if view == "front":
        feedback.append("Front view: watch both legs extend evenly.")
    elif view == "left":
        feedback.append("Left side: ensure left knee tracks smoothly.")
    else:
        feedback.append("Right side: ensure right knee tracks smoothly.")

    if not feedback:
        feedback.append("Great leg extension form!")

    return {"rep_count": rep_count, "form_analysis": fa, "feedback": feedback}

def analyze_leg_curls(video_path, output_video_path=None):
    cap    = cv2.VideoCapture(video_path)
    w, h   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    rep_count    = 0
    stage        = None  # "down" or "up"
    good_frames  = 0
    knee_angles  = []
    neck_angles  = []
    orientations = []

    def calc_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ang = abs(np.degrees(
            np.arctan2(c[1]-b[1], c[0]-b[0]) -
            np.arctan2(a[1]-b[1], a[0]-b[0])
        ))
        return 360 - ang if ang > 180 else ang

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res     = pose_tracker.process(img_rgb)
        out     = frame.copy()
        if res.pose_landmarks:
            good_frames += 1
            lm = res.pose_landmarks.landmark
            orientations.append(detect_orientation(lm))
            def nc(idx): return [lm[idx].x, lm[idx].y]

            # Knee: hip→knee→ankle
            knee_ang = calc_angle(nc(mp_pose.PoseLandmark.LEFT_HIP.value),
                                  nc(mp_pose.PoseLandmark.LEFT_KNEE.value),
                                  nc(mp_pose.PoseLandmark.LEFT_ANKLE.value))
            knee_angles.append(knee_ang)

            # Neck
            neck_ang = calc_angle(nc(mp_pose.PoseLandmark.LEFT_EAR.value),
                                  nc(mp_pose.PoseLandmark.NOSE.value),
                                  nc(mp_pose.PoseLandmark.RIGHT_EAR.value))
            neck_angles.append(neck_ang)

            mp_drawing.draw_landmarks(out, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(out, f"Knee: {knee_ang:.1f}°", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            # rep detect: down >160°, up <60°
            if knee_ang > 160 and stage == "up":
                stage = "down"
            elif knee_ang < 60 and (stage == "down" or stage is None):
                stage = "up"
                rep_count += 1
                print(f"Leg Curl #{rep_count}")

        cv2.putText(out, f"Reps: {rep_count}", (10, h-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        if writer: writer.write(out)

    cap.release()
    if writer: writer.release()
    if good_frames < 10:
        return {"rep_count": 0, "error": "Too few detections."}

    fa = {
        "min_knee_deg"   : min(knee_angles),
        "max_knee_deg"   : max(knee_angles),
        "avg_neck_deg"   : sum(neck_angles)/len(neck_angles),
        "dominant_view"  : max(set(orientations), key=orientations.count),
        "frames_analyzed": good_frames
    }
    feedback = []
    if fa["max_knee_deg"] < 160:
        feedback.append("Extend leg more at start—aim for >160°.")
    if fa["min_knee_deg"] > 60:
        feedback.append("Curl heels closer to glutes—aim <60°.")
    if fa["avg_neck_deg"] < 170:
        feedback.append("Keep head neutral.")

    view = fa["dominant_view"]
    if view == "front":
        feedback.append("Front: ensure both legs curl evenly.")
    elif view == "left":
        feedback.append("Left: check left knee flexion depth.")
    else:
        feedback.append("Right: check right knee flexion depth.")

    if not feedback:
        feedback.append("Great leg curl form!")

    return {"rep_count": rep_count, "form_analysis": fa, "feedback": feedback}

def analyze_hip_abduction_adduction(video_path, output_video_path=None):
    cap       = cv2.VideoCapture(video_path)
    w, h      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps       = cap.get(cv2.CAP_PROP_FPS)
    writer    = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps,(w,h))

    rep_count      = 0
    stage          = None      # "adduct" or "abduct"
    good_frames    = 0
    hip_angles     = []
    knee_angles    = []
    shoulder_diff  = []
    neck_angles    = []
    orientations   = []

    def calc_angle(a,b,c):
        a,b,c = np.array(a),np.array(b),np.array(c)
        ang = abs(np.degrees(
            np.arctan2(c[1]-b[1],c[0]-b[0]) -
            np.arctan2(a[1]-b[1],a[0]-b[0])
        ))
        return 360-ang if ang>180 else ang

    while True:
        ret,frame = cap.read()
        if not ret: break
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res = pose_tracker.process(img)
        out = frame.copy()
        if res.pose_landmarks:
            good_frames+=1
            lm = res.pose_landmarks.landmark
            orientations.append(detect_orientation(lm))
            # coords
            def nc(idx): return [lm[idx].x, lm[idx].y]
            # hip abd/adduction: shoulder→hip→knee
            hip = calc_angle(
                nc(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
                nc(mp_pose.PoseLandmark.LEFT_HIP.value),
                nc(mp_pose.PoseLandmark.LEFT_KNEE.value)
            )
            hip_angles.append(hip)
            # knee stability: hip→knee→ankle
            knee = calc_angle(
                nc(mp_pose.PoseLandmark.LEFT_HIP.value),
                nc(mp_pose.PoseLandmark.LEFT_KNEE.value),
                nc(mp_pose.PoseLandmark.LEFT_ANKLE.value)
            )
            knee_angles.append(knee)
            # shoulder level: diff in shoulder y
            s_l = nc(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            s_r = nc(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            shoulder_diff.append(abs((s_l[1]-s_r[1])*h))
            # neck align
            neck = calc_angle(
                nc(mp_pose.PoseLandmark.LEFT_EAR.value),
                nc(mp_pose.PoseLandmark.NOSE.value),
                nc(mp_pose.PoseLandmark.RIGHT_EAR.value)
            )
            neck_angles.append(neck)

            mp_drawing.draw_landmarks(out,res.pose_landmarks,mp_pose.POSE_CONNECTIONS)
            cv2.putText(out,f"Hip: {hip:.1f}°",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

            # rep detection: adduct when hip<10°, abduct when >25°
            if hip < 10 and stage=="abduct":
                stage="adduct"
            elif hip > 25 and (stage=="adduct" or stage is None):
                stage="abduct"
                rep_count+=1
                print(f"Hip Ab/Adduction #{rep_count}")

        cv2.putText(out,f"Reps: {rep_count}",(10,h-30),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
        if writer: writer.write(out)

    cap.release()
    if writer: writer.release()

    if good_frames<10:
        return {"rep_count":0,"error":"Too few detections."}

    fa = {
        "min_hip_deg"    : min(hip_angles),
        "max_hip_deg"    : max(hip_angles),
        "avg_knee_deg"   : sum(knee_angles)/len(knee_angles),
        "avg_shoulder_dy": sum(shoulder_diff)/len(shoulder_diff),
        "avg_neck_deg"   : sum(neck_angles)/len(neck_angles),
        "dominant_view"  : max(set(orientations),key=orientations.count),
        "frames_analyzed": good_frames
    }
    fb=[]
    if fa["max_hip_deg"]<25:
        fb.append("Increase abduction range—aim ~30°.")
    if fa["min_hip_deg"]>10:
        fb.append("Bring leg fully across midline on adduction (<10°).")
    if fa["avg_shoulder_dy"]>0.05*h:
        fb.append("Keep shoulders level—avoid hip hiking.")
    if fa["avg_neck_deg"]<170:
        fb.append("Maintain neutral neck.")

    v=fa["dominant_view"]
    if v=="front":
        fb.append("Front view: watch even left/right range.")
    elif v=="left":
        fb.append("Left side: ensure left hip moves through full range.")
    else:
        fb.append("Right side: ensure right hip moves through full range.")

    if not fb: fb.append("Great hip abd/adduction form!")
    return {"rep_count":rep_count,"form_analysis":fa,"feedback":fb}

def analyze_calf_raise_machine(video_path,output_video_path=None):
    cap      = cv2.VideoCapture(video_path)
    w,h      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    writer   = None
    if output_video_path:
        fourcc=cv2.VideoWriter_fourcc(*'mp4v')
        writer=cv2.VideoWriter(output_video_path,fourcc,fps,(w,h))

    rep_count   = 0
    stage       = None  # "down" or "up"
    good_frames = 0
    ankle_angles= []
    knee_angles = []
    neck_angles = []
    orientations= []

    def calc_angle(a,b,c):
        a,b,c=np.array(a),np.array(b),np.array(c)
        ang=abs(np.degrees(np.arctan2(c[1]-b[1],c[0]-b[0])-
                            np.arctan2(a[1]-b[1],a[0]-b[0])))
        return 360-ang if ang>180 else ang

    while True:
        ret,frame=cap.read()
        if not ret: break
        img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res=pose_tracker.process(img)
        out=frame.copy()
        if res.pose_landmarks:
            good_frames+=1
            lm=res.pose_landmarks.landmark
            orientations.append(detect_orientation(lm))
            def nc(i): return [lm[i].x,lm[i].y]
            # ankle angle: ankle→heel→foot_index
            ankle=calc_angle(
                nc(mp_pose.PoseLandmark.LEFT_ANKLE.value),
                nc(mp_pose.PoseLandmark.LEFT_HEEL.value),
                nc(mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value)
            )
            ankle_angles.append(ankle)
            # knee lock: hip→knee→ankle
            knee=calc_angle(
                nc(mp_pose.PoseLandmark.LEFT_HIP.value),
                nc(mp_pose.PoseLandmark.LEFT_KNEE.value),
                nc(mp_pose.PoseLandmark.LEFT_ANKLE.value)
            )
            knee_angles.append(knee)
            # neck
            neck=calc_angle(
                nc(mp_pose.PoseLandmark.LEFT_EAR.value),
                nc(mp_pose.PoseLandmark.NOSE.value),
                nc(mp_pose.PoseLandmark.RIGHT_EAR.value)
            )
            neck_angles.append(neck)

            mp_drawing.draw_landmarks(out,res.pose_landmarks,mp_pose.POSE_CONNECTIONS)
            cv2.putText(out,f"Ankle: {ankle:.1f}°",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            # rep detect: down<100°, up>150°
            if ankle<100 and stage=="up":
                stage="down"
            elif ankle>150 and (stage=="down" or stage is None):
                stage="up"; rep_count+=1; print(f"Calf Raise #{rep_count}")

        cv2.putText(out,f"Reps: {rep_count}",(10,h-30),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
        if writer: writer.write(out)

    cap.release(); writer and writer.release() if writer else None

    if good_frames<10:
        return {"rep_count":0,"error":"Too few detections."}

    fa={
     "min_ankle_deg":min(ankle_angles),
     "max_ankle_deg":max(ankle_angles),
     "avg_knee_deg" :sum(knee_angles)/len(knee_angles),
     "avg_neck_deg" :sum(neck_angles)/len(neck_angles),
     "dominant_view":max(set(orientations),key=orientations.count),
     "frames_analyzed":good_frames
    }
    fb=[]
    if fa["max_ankle_deg"]<150: fb.append("Raise heels higher—aim >150° plantarflexion.")
    if fa["min_ankle_deg"]>100: fb.append("Lower heels fully below neutral (<100°).")
    if abs(fa["avg_knee_deg"]-180)>5: fb.append("Keep knees locked (~180°).")
    if fa["avg_neck_deg"]<170: fb.append("Keep head neutral.")

    v=fa["dominant_view"]
    if v=="front": fb.append("Front: watch both calves evenly.")
    elif v=="left": fb.append("Left: ensure left heel raises fully.")
    else: fb.append("Right: ensure right heel raises fully.")
    if not fb: fb.append("Great calf raise form!")
    return {"rep_count":rep_count,"form_analysis":fa,"feedback":fb}

def analyze_cable_crunches(video_path,output_video_path=None):
    cap=cv2.VideoCapture(video_path)
    w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=cap.get(cv2.CAP_PROP_FPS)
    writer=None
    if output_video_path:
        fourcc=cv2.VideoWriter_fourcc(*'mp4v')
        writer=cv2.VideoWriter(output_video_path,fourcc,fps,(w,h))

    rep_count=0;stage=None;gf=0
    hip_angles=[];neck_angles=[];orientations=[]

    def angle(a,b,c):
        a,b,c=np.array(a),np.array(b),np.array(c)
        ang=abs(np.degrees(np.arctan2(c[1]-b[1],c[0]-b[0])-
                           np.arctan2(a[1]-b[1],a[0]-b[0])))
        return 360-ang if ang>180 else ang

    while True:
        ret,frame=cap.read()
        if not ret:break
        img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res=pose_tracker.process(img);out=frame.copy()
        if res.pose_landmarks:
            gf+=1;lm=res.pose_landmarks.landmark
            orientations.append(detect_orientation(lm))
            def nc(i):return[lm[i].x,lm[i].y]
            # hip flexion: shoulder→hip→knee
            hip=angle(nc(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
                      nc(mp_pose.PoseLandmark.LEFT_HIP.value),
                      nc(mp_pose.PoseLandmark.LEFT_KNEE.value))
            hip_angles.append(hip)
            # neck tuck: ear→nose→ear
            neck=angle(nc(mp_pose.PoseLandmark.LEFT_EAR.value),
                       nc(mp_pose.PoseLandmark.NOSE.value),
                       nc(mp_pose.PoseLandmark.RIGHT_EAR.value))
            neck_angles.append(neck)
            mp_drawing.draw_landmarks(out,res.pose_landmarks,mp_pose.POSE_CONNECTIONS)
            cv2.putText(out,f"Hip: {hip:.1f}°",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            # rep: down when hip>160°, up when hip<100°
            if hip>160 and stage=="up":
                stage="down"
            elif hip<100 and(stage=="down" or stage is None):
                stage="up";rep_count+=1;print(f"Cable Crunch #{rep_count}")
        cv2.putText(out,f"Reps: {rep_count}",(10,h-30),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
        if writer:writer.write(out)
    cap.release(); writer and writer.release() if writer else None

    if gf<10: return{"rep_count":0,"error":"Too few detections."}
    fa={
     "min_hip_deg":min(hip_angles),
     "max_hip_deg":max(hip_angles),
     "avg_neck_deg":sum(neck_angles)/len(neck_angles),
     "dominant_view":max(set(orientations),key=orientations.count),
     "frames_analyzed":gf
    }
    fb=[]
    if fa["max_hip_deg"]<160: fb.append("Crunch deeper—aim hip flexion ~90°.")
    if fa["min_hip_deg"]>100: fb.append("Return fully to upright (>160°).")
    if fa["avg_neck_deg"]<170: fb.append("Tuck chin—keep neck neutral.")

    v=fa["dominant_view"]
    if v=="front": fb.append("Front: watch even left/right movement.")
    elif v=="left": fb.append("Left: ensure side crunch is visible.")
    else: fb.append("Right: ensure side crunch is visible.")
    if not fb: fb.append("Great cable crunch form!")
    return{"rep_count":rep_count,"form_analysis":fa,"feedback":fb}

def analyze_ab_machine(video_path,output_video_path=None):
    cap=cv2.VideoCapture(video_path)
    w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=cap.get(cv2.CAP_PROP_FPS)
    writer=None
    if output_video_path:
        fourcc=cv2.VideoWriter_fourcc(*'mp4v')
        writer=cv2.VideoWriter(output_video_path,fourcc,fps,(w,h))

    rep_count=0;stage=None;gf=0
    hip_angles=[];neck_angles=[];orientations=[]

    def angle(a,b,c):
        a,b,c=np.array(a),np.array(b),np.array(c)
        ang=abs(np.degrees(np.arctan2(c[1]-b[1],c[0]-b[0])-
                           np.arctan2(a[1]-b[1],a[0]-b[0])))
        return 360-ang if ang>180 else ang

    while True:
        ret,frame=cap.read()
        if not ret:break
        img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res=pose_tracker.process(img);out=frame.copy()
        if res.pose_landmarks:
            gf+=1;lm=res.pose_landmarks.landmark
            orientations.append(detect_orientation(lm))
            def nc(i):return[lm[i].x,lm[i].y]
            hip=angle(nc(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
                      nc(mp_pose.PoseLandmark.LEFT_HIP.value),
                      nc(mp_pose.PoseLandmark.LEFT_KNEE.value))
            hip_angles.append(hip)
            neck=angle(nc(mp_pose.PoseLandmark.LEFT_EAR.value),
                       nc(mp_pose.PoseLandmark.NOSE.value),
                       nc(mp_pose.PoseLandmark.RIGHT_EAR.value))
            neck_angles.append(neck)
            mp_drawing.draw_landmarks(out,res.pose_landmarks,mp_pose.POSE_CONNECTIONS)
            cv2.putText(out,f"Hip: {hip:.1f}°",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            if hip>160 and stage=="up":
                stage="down"
            elif hip<100 and(stage=="down" or stage is None):
                stage="up";rep_count+=1;print(f"Ab Machine #{rep_count}")
        cv2.putText(out,f"Reps: {rep_count}",(10,h-30),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
        if writer:writer.write(out)
    cap.release(); writer and writer.release() if writer else None

    if gf<10:return{"rep_count":0,"error":"Too few detections."}
    fa={
     "min_hip_deg":min(hip_angles),
     "max_hip_deg":max(hip_angles),
     "avg_neck_deg":sum(neck_angles)/len(neck_angles),
     "dominant_view":max(set(orientations),key=orientations.count),
     "frames_analyzed":gf
    }
    fb=[]
    if fa["max_hip_deg"]<160:fb.append("Crunch deeper—aim ~90° hip flexion.")
    if fa["min_hip_deg"]>100:fb.append("Return upright (>160°).")
    if fa["avg_neck_deg"]<170:fb.append("Tuck chin—keep neck neutral.")
    v=fa["dominant_view"]
    if v=="front":fb.append("Front: watch even movement.")
    elif v=="left":fb.append("Left: ensure side flexion visible.")
    else:fb.append("Right: ensure side flexion visible.")
    if not fb:fb.append("Great ab-machine form!")
    return{"rep_count":rep_count,"form_analysis":fa,"feedback":fb}

def analyze_hanging_leg_raises(video_path,output_video_path=None):
    cap=cv2.VideoCapture(video_path)
    w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=cap.get(cv2.CAP_PROP_FPS)
    writer=None
    if output_video_path:
        fourcc=cv2.VideoWriter_fourcc(*'mp4v')
        writer=cv2.VideoWriter(output_video_path,fourcc,fps,(w,h))

    rep_count=0;stage=None;gf=0
    hip_angles=[];neck_angles=[];orientations=[]

    def angle(a,b,c):
        a,b,c=np.array(a),np.array(b),np.array(c)
        ang=abs(np.degrees(np.arctan2(c[1]-b[1],c[0]-b[0])-
                           np.arctan2(a[1]-b[1],a[0]-b[0])))
        return 360-ang if ang>180 else ang

    while True:
        ret,frame=cap.read()
        if not ret:break
        img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res=pose_tracker.process(img);out=frame.copy()
        if res.pose_landmarks:
            gf+=1;lm=res.pose_landmarks.landmark
            orientations.append(detect_orientation(lm))
            def nc(i):return[lm[i].x,lm[i].y]
            # hip flexion: shoulder→hip→knee
            hip=angle(nc(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
                      nc(mp_pose.PoseLandmark.LEFT_HIP.value),
                      nc(mp_pose.PoseLandmark.LEFT_KNEE.value))
            hip_angles.append(hip)
            # neck
            neck=angle(nc(mp_pose.PoseLandmark.LEFT_EAR.value),
                       nc(mp_pose.PoseLandmark.NOSE.value),
                       nc(mp_pose.PoseLandmark.RIGHT_EAR.value))
            neck_angles.append(neck)
            mp_drawing.draw_landmarks(out,res.pose_landmarks,mp_pose.POSE_CONNECTIONS)
            cv2.putText(out,f"Hip: {hip:.1f}°",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            # rep: down hip>160°, up<100°
            if hip>160 and stage=="up":
                stage="down"
            elif hip<100 and(stage=="down" or stage is None):
                stage="up";rep_count+=1;print(f"Hanging Raise #{rep_count}")
        cv2.putText(out,f"Reps: {rep_count}",(10,h-30),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
        if writer:writer.write(out)
    cap.release(); writer and writer.release() if writer else None

    if gf<10:return{"rep_count":0,"error":"Too few detections."}
    fa={
     "min_hip_deg":min(hip_angles),
     "max_hip_deg":max(hip_angles),
     "avg_neck_deg":sum(neck_angles)/len(neck_angles),
     "dominant_view":max(set(orientations),key=orientations.count),
     "frames_analyzed":gf
    }
    fb=[]
    if fa["max_hip_deg"]<90:fb.append("Lift knees/legs higher—aim ~90° hip flexion.")
    if fa["min_hip_deg"]>160:fb.append("Lower fully (<160°).")
    if fa["avg_neck_deg"]<170:fb.append("Keep neck neutral.")
    v=fa["dominant_view"]
    if v=="front":fb.append("Front: watch symmetry.")
    elif v=="left":fb.append("Left: ensure left leg lifts fully.")
    else:fb.append("Right: ensure right leg lifts fully.")
    if not fb:fb.append("Great hanging leg raise form!")
    return{"rep_count":rep_count,"form_analysis":fa,"feedback":fb}

def analyze_assisted_pullups(video_path,output_video_path=None):
    cap=cv2.VideoCapture(video_path)
    w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=cap.get(cv2.CAP_PROP_FPS)
    writer=None
    if output_video_path:
        fourcc=cv2.VideoWriter_fourcc(*'mp4v')
        writer=cv2.VideoWriter(output_video_path,fourcc,fps,(w,h))

    rep_count=0;stage=None;gf=0
    elbow_angles=[];neck_angles=[];orientations=[]

    def angle(a,b,c):
        a,b,c=np.array(a),np.array(b),np.array(c)
        ang=abs(np.degrees(np.arctan2(c[1]-b[1],c[0]-b[0])-
                           np.arctan2(a[1]-b[1],a[0]-b[0])))
        return 360-ang if ang>180 else ang

    while True:
        ret,frame=cap.read()
        if not ret:break
        img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res=pose_tracker.process(img);out=frame.copy()
        if res.pose_landmarks:
            gf+=1;lm=res.pose_landmarks.landmark
            orientations.append(detect_orientation(lm))
            def nc(i):return[lm[i].x,lm[i].y]
            # elbow flexion
            el=angle(nc(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
                     nc(mp_pose.PoseLandmark.LEFT_ELBOW.value),
                     nc(mp_pose.PoseLandmark.LEFT_WRIST.value))
            elbow_angles.append(el)
            # neck
            neck=angle(nc(mp_pose.PoseLandmark.LEFT_EAR.value),
                       nc(mp_pose.PoseLandmark.NOSE.value),
                       nc(mp_pose.PoseLandmark.RIGHT_EAR.value))
            neck_angles.append(neck)
            mp_drawing.draw_landmarks(out,res.pose_landmarks,mp_pose.POSE_CONNECTIONS)
            cv2.putText(out,f"Elbow: {el:.1f}°",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            # rep: down when el>160, up when el<50
            if el>160 and stage=="up":
                stage="down"
            elif el<50 and(stage=="down" or stage is None):
                stage="up";rep_count+=1;print(f"Pull-Up #{rep_count}")
        cv2.putText(out,f"Reps: {rep_count}",(10,h-30),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
        if writer:writer.write(out)
    cap.release(); writer and writer.release() if writer else None

    if gf<10:return{"rep_count":0,"error":"Too few detections."}
    fa={
     "min_elbow_deg":min(elbow_angles),
     "max_elbow_deg":max(elbow_angles),
     "avg_neck_deg" :sum(neck_angles)/len(neck_angles),
     "dominant_view":max(set(orientations),key=orientations.count),
     "frames_analyzed":gf
    }
    fb=[]
    if fa["max_elbow_deg"]<160:fb.append("Lower fully—arms straight >160°.")
    if fa["min_elbow_deg"]>50:fb.append("Pull higher—chin over bar (<50° elbow).")
    if fa["avg_neck_deg"]<170:fb.append("Keep neck neutral.")
    v=fa["dominant_view"]
    if v=="front":fb.append("Front: watch even left/right pull.")
    elif v=="left":fb.append("Left: ensure left side pulls evenly.")
    else:fb.append("Right: ensure right side pulls evenly.")
    if not fb:fb.append("Great pull-up form!")
    return{"rep_count":rep_count,"form_analysis":fa,"feedback":fb}



