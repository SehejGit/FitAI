import mediapipe as mp
import cv2
import numpy as np
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def detect_orientation(lm):
    """
    Determine whether subject is facing 'front', 'left', or 'right'.
    """
    nose = lm[mp_pose.PoseLandmark.NOSE.value]
    ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    mid_x = (ls.x + rs.x) / 2
    delta = nose.x - mid_x
    if abs(delta) < 0.05:
        return "front"
    return "right" if delta < 0 else "left"

def analyze_plank(video_path, output_video_path=None):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    plank_active = False
    start_time = None
    duration = 0
    frames_without_detection = 0
    good_frames = 0
    cooldown_frames = 0  # allows for brief loss of pose if user moves
    
    # Metrics to track
    alignment_scores = []
    hip_drops = []
    shoulder_heights = []
    orientations = []
    back_angles = []
    neck_angles = []

    print(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}")

    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        annotated_image = image.copy()

        if results.pose_landmarks:
            good_frames += 1
            frames_without_detection = 0
            cooldown_frames = 0

            landmarks = results.pose_landmarks.landmark
            orientations.append(detect_orientation(landmarks))

            # Get key points
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                   landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]

            # Calculate metrics
            # 1. Body alignment (shoulder-hip-ankle)
            left_alignment_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
            right_alignment_angle = calculate_angle(right_shoulder, right_hip, right_ankle)
            alignment_score = (180 - abs(left_alignment_angle - 180) + 180 - abs(right_alignment_angle - 180)) / 360
            alignment_scores.append(alignment_score)

            # 2. Hip drop (difference between shoulder and hip height)
            avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            avg_hip_y = (left_hip[1] + right_hip[1]) / 2
            hip_drop = (avg_hip_y - avg_shoulder_y) * frame_height  # in pixels
            hip_drops.append(hip_drop)

            # 3. Shoulder height difference (uneven shoulders)
            shoulder_height_diff = abs(left_shoulder[1] - right_shoulder[1]) * frame_height
            shoulder_heights.append(shoulder_height_diff)

            # 4. Back angle (shoulder-hip-knee, for sagging detection)
            left_back_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            right_back_angle = calculate_angle(right_shoulder, right_hip, right_knee)
            back_angles.append((left_back_angle + right_back_angle) / 2)

            # 5. Neck angle (ear-nose-ear)
            neck_angle = calculate_angle(left_ear, nose, right_ear)
            neck_angles.append(neck_angle)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # Display metrics
            cv2.putText(annotated_image, f"Alignment: {alignment_score * 100:.1f}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_image, f"Hip Drop: {hip_drop:.1f}px", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_image, f"Back Angle: {back_angles[-1]:.1f}°", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            # Determine if plank is active (shoulders above hips)
            if left_shoulder[1] < left_hip[1] and right_shoulder[1] < right_hip[1]:
                if not plank_active:
                    plank_active = True
                    start_time = cap.get(cv2.CAP_PROP_POS_MSEC)
                if start_time is not None:
                    current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
                    duration = (current_time - start_time) / 1000  # seconds
            else:
                if plank_active and cooldown_frames < fps // 2:  # allow 0.5s grace
                    cooldown_frames += 1
                else:
                    plank_active = False
                    start_time = None
                    duration = 0
        else:
            frames_without_detection += 1
            if frames_without_detection % 30 == 0:
                print(f"No pose detection for {frames_without_detection} frames")
            if plank_active and cooldown_frames < fps // 2:
                cooldown_frames += 1
            else:
                plank_active = False
                start_time = None
                duration = 0

        cv2.putText(annotated_image, f'Duration: {duration:.1f}s', (10, frame_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        if output_video_path:
            out.write(annotated_image)

    cap.release()
    if output_video_path:
        out.release()

    if good_frames < 10:
        return {
            "duration": 0,
            "error": "Not enough valid pose detections. Check video quality and positioning."
        }

    # Calculate averages
    avg_alignment = sum(alignment_scores) / len(alignment_scores)
    avg_hip_drop = sum(hip_drops) / len(hip_drops)
    avg_shoulder_diff = sum(shoulder_heights) / len(shoulder_heights)
    avg_back_angle = sum(back_angles) / len(back_angles)
    avg_neck_angle = sum(neck_angles) / len(neck_angles)
    dominant_view = max(set(orientations), key=orientations.count) if orientations else "unknown"

    # Prepare feedback
    feedback = {
        "duration": duration,
        "form_analysis": {
            "avg_body_alignment_score": avg_alignment * 100,
            "avg_hip_drop_px": avg_hip_drop,
            "avg_shoulder_height_diff_px": avg_shoulder_diff,
            "avg_back_angle": avg_back_angle,
            "avg_neck_angle": avg_neck_angle,
            "dominant_view": dominant_view,
            "frames_analyzed": good_frames
        },
        "feedback": []
    }

    # General feedback
    if avg_alignment < 0.85:
        feedback["feedback"].append("Maintain a straighter line from shoulders to ankles. Avoid sagging or arching your back.")
    if avg_hip_drop > 0.1 * frame_height:  # More than 10% of frame height
        feedback["feedback"].append("Engage your core to prevent your hips from sagging.")
    if avg_shoulder_diff > 0.05 * frame_height:
        feedback["feedback"].append("Keep your shoulders level - one side appears higher than the other.")
    if avg_back_angle < 160:
        feedback["feedback"].append("Keep your back straighter - you may be rounding your shoulders or arching your lower back.")
    if avg_neck_angle < 160:
        feedback["feedback"].append("Keep your neck in line with your spine - avoid looking forward, dropping your head, or any movements.")

    # View-specific feedback
    if dominant_view == "front":
        feedback["feedback"].append("Front view: Ensure your hands are shoulder-width apart and body is straight.")
    elif dominant_view == "left":
        feedback["feedback"].append("Left side view: Check that your hips aren't sagging and your shoulders are directly above your elbows.")
    elif dominant_view == "right":
        feedback["feedback"].append("Right side view: Check that your hips aren't sagging and your shoulders are directly above your elbows.")

    return feedback