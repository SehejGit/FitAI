import mediapipe as mp
import cv2
import numpy as np
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

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
    alignment_scores = []
    cooldown_frames = 0  # allows for brief loss of pose if user moves

    print(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}")

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

            # Only draw body landmarks (not face)
            body_landmarks = [
                mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE
            ]
            # Draw only body landmarks and connections
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            landmarks = results.pose_landmarks.landmark

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            def calculate_angle(a, b, c):
                a = np.array(a)
                b = np.array(b)
                c = np.array(c)
                radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                angle = np.abs(radians * 180.0 / np.pi)
                if angle > 180.0:
                    angle = 360 - angle
                return angle

            # Check body alignment (shoulder-hip-ankle angle should be close to 180)
            left_alignment_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
            right_alignment_angle = calculate_angle(right_shoulder, right_hip, right_ankle)
            alignment_score = (180 - abs(left_alignment_angle - 180) + 180 - abs(right_alignment_angle - 180)) / 360
            alignment_scores.append(alignment_score)

            cv2.putText(annotated_image, f"Alignment: {alignment_score * 100:.1f}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # Determine if plank is active (shoulders above hips, ignore feet)
            if left_shoulder[1] < left_hip[1] and right_shoulder[1] < right_hip[1]:
                if not plank_active:
                    plank_active = True
                    start_time = cap.get(cv2.CAP_PROP_POS_MSEC)
                if start_time is not None:
                    current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
                    duration = (current_time - start_time) / 1000  # seconds
            else:
                # Instead of immediately resetting, allow a short cooldown
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
            # Allow brief detection loss
            if plank_active and cooldown_frames < fps // 2:
                cooldown_frames += 1
            else:
                plank_active = False
                start_time = None
                duration = 0

        cv2.putText(annotated_image, f'Duration: {duration:.1f}s', (10, 70),
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

    avg_alignment = sum(alignment_scores) / max(len(alignment_scores), 1)

    feedback = {
        "duration": duration,
        "form_analysis": {
            "avg_body_alignment_score": avg_alignment * 100,
            "frames_analyzed": good_frames
        },
        "feedback": []
    }

    if avg_alignment < 0.85:
        feedback["feedback"].append("Maintain a straighter line from your shoulders to your ankles. Avoid sagging or arching your back.")

    if not feedback["feedback"]:
        feedback["feedback"].append("Great plank form! You held a good straight line.")

    return feedback
