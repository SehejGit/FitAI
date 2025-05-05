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

def analyze_russian_twist(video_path, output_video_path=None):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Russian twist variables
    twist_count = 0
    last_direction = None
    twist_state = "none"  # Track the twist state: none, left, center, right
    
    # Metrics to track
    torso_angles = []
    hip_positions = []
    orientations = []
    v_positions = []  # V-position (legs raised)
    
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
            
            # Calculate midpoints
            mid_shoulder = [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2]
            mid_hip = [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2]
            
            # Calculate metrics for Russian twist
            # 1. Torso angle (vertical to torso)
            vertical = [mid_shoulder[0], 0]  # Point directly above mid_shoulder
            torso_angle = calculate_angle(vertical, mid_shoulder, mid_hip)
            torso_angles.append(torso_angle)
            
            # 2. V-position check (knees raised)
            knee_height = (left_knee[1] + right_knee[1])/2
            hip_height = (left_hip[1] + right_hip[1])/2
            v_position = knee_height < hip_height
            v_positions.append(v_position)
            
            # 3. Rotation detection (for counting twists)
            shoulder_vector = [right_shoulder[0] - left_shoulder[0], right_shoulder[1] - left_shoulder[1]]
            shoulder_angle = math.degrees(math.atan2(shoulder_vector[1], shoulder_vector[0]))
            
            # Determine twist direction with clear thresholds
            if abs(shoulder_angle) < 10:  # Smaller threshold for center position
                current_direction = "center"
            elif shoulder_angle > 20:  # Threshold for right twist
                current_direction = "right"
            elif shoulder_angle < -20:  # Threshold for left twist
                current_direction = "left"
            else:
                current_direction = last_direction  # Maintain previous direction if in transition
            
            # State machine for counting complete twists
            if twist_state == "none":
                if current_direction == "left":
                    twist_state = "left"
                elif current_direction == "right":
                    twist_state = "right"
            elif twist_state == "left":
                if current_direction == "center":
                    twist_state = "left_center"
                elif current_direction == "right":
                    # Missed center, but still count it
                    twist_count += 1
                    twist_state = "right"
            elif twist_state == "left_center":
                if current_direction == "right":
                    twist_count += 1
                    twist_state = "right"
                elif current_direction == "left":
                    twist_state = "left"  # Went back to left
            elif twist_state == "right":
                if current_direction == "center":
                    twist_state = "right_center"
                elif current_direction == "left":
                    # Missed center, but still count it
                    twist_count += 1
                    twist_state = "left"
            elif twist_state == "right_center":
                if current_direction == "left":
                    twist_count += 1
                    twist_state = "left"
                elif current_direction == "right":
                    twist_state = "right"  # Went back to right
            
            last_direction = current_direction

            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # Display metrics
            cv2.putText(annotated_image, f"Twist Count: {twist_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_image, f"Torso Angle: {torso_angle:.1f}°", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_image, f"V-Position: {'Yes' if v_position else 'No'}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_image, f"Direction: {current_direction}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_image, f"State: {twist_state}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

        if output_video_path:
            out.write(annotated_image)

    cap.release()
    if output_video_path:
        out.release()
    
    # Calculate averages and prepare feedback
    if len(torso_angles) < 10:
        return {
            "twist_count": 0,
            "error": "Not enough valid pose detections. Check video quality and positioning."
        }

    avg_torso_angle = sum(torso_angles) / len(torso_angles)
    v_position_percent = sum(1 for v in v_positions if v) / len(v_positions) * 100
    dominant_view = max(set(orientations), key=orientations.count) if orientations else "unknown"
    
    min_torso_angle = min(torso_angles)
    max_torso_angle = max(torso_angles)
    angle_variation = max_torso_angle - min_torso_angle
    
    # Calculate speed metrics (twists per second)
    video_duration = len(torso_angles) / fps
    twists_per_second = twist_count / video_duration if video_duration > 0 else 0

    feedback = {
        "twist_count": twist_count,
        "form_analysis": {
            "avg_torso_angle": avg_torso_angle,
            "min_torso_angle": min_torso_angle,
            "max_torso_angle": max_torso_angle,
            "angle_variation": angle_variation,
            "v_position_percent": v_position_percent,
            "dominant_view": dominant_view,
            "twists_per_second": twists_per_second,
            "video_duration_seconds": video_duration
        },
        "feedback": []
    }

    # 1. Torso angle feedback
    if avg_torso_angle < 30:
        feedback["feedback"].append("Lean back more (45-60° from vertical) to properly engage your core.")
    elif avg_torso_angle > 70:
        feedback["feedback"].append("Excellent torso angle! You're maintaining good lean for core engagement.")
    
    # 2. V-position feedback
    if v_position_percent < 50:
        feedback["feedback"].append("Keep your feet elevated throughout the exercise for better core engagement.")
    elif v_position_percent < 80:
        feedback["feedback"].append("Good leg elevation, but try to maintain it consistently throughout.")
    else:
        feedback["feedback"].append("Excellent leg elevation maintenance!")
    
    
    # 4. Speed feedback
    if twists_per_second > 0.8:
        feedback["feedback"].append("Slow down your twists for better control and muscle engagement.")
    elif twists_per_second < 0.3:
        feedback["feedback"].append("Consider increasing your pace slightly for better cardio benefits.")
    else:
        feedback["feedback"].append("Good pace - controlled movements with proper form.")
    
    # 5. Range of motion feedback
    if angle_variation < 30:
        feedback["feedback"].append("Increase your rotation range - try to touch the floor on each side.")
    elif angle_variation > 90:
        feedback["feedback"].append("Great range of motion in your twists!")
    
    # 6. General form feedback
    if twist_count > 0 and len(feedback["feedback"]) < 3:
        feedback["feedback"].append("Overall good form! Focus on controlled breathing during each twist.")

    return feedback
