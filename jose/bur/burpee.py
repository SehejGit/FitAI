import mediapipe as mp
import cv2
import numpy as np
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def analyze_burpee(video_path, output_video_path=None):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Burpee variables
    burpee_count = 0
    burpee_state = "standing"  # standing -> squat -> plank -> squat -> jump -> standing
    state_start_time = 0
    frame_count = 0
    
    # Track metrics for form feedback
    jump_heights = []
    plank_durations = []
    squat_depths = []
    
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
            
        frame_count += 1
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        annotated_image = image.copy()

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Get key points
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                   landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
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
            
            # Calculate midpoints
            mid_shoulder = [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2]
            mid_hip = [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2]
            mid_ankle = [(left_ankle[0] + right_ankle[0])/2, (left_ankle[1] + right_ankle[1])/2]
            
            # Calculate metrics for positions
            # 1. Body angle (vertical = standing, horizontal = plank)
            vertical_angle = calculate_angle([mid_shoulder[0], 0], mid_shoulder, mid_hip)
            
            # 2. Knee angle (straight = standing/plank, bent = squat)
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            knee_angle = (left_knee_angle + right_knee_angle) / 2
            
            # 3. Hip height (for squat depth and jump detection)
            hip_height = mid_hip[1]
            ankle_height = mid_ankle[1]
            
            # 4. Wrist position relative to shoulders (for plank detection)
            wrist_shoulder_y_diff = ((left_wrist[1] + right_wrist[1])/2) - ((left_shoulder[1] + right_shoulder[1])/2)
            
            # Detect positions
            is_standing = vertical_angle < 30 and knee_angle > 160 and hip_height < 0.7
            is_squat = knee_angle < 120 and hip_height > 0.6
            is_plank = vertical_angle > 60 and knee_angle > 160 and wrist_shoulder_y_diff < 0.15
            is_jump = hip_height < 0.5 and ankle_height < 0.75  # Both hips and ankles are high (in the air)
            
            # Record metrics for feedback
            if is_squat:
                squat_depths.append(hip_height)
            if is_jump:
                jump_heights.append(1.0 - hip_height)  # Normalize so higher value = higher jump
            
            # State machine for burpee detection
            if burpee_state == "standing":
                if is_squat:
                    burpee_state = "squat_down"
                    state_start_time = frame_count
            
            elif burpee_state == "squat_down":
                if is_plank:
                    burpee_state = "plank"
                    state_start_time = frame_count
                elif is_standing:  # Reset if they stand back up without completing
                    burpee_state = "standing"
            
            elif burpee_state == "plank":
                plank_duration = frame_count - state_start_time
                plank_durations.append(plank_duration)
                
                if is_squat:
                    burpee_state = "squat_up"
                    state_start_time = frame_count
                elif is_standing:  # Skip squat and go straight to standing
                    burpee_state = "standing"
            
            elif burpee_state == "squat_up":
                if is_jump:
                    burpee_state = "jump"
                    state_start_time = frame_count
                elif is_standing:  # Skip jump
                    burpee_state = "standing"
                    burpee_count += 1  # Still count as a rep, but will note in feedback
            
            elif burpee_state == "jump":
                if is_standing:
                    burpee_state = "standing"
                    burpee_count += 1
                    state_start_time = frame_count

            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # Display metrics
            cv2.putText(annotated_image, f"Burpee Count: {burpee_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_image, f"State: {burpee_state}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_image, f"Vertical Angle: {vertical_angle:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_image, f"Knee Angle: {knee_angle:.1f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        if output_video_path:
            out.write(annotated_image)

    cap.release()
    if output_video_path:
        out.release()

    # Calculate averages and prepare feedback
    if frame_count < 10:
        return {
            "burpee_count": 0,
            "error": "Not enough valid frames. Check video quality and positioning."
        }

    avg_squat_depth = sum(squat_depths) / len(squat_depths) if squat_depths else 0
    avg_jump_height = sum(jump_heights) / len(jump_heights) if jump_heights else 0
    avg_plank_duration = sum(plank_durations) / len(plank_durations) if plank_durations else 0
    avg_plank_duration_seconds = avg_plank_duration / fps if fps > 0 else 0

    feedback = {
        "burpee_count": burpee_count,
        "form_analysis": {
            "avg_squat_depth": avg_squat_depth,
            "avg_jump_height": avg_jump_height,
            "avg_plank_duration_seconds": avg_plank_duration_seconds
        },
        "feedback": []
    }

    # Generate feedback
    if avg_squat_depth < 0.65:
        feedback["feedback"].append("Deepen your squat for a full range of motion.")
    if avg_jump_height < 0.2:
        feedback["feedback"].append("Push through your heels for a more explosive jump.")
    if avg_plank_duration_seconds < 0.5:
        feedback["feedback"].append("Ensure a complete plank position in the middle of each burpee.")
    if burpee_count < 3:
        feedback["feedback"].append("Focus on completing the full sequence: squat, plank, squat, jump.")

    return feedback
