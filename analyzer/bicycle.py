import mediapipe as mp
import cv2
import numpy as np
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def analyze_bicycle_crunch(video_path, output_video_path=None):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Bicycle crunch variables
    crunch_count = 0
    twist_state = "none"  # Track the twist state: none, left, center, right
    last_direction = None
    
    # Metrics to track
    elbow_knee_distances = []
    leg_extensions = []
    shoulder_elevations = []
    torso_rotations = []
    neck_positions = []
    breath_patterns = []
    hip_stability = []
    
    print(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}")

    def calculate_distance(point1, point2):
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        annotated_image = image.copy()

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Get key points
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
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
            
            # 1. Check if shoulders are elevated (off the ground)
            shoulder_height = (left_shoulder[1] + right_shoulder[1])/2
            hip_height = (left_hip[1] + right_hip[1])/2
            shoulder_elevated = shoulder_height < hip_height - 0.05  # Shoulders should be slightly raised
            shoulder_elevations.append(shoulder_elevated)
            
            # 2. Calculate distances between elbows and opposite knees
            left_elbow_right_knee_dist = calculate_distance(left_elbow, right_knee)
            right_elbow_left_knee_dist = calculate_distance(right_elbow, left_knee)
            min_elbow_knee_dist = min(left_elbow_right_knee_dist, right_elbow_left_knee_dist)
            elbow_knee_distances.append(min_elbow_knee_dist)
            
            # 3. Check leg extension (one leg should be extended while the other is bent)
            left_leg_extended = calculate_distance(left_hip, left_ankle) > calculate_distance(left_hip, left_knee) * 1.5
            right_leg_extended = calculate_distance(right_hip, right_ankle) > calculate_distance(right_hip, right_knee) * 1.5
            leg_extension = left_leg_extended or right_leg_extended
            leg_extensions.append(leg_extension)


            shoulder_hip_angle = math.atan2(mid_shoulder[1] - mid_hip[1], mid_shoulder[0] - mid_hip[0])
            proper_rotation = abs(shoulder_hip_angle) > 0.05  # Should rotate sufficiently
            torso_rotations.append(proper_rotation)

            # Check neck position
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            neck_neutral = abs(nose[0] - mid_shoulder[0]) < 0.1  # Neck should be neutral
            neck_positions.append(neck_neutral)

            # Check hip stability
            hip_stable = abs(left_hip[0] - right_hip[0]) < 0.15  # Limited side-to-side movement
            hip_stability.append(hip_stable)
            
            # Determine crunch direction based on which elbow is closer to opposite knee
            if left_elbow_right_knee_dist < right_elbow_left_knee_dist:
                current_direction = "left"  # Left elbow to right knee
            else:
                current_direction = "right"  # Right elbow to left knee
                
            # State machine for counting complete bicycle crunches
            if twist_state == "none":
                if current_direction == "left" and min_elbow_knee_dist < 0.2 and leg_extension:
                    twist_state = "left"
                elif current_direction == "right" and min_elbow_knee_dist < 0.2 and leg_extension:
                    twist_state = "right"
            elif twist_state == "left":
                if current_direction == "right" and min_elbow_knee_dist < 0.2 and leg_extension:
                    crunch_count += 1
                    twist_state = "right"
            elif twist_state == "right":
                if current_direction == "left" and min_elbow_knee_dist < 0.2 and leg_extension:
                    crunch_count += 1
                    twist_state = "left"
            
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
            cv2.putText(annotated_image, f"Crunch Count: {crunch_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_image, f"Direction: {current_direction}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_image, f"State: {twist_state}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_image, f"Shoulders Up: {'Yes' if shoulder_elevated else 'No'}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_image, f"Leg Extended: {'Yes' if leg_extension else 'No'}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        if output_video_path:
            out.write(annotated_image)

    cap.release()
    if output_video_path:
        out.release()

    # Calculate averages and prepare feedback
    if len(shoulder_elevations) < 10:
        return {
            "crunch_count": 0,
            "error": "Not enough valid pose detections. Check video quality and positioning."
        }

    shoulder_elevation_percent = sum(1 for s in shoulder_elevations if s) / len(shoulder_elevations) * 100
    leg_extension_percent = sum(1 for l in leg_extensions if l) / len(leg_extensions) * 100
    avg_elbow_knee_dist = sum(elbow_knee_distances) / len(elbow_knee_distances)

    feedback = {
        "crunch_count": crunch_count,
        "form_analysis": {
            "shoulder_elevation_percent": shoulder_elevation_percent,
            "leg_extension_percent": leg_extension_percent,
            "avg_elbow_knee_dist": avg_elbow_knee_dist
        },
        "feedback": []
    }

    # Generate feedback
    if shoulder_elevation_percent < 10: #10 is the best?
        feedback["feedback"].append("Keep your shoulders lifted off the ground throughout the exercise.")
    if leg_extension_percent < 40:
        feedback["feedback"].append("Focus on fully extending one leg while bringing the other knee toward your chest.")
    if avg_elbow_knee_dist > 0.25:
        feedback["feedback"].append("Try to bring your elbow closer to the opposite knee for a more effective crunch.")
    if crunch_count < 5:
        feedback["feedback"].append("Maintain a steady rhythm, alternating sides in a pedaling motion.")

    if sum(1 for r in torso_rotations if r) / len(torso_rotations) * 100 < 50:
        feedback["feedback"].append("Increase your torso rotation to engage obliques more effectively.")
    
    if sum(1 for n in neck_positions if n) / len(neck_positions) * 100 < 60:
        feedback["feedback"].append("Keep your neck relaxed and chin slightly tucked to avoid strain.")
        
    if sum(1 for h in hip_stability if h) / len(hip_stability) * 100 < 60:
        feedback["feedback"].append("Stabilize your hips more to better isolate your core muscles.")
        
    feedback["feedback"].append("Tip: Remember to breathe out as you crunch and breathe in as you extend your legs.")

    return feedback
