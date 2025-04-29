import mediapipe as mp
import cv2
import numpy as np
import math

def analyze_bicep_curl(video_path, output_video_path=None):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup output video writer if path is provided
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Variables to track bicep curl state
    curl_count = 0
    curl_stage = None  # "up" or "down"
    good_frames = 0
    
    # Lists to store angles and positions for analysis
    elbow_angles = []
    shoulder_stability = []
    hip_stability = []
    wrist_angles = []
    rep_depths = []  # To track curl depth
    
    # Calculate angle function
    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle
    
    # Convert normalized coordinates to pixel coordinates
    def normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height):
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
            
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        # Draw pose landmarks on the image
        annotated_image = image.copy()
        
        if results.pose_landmarks:
            good_frames += 1
            
            mp_drawing.draw_landmarks(
                annotated_image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
            
            landmarks = results.pose_landmarks.landmark
            
            # Get key points for curl analysis
            # We'll focus on right arm for simplicity
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            
            # Get elbow angle (shoulder-elbow-wrist)
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            elbow_angles.append(elbow_angle)
            
            # Track shoulder stability - shoulder should remain still relative to hip
            shoulder_hip_distance = np.sqrt((shoulder[0] - hip[0])**2 + (shoulder[1] - hip[1])**2)
            shoulder_stability.append(shoulder_hip_distance)
            
            # Measure wrist angle relative to forearm
            wrist_deviation = abs(elbow_angle - 180) if elbow_angle > 90 else elbow_angle
            wrist_angles.append(wrist_deviation)
            
            # Visualize elbow angle
            shoulder_px = normalized_to_pixel_coordinates(shoulder[0], shoulder[1], frame_width, frame_height)
            elbow_px = normalized_to_pixel_coordinates(elbow[0], elbow[1], frame_width, frame_height)
            wrist_px = normalized_to_pixel_coordinates(wrist[0], wrist[1], frame_width, frame_height)
            
            # Draw angle text
            cv2.putText(annotated_image, 
                        f"Elbow: {elbow_angle:.1f}°",
                        (elbow_px[0] - 50, elbow_px[1] + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Determine curl stage based on elbow angle
            if elbow_angle > 160 and curl_stage == "up":
                curl_stage = "down"
                # Draw curl stage
                cv2.putText(annotated_image, 'DOWN', (50, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # If we were in "down" position and now angle is < 60, we're in "up" position
            elif elbow_angle < 60 and (curl_stage == "down" or curl_stage is None):
                curl_stage = "up"
                curl_count += 1
                rep_depths.append(elbow_angle)  # Store depth of curl
                cv2.putText(annotated_image, 'UP', (50, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
        # Display curl count
        cv2.putText(annotated_image, f'Curls: {curl_count}', (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    
        # Write frame to output video
        if output_video_path:
            out.write(annotated_image)
                
    cap.release()
    if output_video_path:
        out.release()
    
    # Only analyze if we have enough valid frames
    if good_frames < 10:
        return {
            "curl_count": 0,
            "error": "Not enough valid pose detections. Check video quality and positioning."
        }
        
    # Calculate metrics for analysis
    avg_shoulder_stability = np.std(shoulder_stability) * 100  # Convert to percentage variation
    avg_wrist_deviation = np.mean(wrist_angles)
    avg_curl_depth = np.mean(rep_depths) if rep_depths else 0
    full_extension = np.max(elbow_angles) > 150
    
    # Generate feedback
    feedback = {
        "curl_count": curl_count,
        "form_analysis": {
            "curl_depth": avg_curl_depth,
            "full_extension": full_extension,
            "shoulder_stability": 100 - avg_shoulder_stability,  # Higher is better
            "wrist_stability": 100 - avg_wrist_deviation,  # Higher is better
            "frames_analyzed": good_frames
        },
        "feedback": []
    }
    
    # Add specific feedback based on measurements
    if avg_curl_depth > 45:
        feedback["feedback"].append("You're not curling the weight fully to your shoulder. Try to bring the weight closer to your shoulder.")
    
    if not full_extension:
        feedback["feedback"].append("You're not fully extending your arm at the bottom of the curl. Straighten your arm more for full range of motion.")
    
    if avg_shoulder_stability > 10:  # If shoulder position varies too much
        feedback["feedback"].append("Keep your upper arm and shoulder more stable during the curl. Avoid swinging.")
    
    if avg_wrist_deviation > 30:
        feedback["feedback"].append("Keep your wrist straight throughout the movement. Avoid bending your wrist.")
    
    if not feedback["feedback"]:
        feedback["feedback"].append("Great form! Your bicep curls show good depth, stable shoulders, and proper wrist position.")
        
    return feedback