import mediapipe as mp
import cv2
import numpy as np
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def analyze_pushup(video_path, output_video_path=None):
    """
    Analyzes pushup form from a video using MediaPipe pose detection.
    
    Args:
        video_path (str): Path to the input video file
        output_video_path (str, optional): Path where the analyzed video will be saved
        
    Returns:
        dict: Analysis results including pushup count, form metrics, and feedback
    """
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup output video writer if path is provided
    if output_video_path:
        # Changed from 'mp4v' to 'avc1' for MOV format
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Variables to track pushup state
    pushup_count = 0
    pushup_stage = None  # "up" or "down"
    frames_without_detection = 0
    good_frames = 0
    
    # Lists to store angles for analysis
    elbow_angles = []
    body_alignment_scores = []
    
    # Add debugging info
    print(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}")
    
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
            frames_without_detection = 0
            
            mp_drawing.draw_landmarks(
                annotated_image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
            
            landmarks = results.pose_landmarks.landmark
            
            # Debug output to see what landmarks are detected
            if good_frames == 1:
                print(f"Detected {len(landmarks)} landmarks")
            
            # Get key points for pushup analysis
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
            
            # Get right elbow angle (shoulder-elbow-wrist)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            elbow_angles.append(right_elbow_angle)
            
            # Convert normalized coordinates to pixel coordinates for visualization
            def normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height):
                x_px = min(math.floor(normalized_x * image_width), image_width - 1)
                y_px = min(math.floor(normalized_y * image_height), image_height - 1)
                return x_px, y_px
            
            # Visualize elbow angle
            r_shoulder_px = normalized_to_pixel_coordinates(right_shoulder[0], right_shoulder[1], frame_width, frame_height)
            r_elbow_px = normalized_to_pixel_coordinates(right_elbow[0], right_elbow[1], frame_width, frame_height)
            r_wrist_px = normalized_to_pixel_coordinates(right_wrist[0], right_wrist[1], frame_width, frame_height)
            
            # Draw angle text
            cv2.putText(annotated_image, 
                        f"Elbow: {right_elbow_angle:.1f}°",
                        (r_elbow_px[0] - 50, r_elbow_px[1] + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Body alignment check
            # Get hip, shoulder, and ankle points
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # Calculate body alignment
            hip_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
            hip_ankle_angle = calculate_angle(right_shoulder, right_hip, right_ankle)
            
            # Better alignment calculation
            # Perfect alignment would have both angles close to 180
            alignment_score = (180 - abs(hip_shoulder_angle - 180) - abs(hip_ankle_angle - 180))/180
            body_alignment_scores.append(alignment_score)
            
            # Visualize body alignment
            cv2.putText(annotated_image, 
                        f"Alignment: {alignment_score*100:.1f}%",
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Determine pushup stage based on elbow angle
            if right_elbow_angle > 150 and pushup_stage == "down":
                pushup_stage = "up"
                # Draw pushup count
                cv2.putText(annotated_image, 'UP', (50, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # If we were in "up" position and now angle is < 90, we're in "down" position
            elif right_elbow_angle < 100 and (pushup_stage == "up" or pushup_stage is None):
                pushup_stage = "down"
                pushup_count += 1
                cv2.putText(annotated_image, 'DOWN', (50, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                print(f"Pushup #{pushup_count} detected at frame with elbow angle {right_elbow_angle:.1f}")
        else:
            frames_without_detection += 1
            if frames_without_detection % 30 == 0:
                print(f"No pose detection for {frames_without_detection} frames")
                
        # Display pushup count
        cv2.putText(annotated_image, f'Pushups: {pushup_count}', (10, 100), 
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
            "pushup_count": 0,
            "error": "Not enough valid pose detections. Check video quality and positioning."
        }
        
    # Analyze the data collected
    if elbow_angles:
        avg_elbow_angle_bottom = min(elbow_angles)
        avg_elbow_angle_top = max(elbow_angles)
    else:
        avg_elbow_angle_bottom = 0
        avg_elbow_angle_top = 0
        
    avg_alignment = sum(body_alignment_scores) / max(len(body_alignment_scores), 1)
    
    # Generate feedback
    feedback = {
        "pushup_count": pushup_count,
        "form_analysis": {
            "elbow_angle_at_bottom": avg_elbow_angle_bottom,
            "elbow_angle_at_top": avg_elbow_angle_top,
            "body_alignment_score": avg_alignment * 100,
            "frames_analyzed": good_frames
        },
        "feedback": []
    }
    
    # Add specific feedback based on measurements
    if avg_elbow_angle_bottom > 100:
        feedback["feedback"].append("You're not going deep enough. Try to lower your body until your elbows are at 90 degrees.")
    
    if avg_elbow_angle_top < 150:
        feedback["feedback"].append("You're not fully extending your arms at the top of the pushup.")
    
    if avg_alignment < 0.8:
        feedback["feedback"].append("Keep your body straighter throughout the movement. Your hips are sagging or piking.")
    
    if not feedback["feedback"]:
        feedback["feedback"].append("Great form! Your pushups have good depth and body alignment.")
        
    return feedback