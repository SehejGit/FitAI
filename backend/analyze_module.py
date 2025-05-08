import mediapipe as mp
import cv2
import numpy as np
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Convert normalized coordinates to pixel coordinates for visualization
def normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height):
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

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

def calculate_alignment(p1, p2, p3):
    """Calculate alignment of three points (how close they are to a straight line)"""
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    # Calculate vectors
    v1 = p2 - p1
    v2 = p3 - p2
    # Normalize vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    # Calculate dot product (1 = perfect alignment, -1 = opposite direction)
    alignment = np.dot(v1, v2)
    return alignment

def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def detect_orientation(lm):
    nose = lm[mp_pose.PoseLandmark.NOSE.value]
    ls   = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs   = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    mid_x = (ls.x + rs.x) / 2
    delta = nose.x - mid_x
    if abs(delta) < 0.05:
        return "front"
    return "right" if delta < 0 else "left"

def analyze_pushups(video_path, output_video_path=None):
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
            
            
            # Get right elbow angle (shoulder-elbow-wrist)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            elbow_angles.append(right_elbow_angle)
            
            
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

def analyze_squats(video_path, output_video_path=None):
    """
    Analyzes squat form from a video using MediaPipe pose detection.
    
    Args:
        video_path (str): Path to the input video file
        output_video_path (str, optional): Path where the analyzed video will be saved
        
    Returns:
        dict: Analysis results including squat count, form metrics, and feedback
    """
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup output video writer if path is provided
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Variables to track squat state
    squat_count = 0
    squat_stage = None  # "up" or "down"
    frames_without_detection = 0
    good_frames = 0
    
    # Lists to store angles for analysis
    knee_angles = []
    hip_angles = []
    ankle_angles = []
    back_alignment_scores = []
    
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
            
            # Get key points for squat analysis
            # Hip, knee, and ankle landmarks for squat form analysis
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            
            
            # Get knee angle (hip-knee-ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            knee_angles.append(right_knee_angle)
            
            # Get hip angle (shoulder-hip-knee)
            right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
            hip_angles.append(right_hip_angle)
            
            
            # Visualize knee angle
            r_hip_px = normalized_to_pixel_coordinates(right_hip[0], right_hip[1], frame_width, frame_height)
            r_knee_px = normalized_to_pixel_coordinates(right_knee[0], right_knee[1], frame_width, frame_height)
            r_ankle_px = normalized_to_pixel_coordinates(right_ankle[0], right_ankle[1], frame_width, frame_height)
            
            # Draw angle text
            cv2.putText(annotated_image, 
                        f"Knee: {right_knee_angle:.1f}°",
                        (r_knee_px[0] - 50, r_knee_px[1] + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(annotated_image, 
                        f"Hip: {right_hip_angle:.1f}°",
                        (r_hip_px[0] - 50, r_hip_px[1] + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Back alignment check - we want to ensure the back is straight during a squat
            # Get shoulder and hip points to check back angle relative to vertical
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            
            # Calculate torso angle relative to vertical (approximation)
            back_angle = abs(90 - calculate_angle([left_shoulder[0], 0], left_shoulder, left_hip))
            back_alignment = max(0, 1 - back_angle / 45)  # Higher is better, 1 is vertical
            back_alignment_scores.append(back_alignment)
            
            # Visualize back alignment
            cv2.putText(annotated_image, 
                        f"Back: {back_alignment*100:.1f}%",
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Determine squat stage based on knee angle
            if right_knee_angle > 160 and squat_stage == "down":
                squat_stage = "up"
                # Draw squat state
                cv2.putText(annotated_image, 'UP', (50, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # If we were in "up" position and now angle is < 110, we're in "down" position
            elif right_knee_angle < 110 and (squat_stage == "up" or squat_stage is None):
                squat_stage = "down"
                squat_count += 1
                cv2.putText(annotated_image, 'DOWN', (50, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                print(f"Squat #{squat_count} detected at frame with knee angle {right_knee_angle:.1f}")
        else:
            frames_without_detection += 1
            if frames_without_detection % 30 == 0:
                print(f"No pose detection for {frames_without_detection} frames")
                
        # Display squat count
        cv2.putText(annotated_image, f'Squats: {squat_count}', (10, 100), 
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
            "squat_count": 0,
            "error": "Not enough valid pose detections. Check video quality and positioning."
        }
        
    # Analyze the data collected
    if knee_angles:
        avg_knee_angle_bottom = min(knee_angles)
        avg_knee_angle_top = max(knee_angles)
    else:
        avg_knee_angle_bottom = 0
        avg_knee_angle_top = 0
        
    if hip_angles:
        avg_hip_angle_bottom = min(hip_angles)
    else:
        avg_hip_angle_bottom = 0
        
    avg_back_alignment = sum(back_alignment_scores) / max(len(back_alignment_scores), 1)
    
    # Generate feedback
    feedback = {
        "squat_count": squat_count,
        "form_analysis": {
            "knee_angle_at_bottom": avg_knee_angle_bottom,
            "knee_angle_at_top": avg_knee_angle_top,
            "hip_angle_at_bottom": avg_hip_angle_bottom,
            "back_alignment_score": avg_back_alignment * 100,
            "frames_analyzed": good_frames
        },
        "feedback": []
    }
    
    # Add specific feedback based on measurements
    if avg_knee_angle_bottom > 110:
        feedback["feedback"].append("You're not squatting deep enough. Try to lower your body until your thighs are parallel to the ground.")
    
    if avg_knee_angle_bottom < 80:
        feedback["feedback"].append("You're squatting too deep which could stress your knees. Aim for thighs parallel to the ground.")
    
    if avg_back_alignment < 0.7:
        feedback["feedback"].append("Keep your back straighter throughout the movement. You're leaning forward too much.")
    
    if avg_hip_angle_bottom > 100:
        feedback["feedback"].append("Focus on pushing your hips back more as you squat down.")
    
    if not feedback["feedback"]:
        feedback["feedback"].append("Great form! Your squats have good depth and your back alignment is excellent.")
        
    return feedback

def analyze_planks(video_path, output_video_path=None):
    """
    Analyzes plank form from a video using MediaPipe pose detection.
    
    Args:
        video_path (str): Path to the input video file
        output_video_path (str, optional): Path where the analyzed video will be saved
        
    Returns:
        dict: Analysis results including plank duration, form metrics, and feedback
    """
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup output video writer if path is provided
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Variables to track plank state
    plank_started = False
    plank_ended = False
    plank_start_frame = 0
    plank_end_frame = 0
    current_frame = 0
    frames_without_detection = 0
    good_frames = 0
    
    # Lists to store angles and alignment scores for analysis
    body_alignment_scores = []
    hip_height_scores = []
    
    # Add debugging info
    print(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
            
        current_frame += 1
            
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
            
            # Get key points for plank analysis
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            
            
            # Check for plank position - body should be straight
            # Shoulder to hip to ankle should form a relatively straight line
            body_angle = calculate_angle(right_shoulder, right_hip, right_ankle)
            body_alignment = max(0, 1 - abs(body_angle - 180) / 30)  # Higher is better, 1 is perfect alignment
            body_alignment_scores.append(body_alignment)
            
            # Check hip height - hips shouldn't sag or pike
            hip_knee_ankle_angle = calculate_angle(right_hip, right_knee, right_ankle)
            hip_height_score = max(0, 1 - abs(hip_knee_ankle_angle - 180) / 30)
            hip_height_scores.append(hip_height_score)
            
            # Visualize body alignment
            cv2.putText(annotated_image, 
                        f"Alignment: {body_alignment*100:.1f}%",
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.putText(annotated_image, 
                        f"Hip position: {hip_height_score*100:.1f}%",
                        (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Detect plank position
            # If both alignment scores are good, consider it a plank
            is_plank_position = body_alignment > 0.75 and hip_height_score > 0.75
            
            if is_plank_position and not plank_started:
                plank_started = True
                plank_start_frame = current_frame
                print(f"Plank started at frame {plank_start_frame}")
            
            # If we were in plank position but no longer are, mark the end
            if plank_started and not is_plank_position and not plank_ended:
                plank_ended = True
                plank_end_frame = current_frame
                print(f"Plank ended at frame {plank_end_frame}")
                
            # Display plank status
            status_text = "PLANK ACTIVE" if is_plank_position else "NOT IN PLANK"
            cv2.putText(annotated_image, status_text, (10, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        
            # Show duration if plank is active
            if plank_started and not plank_ended:
                duration_frames = current_frame - plank_start_frame
                duration_seconds = duration_frames / fps
                cv2.putText(annotated_image, f'Duration: {duration_seconds:.1f}s', (10, 140), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            frames_without_detection += 1
            if frames_without_detection % 30 == 0:
                print(f"No pose detection for {frames_without_detection} frames")
                
        # Write frame to output video
        if output_video_path:
            out.write(annotated_image)
                
    cap.release()
    if output_video_path:
        out.release()
    
    # Only analyze if we have enough valid frames
    if good_frames < 10:
        return {
            "plank_duration": 0,
            "error": "Not enough valid pose detections. Check video quality and positioning."
        }
    
    # Calculate plank duration
    if not plank_ended and plank_started:
        plank_end_frame = current_frame  # If video ended while still in plank
        
    plank_frames = plank_end_frame - plank_start_frame if plank_started else 0
    plank_duration = plank_frames / fps if plank_frames > 0 else 0
    
    # Get average alignment scores
    avg_body_alignment = sum(body_alignment_scores) / max(len(body_alignment_scores), 1)
    avg_hip_height = sum(hip_height_scores) / max(len(hip_height_scores), 1)
    
    # Generate feedback
    feedback = {
        "plank_duration": plank_duration,
        "form_analysis": {
            "body_alignment_score": avg_body_alignment * 100,
            "hip_position_score": avg_hip_height * 100,
            "frames_analyzed": good_frames
        },
        "feedback": []
    }
    
    # Add specific feedback based on measurements
    if avg_body_alignment < 0.8:
        feedback["feedback"].append("Work on keeping your body in a straight line from shoulders to ankles.")
    
    if avg_hip_height < 0.8:
        feedback["feedback"].append("Your hips are either sagging too low or piking too high. Keep them in line with your shoulders and ankles.")
    
    if plank_duration < 10:
        feedback["feedback"].append(f"Your plank duration was {plank_duration:.1f} seconds. Aim to hold for at least 30 seconds.")
    elif plank_duration < 30:
        feedback["feedback"].append(f"Your plank duration was {plank_duration:.1f} seconds. Good effort, but try to work up to 60 seconds.")
    
    if not feedback["feedback"]:
        feedback["feedback"].append(f"Excellent plank form and duration ({plank_duration:.1f} seconds)! Your body alignment is strong.")
        
    return feedback

def analyze_dumbbell_press(video_path, output_video_path=None):
    """
    Analyzes dumbbell press form from a video using MediaPipe pose detection.
    
    Args:
        video_path (str): Path to the input video file
        output_video_path (str, optional): Path where the analyzed video will be saved
        
    Returns:
        dict: Analysis results including rep count, form metrics, and feedback
    """
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup output video writer if path is provided
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Variables to track press state
    press_count = 0
    press_stage = None  # "up" or "down"
    frames_without_detection = 0
    good_frames = 0
    
    # Lists to store angles for analysis
    elbow_angles = []
    arm_symmetry_scores = []
    back_arch_scores = []
    
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
            
            # Get key points for dumbbell press analysis
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
            
            
            # Get elbow angles (shoulder-elbow-wrist)
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Average elbow angle for press detection
            avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
            elbow_angles.append(avg_elbow_angle)
            
            # Check arm symmetry (difference between left and right elbow angles)
            arm_symmetry = max(0, 1 - abs(left_elbow_angle - right_elbow_angle) / 20)  # Lower difference is better
            arm_symmetry_scores.append(arm_symmetry)
            
            
            # Visualize elbow angles
            l_elbow_px = normalized_to_pixel_coordinates(left_elbow[0], left_elbow[1], frame_width, frame_height)
            r_elbow_px = normalized_to_pixel_coordinates(right_elbow[0], right_elbow[1], frame_width, frame_height)
            
            # Draw angle text
            cv2.putText(annotated_image, 
                        f"L Elbow: {left_elbow_angle:.1f}°",
                        (l_elbow_px[0] - 50, l_elbow_px[1] + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                        
            cv2.putText(annotated_image, 
                        f"R Elbow: {right_elbow_angle:.1f}°",
                        (r_elbow_px[0] - 50, r_elbow_px[1] + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Check for back arch - use hip and shoulder landmarks
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            
            # Simplified back arch check - this would be better with side view
            back_arch = calculate_angle(
                [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2], 
                [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2],
                [0, 0]  # Arbitrary point above to get vertical angle
            )
            
            # Score for back arch - keep the back straight
            back_arch_score = max(0, 1 - abs(back_arch - 90) / 15)
            back_arch_scores.append(back_arch_score)
            
            # Visualize symmetry and back arch
            cv2.putText(annotated_image, 
                        f"Symmetry: {arm_symmetry*100:.1f}%",
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                        
            cv2.putText(annotated_image, 
                        f"Back: {back_arch_score*100:.1f}%",
                        (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            # Determine press stage based on elbow angle
            if avg_elbow_angle > 160 and press_stage == "down":
                press_stage = "up"
                # Draw press state
                cv2.putText(annotated_image, 'UP', (50, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # If we were in "up" position and now angle is < 90, we're in "down" position
            elif avg_elbow_angle < 100 and (press_stage == "up" or press_stage is None):
                press_stage = "down"
                press_count += 1
                cv2.putText(annotated_image, 'DOWN', (50, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                print(f"Press #{press_count} detected at frame with avg elbow angle {avg_elbow_angle:.1f}")
        else:
            frames_without_detection += 1
            if frames_without_detection % 30 == 0:
                print(f"No pose detection for {frames_without_detection} frames")
                
        # Display press count
        cv2.putText(annotated_image, f'Presses: {press_count}', (10, 100), 
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
            "press_count": 0,
            "error": "Not enough valid pose detections. Check video quality and positioning."
        }
        
    # Analyze the data collected
    if elbow_angles:
        avg_elbow_angle_bottom = min(elbow_angles)
        avg_elbow_angle_top = max(elbow_angles)
    else:
        avg_elbow_angle_bottom = 0
        avg_elbow_angle_top = 0
        
    avg_arm_symmetry = sum(arm_symmetry_scores) / max(len(arm_symmetry_scores), 1)
    avg_back_arch = sum(back_arch_scores) / max(len(back_arch_scores), 1)
    
    # Generate feedback
    feedback = {
        "press_count": press_count,
        "form_analysis": {
            "elbow_angle_at_bottom": avg_elbow_angle_bottom,
            "elbow_angle_at_top": avg_elbow_angle_top,
            "arm_symmetry_score": avg_arm_symmetry * 100,
            "back_position_score": avg_back_arch * 100,
            "frames_analyzed": good_frames
        },
        "feedback": []
    }
    
    # Add specific feedback based on measurements
    if avg_elbow_angle_bottom > 100:
        feedback["feedback"].append("You're not lowering the weights enough. Try to bring your elbows to at least 90 degrees.")
    
    if avg_elbow_angle_top < 160:
        feedback["feedback"].append("You're not fully extending your arms at the top of the press.")
    
    if avg_arm_symmetry < 0.8:
        feedback["feedback"].append("Try to keep your arms more symmetrical throughout the movement.")
    
    if avg_back_arch < 0.8:
        feedback["feedback"].append("Watch your back position - avoid excessive arching during the press.")
    
    if not feedback["feedback"]:
        feedback["feedback"].append("Great form! Your dumbbell press shows good range of motion and symmetry.")
        
    return feedback

def analyze_banded_squat(video_path, output_video_path=None):
    """
    Analyzes banded squat form from a video using MediaPipe pose detection.
    
    Args:
        video_path (str): Path to the input video file
        output_video_path (str, optional): Path where the analyzed video will be saved
        
    Returns:
        dict: Analysis results including squat count, form metrics, and feedback
    """
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup output video writer if path is provided
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Variables to track squat state
    squat_count = 0
    squat_stage = None  # "up" or "down"
    frames_without_detection = 0
    good_frames = 0
    
    # Lists to store angles for analysis
    knee_angles = []
    hip_angles = []
    knee_tracking_scores = []  # For knee alignment
    foot_width_scores = []     # For stance width
    
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
            
            # Get key points for banded squat analysis
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
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_foot = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            right_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
            
            
            # Get knee angles (hip-knee-ankle)
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            
            # Average knee angle for squat detection
            avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
            knee_angles.append(avg_knee_angle)
            
            # Get hip angles (shoulder-hip-knee)
            left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
            
            # Average hip angle for analysis
            avg_hip_angle = (left_hip_angle + right_hip_angle) / 2
            hip_angles.append(avg_hip_angle)
            
            
            # Visualize knee angles
            l_knee_px = normalized_to_pixel_coordinates(left_knee[0], left_knee[1], frame_width, frame_height)
            r_knee_px = normalized_to_pixel_coordinates(right_knee[0], right_knee[1], frame_width, frame_height)
            
            # Draw angle text
            cv2.putText(annotated_image, 
                        f"L Knee: {left_knee_angle:.1f}°",
                        (l_knee_px[0] - 50, l_knee_px[1] + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                        
            cv2.putText(annotated_image, 
                        f"R Knee: {right_knee_angle:.1f}°",
                        (r_knee_px[0] - 50, r_knee_px[1] + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Check knee tracking - knees should be aligned with toes
            # Calculate knee alignment relative to ankle and hip
            left_knee_tracking = calculate_angle(
                [left_hip[0], left_hip[1]], 
                [left_knee[0], left_knee[1]],
                [left_ankle[0], left_ankle[1]]
            )
            right_knee_tracking = calculate_angle(
                [right_hip[0], right_hip[1]], 
                [right_knee[0], right_knee[1]],
                [right_ankle[0], right_ankle[1]]
            )
            
            # Score for knee tracking - ideal is around 180 degrees (straight line from hip through knee to ankle)
            left_knee_tracking_score = max(0, 1 - abs(left_knee_tracking - 180) / 20)
            right_knee_tracking_score = max(0, 1 - abs(right_knee_tracking - 180) / 20)
            avg_knee_tracking_score = (left_knee_tracking_score + right_knee_tracking_score) / 2
            knee_tracking_scores.append(avg_knee_tracking_score)
            
            # Check foot width (wider stance is typically better for banded squats)
            foot_width = abs(right_foot[0] - left_foot[0])  # Normalized width between feet
            optimal_width = 0.25  # This would need calibration
            foot_width_score = max(0, 1 - abs(foot_width - optimal_width) / optimal_width)
            foot_width_scores.append(foot_width_score)
            
            # Visualize knee tracking and stance
            cv2.putText(annotated_image, 
                        f"Knee align: {avg_knee_tracking_score*100:.1f}%",
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                        
            cv2.putText(annotated_image, 
                        f"Stance: {foot_width_score*100:.1f}%",
                        (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Determine squat stage based on knee angle
            if avg_knee_angle > 160 and squat_stage == "down":
                squat_stage = "up"
                # Draw squat state
                cv2.putText(annotated_image, 'UP', (50, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # If we were in "up" position and now angle is < 120 (less deep due to band resistance)
            elif avg_knee_angle < 120 and (squat_stage == "up" or squat_stage is None):
                squat_stage = "down"
                squat_count += 1
                cv2.putText(annotated_image, 'DOWN', (50, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                print(f"Squat #{squat_count} detected at frame with avg knee angle {avg_knee_angle:.1f}")
        else:
            frames_without_detection += 1
            if frames_without_detection % 30 == 0:
                print(f"No pose detection for {frames_without_detection} frames")
                
        # Display squat count
        cv2.putText(annotated_image, f'Squats: {squat_count}', (10, 120), 
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
            "squat_count": 0,
            "error": "Not enough valid pose detections. Check video quality and positioning."
        }
        
    # Analyze the data collected
    if knee_angles:
        avg_knee_angle_bottom = min(knee_angles)
        avg_knee_angle_top = max(knee_angles)
    else:
        avg_knee_angle_bottom = 0
        avg_knee_angle_top = 0
        
    if hip_angles:
        avg_hip_angle_bottom = min(hip_angles)
    else:
        avg_hip_angle_bottom = 0
        
    avg_knee_tracking = sum(knee_tracking_scores) / max(len(knee_tracking_scores), 1)
    avg_foot_width = sum(foot_width_scores) / max(len(foot_width_scores), 1)
    
    # Generate feedback
    feedback = {
        "squat_count": squat_count,
        "form_analysis": {
            "knee_angle_at_bottom": avg_knee_angle_bottom,
            "knee_angle_at_top": avg_knee_angle_top,
            "hip_angle_at_bottom": avg_hip_angle_bottom,
            "knee_tracking_score": avg_knee_tracking * 100,
            "stance_width_score": avg_foot_width * 100,
            "frames_analyzed": good_frames
        },
        "feedback": []
    }
    
    # Add specific feedback based on measurements
    if avg_knee_angle_bottom > 120:
        feedback["feedback"].append("Try to squat deeper while maintaining band tension. Aim for thighs parallel to the ground.")
    
    if avg_knee_tracking < 0.8:
        feedback["feedback"].append("Keep your knees tracking over your toes throughout the movement.")
    
    if avg_foot_width < 0.7:
        feedback["feedback"].append("Consider using a wider stance for better stability with the resistance band.")
    
    if avg_hip_angle_bottom > 120:
        feedback["feedback"].append("Focus on pushing your hips back more as you squat down.")
    
    if not feedback["feedback"]:
        feedback["feedback"].append("Great form on your banded squats! Your knee tracking and stance width are excellent.")
        
    return feedback

def analyze_leg_press(video_path, output_video_path=None):
    """
    Analyzes leg press form from a video using MediaPipe pose detection.
    
    Args:
        video_path (str): Path to the input video file
        output_video_path (str, optional): Path where the analyzed video will be saved
        
    Returns:
        dict: Analysis results including rep count, form metrics, and feedback
    """
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup output video writer if path is provided
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Variables to track leg press state
    press_count = 0
    press_stage = None  # "extended" or "contracted"
    frames_without_detection = 0
    good_frames = 0
    
    # Lists to store angles for analysis
    knee_angles = []
    leg_symmetry_scores = []
    hip_position_scores = []
    
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
            
            # Get key points for leg press analysis
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
            
            
            # Get knee angles (hip-knee-ankle)
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            
            # Average knee angle for leg press detection
            avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
            knee_angles.append(avg_knee_angle)
            
            # Check leg symmetry (difference between left and right knee angles)
            leg_symmetry = max(0, 1 - abs(left_knee_angle - right_knee_angle) / 15)  # Lower difference is better
            leg_symmetry_scores.append(leg_symmetry)
            
            
            # Visualize knee angles
            l_knee_px = normalized_to_pixel_coordinates(left_knee[0], left_knee[1], frame_width, frame_height)
            r_knee_px = normalized_to_pixel_coordinates(right_knee[0], right_knee[1], frame_width, frame_height)
            
            # Draw angle text
            cv2.putText(annotated_image, 
                        f"L Knee: {left_knee_angle:.1f}°",
                        (l_knee_px[0] - 50, l_knee_px[1] + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                        
            cv2.putText(annotated_image, 
                        f"R Knee: {right_knee_angle:.1f}°",
                        (r_knee_px[0] - 50, r_knee_px[1] + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Check hip position - hips should remain on the seat (minimal vertical movement)
            # This is simplified since the actual position depends on the machine
            hip_y_position = (left_hip[1] + right_hip[1]) / 2
            
            # Store hip positions for later analysis
            if good_frames == 1:
                initial_hip_y = hip_y_position
            
            hip_movement = abs(hip_y_position - initial_hip_y) if good_frames > 1 else 0
            hip_position_score = max(0, 1 - hip_movement / 0.05)  # Smaller movement is better
            hip_position_scores.append(hip_position_score)
            
            # Visualize symmetry and hip position
            cv2.putText(annotated_image, 
                        f"Symmetry: {leg_symmetry*100:.1f}%",
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                        
            cv2.putText(annotated_image, 
                        f"Hip stability: {hip_position_score*100:.1f}%",
                        (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Determine leg press stage based on knee angle
            if avg_knee_angle > 150 and press_stage == "contracted":
                press_stage = "extended"
                # Draw press state
                cv2.putText(annotated_image, 'EXTENDED', (50, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # If we were in "extended" position and now angle is < 110, we're in "contracted" position
            elif avg_knee_angle < 110 and (press_stage == "extended" or press_stage is None):
                press_stage = "contracted"
                press_count += 1
                cv2.putText(annotated_image, 'CONTRACTED', (50, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                print(f"Leg press #{press_count} detected at frame with avg knee angle {avg_knee_angle:.1f}")
        else:
            frames_without_detection += 1
            if frames_without_detection % 30 == 0:
                print(f"No pose detection for {frames_without_detection} frames")
                
        # Display press count
        cv2.putText(annotated_image, f'Presses: {press_count}', (10, 120), 
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
            "press_count": 0,
            "error": "Not enough valid pose detections. Check video quality and positioning."
        }
        
    # Analyze the data collected
    if knee_angles:
        avg_knee_angle_contracted = min(knee_angles)
        avg_knee_angle_extended = max(knee_angles)
    else:
        avg_knee_angle_contracted = 0
        avg_knee_angle_extended = 0
        
    avg_leg_symmetry = sum(leg_symmetry_scores) / max(len(leg_symmetry_scores), 1)
    avg_hip_position = sum(hip_position_scores) / max(len(hip_position_scores), 1)
    
    # Generate feedback
    feedback = {
        "press_count": press_count,
        "form_analysis": {
            "knee_angle_at_contraction": avg_knee_angle_contracted,
            "knee_angle_at_extension": avg_knee_angle_extended,
            "leg_symmetry_score": avg_leg_symmetry * 100,
            "hip_stability_score": avg_hip_position * 100,
            "frames_analyzed": good_frames
        },
        "feedback": []
    }
    
    # Add specific feedback based on measurements
    if avg_knee_angle_contracted > 110:
        feedback["feedback"].append("Try to bring the weight closer to your body for a deeper contraction.")
    
    if avg_knee_angle_extended < 150:
        feedback["feedback"].append("Extend your legs more fully at the top of the movement, but avoid locking your knees.")
    
    if avg_leg_symmetry < 0.85:
        feedback["feedback"].append("Work on pushing evenly with both legs for better symmetry.")
    
    if avg_hip_position < 0.8:
        feedback["feedback"].append("Keep your hips firmly planted on the seat throughout the movement.")
    
    if not feedback["feedback"]:
        feedback["feedback"].append("Excellent leg press form! Your range of motion and stability are very good.")
        
    return feedback

def analyze_burpees(video_path, output_video_path=None):
    """
    Analyzes burpee form from a video using MediaPipe pose detection.
    
    Args:
        video_path (str): Path to the input video file
        output_video_path (str, optional): Path where the analyzed video will be saved
        
    Returns:
        dict: Analysis results including burpee count, form metrics, and feedback
    """
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup output video writer if path is provided
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Variables to track burpee state
    burpee_count = 0
    burpee_stage = "stand"  # "stand", "squat", "plank", "pushup", "squat_up", "jump"
    frames_without_detection = 0
    good_frames = 0
    
    # Lists to store metrics for analysis
    vertical_position_history = []  # To track body position up/down
    plank_quality_scores = []       # For plank position quality
    jump_height_scores = []         # For jump height
    
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
            
            # Get key points for burpee analysis
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                   landmarks[mp_pose.PoseLandmark.NOSE.value].y]
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
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            
            # Calculate key metrics for burpee analysis
            
            # 1. Vertical position of hips (normalized y-coordinate, higher value = lower position)
            hip_y = (left_hip[1] + right_hip[1]) / 2
            vertical_position_history.append(hip_y)
            
            # 2. Knee angles (hip-knee-ankle)
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
            
            # 3. Check for plank position
            # In plank, shoulders should be over wrists, and body should be straight
            shoulder_to_wrist_x_diff = abs((left_shoulder[0] + right_shoulder[0])/2 - (left_wrist[0] + right_wrist[0])/2)
            body_straight_angle = calculate_angle(
                [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2],
                [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2],
                [(left_ankle[0] + right_ankle[0])/2, (left_ankle[1] + right_ankle[1])/2]
            )
            plank_quality = (max(0, 1 - shoulder_to_wrist_x_diff / 0.2) + max(0, 1 - abs(body_straight_angle - 180) / 30)) / 2
            plank_quality_scores.append(plank_quality)
            
            
            # Detect burpee stages based on body position and angles
            
            # Track min/max hip positions to detect stages
            if len(vertical_position_history) > 10:
                recent_positions = vertical_position_history[-10:]
                min_recent_pos = min(recent_positions)
                max_recent_pos = max(recent_positions)
                
                # Standing position
                is_standing = hip_y < 0.6 and avg_knee_angle > 160
                
                # Squat position
                is_squatting = hip_y > 0.6 and avg_knee_angle < 120
                
                # Plank/pushup position
                is_plank = shoulder_to_wrist_x_diff < 0.2 and body_straight_angle > 150 and hip_y > 0.5
                
                # Jump detection (sudden upward movement of hips)
                is_jumping = False
                if len(vertical_position_history) > 5:
                    if vertical_position_history[-5] - vertical_position_history[-1] > 0.1:  # Upward movement
                        is_jumping = True
                        # Calculate jump height based on hip movement
                        jump_height = (vertical_position_history[-5] - min(vertical_position_history[-5:]))
                        jump_height_scores.append(jump_height)
                
                # State machine for burpee stages
                if burpee_stage == "stand" and is_squatting:
                    burpee_stage = "squat"
                    print("Detected squat position")
                
                elif burpee_stage == "squat" and is_plank:
                    burpee_stage = "plank"
                    print("Detected plank position")
                
                elif burpee_stage == "plank" and is_squatting:
                    burpee_stage = "squat_up"
                    print("Detected squat up position")
                
                elif burpee_stage == "squat_up" and is_jumping:
                    burpee_stage = "jump"
                    print("Detected jump")
                
                elif burpee_stage == "jump" and is_standing:
                    burpee_stage = "stand"
                    burpee_count += 1
                    print(f"Completed burpee #{burpee_count}")
            
            # Visualize metrics
            cv2.putText(annotated_image, 
                        f"Stage: {burpee_stage.upper()}",
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                        
            if burpee_stage == "plank":
                cv2.putText(annotated_image, 
                            f"Plank quality: {plank_quality*100:.1f}%",
                            (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            if burpee_stage == "jump" and jump_height_scores:
                cv2.putText(annotated_image, 
                            f"Jump height: {jump_height_scores[-1]*100:.1f}",
                            (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            frames_without_detection += 1
            if frames_without_detection % 30 == 0:
                print(f"No pose detection for {frames_without_detection} frames")
                
        # Display burpee count
        cv2.putText(annotated_image, f'Burpees: {burpee_count}', (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    
        # Write frame to output video
        if output_video_path:
            out.write(annotated_image)
                
    cap.release()
    if output_video_path:
        out.release()
    
    # Only analyze if we have enough valid frames
    if good_frames < 30:  # Burpees need more frames for full analysis
        return {
            "burpee_count": 0,
            "error": "Not enough valid pose detections. Check video quality and positioning."
        }
    
    # Calculate average metrics
    avg_plank_quality = sum(plank_quality_scores) / max(len(plank_quality_scores), 1)
    
    # Calculate jump height metrics if jumps were detected
    avg_jump_height = sum(jump_height_scores) / max(len(jump_height_scores), 1) if jump_height_scores else 0
    
    # Generate feedback
    feedback = {
        "burpee_count": burpee_count,
        "form_analysis": {
            "plank_quality_score": avg_plank_quality * 100,
            "avg_jump_height": avg_jump_height * 100,
            "frames_analyzed": good_frames
        },
        "feedback": []
    }
    
    # Add specific feedback based on measurements
    if avg_plank_quality < 0.7:
        feedback["feedback"].append("Work on your plank position during burpees. Keep your shoulders over your wrists and body in a straight line.")
    
    if avg_jump_height < 0.05:
        feedback["feedback"].append("Try to add more explosive power to your jumps at the end of each burpee.")
    
    if burpee_count < 3 and good_frames > 300:  # If few burpees over many frames
        feedback["feedback"].append("Consider increasing your pace to get more benefit from the exercise.")
    
    if not feedback["feedback"]:
        feedback["feedback"].append("Great burpee form! Your plank position is strong and your transitions between stages are smooth.")
        
    return feedback

def analyze_bicycle_crunch(video_path, output_video_path=None):
    """
    Analyzes bicycle crunch form from a video using MediaPipe pose detection.
    
    Args:
        video_path (str): Path to the input video file
        output_video_path (str, optional): Path where the analyzed video will be saved
        
    Returns:
        dict: Analysis results including rep count, form metrics, and feedback
    """
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup output video writer if path is provided
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Variables to track bicycle crunch state
    crunch_count = 0
    crunch_stage = None  # "left", "right", "neutral"
    last_elbow_knee_diff = 0
    frames_without_detection = 0
    good_frames = 0
    side_switch_detected = False
    
    # Lists to store metrics for analysis
    upper_back_lift_scores = []      # For keeping shoulders off ground
    elbow_to_knee_scores = []        # For proper elbow to opposite knee connection
    hip_angle_scores = []            # For proper leg extension
    
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
            
            # Get key points for bicycle crunch analysis
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
            
            # Calculate key metrics for bicycle crunch analysis
            
            # 1. Upper back lift (shoulders off ground)
            # Check the vertical position of shoulders relative to hips
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_y = (left_hip[1] + right_hip[1]) / 2
            upper_back_lift = max(0, min(1, (hip_y - shoulder_y) / 0.1))  # Normalizing
            upper_back_lift_scores.append(upper_back_lift)
            
            # 2. Hip angles (hip-knee-ankle) for leg extension
            left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
            
            # 3. Check for elbow to opposite knee proximity
            left_elbow_to_right_knee = np.sqrt((left_elbow[0] - right_knee[0])**2 + (left_elbow[1] - right_knee[1])**2)
            right_elbow_to_left_knee = np.sqrt((right_elbow[0] - left_knee[0])**2 + (right_elbow[1] - left_knee[1])**2)
            
            # Lower distance indicates better elbow-knee connection (normalized)
            left_connection = max(0, 1 - left_elbow_to_right_knee / 0.3)
            right_connection = max(0, 1 - right_elbow_to_left_knee / 0.3)
            
            # 4. Determine which side is active (elbow closer to opposite knee)
            elbow_knee_diff = left_elbow_to_right_knee - right_elbow_to_left_knee
            
            # Detect side switch for rep counting
            if elbow_knee_diff * last_elbow_knee_diff < 0:  # Sign change indicates side switch
                side_switch_detected = True
                
                if crunch_stage == "left":
                    crunch_stage = "right"
                    # Add the connection quality to scores
                    elbow_to_knee_scores.append(right_connection)
                    # Add hip angle to scores
                    hip_angle_scores.append(left_hip_angle)  # Extended leg angle
                elif crunch_stage == "right":
                    crunch_stage = "left" 
                    elbow_to_knee_scores.append(left_connection)
                    hip_angle_scores.append(right_hip_angle)  # Extended leg angle
                    crunch_count += 1  # Count each complete cycle (both sides) as one rep
                    print(f"Bicycle crunch #{crunch_count} completed")
                else:
                    # First detection
                    crunch_stage = "left" if elbow_knee_diff < 0 else "right"
            
            # Update last difference for next frame comparison
            last_elbow_knee_diff = elbow_knee_diff
            
            
            # Visualize metrics
            active_side = "LEFT" if crunch_stage == "left" else "RIGHT"
            cv2.putText(annotated_image, 
                        f"Active: {active_side}",
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            connection_score = left_connection if crunch_stage == "left" else right_connection
            cv2.putText(annotated_image, 
                        f"Connection: {connection_score*100:.1f}%",
                        (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.putText(annotated_image, 
                        f"Back lift: {upper_back_lift*100:.1f}%",
                        (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            frames_without_detection += 1
            if frames_without_detection % 30 == 0:
                print(f"No pose detection for {frames_without_detection} frames")
                
        # Display bicycle crunch count
        cv2.putText(annotated_image, f'Crunches: {crunch_count}', (10, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    
        # Write frame to output video
        if output_video_path:
            out.write(annotated_image)
                
    cap.release()
    if output_video_path:
        out.release()
    
    # Only analyze if we have enough valid frames
    if good_frames < 20:
        return {
            "crunch_count": 0,
            "error": "Not enough valid pose detections. Check video quality and positioning."
        }
    
    # Calculate average metrics
    avg_upper_back_lift = sum(upper_back_lift_scores) / max(len(upper_back_lift_scores), 1)
    avg_elbow_to_knee = sum(elbow_to_knee_scores) / max(len(elbow_to_knee_scores), 1)
    avg_hip_angle = sum(hip_angle_scores) / max(len(hip_angle_scores), 1)
    
    # Generate feedback
    feedback = {
        "crunch_count": crunch_count,
        "form_analysis": {
            "upper_back_lift_score": avg_upper_back_lift * 100,
            "elbow_to_knee_connection": avg_elbow_to_knee * 100,
            "leg_extension_angle": avg_hip_angle,
            "frames_analyzed": good_frames
        },
        "feedback": []
    }
    
    # Add specific feedback based on measurements
    if avg_upper_back_lift < 0.7:
        feedback["feedback"].append("Keep your shoulder blades off the ground throughout the exercise for better core engagement.")
    
    if avg_elbow_to_knee < 0.7:
        feedback["feedback"].append("Try to bring your elbow closer to your opposite knee on each repetition.")
    
    if avg_hip_angle < 160:
        feedback["feedback"].append("Extend your legs more fully when straightening them for better hip flexor engagement.")
    
    if not feedback["feedback"]:
        feedback["feedback"].append("Excellent bicycle crunch form! You're maintaining good shoulder elevation and proper elbow-to-knee connection.")
        
    return feedback

def analyze_band_pull_apart(video_path, output_video_path=None):
    """
    Analyzes band pull-apart form from a video using MediaPipe pose detection.
    
    Args:
        video_path (str): Path to the input video file
        output_video_path (str, optional): Path where the analyzed video will be saved
        
    Returns:
        dict: Analysis results including rep count, form metrics, and feedback
    """
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup output video writer if path is provided
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Variables to track band pull-apart state
    rep_count = 0
    pull_stage = None  # "together" or "apart"
    frames_without_detection = 0
    good_frames = 0
    
    # Lists to store metrics for analysis
    wrist_distance_scores = []          # For tracking band stretch
    shoulder_retraction_scores = []     # For proper shoulder retraction
    arm_height_consistency_scores = []  # For maintaining consistent arm height
    
    # Add debugging info
    print(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}")
    
    # Initial arm height reference
    initial_arm_height = None
    
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
            
            # Get key points for band pull-apart analysis
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

            # Calculate key metrics for band pull-apart analysis
            
            # 1. Wrist distance (normalized x-coordinate distance)
            wrist_distance = abs(right_wrist[0] - left_wrist[0])
            wrist_distance_scores.append(wrist_distance)
            
            # 2. Shoulder retraction (distance between shoulders)
            shoulder_distance = abs(right_shoulder[0] - left_shoulder[0])
            
            # Score based on increase in shoulder width when band is pulled apart
            if pull_stage == "together":
                initial_shoulder_distance = shoulder_distance
            elif pull_stage == "apart":
                shoulder_retraction = max(0, min(1, (shoulder_distance - initial_shoulder_distance) / 0.05))
                shoulder_retraction_scores.append(shoulder_retraction)
            
            # 3. Arm height consistency
            # Arms should stay at consistent height throughout movement
            current_arm_height = (left_elbow[1] + right_elbow[1]) / 2
            
            if initial_arm_height is None and good_frames > 5:  # Set initial reference after a few frames
                initial_arm_height = current_arm_height
            
            if initial_arm_height is not None:
                arm_height_consistency = max(0, 1 - abs(current_arm_height - initial_arm_height) / 0.05)
                arm_height_consistency_scores.append(arm_height_consistency)
            
            
            # Determine pull stage based on wrist distance
            if wrist_distance > 0.5 and pull_stage == "together":
                pull_stage = "apart"
                print("Band pulled apart")
            
            # If we were in "apart" position and now wrists are close
            elif wrist_distance < 0.3 and pull_stage == "apart":
                pull_stage = "together"
                rep_count += 1
                print(f"Band pull-apart #{rep_count} completed")
            
            # Set initial stage if not yet set
            if pull_stage is None:
                pull_stage = "together" if wrist_distance < 0.3 else "apart"
            
            # Visualize metrics
            cv2.putText(annotated_image, 
                        f"Stage: {pull_stage.upper()}",
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.putText(annotated_image, 
                        f"Wrist distance: {wrist_distance:.2f}",
                        (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            if pull_stage == "apart" and len(shoulder_retraction_scores) > 0:
                cv2.putText(annotated_image, 
                            f"Shoulder retraction: {shoulder_retraction_scores[-1]*100:.1f}%",
                            (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            frames_without_detection += 1
            if frames_without_detection % 30 == 0:
                print(f"No pose detection for {frames_without_detection} frames")
                
        # Display rep count
        cv2.putText(annotated_image, f'Reps: {rep_count}', (10, 130), 
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
            "rep_count": 0,
            "error": "Not enough valid pose detections. Check video quality and positioning."
        }
    
    # Calculate average metrics
    max_wrist_distance = max(wrist_distance_scores) if wrist_distance_scores else 0
    avg_shoulder_retraction = sum(shoulder_retraction_scores) / max(len(shoulder_retraction_scores), 1)
    avg_arm_height_consistency = sum(arm_height_consistency_scores) / max(len(arm_height_consistency_scores), 1)
    
    # Generate feedback
    feedback = {
        "rep_count": rep_count,
        "form_analysis": {
            "max_band_stretch": max_wrist_distance,
            "shoulder_retraction_score": avg_shoulder_retraction * 100,
            "arm_height_consistency": avg_arm_height_consistency * 100,
            "frames_analyzed": good_frames
        },
        "feedback": []
    }
    
    # Add specific feedback based on measurements
    if max_wrist_distance < 0.6:
        feedback["feedback"].append("Try to pull the band apart with more range of motion for better activation of the upper back muscles.")
    
    if avg_shoulder_retraction < 0.7:
        feedback["feedback"].append("Focus on squeezing your shoulder blades together at the end of each repetition.")
    
    if avg_arm_height_consistency < 0.8:
        feedback["feedback"].append("Keep your arms at a consistent height throughout the movement.")
    
    if not feedback["feedback"]:
        feedback["feedback"].append("Great form on your band pull-aparts! You're achieving good range of motion and shoulder retraction.")
        
    return feedback

def analyze_bench_press(video_path, output_video_path=None):
    """
    Analyzes bench press form from a video using MediaPipe pose detection.
    
    Args:
        video_path (str): Path to the input video file
        output_video_path (str, optional): Path where the analyzed video will be saved
        
    Returns:
        dict: Analysis results including rep count, form metrics, and feedback
    """
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup output video writer if path is provided
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Variables to track bench press state
    rep_count = 0
    press_stage = None  # "down" or "up"
    frames_without_detection = 0
    good_frames = 0
    
    # Lists to store metrics for analysis
    elbow_angles = []              # For tracking proper depth
    wrist_alignment_scores = []    # For bar path - wrists should be over elbows
    arm_symmetry_scores = []       # For even pressing
    bar_path_scores = []           # For vertical bar path
    
    # Add debugging info
    print(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}")
    
    # Previous wrist positions for tracking bar path
    prev_wrist_pos = None
    
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
            
            # Get key points for bench press analysis
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
            
            # Calculate key metrics for bench press analysis
            
            # 1. Elbow angles (shoulder-elbow-wrist)
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Average elbow angle for press detection
            avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
            elbow_angles.append(avg_elbow_angle)
            
            # 2. Wrist alignment (wrists should be stacked over elbows)
            left_wrist_alignment = 1 - abs(left_wrist[0] - left_elbow[0]) / 0.1  # Normalized score
            right_wrist_alignment = 1 - abs(right_wrist[0] - right_elbow[0]) / 0.1
            avg_wrist_alignment = (max(0, left_wrist_alignment) + max(0, right_wrist_alignment)) / 2
            wrist_alignment_scores.append(avg_wrist_alignment)
            
            # 3. Arm symmetry (difference between left and right elbow angles)
            arm_symmetry = max(0, 1 - abs(left_elbow_angle - right_elbow_angle) / 15)  # Lower difference is better
            arm_symmetry_scores.append(arm_symmetry)
            
            # 4. Bar path (vertical movement of wrists)
            curr_wrist_pos = [(left_wrist[0] + right_wrist[0])/2, (left_wrist[1] + right_wrist[1])/2]
            
            if prev_wrist_pos is not None:
                # Check horizontal deviation (should be minimal for good bar path)
                horiz_deviation = abs(curr_wrist_pos[0] - prev_wrist_pos[0])
                bar_path_score = max(0, 1 - horiz_deviation / 0.03)  # Normalize
                bar_path_scores.append(bar_path_score)
            
            prev_wrist_pos = curr_wrist_pos
            
            # Visualize elbow angles
            l_elbow_px = normalized_to_pixel_coordinates(left_elbow[0], left_elbow[1], frame_width, frame_height)
            r_elbow_px = normalized_to_pixel_coordinates(right_elbow[0], right_elbow[1], frame_width, frame_height)
            
            # Draw angle text
            cv2.putText(annotated_image, 
                        f"L Elbow: {left_elbow_angle:.1f}°",
                        (l_elbow_px[0] - 50, l_elbow_px[1] + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                        
            cv2.putText(annotated_image, 
                        f"R Elbow: {right_elbow_angle:.1f}°",
                        (r_elbow_px[0] - 50, r_elbow_px[1] + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Determine press stage based on elbow angle
            if avg_elbow_angle > 160 and press_stage == "down":
                press_stage = "up"
                # Draw press state
                cv2.putText(annotated_image, 'UP', (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # If we were in "up" position and now angle is < 90, we're in "down" position
            elif avg_elbow_angle < 90 and (press_stage == "up" or press_stage is None):
                press_stage = "down"
                rep_count += 1
                cv2.putText(annotated_image, 'DOWN', (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                print(f"Bench press #{rep_count} detected at depth with avg elbow angle {avg_elbow_angle:.1f}")
            
            # Visualize metrics
            cv2.putText(annotated_image, 
                        f"Wrist alignment: {avg_wrist_alignment*100:.1f}%",
                        (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.putText(annotated_image, 
                        f"Symmetry: {arm_symmetry*100:.1f}%",
                        (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            if len(bar_path_scores) > 0:
                cv2.putText(annotated_image, 
                            f"Bar path: {bar_path_scores[-1]*100:.1f}%",
                            (10, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            frames_without_detection += 1
            if frames_without_detection % 30 == 0:
                print(f"No pose detection for {frames_without_detection} frames")
                
        # Display rep count
        cv2.putText(annotated_image, f'Reps: {rep_count}', (10, 190), 
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
            "rep_count": 0,
            "error": "Not enough valid pose detections. Check video quality and positioning."
        }
    
    # Calculate average metrics
    min_elbow_angle = min(elbow_angles) if elbow_angles else 0
    max_elbow_angle = max(elbow_angles) if elbow_angles else 0
    avg_wrist_alignment = sum(wrist_alignment_scores) / max(len(wrist_alignment_scores), 1)
    avg_arm_symmetry = sum(arm_symmetry_scores) / max(len(arm_symmetry_scores), 1)
    avg_bar_path = sum(bar_path_scores) / max(len(bar_path_scores), 1) if bar_path_scores else 0
    
    # Generate feedback
    feedback = {
        "rep_count": rep_count,
        "form_analysis": {
            "elbow_angle_at_bottom": min_elbow_angle,
            "elbow_angle_at_top": max_elbow_angle,
            "wrist_alignment_score": avg_wrist_alignment * 100,
            "arm_symmetry_score": avg_arm_symmetry * 100,
            "bar_path_score": avg_bar_path * 100,
            "frames_analyzed": good_frames
        },
        "feedback": []
    }
    
    # Add specific feedback based on measurements
    if min_elbow_angle > 90:
        feedback["feedback"].append("You're not lowering the bar enough. Aim for approximately 90 degrees at the elbow for proper depth.")
    
    if max_elbow_angle < 160:
        feedback["feedback"].append("Make sure to fully extend your arms at the top of each repetition, but avoid locking your elbows.")
    
    if avg_wrist_alignment < 0.8:
        feedback["feedback"].append("Keep your wrists stacked over your elbows for better leverage and to reduce wrist strain.")
    
    if avg_arm_symmetry < 0.85:
        feedback["feedback"].append("Focus on pressing the bar evenly with both arms for better balance and strength development.")
    
    if avg_bar_path < 0.8:
        feedback["feedback"].append("Work on maintaining a more vertical bar path to maximize efficiency.")
    
    if not feedback["feedback"]:
        feedback["feedback"].append("Excellent bench press form! Your bar path, depth, and symmetry are all very good.")
        
    return feedback

def analyze_jumping_jacks(video_path, output_video_path=None):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    is_vertical = frame_height > frame_width

    out = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_w, out_h = (frame_height, frame_width) if is_vertical else (frame_width, frame_height)
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))

    jumping_jack_count = 0
    jumping_jack_stage = None
    frames_without_detection = 0
    good_frames = 0
    
    # Track movement to avoid false positives
    first_real_spread_detected = False
    spread_confidence = 0  # Used to track confidence in spread detection
    stable_starting_position = False  # New flag to ensure user is stable before counting
    stable_position_frames = 0  # Count frames in stable position
    
    arm_spreads = []
    leg_spreads = []
    orientations = []
    shoulder_heights = []
    neck_angles = []
    state_changes = []
    recent_arm_spreads = []
    recent_leg_spreads = []
    window_size = 5

    print(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        if is_vertical:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
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
            orientations.append(detect_orientation(landmarks))

            # Key points
            def nc(idx): return [landmarks[idx].x, landmarks[idx].y]
            ls, rs = nc(mp_pose.PoseLandmark.LEFT_SHOULDER.value), nc(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            lw, rw = nc(mp_pose.PoseLandmark.LEFT_WRIST.value), nc(mp_pose.PoseLandmark.RIGHT_WRIST.value)
            lh, rh = nc(mp_pose.PoseLandmark.LEFT_HIP.value), nc(mp_pose.PoseLandmark.RIGHT_HIP.value)
            la, ra = nc(mp_pose.PoseLandmark.LEFT_ANKLE.value), nc(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
            leye   = nc(mp_pose.PoseLandmark.LEFT_EYE.value)
            reye   = nc(mp_pose.PoseLandmark.RIGHT_EYE.value)
            nose   = nc(mp_pose.PoseLandmark.NOSE.value)

            # Distances (normalized)
            shoulder_width = calculate_distance(ls, rs)
            hip_width = calculate_distance(lh, rh)
            wrist_distance = calculate_distance(lw, rw)
            ankle_distance = calculate_distance(la, ra)
            arm_spread_ratio = wrist_distance / max(shoulder_width, 0.01)
            leg_spread_ratio = ankle_distance / max(hip_width, 0.01)

            recent_arm_spreads.append(arm_spread_ratio)
            recent_leg_spreads.append(leg_spread_ratio)
            if len(recent_arm_spreads) > window_size:
                recent_arm_spreads.pop(0)
                recent_leg_spreads.pop(0)
            smoothed_arm_ratio = sum(recent_arm_spreads) / len(recent_arm_spreads)
            smoothed_leg_ratio = sum(recent_leg_spreads) / len(recent_leg_spreads)
            arm_spreads.append(arm_spread_ratio)
            leg_spreads.append(leg_spread_ratio)

            # Shoulder height diff (for symmetry)
            shoulder_heights.append(abs((ls[1] - rs[1]) * (frame_height if not is_vertical else frame_width)))
            # Neck angle (left eye - nose - right eye)
            def calc_angle(a, b, c):
                a, b, c = np.array(a), np.array(b), np.array(c)
                ang = abs(np.degrees(
                    np.arctan2(c[1]-b[1], c[0]-b[0]) -
                    np.arctan2(a[1]-b[1], a[0]-b[0])
                ))
                return 360 - ang if ang > 180 else ang
            neck_angle = calc_angle(leye, nose, reye)
            neck_angles.append(neck_angle)

            # Arm/leg position logic with more robust criteria
            left_wrist_above = lw[1] < ls[1] - 0.05
            right_wrist_above = rw[1] < rs[1] - 0.05

            if is_vertical:
                spread_threshold = 2.2
                together_threshold = 1.6
            else:
                spread_threshold = 1.8
                together_threshold = 1.4

            is_definite_spread = (left_wrist_above and right_wrist_above and smoothed_arm_ratio > spread_threshold) or (smoothed_leg_ratio > spread_threshold)
            is_probable_spread = smoothed_leg_ratio > 1.8 or (left_wrist_above and right_wrist_above and smoothed_arm_ratio > 1.5)
            is_together_position = smoothed_leg_ratio < together_threshold and not (left_wrist_above and right_wrist_above)
            
            # Check for stable starting position
            if is_together_position and not stable_starting_position:
                stable_position_frames += 1
                if stable_position_frames >= 10:  # Require 10 frames of stable position
                    stable_starting_position = True
                    state_changes.append(f"Frame {good_frames}: Stable starting position detected")
            elif not is_together_position:
                stable_position_frames = 0
            
            # Update confidence in spread position detection
            if is_definite_spread:
                spread_confidence = 3  # Very confident
                if stable_starting_position:  # Only mark as real spread if we started in stable position
                    first_real_spread_detected = True
            elif is_probable_spread and spread_confidence < 3:
                spread_confidence = min(spread_confidence + 1, 2)  # Build confidence
            elif not is_probable_spread and spread_confidence > 0:
                spread_confidence -= 1  # Reduce confidence
                
            # Only consider as spread position if we have enough confidence
            is_spread_position = is_probable_spread and (first_real_spread_detected or spread_confidence >= 2)

            # Modified state machine with confidence check
            if is_spread_position and (jumping_jack_stage == "together" or jumping_jack_stage is None):
                old_stage = jumping_jack_stage
                jumping_jack_stage = "spread"
                transition_msg = f"Frame {good_frames}: {old_stage} -> spread, arm: {smoothed_arm_ratio:.2f}, leg: {smoothed_leg_ratio:.2f}"
                state_changes.append(transition_msg)
            elif is_together_position and jumping_jack_stage == "spread":
                old_stage = jumping_jack_stage
                jumping_jack_stage = "together"
                # Only count if we've detected at least one real spread position before
                if first_real_spread_detected and stable_starting_position:
                    jumping_jack_count += 1
                    transition_msg = f"Frame {good_frames}: spread -> together, COUNTED JUMPING JACK #{jumping_jack_count}, arm: {smoothed_arm_ratio:.2f}, leg: {smoothed_leg_ratio:.2f}"
                else:
                    transition_msg = f"Frame {good_frames}: spread -> together, POSITIONING (not counted), arm: {smoothed_arm_ratio:.2f}, leg: {smoothed_leg_ratio:.2f}"
                state_changes.append(transition_msg)

            # Overlays
            cv2.putText(annotated_image, f"Arm spread: {arm_spread_ratio:.2f}x", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(annotated_image, f"Leg spread: {leg_spread_ratio:.2f}x", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(annotated_image, f"Neck: {neck_angle:.1f}°", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.putText(annotated_image, f"View: {orientations[-1]}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
            
            count_text = f'Jumping Jacks: {jumping_jack_count}'
            if not stable_starting_position:
                count_text += " (Stand still to begin...)"
            elif not first_real_spread_detected:
                count_text += " (Get in position...)"
            cv2.putText(annotated_image, count_text, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Write frame to output video
            if out:
                out.write(annotated_image)
        else:
            frames_without_detection += 1
            if frames_without_detection % 30 == 0:
                print(f"No pose detection for {frames_without_detection} frames")

    cap.release()
    if out:
        out.release()

    if good_frames < 10:
        return {
            "jumping_jack_count": 0,
            "error": "Not enough valid pose detections. Check video quality and positioning."
        }

    max_arm_spread = max(arm_spreads) if arm_spreads else 0
    min_arm_spread = min(arm_spreads) if arm_spreads else 0
    max_leg_spread = max(leg_spreads) if leg_spreads else 0
    min_leg_spread = min(leg_spreads) if leg_spreads else 0
    avg_neck_deg = sum(neck_angles)/len(neck_angles) if neck_angles else 0
    avg_shoulder_dy = sum(shoulder_heights)/len(shoulder_heights) if shoulder_heights else 0
    dominant_view = max(set(orientations), key=orientations.count) if orientations else "unknown"

    feedback = []
    if max_arm_spread < 1.8:
        feedback.append("Raise your arms higher-hands should meet above your head.")
    if max_leg_spread < 1.8:
        feedback.append("Spread your legs wider at the top of the movement.")
    if min_arm_spread > 1.0:
        feedback.append("Bring your arms fully down to your sides at the bottom.")
    if min_leg_spread > 1.1:
        feedback.append("Bring your feet together at the bottom of each rep.")
    if avg_neck_deg < 165:
        feedback.append("Keep your neck neutral-look forward, not down or up.")
    if avg_shoulder_dy > 0.05 * (frame_height if not is_vertical else frame_width):
        feedback.append("Keep shoulders level-avoid shrugging or tilting.")
    # Orientation-specific feedback
    if dominant_view == "front":
        feedback.append("Front view: check for even arm/leg motion and symmetry.")
    elif dominant_view == "left":
        feedback.append("Side view (left): ensure arms reach overhead and torso stays upright.")
    elif dominant_view == "right":
        feedback.append("Side view (right): ensure arms reach overhead and torso stays upright.")
    if not feedback:
        feedback.append("Excellent jumping jack form!")

    return {
        "jumping_jack_count": jumping_jack_count,
        "form_analysis": {
            "max_arm_spread_ratio": max_arm_spread,
            "min_arm_spread_ratio": min_arm_spread,
            "max_leg_spread_ratio": max_leg_spread,
            "min_leg_spread_ratio": min_leg_spread,
            "avg_neck_deg": avg_neck_deg,
            "avg_shoulder_dy": avg_shoulder_dy,
            "dominant_view": dominant_view,
            "frames_analyzed": good_frames
        },
        "feedback": feedback,
        "state_transitions": state_changes[:20]
    }

def analyze_mountain_climbers(video_path, output_video_path=None):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Video writer setup
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) if output_video_path else None
    
    # Metrics
    rep_count = 0
    stage = None  # "up" or "down"
    knee_angles = []
    hip_angles = []
    core_alignment = []
    shoulder_stability = []
    back_straightness = []  # New metric for back alignment
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        annotated_image = frame.copy()
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Get key points
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # Get back points for straightness analysis
            neck = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                   landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            mid_shoulder = [(shoulder_r[0] + shoulder_l[0])/2, 
                           (shoulder_r[1] + shoulder_l[1])/2]
            mid_hip = [(hip_r[0] + hip_l[0])/2, 
                      (hip_r[1] + hip_l[1])/2]
            
            # Calculate angles
            knee_angle_r = calculate_angle(hip_r, knee_r, ankle_r)
            knee_angle_l = calculate_angle(hip_l, knee_l, ankle_l)
            hip_angle_r = calculate_angle(shoulder_r, hip_r, knee_r)
            hip_angle_l = calculate_angle(shoulder_l, hip_l, knee_l)
            
            # Calculate back straightness (shoulder to hip to knee alignment)
            # For plank position in mountain climbers, we want this to be close to 180 degrees
            back_angle = calculate_angle(mid_shoulder, mid_hip, knee_r)  # Using right knee as reference
            back_straightness_score = abs(180 - back_angle)  # Lower is better (0 = perfectly straight)
            
            # Alternative method: check alignment of shoulder-hip-knee
            shoulder_hip_alignment = calculate_alignment(mid_shoulder, mid_hip, knee_r)
            
            # Core metrics
            avg_knee_angle = (knee_angle_r + knee_angle_l) / 2
            avg_hip_angle = (hip_angle_r + hip_angle_l) / 2
            shoulder_diff = abs(shoulder_r[1] - shoulder_l[1]) * height
            
            # Rep counting logic
            if avg_knee_angle < 90 and stage != "down":
                stage = "down"
            elif avg_knee_angle > 120 and stage == "down":
                stage = "up"
                rep_count += 1
            
            # Store metrics
            knee_angles.append(avg_knee_angle)
            hip_angles.append(avg_hip_angle)
            shoulder_stability.append(shoulder_diff)
            back_straightness.append(back_straightness_score)
            
            # Visual feedback
            cv2.putText(annotated_image, f"Reps: {rep_count}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(annotated_image, f"Knee Angle: {avg_knee_angle:.1f}°", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.putText(annotated_image, f"Hip Angle: {avg_hip_angle:.1f}°", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            
            # Add back straightness feedback
            back_color = (0, 255, 0) if back_straightness_score < 20 else (0, 165, 255) if back_straightness_score < 40 else (0, 0, 255)
            cv2.putText(annotated_image, f"Back Alignment: {back_straightness_score:.1f}°", (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, back_color, 2)
            
            # Draw line along the back to visualize alignment
            spine_start = (int(mid_shoulder[0] * width), int(mid_shoulder[1] * height))
            spine_mid = (int(mid_hip[0] * width), int(mid_hip[1] * height))
            spine_end = (int(knee_r[0] * width), int(knee_r[1] * height))
            
            cv2.line(annotated_image, spine_start, spine_mid, back_color, 2)
            cv2.line(annotated_image, spine_mid, spine_end, back_color, 2)
            
            # Draw landmarks
            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        if writer:
            writer.write(annotated_image)
    
    cap.release()
    if writer:
        writer.release()
    
    # Analysis and feedback
    if len(knee_angles) < 10:
        return {"error": "Insufficient data - check video quality"}
    
    avg_knee = sum(knee_angles) / len(knee_angles)
    avg_hip = sum(hip_angles) / len(hip_angles)
    avg_shoulder_diff = sum(shoulder_stability) / len(shoulder_stability)
    avg_back_straightness = sum(back_straightness) / len(back_straightness)
 
    feedback = {
        "rep_count": rep_count,
        "form_analysis": {
            "avg_knee_angle": avg_knee,
            "avg_hip_angle": avg_hip,
            "avg_shoulder_stability_px": avg_shoulder_diff,
            "avg_back_straightness": avg_back_straightness,
            "frames_analyzed": len(knee_angles)
        },
        "feedback": []
    }
    
    # Knee feedback (more nuanced)
    if avg_knee > 140:
        feedback["feedback"].append("Bring knees higher toward chest - aim for 90-120° knee bend")
    elif avg_knee > 120:
        feedback["feedback"].append("Good knee range - could bring slightly higher for full engagement")
    elif avg_knee < 70:
        feedback["feedback"].append("Avoid over-bending knees - maintain controlled motion")
    else:
        feedback["feedback"].append("Excellent knee movement - good range of motion")
        
    # Hip stability feedback - can play around with this more
    if avg_hip < 150:
        feedback["feedback"].append("Engage core to stabilize hips - slight rocking detected")
    elif avg_hip < 170:
        feedback["feedback"].append("Moderate hip stability - focus on keeping hips level")
    else:
        feedback["feedback"].append("Excellent hip stability - minimal movement detected")
        
    # Shoulder stability
    if avg_shoulder_diff > 0.15 * height:
        feedback["feedback"].append("Significant shoulder movement - keep shoulders square")
    elif avg_shoulder_diff > 0.08 * height:
        feedback["feedback"].append("Minor shoulder tilt - focus on balanced movement")
    else:
        feedback["feedback"].append("Excellent shoulder stability - maintaining good position")
    
    # Revised back straightness feedback 
    if avg_back_straightness > 45:
        feedback["feedback"].append("Noticeable back arch/sag - engage core to flatten back")
    elif avg_back_straightness > 25:
        feedback["feedback"].append("Moderate back alignment - focus on straight line from shoulders to knees")
    elif avg_back_straightness > 15:
        feedback["feedback"].append("Good back alignment - minor adjustments could improve form")
    else:
        feedback["feedback"].append("Excellent back alignment - maintaining perfect plank position")
    
    # Overall form assessment
    good_metrics = 0
    if avg_knee <= 140 and avg_knee >= 70: good_metrics += 1
    if avg_hip >= 150: good_metrics += 1
    if avg_shoulder_diff <= 0.15 * height: good_metrics += 1
    if avg_back_straightness <= 45: good_metrics += 1
    
    if good_metrics == 4:
        feedback["feedback"].append("EXCELLENT FORM - Maintain all aspects of your technique")
    elif good_metrics >= 2:
        feedback["feedback"].append("GOOD FORM - Focus on the highlighted corrections")
    else:
        feedback["feedback"].insert(0, "NEEDS WORK - Prioritize these corrections:")
    
    return feedback

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





