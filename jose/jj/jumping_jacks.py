import cv2
import mediapipe as mp
import numpy as np
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def calculate_distance(p1, p2):
    """Helper function to calculate distance between two points"""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height):
    """Convert normalized coordinates to pixel coordinates"""
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

def analyze_jumping_jacks(video_path, output_video_path=None, debug=False, already_positioned=False):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup output video writer if path is provided
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Variables to track jumping jack state
    jumping_jack_count = 0
    jumping_jack_stage = None  # "spread" or "together"
    frames_without_detection = 0
    good_frames = 0
    
    # Flag to determine if the first cycle has been completed
    first_cycle_complete = already_positioned  # Skip initialization if already positioned
    
    # Lists to store measurements for analysis
    arm_spreads = []  # Distance between wrists relative to shoulder width
    leg_spreads = []  # Distance between ankles relative to hip width
    
    # For tracking state transitions
    state_changes = []
    
    # For smoothing measurements
    recent_arm_spreads = []
    recent_leg_spreads = []
    window_size = 5  # Number of frames to average
    
    # debugging info - might delete this later
    print(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}")
    print(f"Already positioned mode: {already_positioned}")
    
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
            
            # Get key points for jumping jack analysis
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # Calculate normalized distances
            shoulder_width = calculate_distance(left_shoulder, right_shoulder)
            hip_width = calculate_distance(left_hip, right_hip)
            
            wrist_distance = calculate_distance(left_wrist, right_wrist)
            ankle_distance = calculate_distance(left_ankle, right_ankle)
            
            # Normalize by shoulder/hip width to account for different distances from camera
            arm_spread_ratio = wrist_distance / max(shoulder_width, 0.01)  # Avoid division by zero
            leg_spread_ratio = ankle_distance / max(hip_width, 0.01)
            
            # Apply smoothing with a rolling window
            recent_arm_spreads.append(arm_spread_ratio)
            recent_leg_spreads.append(leg_spread_ratio)
            
            if len(recent_arm_spreads) > window_size:
                recent_arm_spreads.pop(0)
                recent_leg_spreads.pop(0)
                
            # Get smoothed values
            smoothed_arm_ratio = sum(recent_arm_spreads) / len(recent_arm_spreads)
            smoothed_leg_ratio = sum(recent_leg_spreads) / len(recent_leg_spreads)
            
            # Use smoothed values for state detection but store original for analysis
            arm_spreads.append(arm_spread_ratio)
            leg_spreads.append(leg_spread_ratio)

            # Visualize arm and leg spread
            cv2.putText(annotated_image, 
                        f"Arm spread: {arm_spread_ratio:.2f}x",
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(annotated_image, 
                        f"Leg spread: {leg_spread_ratio:.2f}x",
                        (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Determine jumping jack stage using smoothed values
            # For arms: calculate the vertical position of wrists relative to shoulders
            # We want to detect if arms are raised up
            left_wrist_above = left_wrist[1] < left_shoulder[1] - 0.05  # Ensure wrist is clearly above shoulder
            right_wrist_above = right_wrist[1] < right_shoulder[1] - 0.05
            
            # Calculate vertical position of wrists
            wrist_height_ratio = ((left_shoulder[1] - left_wrist[1]) + (right_shoulder[1] - right_wrist[1])) / 2
            
            # Combined condition for the "spread" position - Check EITHER arms are up OR legs are wide
            is_spread_position = False
            
            # Main condition: if legs are significantly spread OR arms are raised high
            if smoothed_leg_ratio > 1.8 or (left_wrist_above and right_wrist_above and smoothed_arm_ratio > 1.5):
                is_spread_position = True
                
            # Combined condition for the "together" position
            # Legs close together AND arms not raised
            is_together_position = smoothed_leg_ratio < 1.4 and not (left_wrist_above and right_wrist_above)
            
            # Add initialization status to the state display
            state_text = f"Spread: {is_spread_position}, Together: {is_together_position}, Stage: {jumping_jack_stage}"
            if not first_cycle_complete:
                state_text += " (Initializing)"
                
            cv2.putText(annotated_image, state_text, (10, frame_height - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Additional debugging info
            metrics_text = f"Arm: {smoothed_arm_ratio:.2f}x, Leg: {smoothed_leg_ratio:.2f}x"
            cv2.putText(annotated_image, metrics_text, (10, frame_height - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # State machine for counting - Modified to handle initialization
            if is_spread_position and (jumping_jack_stage == "together" or jumping_jack_stage is None):
                old_stage = jumping_jack_stage
                jumping_jack_stage = "spread"
                cv2.putText(annotated_image, 'SPREAD', (50, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                # Log state transition
                transition_msg = f"Frame {good_frames}: {old_stage} -> spread, arm: {smoothed_arm_ratio:.2f}, leg: {smoothed_leg_ratio:.2f}"
                state_changes.append(transition_msg)
                if debug:
                    print(transition_msg)
            
            elif is_together_position and jumping_jack_stage == "spread":
                old_stage = jumping_jack_stage
                jumping_jack_stage = "together"
                
                # Only increment count if we've completed initialization
                if first_cycle_complete:
                    jumping_jack_count += 1
                    transition_msg = f"Frame {good_frames}: spread -> together, COUNTED JUMPING JACK #{jumping_jack_count}, arm: {smoothed_arm_ratio:.2f}, leg: {smoothed_leg_ratio:.2f}"
                else:
                    first_cycle_complete = True  # Mark that initialization is complete
                    transition_msg = f"Frame {good_frames}: spread -> together, INITIALIZATION COMPLETE (not counted), arm: {smoothed_arm_ratio:.2f}, leg: {smoothed_leg_ratio:.2f}"
                
                state_changes.append(transition_msg)
                if debug:
                    print(transition_msg)
                    
                cv2.putText(annotated_image, 'TOGETHER', (50, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            frames_without_detection += 1
            if frames_without_detection % 30 == 0:
                print(f"No pose detection for {frames_without_detection} frames")
                
        # Display jumping jack count and initialization status
        count_text = f'Jumping Jacks: {jumping_jack_count}'
        if not first_cycle_complete:
            count_text += " (Initializing...)"
            
        cv2.putText(annotated_image, count_text, (10, 120), 
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
            "jumping_jack_count": 0,
            "error": "Not enough valid pose detections. Check video quality and positioning."
        }
        
    # Analyze the data collected
    max_arm_spread = max(arm_spreads) if arm_spreads else 0
    min_arm_spread = min(arm_spreads) if arm_spreads else 0
    max_leg_spread = max(leg_spreads) if leg_spreads else 0
    min_leg_spread = min(leg_spreads) if leg_spreads else 0
    
    # Generate feedback
    feedback = {
        "jumping_jack_count": jumping_jack_count,
        "form_analysis": {
            "max_arm_spread_ratio": max_arm_spread,
            "min_arm_spread_ratio": min_arm_spread,
            "max_leg_spread_ratio": max_leg_spread,
            "min_leg_spread_ratio": min_leg_spread,
            "frames_analyzed": good_frames
        },
        "feedback": [],
        "state_transitions": state_changes[:20]  # Include up to 20 state transitions for debugging
    }
    
    if max_arm_spread < 1.8:
        feedback["feedback"].append("Try to raise your arms higher. For optimal form, your hands should meet or nearly meet above your head.")
    
    if max_leg_spread < 1.8:
        feedback["feedback"].append("Your legs could spread wider for a more effective jumping jack.")
    
    if min_arm_spread > 1.0:
        feedback["feedback"].append("Make sure to bring your arms all the way down to your sides in the starting position.")
    
    if min_leg_spread > 1.1:
        feedback["feedback"].append("Try to bring your feet closer together in the starting position.")
    
    if not feedback["feedback"]:
        feedback["feedback"].append("Great form! Your jumping jacks have good range of motion in both arms and legs.")
        
    return feedback