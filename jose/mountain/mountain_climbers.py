import mediapipe as mp
import cv2
import numpy as np
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360 - angle if angle > 180 else angle

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
