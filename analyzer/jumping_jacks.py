import cv2
import mediapipe as mp
import numpy as np
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

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
