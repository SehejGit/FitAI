import streamlit as st
import random
import pandas as pd
import io
import json
import datetime
import os
import tempfile
import importlib
import sys
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import base64

# Set page configuration
st.set_page_config(
    page_title="Fitness Buddy",
    page_icon="💪",
    layout="wide"
)

# Initialize session state for tracking workouts
if 'saved_workouts' not in st.session_state:
    st.session_state.saved_workouts = []

if 'current_workout' not in st.session_state:
    st.session_state.current_workout = None

if 'workout_logs' not in st.session_state:
    st.session_state.workout_logs = {}

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Generate"

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

if 'processed_video_path' not in st.session_state:
    st.session_state.processed_video_path = None

# Exercise library
EXERCISE_LIBRARY = {
    "bodyweight": {
        "upper": ["Push-ups", "Tricep dips", "Plank shoulder taps", "Pike push-ups"],
        "lower": ["Squats", "Lunges", "Glute bridges", "Calf raises"],
        "core": ["Planks", "Mountain climbers", "Bicycle crunches", "Russian twists"],
        "full_body": ["Burpees", "Jumping jacks", "Mountain climbers", "Bear crawls"]
    },
    "dumbbell": {
        "upper": ["Dumbbell press", "Dumbbell rows", "Shoulder press", "Bicep curls", "Tricep extensions"],
        "lower": ["Goblet squats", "Dumbbell lunges", "Romanian deadlifts", "Step-ups"],
        "core": ["Dumbbell russian twists", "Weighted sit-ups", "Side bends"],
        "full_body": ["Dumbbell thrusters", "Renegade rows", "Man makers"]
    },
    "bands": {
        "upper": ["Band pull-aparts", "Banded push-ups", "Band bicep curls", "Band tricep pushdowns"],
        "lower": ["Banded squats", "Banded glute bridges", "Lateral band walks", "Banded hip thrusts"],
        "core": ["Banded russian twists", "Banded mountain climbers", "Pallof press"],
        "full_body": ["Banded jumping jacks", "Banded burpees"]
    },
    "gym": {
        "upper": ["Bench press", "Lat pulldowns", "Cable rows", "Shoulder press machine", "Chest fly machine"],
        "lower": ["Leg press", "Leg extensions", "Leg curls", "Hip abduction/adduction", "Calf raise machine"],
        "core": ["Cable crunches", "Ab machine", "Hanging leg raises"],
        "full_body": ["Assisted pull-ups", "Rowing machine", "Elliptical"]
    },
    "cardio": ["Running in place", "High knees", "Jumping jacks", "Shadow boxing", 
               "Treadmill", "Stair climber", "Stationary bike", "Elliptical"]
}

# Available exercise analysis modules
# This maps display names to the actual Python module names
EXERCISE_ANALYSIS_MODULES = {
    "Push-up": "pushup",
    "Squat": "squat",
    "Deadlift": "deadlift",
    "Plank": "plank",
    "Lunge": "lunge", 
    "Bench Press": "bench_press",
    "Shoulder Press": "shoulder_press",
    "Pull-up": "pullup",
    "Bicep Curl": "bicep_curl",
    "Tricep Extension": "tricep_extension"
    # Add all your exercise modules here
}

# Workout plan generator functions
def determine_exercises(equipment, fitness_goal, fitness_level, injuries):
    """Select appropriate exercises based on user inputs"""
    has_injuries = injuries.strip() != ""
    
    # Start with bodyweight exercises
    selected_exercises = {
        "upper": EXERCISE_LIBRARY["bodyweight"]["upper"].copy(),
        "lower": EXERCISE_LIBRARY["bodyweight"]["lower"].copy(),
        "core": EXERCISE_LIBRARY["bodyweight"]["core"].copy(),
        "full_body": EXERCISE_LIBRARY["bodyweight"]["full_body"].copy(),
        "cardio": ["Running in place", "High knees", "Jumping jacks", "Shadow boxing"]
    }
    
    # Add equipment-based exercises
    if "Dumbbells" in equipment:
        selected_exercises["upper"].extend(EXERCISE_LIBRARY["dumbbell"]["upper"])
        selected_exercises["lower"].extend(EXERCISE_LIBRARY["dumbbell"]["lower"])
        selected_exercises["core"].extend(EXERCISE_LIBRARY["dumbbell"]["core"])
        selected_exercises["full_body"].extend(EXERCISE_LIBRARY["dumbbell"]["full_body"])
    
    if "Kettlebells" in equipment:
        # Add kettlebell exercises (similar to dumbbell but with kettlebell-specific movements)
        kettlebell_exercises = {
            "upper": ["Kettlebell press", "Kettlebell rows", "Kettlebell high pull", "Kettlebell halos"],
            "lower": ["Kettlebell swings", "Kettlebell goblet squats", "Kettlebell lunges", "Kettlebell deadlifts"],
            "core": ["Kettlebell windmill", "Kettlebell Turkish get-up", "Kettlebell around the world"],
            "full_body": ["Kettlebell clean and press", "Kettlebell snatch", "Kettlebell swing to press"]
        }
        selected_exercises["upper"].extend(kettlebell_exercises["upper"])
        selected_exercises["lower"].extend(kettlebell_exercises["lower"])
        selected_exercises["core"].extend(kettlebell_exercises["core"])
        selected_exercises["full_body"].extend(kettlebell_exercises["full_body"])
    
    if "Resistance bands" in equipment:
        selected_exercises["upper"].extend(EXERCISE_LIBRARY["bands"]["upper"])
        selected_exercises["lower"].extend(EXERCISE_LIBRARY["bands"]["lower"])
        selected_exercises["core"].extend(EXERCISE_LIBRARY["bands"]["core"])
        selected_exercises["full_body"].extend(EXERCISE_LIBRARY["bands"]["full_body"])
    
    if "Pull-up bar" in equipment:
        # Add pull-up bar specific exercises
        pullup_exercises = {
            "upper": ["Pull-ups", "Chin-ups", "Hanging scapular retractions", "Hanging L-sits", "Negative pull-ups"],
            "core": ["Hanging knee raises", "Hanging leg raises", "Windshield wipers"]
        }
        selected_exercises["upper"].extend(pullup_exercises["upper"])
        selected_exercises["core"].extend(pullup_exercises["core"])
    
    if "Bench" in equipment:
        # Add bench-specific exercises
        bench_exercises = {
            "upper": ["Bench press", "Bench flyes", "Incline push-ups", "Decline push-ups", "Seated shoulder press"],
            "lower": ["Step-ups", "Bulgarian split squats", "Box jumps", "Bench hip thrusts"],
            "core": ["Bench leg raises", "Seated Russian twists", "Decline sit-ups"]
        }
        selected_exercises["upper"].extend(bench_exercises["upper"])
        selected_exercises["lower"].extend(bench_exercises["lower"])
        selected_exercises["core"].extend(bench_exercises["core"])
    
    if "Full gym access" in equipment:
        selected_exercises["upper"].extend(EXERCISE_LIBRARY["gym"]["upper"])
        selected_exercises["lower"].extend(EXERCISE_LIBRARY["gym"]["lower"])
        selected_exercises["core"].extend(EXERCISE_LIBRARY["gym"]["core"])
        selected_exercises["full_body"].extend(EXERCISE_LIBRARY["gym"]["full_body"])
        selected_exercises["cardio"].extend(["Treadmill", "Stair climber", "Stationary bike", "Elliptical", "Rowing machine", "Jacob's ladder"])
    
    # Remove high-impact exercises if user has injuries
    if has_injuries:
        high_impact = ["Burpees", "Jumping", "Jump", "Running", "High knees", "Sprint", "Box jump", "Plyo"]
        
        # Check for specific injuries and remove related exercises
        specific_limitations = {
            "knee": ["Squat", "Lunge", "Jump", "Run", "Extension", "Leg press", "Step-up"],
            "back": ["Deadlift", "Row", "Twist", "Bend", "Swing", "Rotation", "Clean", "Snatch"],
            "shoulder": ["Press", "Push-up", "Pull-up", "Raise", "Fly", "Dip", "Clean", "Snatch", "Handstand"],
            "wrist": ["Push-up", "Plank", "Press", "Curl", "Extension", "Push", "Pull"],
            "ankle": ["Jump", "Run", "Sprint", "Hop", "Lunge", "Step", "Calf raise"],
            "hip": ["Squat", "Lunge", "Deadlift", "Thrust", "Bridge", "Leg raise"],
            "neck": ["Shoulder press", "Pull-up", "Deadlift", "Turkish get-up", "Handstand"]
        }
        
        # Check if any specific injuries are mentioned
        for limitation, excluded_moves in specific_limitations.items():
            if limitation.lower() in injuries.lower():
                for area in selected_exercises:
                    selected_exercises[area] = [ex for ex in selected_exercises[area] 
                                             if not any(move in ex for move in excluded_moves)]
        
        # Remove general high-impact exercises
        for area in selected_exercises:
            selected_exercises[area] = [ex for ex in selected_exercises[area] 
                                      if not any(impact in ex for impact in high_impact)]
        
        # Add low-impact alternatives if a category becomes too small
        for area in selected_exercises:
            if len(selected_exercises[area]) < 3:
                low_impact_alternatives = {
                    "upper": ["Wall push-ups", "Seated band rows", "Isometric chest press", "Shoulder shrugs"],
                    "lower": ["Seated leg extensions", "Seated calf raises", "Lying leg curls", "Glute bridges"],
                    "core": ["Dead bug", "Bird dog", "Modified planks", "Seated core rotations"],
                    "full_body": ["Modified burpee (no jump)", "Step-out-step-in", "Modified jumping jack"],
                    "cardio": ["Seated punches", "Chair marches", "Seated knee lifts", "Arm circles"]
                }
                selected_exercises[area].extend(low_impact_alternatives.get(area, []))
    
    # Modify based on fitness goal
    if fitness_goal == "Weight Loss":
        # Add more compound and cardio exercises
        cardio_emphasis = [
            "Mountain climbers", "Jumping jacks", "High knees", "Burpees", 
            "Jump rope", "Jumping squats", "Skaters", "Lateral bounds",
            "Squat thrusts", "Plank jacks"
        ]
        # Only add if no injuries
        if not has_injuries:
            selected_exercises["cardio"].extend([ex for ex in cardio_emphasis 
                                              if ex not in selected_exercises["cardio"]])
            selected_exercises["full_body"].extend([ex for ex in cardio_emphasis 
                                                 if ex not in selected_exercises["full_body"]])
    
    elif fitness_goal == "Muscle Gain":
        # Prioritize resistance exercises with progressive overload potential
        if any(equip in equipment for equip in ["Dumbbells", "Kettlebells", "Full gym access"]):
            strength_emphasis = {
                "upper": ["Bench press", "Overhead press", "Rows", "Pull-ups", "Dips"],
                "lower": ["Squats", "Deadlifts", "Lunges", "Step-ups", "Hip thrusts"],
                "full_body": ["Clean and press", "Thrusters", "Man makers", "Snatches"]
            }
            
            for area in ["upper", "lower", "full_body"]:
                # Add strength exercises with the equipment name if available
                for ex in strength_emphasis[area]:
                    if "Dumbbells" in equipment:
                        selected_exercises[area].append(f"Dumbbell {ex.lower()}")
                    if "Kettlebells" in equipment:
                        selected_exercises[area].append(f"Kettlebell {ex.lower()}")
    
    elif fitness_goal == "Endurance":
        # Add more repetitive, lower-intensity exercises
        endurance_emphasis = {
            "cardio": ["Steady state running", "Cycling", "Rowing", "Power walking", "Swimming"],
            "upper": ["High-rep push-ups", "High-rep rows", "Band pull-aparts", "Arm circles"],
            "lower": ["Walking lunges", "Step-ups", "Bodyweight squats", "Calf raises"],
            "full_body": ["Circuit training", "Jumping jacks", "Modified burpees"]
        }
        
        for area, exercises in endurance_emphasis.items():
            if area in selected_exercises:
                selected_exercises[area].extend([ex for ex in exercises 
                                             if ex not in selected_exercises[area]])
    
    # Ensure there are no duplicates by converting to set and back to list
    for area in selected_exercises:
        selected_exercises[area] = list(set(selected_exercises[area]))
    
    return selected_exercises

def create_workout_schedule(exercises, days_per_week, time_per_session, fitness_level):
    """Create a personalized workout schedule"""
    workout_days = []
    
    # Number of exercises based on time per session (adjusted for more realistic workout volume)
    # For a typical workout:
    # - 30 min: ~5-6 exercises
    # - 45 min: ~6-8 exercises
    # - 60 min: ~8-10 exercises
    # - 90 min: ~10-12 exercises
    if time_per_session == 30:
        exercises_per_workout = 6
    elif time_per_session == 45:
        exercises_per_workout = 7
    elif time_per_session == 60:
        exercises_per_workout = 9
    else:  # 90 minutes
        exercises_per_workout = 11
    
    # Determine sets and reps based on fitness level
    if fitness_level == "Beginner":
        sets_reps = {"sets": 2, "reps": "8-10", "rest": "60 sec"}
    elif fitness_level == "Intermediate":
        sets_reps = {"sets": 3, "reps": "10-12", "rest": "45 sec"}
    else:  # Advanced
        sets_reps = {"sets": 4, "reps": "12-15", "rest": "30 sec"}
    
    # Define workout types
    workout_types = [
        {"name": "Full Body", "focus": ["upper", "lower", "core"]},
        {"name": "Upper Body", "focus": ["upper", "core"]},
        {"name": "Lower Body", "focus": ["lower", "core"]},
        {"name": "HIIT & Cardio", "focus": ["full_body", "cardio"]},
        {"name": "Core & Mobility", "focus": ["core", "full_body"]}
    ]
    
    # Ensure daysPerWeek is a number
    days = int(days_per_week) if isinstance(days_per_week, (int, str)) else 3
    
    # Distribute workout types throughout the week
    for i in range(days):
        workout_type = workout_types[i % len(workout_types)]
        workout = {
            "day": f"Day {i + 1}",
            "name": workout_type["name"],
            "exercises": []
        }
        
        # Add exercises from each focus area
        for area in workout_type["focus"]:
            if area == "cardio":
                # Handle cardio exercises differently
                cardio_exercises = exercises["cardio"].copy()
                if cardio_exercises:
                    # For cardio, select 1-2 exercises and assign duration instead of reps
                    num_cardio = min(2, len(cardio_exercises))
                    for _ in range(num_cardio):
                        if cardio_exercises:
                            exercise = random.choice(cardio_exercises)
                            # Check if this is a machine-based cardio
                            machine_cardio = ["Treadmill", "Elliptical", "Stationary bike", "Rowing machine", 
                                            "Stair climber", "Jacob's ladder"]
                            
                            if any(machine in exercise for machine in machine_cardio):
                                # For cardio machines, use duration instead of reps
                                if fitness_level == "Beginner":
                                    duration = "10-15 minutes"
                                elif fitness_level == "Intermediate":
                                    duration = "15-20 minutes"
                                else:  # Advanced
                                    duration = "20-30 minutes"
                                
                                workout["exercises"].append({
                                    "name": exercise,
                                    "sets": 1,
                                    "reps": duration,
                                    "rest": "60 sec"
                                })
                            else:
                                # For cardio movements (not machines), use reps
                                workout["exercises"].append({
                                    "name": exercise,
                                    "sets": sets_reps["sets"],
                                    "reps": sets_reps["reps"],
                                    "rest": sets_reps["rest"]
                                })
                            
                            cardio_exercises.remove(exercise)
            else:
                # Make a copy to avoid modifying the original
                area_exercises = exercises[area].copy()
                
                # Calculate exercises per area with better distribution
                # If we have 3 areas and need 7 exercises total, distribute as 2-3-2 or similar
                base_per_area = max(1, (exercises_per_workout - len(workout["exercises"])) // 
                                len([a for a in workout_type["focus"] if a != "cardio"]))
                
                # Ensure we don't exceed our target count
                num_exercises = min(base_per_area, exercises_per_workout - len(workout["exercises"]))
                
                # Add exercises
                for _ in range(min(num_exercises, len(area_exercises))):
                    if area_exercises and len(workout["exercises"]) < exercises_per_workout:  
                        exercise = random.choice(area_exercises)
                        workout["exercises"].append({
                            "name": exercise,
                            "sets": sets_reps["sets"],
                            "reps": sets_reps["reps"],
                            "rest": sets_reps["rest"]
                        })
                        # Remove to avoid duplicates
                        area_exercises.remove(exercise)
        
        # Ensure no duplicate exercises
        existing_exercises = [ex["name"] for ex in workout["exercises"]]
        workout["exercises"] = [ex for i, ex in enumerate(workout["exercises"]) 
                             if ex["name"] not in existing_exercises[:i]]
        
        # Make sure we don't have duplicate cardio machines
        cardio_machines = ["Treadmill", "Elliptical", "Stationary bike", "Rowing machine", "Stair climber"]
        machine_count = sum(1 for ex in workout["exercises"] if any(machine in ex["name"] for machine in cardio_machines))
        if machine_count > 1:
            # Keep only one cardio machine
            kept_one = False
            filtered_exercises = []
            for ex in workout["exercises"]:
                if any(machine in ex["name"] for machine in cardio_machines):
                    if not kept_one:
                        filtered_exercises.append(ex)
                        kept_one = True
                else:
                    filtered_exercises.append(ex)
            workout["exercises"] = filtered_exercises
        
        # Remove duplicate 'Mountain climbers' or other exercises that might appear twice
        seen_exercises = set()
        unique_exercises = []
        for ex in workout["exercises"]:
            if ex["name"] not in seen_exercises:
                seen_exercises.add(ex["name"])
                unique_exercises.append(ex)
        workout["exercises"] = unique_exercises
        
        workout_days.append(workout)
    
    return workout_days

def create_download_link(pdf_bytes, filename):
    """Create a download link for the generated PDF"""
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download Workout Plan</a>'
    return href

def generate_pdf(workout_plan, user_data, time_per_session):
    """Generate PDF from workout plan data"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Add title
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=16,
        alignment=1,  # Center alignment
        spaceAfter=12
    )
    elements.append(Paragraph("Your Personal Workout Plan", title_style))
    elements.append(Spacer(1, 12))
    
    # Add user info
    user_info_style = ParagraphStyle(
        'UserInfo',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6
    )
    
    if user_data.get('name'):
        elements.append(Paragraph(f"Created for: {user_data.get('name')}", user_info_style))
    
    elements.append(Paragraph(f"Fitness Goal: {user_data.get('fitness_goal', 'Not specified')}", user_info_style))
    elements.append(Paragraph(f"Fitness Level: {user_data.get('fitness_level', 'Not specified')}", user_info_style))
    elements.append(Paragraph(f"Available Equipment: {', '.join(user_data.get('equipment', ['Not specified']))}", user_info_style))
    
    if user_data.get('injuries'):
        elements.append(Paragraph(f"Considerations: {user_data.get('injuries')}", user_info_style))
    
    elements.append(Spacer(1, 20))
    
    # Add workout schedule
    heading_style = ParagraphStyle(
        'Heading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=6
    )
    
    sub_heading_style = ParagraphStyle(
        'SubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=6
    )
    
    for day in workout_plan:
        # Day header
        elements.append(Paragraph(f"{day['day']}: {day['name']} Workout", heading_style))
        
        # Exercise table
        data = [["Exercise", "Sets", "Reps", "Rest"]]
        
        for exercise in day['exercises']:
            data.append([
                exercise['name'],
                str(exercise['sets']),
                exercise['reps'],
                exercise['rest']
            ])
        
        table = Table(data, colWidths=[250, 50, 100, 70])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 20))
    
    # Add tips section
    elements.append(Paragraph("Tips for Success", heading_style))
    
    tips = [
        "Start each workout with a 5-minute warm-up (light cardio and dynamic stretching).",
        "End each workout with a 5-minute cool-down (static stretching).",
        "Focus on proper form over heavy weights or high reps.",
        "Track your workouts in a journal or app to monitor progress.",
        "Stay hydrated before, during, and after your workouts.",
        "Allow at least 48 hours of rest for muscle groups between workouts.",
    ]
    
    for tip in tips:
        elements.append(Paragraph(f"• {tip}", styles["Normal"]))
    
    # Build PDF
    doc.build(elements)
    
    # Get PDF bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes

def save_current_workout(plan_name):
    """Save the current workout plan with a name"""
    workout_data = {
        "name": plan_name,
        "date_created": datetime.datetime.now().strftime("%Y-%m-%d"),
        "workout_plan": st.session_state.current_workout,
        "user_data": st.session_state.user_data,
        "time_per_session": st.session_state.time_per_session
    }
    
    st.session_state.saved_workouts.append(workout_data)
    st.success(f"Workout plan '{plan_name}' has been saved!")
    
def log_workout_completion(workout_index, day_index, notes=""):
    """Log the completion of a workout day"""
    workout = st.session_state.saved_workouts[workout_index]
    workout_name = workout["name"]
    day = workout["workout_plan"][day_index]["day"]
    
    # Create a unique key for this workout plan
    if workout_name not in st.session_state.workout_logs:
        st.session_state.workout_logs[workout_name] = []
    
    # Add log entry
    log_entry = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "day": day,
        "notes": notes
    }
    
    st.session_state.workout_logs[workout_name].append(log_entry)
    
    return True

def run_exercise_analysis(video_file, exercise_type, output_path=None):
    """
    Import and run the appropriate exercise analysis module
    
    Args:
        video_file: Uploaded video file from Streamlit
        exercise_type: Type of exercise to analyze
        output_path: Path to save processed video
        
    Returns:
        dict: Analysis results from the exercise module
    """
    # Get the module name for the selected exercise
    if exercise_type in EXERCISE_ANALYSIS_MODULES:
        module_name = EXERCISE_ANALYSIS_MODULES[exercise_type]
    else:
        return {"error": f"No analysis module found for {exercise_type}"}
    
    # Create a temporary file to store the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(video_file.read())
        video_path = temp_file.name
    
    try:
        # Dynamically import the exercise module
        # First, make sure the module's directory is in the Python path
        # Assuming the modules are in a directory called 'exercise_modules'
        exercise_module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exercise_modules')
        
        if exercise_module_dir not in sys.path:
            sys.path.append(exercise_module_dir)
        
        # Import the module
        try:
            module = importlib.import_module(module_name)
            
            # Call the main analysis function
            # Most modules should have an analyze_X function where X is the exercise name
            if hasattr(module, f'analyze_{module_name}'):
                analysis_function = getattr(module, f'analyze_{module_name}')
            elif hasattr(module, 'analyze_exercise'):
                analysis_function = getattr(module, 'analyze_exercise')
            elif hasattr(module, 'analyze_video'):
                analysis_function = getattr(module, 'analyze_video')
            else:
                # If we can't find a standard function name, try to find any function with 'analyze' in the name
                analyze_functions = [f for f in dir(module) if 'analyze' in f.lower() and callable(getattr(module, f))]
                
                if analyze_functions:
                    analysis_function = getattr(module, analyze_functions[0])
                else:
                    return {"error": f"Could not find analysis function in {module_name} module"}
            
            # Call the analyze function
            result = analysis_function(video_path, output_path)
            
            # Clean up the temporary file
            try:
                os.unlink(video_path)
            except:
                pass
                
            return result
            
        except ImportError as e:
            return {"error": f"Could not import {module_name} module: {str(e)}"}
        except Exception as e:
            return {"error": f"Error running analysis: {str(e)}"}
            
    except Exception as e:
        return {"error": f"Error setting up analysis: {str(e)}"}

# Form analysis tab function
def form_analysis_tab():
    st.title("📹 Workout Form Analysis")
    st.subheader("Upload a video to get feedback on your exercise form")
    
    # Exercise selection - show the list of available exercise modules
    exercise_type = st.selectbox(
        "Select exercise to analyze", 
        list(EXERCISE_ANALYSIS_MODULES.keys()),
        index=0
    )
    
    # Exercise instructions - these could be loaded from the modules themselves
    with st.expander("How to record your video for best results"):
        if exercise_type == "Push-up":
            st.markdown("""
            ### Push-up Video Tips
            1. **Camera Position**: Place your camera at side view, approximately 3-4 feet away.
            2. **Lighting**: Ensure the area is well-lit to improve pose detection.
            3. **Clothing**: Wear form-fitting clothing that contrasts with the background.
            4. **Frame**: Make sure your entire body is in the frame throughout the movement.
            5. **Speed**: Perform the push-ups at a controlled pace.
            6. **Repetitions**: Try to complete at least 3-5 repetitions for the best analysis.
            """)
        elif exercise_type == "Squat":
            st.markdown("""
            ### Squat Video Tips
            1. **Camera Position**: Place your camera at side view, approximately 5-6 feet away.
            2. **Lighting**: Ensure the area is well-lit to improve pose detection.
            3. **Clothing**: Wear form-fitting clothing that contrasts with the background.
            4. **Frame**: Make sure your entire body is in the frame throughout the movement.
            5. **Speed**: Perform the squats at a controlled pace.
            6. **Repetitions**: Try to complete at least 3-5 repetitions for the best analysis.
            """)
        else:
            st.markdown(f"""
            ### {exercise_type} Video Tips
            1. **Camera Position**: Place your camera at a position where your full body and movement are visible.
            2. **Lighting**: Ensure the area is well-lit to improve pose detection.
            3. **Clothing**: Wear form-fitting clothing that contrasts with the background.
            4. **Frame**: Make sure your entire body is in the frame throughout the movement.
            5. **Speed**: Perform the exercise at a controlled pace.
            6. **Repetitions**: Try to complete at least 3-5 repetitions for the best analysis.
            """)
    
    # Video upload section
    uploaded_file = st.file_uploader("Upload your exercise video", type=['mp4', 'mov', 'avi'])
    
    if uploaded_file is not None:
        # Create output path for processed video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
            output_path = tmpfile.name
        
        # Run analysis with a progress indicator
        with st.spinner(f"Analyzing your {exercise_type.lower()} form..."):
            # Run the appropriate module's analysis function
            analysis_results = run_exercise_analysis(
                uploaded_file, 
                exercise_type, 
                output_path=output_path
            )
            
            # Store results in session state
            st.session_state.analysis_results = analysis_results
            st.session_state.processed_video_path = output_path
        
        # Display results
        if "error" in analysis_results:
            st.error(analysis_results["error"])
        else:
            st.success("Analysis complete!")
            
            # Check if the output video file exists and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                # Display processed video
                video_file = open(output_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
            else:
                st.warning("No processed video was generated. Showing analysis results only.")
            
            # Create two columns for results display
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Performance Summary")
                
                # Display metrics based on what's available in the results
                if "count" in analysis_results:
                    st.metric("Repetitions Counted", analysis_results["count"])
                
                if "form_analysis" in analysis_results:
                    form = analysis_results["form_analysis"]
                    
                    # Display metrics based on exercise type and available data
                    if exercise_type == "Push-up":
                        if "body_alignment_score" in form:
                            st.metric("Body Alignment", f"{form['body_alignment_score']:.1f}%")
                        if "elbow_angle_at_bottom" in form:
                            st.metric("Elbow Angle (Bottom)", f"{form['elbow_angle_at_bottom']:.1f}°")
                    elif exercise_type == "Squat":
                        if "depth_score" in form:
                            st.metric("Depth Score", f"{form['depth_score']:.1f}%")
                        if "knee_angle_at_bottom" in form:
                            st.metric("Knee Angle (Bottom)", f"{form['knee_angle_at_bottom']:.1f}°")
                    
                    # Display any other metrics that might be in the results
                    for key, value in form.items():
                        if key not in ["body_alignment_score", "elbow_angle_at_bottom", "depth_score", "knee_angle_at_bottom", "frames_analyzed"]:
                            # Format the metric name for display
                            metric_name = " ".join(word.capitalize() for word in key.split("_"))
                            
                            # Format the value based on its type
                            if isinstance(value, float):
                                formatted_value = f"{value:.1f}"
                                # Add units if we can infer them
                                if "angle" in key.lower():
                                    formatted_value += "°"
                                elif "score" in key.lower() or "percentage" in key.lower():
                                    formatted_value += "%"
                            else:
                                formatted_value = str(value)
                                
                            st.metric(metric_name, formatted_value)
            
            with col2:
                st.subheader("Feedback & Tips")
                
                # Display feedback from the analysis
                if "feedback" in analysis_results:
                    for feedback in analysis_results["feedback"]:
                        st.markdown(f"- {feedback}")
                
                # Add general tips based on exercise
                st.markdown("### General Tips")
                if exercise_type == "Push-up":
                    st.markdown("""
                    - Keep your core tight throughout the movement
                    - Breathe out as you push up, breathe in as you lower down
                    - Focus on quality rather than quantity
                    - For more challenge, try diamond push-ups or decline push-ups
                    """)
                elif exercise_type == "Squat":
                    st.markdown("""
                    - Keep your weight in your heels
                    - Push your knees outward as you descend
                    - Breathe in as you lower, breathe out as you rise
                    - Focus on controlled movement rather than speed
                    - For more challenge, try holding weights or single-leg variations
                    """)
                else:
                    # Generic tips
                    st.markdown("""
                    - Focus on proper form over speed or repetitions
                    - Control your breathing with the exercise rhythm
                    - Engage your core throughout the movement
                    - Record yourself regularly to track improvements
                    """)
            
            # Option to save this analysis to workout log
            st.subheader("Log This Workout")
            log_notes = st.text_area("Add notes about this workout session:", 
                                    placeholder="How did it feel? What was challenging?")
            
            # Only show this option if there are saved workouts
            if st.session_state.saved_workouts:
                workout_options = [workout["name"] for workout in st.session_state.saved_workouts]
                selected_workout = st.selectbox("Add to workout plan:", ["-- Select a workout plan --"] + workout_options)
                
                if st.button("Log Exercise Session") and selected_workout != "-- Select a workout plan --":
                    # Find the selected workout
                    workout_index = workout_options.index(selected_workout)
                    
                    # Create a log entry for this workout
                    if selected_workout not in st.session_state.workout_logs:
                        st.session_state.workout_logs[selected_workout] = []
                    
                    # Add log entry
                    log_entry = {
                        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "exercise": exercise_type,
                        "reps": analysis_results.get("count", 0),
                        "notes": log_notes
                    }
                    
                    st.session_state.workout_logs[selected_workout].append(log_entry)
                    st.success(f"Logged {analysis_results.get('count', 0)} {exercise_type} repetitions to {selected_workout}!")
            else:
                st.info("Create a workout plan in the 'Generate' tab to log this exercise session.")
    else:
        # Placeholder for video upload
        st.info("Upload a video to analyze your exercise form. Make sure you're visible in the frame and performing the exercise from the side view for best results.")
        
# Tab functions
def generate_workout_tab():
    st.title("💪 Fitness Buddy")
    st.subheader("Get a personalized weekly workout plan")
    
    # Create columns for the form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=16, max_value=90, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        current_weight = st.number_input("Current Weight (lbs)", min_value=50, max_value=500, value=150)
        goal_weight = st.number_input("Goal Weight (lbs)", min_value=50, max_value=500, value=150)
        height = st.number_input("Height (inches)", min_value=36, max_value=96, value=68)
    
    with col2:
        st.subheader("Fitness Goals & Preferences")
        fitness_goal = st.selectbox("Primary Fitness Goal", 
                                 ["Weight Loss", "Muscle Gain", "Endurance", "General Fitness", "Strength"])
        fitness_level = st.selectbox("Fitness Level", ["Beginner", "Intermediate", "Advanced"])
        days_per_week = st.slider("Days Available Per Week", min_value=2, max_value=6, value=3)
        time_per_session = st.select_slider("Time Available Per Session (minutes)", 
                                       options=[30, 45, 60, 90], value=45)
        injuries = st.text_area("Any Injuries or Limitations?", placeholder="E.g., knee pain, shoulder issues...")
        preferences = st.text_area("Personal Preferences", placeholder="E.g., prefer cardio, hate lunges...")
    
    # Equipment selection
    st.subheader("Available Equipment")
    equipment_options = ["None (bodyweight only)", "Dumbbells", "Resistance bands", 
                      "Kettlebells", "Pull-up bar", "Bench", "Full gym access"]
    equipment = st.multiselect("Select available equipment", equipment_options)
    
    # Generate workout button
    if st.button("Generate Workout Plan", type="primary"):
        if not equipment:
            equipment = ["None (bodyweight only)"]
        
        # Generate the workout plan
        selected_exercises = determine_exercises(equipment, fitness_goal, fitness_level, injuries)
        workout_plan = create_workout_schedule(selected_exercises, days_per_week, time_per_session, fitness_level)
        
        # Store workout plan in session state
        st.session_state.current_workout = workout_plan
        st.session_state.user_data = {
            'name': name,
            'fitness_goal': fitness_goal,
            'fitness_level': fitness_level,
            'equipment': equipment,
            'injuries': injuries
        }
        st.session_state.time_per_session = time_per_session
        
        # Display the workout plan
        st.subheader("Your Personalized Weekly Workout Plan")
        
        # Create tabs for each day
        tabs = st.tabs([f"Day {i+1}: {workout['name']}" for i, workout in enumerate(workout_plan)])
        
        for i, tab in enumerate(tabs):
            with tab:
                day = workout_plan[i]
                
                # Create a header with the workout focus
                st.markdown(f"### {day['name']} Workout")
                
                # Add a description based on the workout type
                workout_descriptions = {
                    "Full Body": "This workout targets all major muscle groups for balanced total-body conditioning.",
                    "Upper Body": "Focus on developing strength and definition in your chest, back, shoulders, and arms.",
                    "Lower Body": "Build strong legs and glutes with these targeted lower body exercises.",
                    "HIIT & Cardio": "Elevate your heart rate and burn calories with these high-intensity movements.",
                    "Core & Mobility": "Strengthen your core and improve flexibility with these targeted exercises."
                }
                
                if day['name'] in workout_descriptions:
                    st.markdown(f"*{workout_descriptions[day['name']]}*")
                
                # Convert to DataFrame for nicer display
                exercises_df = pd.DataFrame([
                    {
                        "Exercise": ex["name"],
                        "Sets": ex["sets"],
                        "Reps": ex["reps"],
                        "Rest": ex["rest"]
                    } for ex in day["exercises"]
                ])
                
                st.table(exercises_df)
                
                # Add workout notes
                if fitness_level == "Beginner":
                    st.info("💡 **Beginner Tips**: Focus on proper form rather than intensity. Take extra rest between sets if needed.")
                elif fitness_level == "Intermediate":
                    st.info("💡 **Workout Tips**: Challenge yourself with the appropriate weight/resistance that makes the last 2-3 reps difficult.")
                else:
                    st.info("💡 **Advanced Tips**: Consider adding progressive overload by increasing weight, reps, or reducing rest time each week.")
        
        # Create a summary section for the complete workout plan
        st.subheader("Weekly Workout Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Workout Days", days_per_week)
        
        with col2:
            total_exercises = sum(len(day["exercises"]) for day in workout_plan)
            st.metric("Total Exercises", total_exercises)
            
        with col3:
            st.metric("Workout Duration", f"{time_per_session} mins")
        
        # Tips for success
        st.subheader("Tips for Success")
        
        # Different tips based on fitness goal
        if fitness_goal == "Weight Loss":
            goal_tips = [
                "Consider adding 10-15 minutes of extra cardio at the end of your workout",
                "Focus on keeping rest periods shorter (30-45 seconds) to maintain elevated heart rate",
                "Nutrition plays a vital role - aim for a small caloric deficit and adequate protein intake",
                "Track your workouts and progress to stay motivated"
            ]
        elif fitness_goal == "Muscle Gain":
            goal_tips = [
                "Ensure you're eating in a slight caloric surplus with plenty of protein (1.6-2.2g per kg of bodyweight)",
                "Prioritize compound movements and progressive overload (gradually increasing weight)",
                "Get adequate sleep (7-9 hours) for optimal recovery and muscle growth",
                "Consider split training as you advance to allow for more volume per muscle group"
            ]
        elif fitness_goal == "Endurance":
            goal_tips = [
                "Focus on maintaining good form even as you fatigue",
                "Gradually increase workout duration over time",
                "Stay well-hydrated before, during, and after workouts",
                "Incorporate active recovery days with light activity"
            ]
        else:  # General Fitness or Strength
            goal_tips = [
                "Focus on consistent progress rather than rapid changes",
                "Mix up your routine every 4-6 weeks to prevent plateaus",
                "Listen to your body and adjust intensity as needed",
                "Balance your training with proper nutrition and recovery"
            ]
            
        # General tips for everyone
        general_tips = [
            "Start each workout with a 5-minute warm-up (light cardio and dynamic stretching)",
            "End each workout with a 5-minute cool-down (static stretching)",
            "Stay hydrated throughout your workouts",
            "Focus on proper form over heavy weights or high reps",
            "Allow at least 48 hours of rest for muscle groups between workouts"
        ]
        
        # Display tips in two columns
        tip_col1, tip_col2 = st.columns(2)
        
        with tip_col1:
            st.markdown("#### General Workout Tips")
            for tip in general_tips:
                st.markdown(f"- {tip}")
                
        with tip_col2:
            st.markdown(f"#### Tips for {fitness_goal}")
            for tip in goal_tips:
                st.markdown(f"- {tip}")
        
        # Add save workout option
        st.subheader("Save This Workout Plan")
        plan_name = st.text_input("Name for this workout plan:", placeholder="E.g., Summer Muscle Plan")
        
        if st.button("Save Workout Plan"):
            if plan_name:
                save_current_workout(plan_name)
                st.success(f"Workout plan '{plan_name}' saved! Go to the 'My Workouts' tab to view it.")
            else:
                st.error("Please enter a name for your workout plan.")
        
        # PDF Download button
        if st.button("Download Workout Plan as PDF"):
            # Generate PDF
            pdf_bytes = generate_pdf(
                workout_plan=st.session_state.current_workout,
                user_data=st.session_state.user_data,
                time_per_session=st.session_state.time_per_session
            )
            
            # Create a download button instead of a link
            st.download_button(
                label="📥 Click here to download your workout plan",
                data=pdf_bytes,
                file_name="workout_plan.pdf",
                mime="application/pdf"
            )
            st.success("Your workout plan PDF is ready! Click the button above to download.")

def my_workouts_tab():
    st.title("My Saved Workout Plans")
    
    if not st.session_state.saved_workouts:
        st.info("You haven't saved any workout plans yet. Go to the 'Generate' tab to create one!")
        return
    
    # Display saved workouts
    for i, workout in enumerate(st.session_state.saved_workouts):
        with st.expander(f"{workout['name']} - Created: {workout['date_created']}"):
            st.subheader(f"{workout['name']} Details")
            st.write(f"**Goal:** {workout['user_data']['fitness_goal']}")
            st.write(f"**Level:** {workout['user_data']['fitness_level']}")
            st.write(f"**Equipment:** {', '.join(workout['user_data']['equipment'])}")
            st.write(f"**Session Duration:** {workout['time_per_session']} minutes")
            
            # Display workout days in tabs
            day_tabs = st.tabs([f"Day {j+1}: {day['name']}" for j, day in enumerate(workout['workout_plan'])])
            
            for j, tab in enumerate(day_tabs):
                with tab:
                    day = workout['workout_plan'][j]
                    st.markdown(f"### {day['name']} Workout")
                    
                    # Create DataFrame for exercises
                    exercises_df = pd.DataFrame([
                        {
                            "Exercise": ex["name"],
                            "Sets": ex["sets"],
                            "Reps": ex["reps"],
                            "Rest": ex["rest"]
                        } for ex in day["exercises"]
                    ])
                    
                    st.table(exercises_df)
                    
                    # Log completion button for this day
                    notes = st.text_area(f"Workout Notes (Day {j+1})", key=f"notes_{i}_{j}", 
                                         placeholder="Enter notes about your workout (weights used, how you felt, etc.)")
                    
                    if st.button(f"Log Completion (Day {j+1})", key=f"log_{i}_{j}"):
                        if log_workout_completion(i, j, notes):
                            st.success(f"Workout Day {j+1} logged successfully!")
            
            # Generate PDF for this saved workout
            if st.button(f"Download as PDF", key=f"pdf_{i}"):
                pdf_bytes = generate_pdf(
                    workout_plan=workout['workout_plan'],
                    user_data=workout['user_data'],
                    time_per_session=workout['time_per_session']
                )
                
                st.download_button(
                    label=f"📥 Download {workout['name']} as PDF",
                    data=pdf_bytes,
                    file_name=f"{workout['name'].replace(' ', '_').lower()}_workout.pdf",
                    mime="application/pdf",
                    key=f"download_{i}"
                )

def progress_tracking_tab():
    st.title("Workout Progress Tracker")
    
    if not st.session_state.workout_logs:
        st.info("You haven't logged any workouts yet. Go to 'My Workouts' tab to log your completed workouts!")
        return
    
    # Display progress metrics
    st.subheader("Workout Completion Statistics")
    
    # Calculate statistics
    total_workouts = sum(len(logs) for logs in st.session_state.workout_logs.values())
    current_week_workouts = sum(
        1 for logs in st.session_state.workout_logs.values() 
        for log in logs 
        if (datetime.datetime.now() - datetime.datetime.strptime(log['date'], "%Y-%m-%d")).days <= 7
    )
    
    # Create metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Workouts Completed", total_workouts)
    
    with col2:
        st.metric("This Week's Workouts", current_week_workouts)
        
    with col3:
        if total_workouts > 0:
            st.metric("Workout Plans Used", len(st.session_state.workout_logs))
        else:
            st.metric("Workout Plans Used", 0)
    
    # Display workout history
    st.subheader("Workout History")
    
    # Create a table with all logged workouts
    if total_workouts > 0:
        log_data = []
        
        for plan_name, logs in st.session_state.workout_logs.items():
            for log in logs:
                log_entry = {
                    "Date": log["date"],
                    "Workout Plan": plan_name,
                }
                
                # Add day or exercise info based on what's available
                if "day" in log:
                    log_entry["Activity"] = log["day"]
                elif "exercise" in log:
                    log_entry["Activity"] = log["exercise"]
                    if "reps" in log:
                        log_entry["Reps"] = log["reps"]
                else:
                    log_entry["Activity"] = "Workout"
                
                log_entry["Notes"] = log["notes"] if log["notes"] else "-"
                log_data.append(log_entry)
        
        # Convert to DataFrame and sort by date (most recent first)
        log_df = pd.DataFrame(log_data)
        log_df = log_df.sort_values("Date", ascending=False)
        
        # Display as table
        st.dataframe(log_df, hide_index=True)
        
        # Option to download workout history
        if st.button("Download Workout History"):
            # Convert DataFrame to CSV
            csv = log_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Workout History CSV",
                data=csv,
                file_name="workout_history.csv",
                mime="text/csv"
            )
    else:
        st.info("No workout history to display yet.")

# Main app
def main():
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Generate", "My Workouts", "Progress Tracker", "Form Analysis"])
    
    with tab1:
        generate_workout_tab()
    
    with tab2:
        my_workouts_tab()
    
    with tab3:
        progress_tracking_tab()
        
    with tab4:
        form_analysis_tab()

if __name__ == "__main__":
    main()