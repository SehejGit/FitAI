import streamlit as st
import random
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Fitness Buddy",
    page_icon="💪",
    layout="wide"
)

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
                            
                # Adjust remaining exercise count after adding cardio
                remaining_exercises = exercises_per_workout - len(workout["exercises"])
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

# Streamlit app
def main():
    # App title
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
        
        # Add a section for tracking progress
        st.subheader("Tracking Your Progress")
        st.markdown("""
        For best results, keep track of your workouts. Record:
        - Weights used
        - Reps completed
        - How you felt during/after the workout
        - Any exercises you want to modify
        
        Aim to progressively increase either weight, reps, or sets each week for continued improvement.
        """)
        
        # Download option (placeholder)
        if st.button("Download Workout Plan as PDF"):
            st.warning("This feature would generate a PDF in a production app. For now, you can take screenshots of your workout plan.")

if __name__ == "__main__":
    main()