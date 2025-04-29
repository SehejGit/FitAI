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
    
    if "Resistance bands" in equipment:
        selected_exercises["upper"].extend(EXERCISE_LIBRARY["bands"]["upper"])
        selected_exercises["lower"].extend(EXERCISE_LIBRARY["bands"]["lower"])
        selected_exercises["core"].extend(EXERCISE_LIBRARY["bands"]["core"])
        selected_exercises["full_body"].extend(EXERCISE_LIBRARY["bands"]["full_body"])
    
    if "Full gym access" in equipment:
        selected_exercises["upper"].extend(EXERCISE_LIBRARY["gym"]["upper"])
        selected_exercises["lower"].extend(EXERCISE_LIBRARY["gym"]["lower"])
        selected_exercises["core"].extend(EXERCISE_LIBRARY["gym"]["core"])
        selected_exercises["full_body"].extend(EXERCISE_LIBRARY["gym"]["full_body"])
        selected_exercises["cardio"].extend(["Treadmill", "Stair climber", "Stationary bike", "Elliptical"])
    
    # Remove high-impact exercises if user has injuries
    if has_injuries:
        high_impact = ["Burpees", "Jumping", "Jump", "Running", "High knees"]
        for area in selected_exercises:
            selected_exercises[area] = [ex for ex in selected_exercises[area] 
                                       if not any(impact in ex for impact in high_impact)]
    
    # Modify based on fitness goal
    if fitness_goal == "Weight Loss":
        # Prioritize cardio and full-body exercises
        pass  # The workout schedule creation will handle this
    elif fitness_goal == "Muscle Gain":
        # Prioritize resistance exercises
        pass  # The workout schedule creation will handle this
    
    return selected_exercises

def create_workout_schedule(exercises, days_per_week, time_per_session, fitness_level):
    """Create a personalized workout schedule"""
    workout_days = []
    
    # Number of exercises based on time per session
    exercises_per_workout = max(3, time_per_session // 10)
    
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
    
    # Distribute workout types throughout the week
    for i in range(days_per_week):
        workout_type = workout_types[i % len(workout_types)]
        workout = {
            "day": f"Day {i + 1}",
            "name": workout_type["name"],
            "exercises": []
        }
        
        # Add exercises from each focus area
        for area in workout_type["focus"]:
            # Make a copy to avoid modifying the original
            area_exercises = exercises[area].copy()
            num_exercises = max(1, exercises_per_workout // len(workout_type["focus"]))
            
            # Add exercises
            for _ in range(min(num_exercises, len(area_exercises))):
                if area_exercises:  # Check if there are exercises available
                    exercise = random.choice(area_exercises)
                    workout["exercises"].append({
                        "name": exercise,
                        "sets": sets_reps["sets"],
                        "reps": sets_reps["reps"],
                        "rest": sets_reps["rest"]
                    })
                    # Remove to avoid duplicates
                    area_exercises.remove(exercise)
        
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
                                           options=[30, 45, 60, 90], value=30)
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
        
        # Tips for success
        st.subheader("Tips for Success")
        tips = [
            "Start each workout with a 5-minute warm-up (light cardio and dynamic stretching)",
            "End each workout with a 5-minute cool-down (static stretching)",
            "Stay hydrated throughout your workouts",
            "Focus on proper form over heavy weights or high reps",
            "Progressively increase intensity as you get stronger",
            "Allow at least 48 hours of rest for muscle groups between workouts"
        ]
        
        for tip in tips:
            st.markdown(f"- {tip}")
        
        # Save workout plan option
        if st.button("Save Workout Plan (PDF)"):
            st.warning("This feature would generate a PDF in a production app")

if __name__ == "__main__":
    main()