import * as React from 'react';
import { useState } from 'react';
import './App.css';
import { Container, Typography, Box } from '@mui/material';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import UserInfoForm from './components/UserInfoForm';
import WorkoutPlan from './components/WorkoutPlan';
import VideoPlayer from './components/VideoPlayer';
import NavigationBar from './components/NavigationBar';
import SavedWorkouts from './components/SavedWorkouts';
import { saveWorkoutPlan } from './services/workoutService';
import AiWorkoutForm from './components/AiWorkoutForm';

// Define types for our exercise library
type ExerciseArea = 'upper' | 'lower' | 'core' | 'full_body' | 'cardio';

// Define types for user input
interface UserInfo {
  name: string;
  age: number;
  gender: string;
  currentWeight: number;
  goalWeight: number;
  height: number;
  fitnessGoal: string;
  fitnessLevel: string;
  daysPerWeek: number;
  timePerSession: number;
  injuries: string;
  preferences: string;
  equipment: string[];
}

// Define types for workout plan
interface Exercise {
  name: string;
  sets: number;
  reps: string;
  rest: string;
}

interface WorkoutDay {
  day: string;
  name: string;
  exercises: Exercise[];
}

// Exercise library data
const EXERCISE_LIBRARY = {
  bodyweight: {
    upper: ["Push-ups", "Tricep dips", "Plank shoulder taps", "Pike push-ups"],
    lower: ["Squats", "Lunges", "Glute bridges", "Calf raises"],
    core: ["Planks", "Mountain climbers", "Bicycle crunches", "Russian twists"],
    full_body: ["Burpees", "Jumping jacks", "Mountain climbers", "Bear crawls"]
  },
  dumbbell: {
    upper: ["Dumbbell press", "Dumbbell rows", "Shoulder press", "Bicep curls", "Tricep extensions"],
    lower: ["Goblet squats", "Dumbbell lunges", "Romanian deadlifts", "Step-ups"],
    core: ["Dumbbell russian twists", "Weighted sit-ups", "Side bends"],
    full_body: ["Dumbbell thrusters", "Renegade rows", "Man makers"]
  },
  bands: {
    upper: ["Band pull-aparts", "Banded push-ups", "Band bicep curls", "Band tricep pushdowns"],
    lower: ["Banded squats", "Banded glute bridges", "Lateral band walks", "Banded hip thrusts"],
    core: ["Banded russian twists", "Banded mountain climbers", "Pallof press"],
    full_body: ["Banded jumping jacks", "Banded burpees"]
  },
  gym: {
    upper: ["Bench press", "Lat pulldowns", "Cable rows", "Shoulder press machine", "Chest fly machine"],
    lower: ["Leg press", "Leg extensions", "Leg curls", "Hip abduction/adduction", "Calf raise machine"],
    core: ["Cable crunches", "Ab machine", "Hanging leg raises"],
    full_body: ["Assisted pull-ups", "Rowing machine", "Elliptical"]
  },
  cardio: ["Running in place", "High knees", "Jumping jacks", "Shadow boxing", 
          "Treadmill", "Stair climber", "Stationary bike", "Elliptical"]
};

const App: React.FC = () => {
  const [userInfo, setUserInfo] = useState<UserInfo | null>(null);
  const [workoutPlan, setWorkoutPlan] = useState<WorkoutDay[] | null>(null);
  const [showWorkout, setShowWorkout] = useState<boolean>(false);
  const [aiInsights, setAiInsights] = useState<any>(null);
  const [useAiGeneration, setUseAiGeneration] = useState<boolean>(false);

  const determineExercises = (equipment: string[], fitnessGoal: string, fitnessLevel: string, injuries: string) => {
    const hasInjuries = injuries.trim() !== "";
    
    // Start with bodyweight exercises
    const selectedExercises = {
      upper: [...EXERCISE_LIBRARY.bodyweight.upper],
      lower: [...EXERCISE_LIBRARY.bodyweight.lower],
      core: [...EXERCISE_LIBRARY.bodyweight.core],
      full_body: [...EXERCISE_LIBRARY.bodyweight.full_body],
      cardio: ["Running in place", "High knees", "Jumping jacks", "Shadow boxing"]
    };
    
    // Add equipment-based exercises
    if (equipment.includes("Dumbbells")) {
      selectedExercises.upper.push(...EXERCISE_LIBRARY.dumbbell.upper);
      selectedExercises.lower.push(...EXERCISE_LIBRARY.dumbbell.lower);
      selectedExercises.core.push(...EXERCISE_LIBRARY.dumbbell.core);
      selectedExercises.full_body.push(...EXERCISE_LIBRARY.dumbbell.full_body);
    }
    
    if (equipment.includes("Resistance bands")) {
      selectedExercises.upper.push(...EXERCISE_LIBRARY.bands.upper);
      selectedExercises.lower.push(...EXERCISE_LIBRARY.bands.lower);
      selectedExercises.core.push(...EXERCISE_LIBRARY.bands.core);
      selectedExercises.full_body.push(...EXERCISE_LIBRARY.bands.full_body);
    }
    
    if (equipment.includes("Full gym access")) {
      selectedExercises.upper.push(...EXERCISE_LIBRARY.gym.upper);
      selectedExercises.lower.push(...EXERCISE_LIBRARY.gym.lower);
      selectedExercises.core.push(...EXERCISE_LIBRARY.gym.core);
      selectedExercises.full_body.push(...EXERCISE_LIBRARY.gym.full_body);
      selectedExercises.cardio.push("Treadmill", "Stair climber", "Stationary bike", "Elliptical");
    }
    
    // Remove high-impact exercises if user has injuries
    if (hasInjuries) {
      const highImpact = ["Burpees", "Jumping", "Jump", "Running", "High knees"];
      for (const area in selectedExercises) {
        selectedExercises[area as ExerciseArea] = selectedExercises[area as ExerciseArea].filter(
          ex => !highImpact.some(impact => ex.includes(impact))
        );
      }
    }
    
    return selectedExercises;
  };

  const createWorkoutSchedule = (
    exercises: { [key in ExerciseArea]: string[] },
    daysPerWeek: number,
    timePerSession: number,
    fitnessLevel: string
  ): WorkoutDay[] => {
    const workoutDays: WorkoutDay[] = [];
    
    // Number of exercises based on time per session
    const exercisesPerWorkout = Math.max(3, Math.floor(timePerSession / 10));
    
    // Determine sets and reps based on fitness level
    let setsReps = { sets: 2, reps: "8-10", rest: "60 sec" };
    if (fitnessLevel === "Intermediate") {
      setsReps = { sets: 3, reps: "10-12", rest: "45 sec" };
    } else if (fitnessLevel === "Advanced") {
      setsReps = { sets: 4, reps: "12-15", rest: "30 sec" };
    }
    
    // Define workout types
    const workoutTypes = [
      { name: "Full Body", focus: ["upper", "lower", "core"] as ExerciseArea[] },
      { name: "Upper Body", focus: ["upper", "core"] as ExerciseArea[] },
      { name: "Lower Body", focus: ["lower", "core"] as ExerciseArea[] },
      { name: "HIIT & Cardio", focus: ["full_body", "cardio"] as ExerciseArea[] },
      { name: "Core & Mobility", focus: ["core", "full_body"] as ExerciseArea[] }
    ];
    
    // Distribute workout types throughout the week
    for (let i = 0; i < daysPerWeek; i++) {
      const workoutType = workoutTypes[i % workoutTypes.length];
      const workout: WorkoutDay = {
        day: `Day ${i + 1}`,
        name: workoutType.name,
        exercises: []
      };
      
      // Add exercises from each focus area
      for (const area of workoutType.focus) {
        // Make a copy to avoid modifying the original
        const areaExercises = [...exercises[area]];
        const numExercises = Math.max(1, Math.floor(exercisesPerWorkout / workoutType.focus.length));
        
        // Add exercises
        for (let j = 0; j < Math.min(numExercises, areaExercises.length); j++) {
          if (areaExercises.length > 0) {
            const randomIndex = Math.floor(Math.random() * areaExercises.length);
            const exercise = areaExercises[randomIndex];
            workout.exercises.push({
              name: exercise,
              sets: setsReps.sets,
              reps: setsReps.reps,
              rest: setsReps.rest
            });
            // Remove to avoid duplicates
            areaExercises.splice(randomIndex, 1);
          }
        }
      }
      
      workoutDays.push(workout);
    }
    
    return workoutDays;
  };

  const handleGenerateWorkout = (formData: UserInfo) => {
    setUserInfo(formData);
    
    if (!formData.equipment || formData.equipment.length === 0) {
      formData.equipment = ["None (bodyweight only)"];
    }
    
    // Generate the workout plan using regular logic
    const selectedExercises = determineExercises(
      formData.equipment,
      formData.fitnessGoal,
      formData.fitnessLevel,
      formData.injuries
    );
    
    const workoutPlan = createWorkoutSchedule(
      selectedExercises,
      formData.daysPerWeek,
      formData.timePerSession,
      formData.fitnessLevel
    );
    
    setWorkoutPlan(workoutPlan);
    setAiInsights(null); // Clear any previous AI insights
    setShowWorkout(true);
  };

  const handleAiGenerateWorkout = (aiGeneratedPlan: WorkoutDay[], insights: any) => {
    setWorkoutPlan(aiGeneratedPlan);
    setAiInsights(insights);
    setUseAiGeneration(true);
    setShowWorkout(true);
  };

  const handleResetWorkout = () => {
    setShowWorkout(false);
    setUseAiGeneration(false);
    setAiInsights(null);
  };

  return (
    <Router>
      <NavigationBar />
      <Container maxWidth="lg" className="App">
        <Box sx={{ my: 4, textAlign: 'center' }}>
          <Typography variant="h2" component="h1" gutterBottom>
            💪 Fitness Buddy
          </Typography>
          <Typography variant="h5" component="h2" gutterBottom>
            Get a personalized weekly workout plan
          </Typography>
        </Box>

        <Routes>
          <Route path="/" element={
            !showWorkout ? (
              <Box>
                <UserInfoForm onGenerateWorkout={handleGenerateWorkout} />
                
                {/* Add AI workout generation option if user has filled the form */}
                {/* {userInfo && (
                  <Box sx={{ mt: 3 }}>
                    <Typography variant="h6" gutterBottom sx={{ textAlign: 'center' }}>
                      Or generate an AI-enhanced workout plan
                    </Typography>
                    <AiWorkoutForm 
                      userInfo={userInfo} 
                      onGenerateWorkout={handleAiGenerateWorkout} 
                    />
                  </Box>
                )} */}
              </Box>
            ) : (
              <WorkoutPlan 
                workoutDays={workoutPlan || []} 
                aiInsights={aiInsights}
                onCreateNewPlan={handleResetWorkout} 
                onSaveWorkout={async (planName) => {
                  try {
                    // Format the workout data according to your API's expected structure
                    const workoutData = {
                      name: planName,
                      date_created: new Date().toISOString().split('T')[0], // YYYY-MM-DD format
                      workout_plan: workoutPlan || [],
                      user_data: {
                        ...(userInfo || {}),
                        equipment: userInfo?.equipment || ["None (bodyweight only)"],
                        fitness_goal: userInfo?.fitnessGoal || "General Fitness",
                        fitness_level: userInfo?.fitnessLevel || "Beginner"
                      },
                      time_per_session: userInfo?.timePerSession || 30
                    };
                    
                    // Call your API to save the workout
                    const result = await saveWorkoutPlan(workoutData);
                    console.log('Workout saved successfully:', result);
                    
                    // Show success message (you already have this UI)
                  } catch (error) {
                    console.error('Error saving workout:', error);
                    // Optionally show an error message to the user
                  }
                }}
              />
            )
          } />
          
          <Route path="/video/:day/:exercise" element={<VideoPlayer />} />
          
          <Route path="*" element={<Navigate to="/" replace />} />
          <Route path="/saved-workouts" element={<SavedWorkouts />} />
        </Routes>
      </Container>
    </Router>
  );
};

export default App;