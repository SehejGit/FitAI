import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  Button, 
  TextField,
  Paper, 
  Tabs, 
  Tab, 
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow,
  Alert,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle
} from '@mui/material';
import { Link } from 'react-router-dom';
import AiInsights from './AiInsights';

// Define prop types
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

interface WorkoutPlanProps {
  workoutDays: WorkoutDay[];
  aiInsights?: any;
  onCreateNewPlan: () => void;
  onSaveWorkout: (planName: string) => Promise<void> | void;
}

const WorkoutPlan: React.FC<WorkoutPlanProps> = ({ workoutDays, aiInsights, onCreateNewPlan, onSaveWorkout }) => {
  const [tabValue, setTabValue] = useState(0);
  const [planName, setPlanName] = useState("");
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleSaveWorkout = async () => {
    if (planName.trim()) {
      try {
        // Call the provided save function, which may be async
        await onSaveWorkout(planName);
        
        // Close dialog and show success message
        setShowSaveDialog(false);
        setSaveSuccess(true);
        
        // Reset success message after 3 seconds
        setTimeout(() => setSaveSuccess(false), 3000);
      } catch (error) {
        console.error('Error saving workout plan:', error);
      }
    }
  };

  // Get workout descriptions based on the workout type
  const getWorkoutDescription = (workoutName: string) => {
    const descriptions: { [key: string]: string } = {
      "Full Body": "This workout targets all major muscle groups for balanced total-body conditioning.",
      "Upper Body": "Focus on developing strength and definition in your chest, back, shoulders, and arms.",
      "Lower Body": "Build strong legs and glutes with these targeted lower body exercises.",
      "HIIT & Cardio": "Elevate your heart rate and burn calories with these high-intensity movements.",
      "Core & Mobility": "Strengthen your core and improve flexibility with these targeted exercises."
    };
    
    return descriptions[workoutName] || "A customized workout to help you reach your fitness goals.";
  };

  const getWorkoutTips = (workoutName: string) => {
    const tips: { [key: string]: string[] } = {
      "Full Body": [
        "Rest at least 48 hours before working the same muscle groups again",
        "Focus on compound movements that work multiple muscle groups",
        "Adjust weights to challenge yourself while maintaining proper form"
      ],
      "Upper Body": [
        "Balance pushing and pulling movements for overall development",
        "Don't neglect the rear deltoids and upper back",
        "For best results, keep your core engaged during all exercises"
      ],
      "Lower Body": [
        "Drive through your heels on squats and lunges",
        "Keep your knees aligned with your toes on all exercises",
        "Engage your glutes at the top of hip hinge movements"
      ],
      "HIIT & Cardio": [
        "Focus on intensity during work intervals",
        "Control your breathing throughout the workout",
        "Modify exercises as needed to match your fitness level"
      ],
      "Core & Mobility": [
        "Focus on controlled movements rather than speed",
        "Breathe through the exercises and avoid holding your breath",
        "Engage your core by pulling your navel toward your spine"
      ]
    };
    
    return tips[workoutName] || [
      "Focus on proper form over heavy weights or high reps",
      "Stay hydrated throughout your workout",
      "Listen to your body and adjust as needed"
    ];
  };

  return (
    <Box sx={{ width: '100%' }}>
      {saveSuccess && (
        <Alert severity="success" sx={{ mb: 3 }}>
          Workout plan saved successfully!
        </Alert>
      )}
      
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs 
          value={tabValue} 
          onChange={handleTabChange} 
          variant="scrollable"
          scrollButtons="auto"
          aria-label="workout days tabs"
        >
          {workoutDays.map((day, index) => (
            <Tab key={index} label={`${day.day}: ${day.name}`} />
          ))}
        </Tabs>
      </Box>
      
      {workoutDays.map((day, index) => (
        <div
          key={index}
          role="tabpanel"
          hidden={tabValue !== index}
          id={`workout-tabpanel-${index}`}
          aria-labelledby={`workout-tab-${index}`}
        >
          {tabValue === index && (
            <Box sx={{ p: 3 }}>
              <Typography variant="h4" gutterBottom>
                {day.name} Workout
              </Typography>
              
              <Typography variant="body1" paragraph>
                {getWorkoutDescription(day.name)}
              </Typography>
              
              <TableContainer component={Paper} sx={{ mb: 4 }}>
                <Table aria-label="workout table">
                  <TableHead>
                    <TableRow>
                      <TableCell>Exercise</TableCell>
                      <TableCell align="center">Sets</TableCell>
                      <TableCell align="center">Reps</TableCell>
                      <TableCell align="center">Rest</TableCell>
                      <TableCell align="center">Video</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {day.exercises.map((exercise, i) => (
                      <TableRow key={i}>
                        <TableCell component="th" scope="row">
                          {exercise.name}
                        </TableCell>
                        <TableCell align="center">{exercise.sets}</TableCell>
                        <TableCell align="center">{exercise.reps}</TableCell>
                        <TableCell align="center">{exercise.rest}</TableCell>
                        <TableCell align="center">
                          <Link to={`/video/${index + 1}/${encodeURIComponent(exercise.name)}`}>
                            Watch Video
                          </Link>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
              
              <Paper sx={{ p: 2, bgcolor: 'info.light', color: 'info.contrastText', mb: 4 }}>
                <Typography variant="h6" gutterBottom>
                  Workout Tips
                </Typography>
                <ul>
                  {getWorkoutTips(day.name).map((tip, i) => (
                    <li key={i}>
                      <Typography variant="body1">{tip}</Typography>
                    </li>
                  ))}
                </ul>
              </Paper>
            </Box>
          )}
        </div>
      ))}
      
      <Box sx={{ mt: 4, display: 'flex', justifyContent: 'space-between' }}>
        <Button 
          variant="contained" 
          color="primary" 
          onClick={() => setShowSaveDialog(true)}
        >
          Save This Workout
        </Button>
        
        <Button 
          variant="outlined" 
          onClick={onCreateNewPlan}
        >
          Create New Plan
        </Button>
      </Box>
      
      {/* Save Workout Dialog */}
      <Dialog open={showSaveDialog} onClose={() => setShowSaveDialog(false)}>
        <DialogTitle>Save Workout Plan</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Give your workout plan a name to save it for future reference.
          </DialogContentText>
          <TextField
            autoFocus
            margin="dense"
            label="Workout Plan Name"
            type="text"
            fullWidth
            variant="outlined"
            value={planName}
            onChange={(e) => setPlanName(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowSaveDialog(false)}>Cancel</Button>
          <Button onClick={handleSaveWorkout} disabled={!planName.trim()}>
            Save
          </Button>
        </DialogActions>
      </Dialog>
      {aiInsights && <AiInsights insights={aiInsights} />}
    </Box>
  );
};

export default WorkoutPlan;