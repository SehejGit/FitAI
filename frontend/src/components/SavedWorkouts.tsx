import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Button,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  TextField,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import { 
  ExpandMore as ExpandMoreIcon,
  Download as DownloadIcon,
  Check as CheckIcon
} from '@mui/icons-material';
import { getSavedWorkouts, logWorkoutCompletion, downloadWorkoutPlan } from '../services/workoutService';

// Define types for workout data
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

interface SavedWorkout {
  id: number;
  name: string;
  date_created: string;
  workout_plan: WorkoutDay[];
  user_data: {
    name?: string;
    fitness_goal: string;
    fitness_level: string;
    equipment: string[];
    injuries?: string;
  };
  time_per_session: number;
}

const SavedWorkouts: React.FC = () => {
  const [workouts, setWorkouts] = useState<SavedWorkout[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [openLogDialog, setOpenLogDialog] = useState<boolean>(false);
  const [selectedWorkout, setSelectedWorkout] = useState<SavedWorkout | null>(null);
  const [selectedDay, setSelectedDay] = useState<string>('');
  const [logNotes, setLogNotes] = useState<string>('');
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  useEffect(() => {
    fetchWorkouts();
  }, []);

  const fetchWorkouts = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const data = await getSavedWorkouts();
      setWorkouts(data.workouts || []);
    } catch (err: any) {
      console.error('Error fetching workouts:', err);
      setError('Failed to load saved workouts. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogWorkout = (workout: SavedWorkout, day: string) => {
    setSelectedWorkout(workout);
    setSelectedDay(day);
    setLogNotes('');
    setOpenLogDialog(true);
  };

  const handleSubmitLog = async () => {
    if (!selectedWorkout) return;
    
    try {
      setIsLoading(true);
      
      const logData = {
        workout_name: selectedWorkout.name,
        date: new Date().toISOString().split('T')[0], // YYYY-MM-DD format
        day: selectedDay,
        notes: logNotes
      };
      
      await logWorkoutCompletion(logData);
      setOpenLogDialog(false);
      setSuccessMessage(`Workout "${selectedDay}" from "${selectedWorkout.name}" logged successfully!`);
      
      // Clear success message after 3 seconds
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err: any) {
      console.error('Error logging workout:', err);
      setError('Failed to log workout. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadPDF = (workoutId: number) => {
    downloadWorkoutPlan(workoutId);
  };

  if (isLoading && workouts.length === 0) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ my: 2 }}>
        {error}
      </Alert>
    );
  }

  if (workouts.length === 0) {
    return (
      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h5" gutterBottom>
          No Saved Workouts
        </Typography>
        <Typography variant="body1">
          You haven't saved any workout plans yet. Generate a new workout plan to get started!
        </Typography>
      </Paper>
    );
  }

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        My Saved Workouts
      </Typography>

      {successMessage && (
        <Alert severity="success" sx={{ mb: 3 }}>
          {successMessage}
        </Alert>
      )}

      <List>
        {workouts.map((workout) => (
          <Paper key={workout.id} sx={{ mb: 3, overflow: 'hidden' }}>
            <Accordion>
              <AccordionSummary
                expandIcon={<ExpandMoreIcon />}
                aria-controls={`workout-${workout.id}-content`}
                id={`workout-${workout.id}-header`}
              >
                <Typography variant="h6">{workout.name}</Typography>
                <Typography variant="body2" sx={{ ml: 2, color: 'text.secondary' }}>
                  Created: {workout.date_created}
                </Typography>
              </AccordionSummary>
              
              <AccordionDetails>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle1">Details:</Typography>
                  <Typography>Goal: {workout.user_data.fitness_goal}</Typography>
                  <Typography>Fitness Level: {workout.user_data.fitness_level}</Typography>
                  <Typography>Equipment: {workout.user_data.equipment.join(', ')}</Typography>
                  <Typography>Session Duration: {workout.time_per_session} minutes</Typography>
                  {workout.user_data.injuries && (
                    <Typography>Considerations: {workout.user_data.injuries}</Typography>
                  )}
                </Box>
                
                <Typography variant="subtitle1" gutterBottom>
                  Workout Schedule:
                </Typography>
                
                <TableContainer sx={{ mb: 3 }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Day</TableCell>
                        <TableCell>Workout Type</TableCell>
                        <TableCell>Exercises</TableCell>
                        <TableCell align="right">Action</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {workout.workout_plan.map((day, index) => (
                        <TableRow key={index}>
                          <TableCell>{day.day}</TableCell>
                          <TableCell>{day.name}</TableCell>
                          <TableCell>{day.exercises.length} exercises</TableCell>
                          <TableCell align="right">
                            <Button
                              size="small"
                              variant="outlined"
                              startIcon={<CheckIcon />}
                              onClick={() => handleLogWorkout(workout, day.day)}
                            >
                              Log
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
                
                <Accordion>
                  <AccordionSummary
                    expandIcon={<ExpandMoreIcon />}
                    aria-controls={`workout-${workout.id}-exercises`}
                    id={`workout-${workout.id}-exercises-header`}
                  >
                    <Typography variant="subtitle1">View All Exercises</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    {workout.workout_plan.map((day, dayIndex) => (
                      <Box key={dayIndex} sx={{ mb: 3 }}>
                        <Typography variant="subtitle2">{day.day}: {day.name}</Typography>
                        <TableContainer>
                          <Table size="small">
                            <TableHead>
                              <TableRow>
                                <TableCell>Exercise</TableCell>
                                <TableCell align="center">Sets</TableCell>
                                <TableCell align="center">Reps</TableCell>
                                <TableCell align="center">Rest</TableCell>
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {day.exercises.map((exercise, exIndex) => (
                                <TableRow key={exIndex}>
                                  <TableCell>{exercise.name}</TableCell>
                                  <TableCell align="center">{exercise.sets}</TableCell>
                                  <TableCell align="center">{exercise.reps}</TableCell>
                                  <TableCell align="center">{exercise.rest}</TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      </Box>
                    ))}
                  </AccordionDetails>
                </Accordion>
                
                <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
                  <Button
                    variant="contained"
                    color="primary"
                    startIcon={<DownloadIcon />}
                    onClick={() => handleDownloadPDF(workout.id)}
                  >
                    Download PDF
                  </Button>
                </Box>
              </AccordionDetails>
            </Accordion>
          </Paper>
        ))}
      </List>
      
      {/* Log Workout Dialog */}
      <Dialog open={openLogDialog} onClose={() => setOpenLogDialog(false)}>
        <DialogTitle>Log Workout Completion</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Log your workout completion to track your progress.
          </DialogContentText>
          {selectedWorkout && (
            <Typography variant="subtitle1" gutterBottom>
              {selectedWorkout.name} - {selectedDay}
            </Typography>
          )}
          <TextField
            autoFocus
            margin="dense"
            label="Notes (optional)"
            fullWidth
            multiline
            rows={3}
            variant="outlined"
            placeholder="How did the workout feel? What weights did you use?"
            value={logNotes}
            onChange={(e) => setLogNotes(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenLogDialog(false)}>Cancel</Button>
          <Button 
            onClick={handleSubmitLog} 
            color="primary" 
            variant="contained"
            disabled={isLoading}
          >
            {isLoading ? <CircularProgress size={24} /> : "Log Workout"}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default SavedWorkouts;