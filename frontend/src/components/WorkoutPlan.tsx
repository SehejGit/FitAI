import * as React from 'react';
import { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider
} from '@mui/material';

import FitnessCenterIcon from '@mui/icons-material/FitnessCenter';
import TimerIcon from '@mui/icons-material/Timer';
import TipsAndUpdatesIcon from '@mui/icons-material/TipsAndUpdates';

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
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`workout-tabpanel-${index}`}
      aria-labelledby={`workout-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `workout-tab-${index}`,
    'aria-controls': `workout-tabpanel-${index}`,
  };
}

const WorkoutPlan: React.FC<WorkoutPlanProps> = ({ workoutDays }) => {
  const [value, setValue] = useState(0);

  const handleChange = (event: React.SyntheticEvent, newValue: number) => {
    setValue(newValue);
  };

  const tips = [
    "Start each workout with a 5-minute warm-up (light cardio and dynamic stretching)",
    "End each workout with a 5-minute cool-down (static stretching)",
    "Stay hydrated throughout your workouts",
    "Focus on proper form over heavy weights or high reps",
    "Progressively increase intensity as you get stronger",
    "Allow at least 48 hours of rest for muscle groups between workouts"
  ];

  return (
    <Box sx={{ width: '100%' }}>
      <Paper elevation={3} sx={{ p: 6, mb: 5 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={value} 
            onChange={handleChange} 
            variant="scrollable"
            scrollButtons="auto"
            aria-label="workout plan tabs"
          >
            {workoutDays.map((day, index) => (
              <Tab 
                key={day.day} 
                label={`${day.day}: ${day.name}`} 
                {...a11yProps(index)} 
              />
            ))}
            <Tab label="Tips for Success" {...a11yProps(workoutDays.length)} />
          </Tabs>
        </Box>

        {workoutDays.map((day, index) => (
          <TabPanel key={day.day} value={value} index={index}>
            <Typography variant="h5" component="h3" gutterBottom>
              {day.name} Workout
            </Typography>
            
            <TableContainer component={Paper} elevation={2}>
              <Table aria-label="workout exercise table">
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
                    <TableRow
                      key={`${day.day}-${exIndex}`}
                      sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
                    >
                      <TableCell component="th" scope="row">
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <FitnessCenterIcon sx={{ mr: 1, color: 'primary.main' }} />
                          {exercise.name}
                        </Box>
                      </TableCell>
                      <TableCell align="center">{exercise.sets}</TableCell>
                      <TableCell align="center">{exercise.reps}</TableCell>
                      <TableCell align="center">
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          <TimerIcon sx={{ mr: 0.5, fontSize: '1rem', color: 'text.secondary' }} />
                          {exercise.rest}
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </TabPanel>
        ))}

        <TabPanel value={value} index={workoutDays.length}>
          <Typography variant="h5" component="h3" gutterBottom>
            Tips for Success
          </Typography>
          
          <List>
            {tips.map((tip, index) => (
              <React.Fragment key={index}>
                <ListItem>
                  <ListItemIcon>
                    <TipsAndUpdatesIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText primary={tip} />
                </ListItem>
                {index < tips.length - 1 && <Divider variant="inset" component="li" />}
              </React.Fragment>
            ))}
          </List>
        </TabPanel>
      </Paper>
    </Box>
  );
};

export default WorkoutPlan;