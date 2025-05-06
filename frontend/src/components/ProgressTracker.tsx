import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  CircularProgress, 
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  Card,
  CardContent,
  Tabs,
  Tab
} from '@mui/material';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { format, startOfWeek, endOfWeek, eachDayOfInterval, parseISO, isWithinInterval, addDays } from 'date-fns';
import { getWorkoutLogs } from '../services/workoutService';
import { Grid } from '@mui/material'

// Define types for workout logs
interface WorkoutLog {
  date: string;
  day: string;
  exercise?: string;
  reps?: number;
  notes: string;
}

interface WorkoutLogs {
  [key: string]: WorkoutLog[];
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
      id={`progress-tabpanel-${index}`}
      aria-labelledby={`progress-tab-${index}`}
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

const ProgressTracker: React.FC = () => {
  const [workoutLogs, setWorkoutLogs] = useState<WorkoutLogs>({});
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [tabValue, setTabValue] = useState<number>(0);
  
  useEffect(() => {
    fetchWorkoutLogs();
  }, []);
  
  const fetchWorkoutLogs = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await getWorkoutLogs();
      
      if (response.success) {
        setWorkoutLogs(response.logs || {});
      } else {
        throw new Error(response.error || 'Failed to fetch workout logs');
      }
    } catch (err: any) {
      console.error('Error fetching workout logs:', err);
      setError('Failed to load workout logs. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };
  
  // Calculate statistics for the dashboard
  const calculateStats = () => {
    const today = new Date();
    const currentWeekStart = startOfWeek(today);
    const currentWeekEnd = endOfWeek(today);
    
    let totalWorkouts = 0;
    let weeklyWorkouts = 0;
    let streakDays = 0;
    let mostFrequentWorkout = '';
    let workoutTypeCount: {[key: string]: number} = {};
    
    // Process all workout logs
    Object.values(workoutLogs).forEach(logs => {
      logs.forEach(log => {
        totalWorkouts++;
        
        // Count weekly workouts
        const logDate = parseISO(log.date);
        if (isWithinInterval(logDate, { start: currentWeekStart, end: currentWeekEnd })) {
          weeklyWorkouts++;
        }
        
        // Count workout types
        const workoutType = log.day.includes('Day') ? log.day.split(':')[1]?.trim() : log.day;
        if (workoutType) {
          workoutTypeCount[workoutType] = (workoutTypeCount[workoutType] || 0) + 1;
        }
      });
    });
    
    // Find most frequent workout
    let maxCount = 0;
    Object.entries(workoutTypeCount).forEach(([type, count]) => {
      if (count > maxCount) {
        maxCount = count;
        mostFrequentWorkout = type;
      }
    });
    
    // Calculate streak (simplified)
    // In a real app, you would want a more sophisticated calculation
    const allDates = Object.values(workoutLogs).flatMap(logs => 
      logs.map(log => log.date)
    ).sort();
    
    if (allDates.length > 0) {
      // Check if worked out today
      const todayFormatted = format(today, 'yyyy-MM-dd');
      if (allDates.includes(todayFormatted)) {
        streakDays = 1;
        
        // Check previous days
        let checkDate = today;
        let keepChecking = true;
        
        while (keepChecking) {
          checkDate = addDays(checkDate, -1);
          const checkDateFormatted = format(checkDate, 'yyyy-MM-dd');
          
          if (allDates.includes(checkDateFormatted)) {
            streakDays++;
          } else {
            keepChecking = false;
          }
        }
      }
    }
    
    return {
      totalWorkouts,
      weeklyWorkouts,
      streakDays,
      mostFrequentWorkout: mostFrequentWorkout || 'None',
      workoutTypeCount
    };
  };
  
  // Prepare weekly chart data
  const prepareWeeklyChartData = () => {
    const last30Days = eachDayOfInterval({
      start: addDays(new Date(), -29),
      end: new Date()
    });
    
    const chartData = last30Days.map(date => {
      const dateStr = format(date, 'yyyy-MM-dd');
      let workoutCount = 0;
      
      Object.values(workoutLogs).forEach(logs => {
        logs.forEach(log => {
          if (log.date === dateStr) {
            workoutCount++;
          }
        });
      });
      
      return {
        date: format(date, 'MMM dd'),
        workouts: workoutCount
      };
    });
    
    return chartData;
  };
  
  // Prepare workout type pie chart data
  const prepareWorkoutTypeData = () => {
    const stats = calculateStats();
    const workoutTypeData = Object.entries(stats.workoutTypeCount).map(([name, value]) => ({
      name,
      value
    }));
    
    return workoutTypeData;
  };
  
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];
  
  const stats = calculateStats();
  const weeklyChartData = prepareWeeklyChartData();
  const workoutTypeData = prepareWorkoutTypeData();
  
  if (isLoading) {
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
  
  const hasLogs = Object.keys(workoutLogs).length > 0;
  
  if (!hasLogs) {
    return (
      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h5" gutterBottom>
          No Workout Logs
        </Typography>
        <Typography variant="body1">
          You haven't logged any workouts yet. Go to the 'My Workouts' tab to log your completed workouts!
        </Typography>
      </Paper>
    );
  }
  
  // Get all workout logs in a flat array
  const allLogs = Object.entries(workoutLogs).flatMap(([planName, logs]) => 
    logs.map(log => ({
      ...log,
      planName
    }))
  ).sort((a, b) => b.date.localeCompare(a.date)); // Sort by date (most recent first)
  
  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Workout Progress Tracker
      </Typography>
      
      {/* Stats Dashboard */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid {...{component: "div", item: true, xs: 12, sm: 6, md: 3} as any}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Workouts
              </Typography>
              <Typography variant="h3">
                {stats.totalWorkouts}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid {...{component: "div", item: true, xs: 12, sm: 6, md: 3} as any}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                This Week
              </Typography>
              <Typography variant="h3">
                {stats.weeklyWorkouts}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid {...{component: "div", item: true, xs: 12, sm: 6, md: 3} as any}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Current Streak
              </Typography>
              <Typography variant="h3">
                {stats.streakDays} days
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid {...{component: "div", item: true, xs: 12, sm: 6, md: 3} as any}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Favorite Workout
              </Typography>
              <Typography variant="h5">
                {stats.mostFrequentWorkout}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      {/* Tabs for different analytics views */}
      <Paper sx={{ width: '100%', mb: 4 }}>
        <Tabs 
          value={tabValue} 
          onChange={handleTabChange} 
          indicatorColor="primary"
          textColor="primary"
          centered
        >
          <Tab label="Activity" />
          <Tab label="Workout Types" />
          <Tab label="History" />
        </Tabs>
        
        {/* Activity Chart Tab */}
        <TabPanel value={tabValue} index={0}>
          <Typography variant="h6" gutterBottom>
            Workout Activity (Last 30 Days)
          </Typography>
          <Box sx={{ height: 400 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={weeklyChartData}
                margin={{
                  top: 20,
                  right: 30,
                  left: 20,
                  bottom: 5,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="workouts" name="Workouts Completed" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        </TabPanel>
        
        {/* Workout Types Tab */}
        <TabPanel value={tabValue} index={1}>
          <Typography variant="h6" gutterBottom>
            Workout Type Distribution
          </Typography>
          <Box sx={{ height: 400 }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={workoutTypeData}
                  cx="50%"
                  cy="50%"
                  labelLine={true}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={120}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {workoutTypeData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value, name) => [value, name]} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Box>
        </TabPanel>
        
        {/* Workout History Tab */}
        <TabPanel value={tabValue} index={2}>
          <Typography variant="h6" gutterBottom>
            Workout History
          </Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Date</TableCell>
                  <TableCell>Workout Plan</TableCell>
                  <TableCell>Activity</TableCell>
                  <TableCell>Notes</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {allLogs.slice(0, 20).map((log, index) => (
                  <TableRow key={index}>
                    <TableCell>{log.date}</TableCell>
                    <TableCell>{log.planName}</TableCell>
                    <TableCell>
                      {log.exercise ? log.exercise : log.day}
                      {log.reps && ` (${log.reps} reps)`}
                    </TableCell>
                    <TableCell>{log.notes || '-'}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
          
          <Box sx={{ mt: 3, display: 'flex', justifyContent: 'center' }}>
            {allLogs.length > 20 && (
              <Button variant="outlined">
                View All ({allLogs.length}) Entries
              </Button>
            )}
          </Box>
        </TabPanel>
      </Paper>
      
      {/* Workout Log Summary */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Fitness Progress Summary
        </Typography>
        
        <Typography paragraph>
          You've completed <strong>{stats.totalWorkouts}</strong> workouts, with <strong>{stats.weeklyWorkouts}</strong> this week.
          Your current streak is <strong>{stats.streakDays}</strong> days, and your most frequent workout type is <strong>{stats.mostFrequentWorkout}</strong>.
        </Typography>
        
        <Typography paragraph>
          {stats.weeklyWorkouts >= 3 
            ? "Great job keeping up with your fitness routine! Consistency is key to making progress."
            : "Try to aim for at least 3-4 workouts per week to build consistency and see better results."}
        </Typography>
        
        <Typography>
          Keep tracking your workouts to monitor your progress and stay motivated!
        </Typography>
      </Paper>
    </Box>
  );
};

export default ProgressTracker;