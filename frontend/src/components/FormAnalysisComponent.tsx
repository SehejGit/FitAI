import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Button, 
  Paper, 
  CircularProgress, 
  Alert,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent
} from '@mui/material';
import {
  FitnessCenter as FitnessCenterIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  VideoLibrary as VideoLibraryIcon,
  Info as InfoIcon
} from '@mui/icons-material';

import Grid from '@mui/material/Grid';

const FormAnalysisComponent = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [filePreview, setFilePreview] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [analysis, setAnalysis] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [annotatedVideoUrl, setAnnotatedVideoUrl] = useState<string | null>(null);
  const [exerciseTypes, setExerciseTypes] = useState<string[]>([]);
  const [selectedExercise, setSelectedExercise] = useState<string>('pushup');
  const [loadingExercises, setLoadingExercises] = useState<boolean>(true);

  // Fetch available exercise types when component mounts
  useEffect(() => {
    const fetchExerciseTypes = async () => {
      try {
        setLoadingExercises(true);
        const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/exercises/`);
        
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        setExerciseTypes(data.available_exercises || []);
        
        // Set default selected exercise if available
        if (data.available_exercises && data.available_exercises.includes('pushup')) {
          setSelectedExercise('pushup');
        } else if (data.available_exercises && data.available_exercises.length > 0) {
          setSelectedExercise(data.available_exercises[0]);
        }
      } catch (err) {
        console.error('Error fetching exercise types:', err);
        setError('Failed to load available exercise types. Please try again later.');
      } finally {
        setLoadingExercises(false);
      }
    };
    
    fetchExerciseTypes();
  }, []);

  const handleExerciseChange = (event: SelectChangeEvent) => {
    setSelectedExercise(event.target.value);
    // Reset results when exercise type changes
    setAnalysis(null);
    setAnnotatedVideoUrl(null);
    setError(null);
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      setSelectedFile(file);
      
      // Create a preview URL for the video
      const fileUrl = URL.createObjectURL(file);
      setFilePreview(fileUrl);
      
      // Reset previous results
      setAnalysis(null);
      setError(null);
      setAnnotatedVideoUrl(null);
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      setError("Please select a video file to analyze");
      return;
    }

    if (!selectedExercise) {
      setError("Please select an exercise type");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('return_video', 'true'); // Request the annotated video

      const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/analyze/${selectedExercise}/`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Analysis result:', data);
      
      if (data.analysis) {
        setAnalysis(data.analysis);
      }
      
      if (data.annotated_video_url) {
        // Construct the full URL for the annotated video
        const baseUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
        setAnnotatedVideoUrl(`${baseUrl}${data.annotated_video_url}`);
      }
    } catch (error: any) {
      console.error('Error analyzing video:', error);
      setError(`Failed to analyze video: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const renderMetrics = (analysis: any) => {
    if (!analysis) return null;
    
    // Define different metrics based on exercise type
    let metrics = [];
    
    // Common metrics
    if (analysis.count !== undefined) {
      metrics.push({ name: 'Repetitions Counted', value: analysis.count || 0 });
    }
    
    if (analysis.form_score !== undefined) {
      metrics.push({ name: 'Form Score', value: `${Math.round((analysis.form_score || 0) * 100)}%` });
    }
    
    // Push-up specific metrics
    if (selectedExercise === 'pushup') {
      if (analysis.elbow_angle !== undefined) {
        metrics.push({ name: 'Elbow Angle', value: `${Math.round(analysis.elbow_angle)}°` });
      }
      if (analysis.back_alignment !== undefined) {
        metrics.push({ name: 'Back Alignment', value: `${Math.round(analysis.back_alignment * 100)}%` });
      }
    }
    
    // Squat specific metrics
    if (selectedExercise === 'squat') {
      if (analysis.knee_angle !== undefined) {
        metrics.push({ name: 'Knee Angle', value: `${Math.round(analysis.knee_angle)}°` });
      }
      if (analysis.depth_score !== undefined) {
        metrics.push({ name: 'Depth Score', value: `${Math.round(analysis.depth_score * 100)}%` });
      }
    }
    
    // If no specific metrics were added, add generic ones from the analysis
    if (metrics.length === 0) {
      // Get all numeric values from analysis
      Object.entries(analysis).forEach(([key, value]) => {
        if (typeof value === 'number' && key !== 'count' && !key.includes('timestamp')) {
          // Format the key for display
          const formattedKey = key
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
          
          // Format the value
          let formattedValue = value as number;
          if (key.includes('angle')) {
            formattedValue = Math.round(formattedValue);
            metrics.push({ name: formattedKey, value: `${formattedValue}°` });
          } else if (key.includes('score') || key.includes('alignment') || key.includes('accuracy')) {
            formattedValue = Math.round(formattedValue * 100);
            metrics.push({ name: formattedKey, value: `${formattedValue}%` });
          } else {
            metrics.push({ name: formattedKey, value: formattedValue.toString() });
          }
        }
      });
    }
    
    // Ensure we display at least 4 metrics (empty if needed)
    while (metrics.length < 4) {
      metrics.push({ name: `Metric ${metrics.length + 1}`, value: 'N/A' });
    }
    
    return (
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {metrics.map((metric, index) => (
          <Grid {...{component: "div", item: true, xs: 6, sm: 3, key: index} as any}>
            <Card elevation={3}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography color="textSecondary" gutterBottom>
                  {metric.name}
                </Typography>
                <Typography variant="h4" component="div">
                  {metric.value}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    );
  };

  const renderFeedback = (analysis: any) => {
    if (!analysis || !analysis.feedback || analysis.feedback.length === 0) {
      return (
        <Alert severity="info" sx={{ mb: 3 }}>
          No specific feedback available for this exercise.
        </Alert>
      );
    }
    
    // Ensure feedback is an array
    const feedbackItems = Array.isArray(analysis.feedback) 
      ? analysis.feedback 
      : [analysis.feedback];
    
    return (
      <Paper elevation={3} sx={{ p: 2, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          <InfoIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Exercise Feedback
        </Typography>
        <List>
          {feedbackItems.map((item: any, index: number) => {
            // Handle different feedback formats
            let message = '';
            let type = 'info';
            
            if (typeof item === 'string') {
              message = item;
            } else if (typeof item === 'object') {
              message = item.message || item.text || JSON.stringify(item);
              type = item.type || 'info';
            }
            
            return (
              <React.Fragment key={index}>
                <ListItem>
                  <ListItemIcon>
                    {type === 'positive' || type === 'good' ? 
                      <CheckCircleIcon color="success" /> : 
                      type === 'negative' || type === 'bad' ? 
                        <ErrorIcon color="warning" /> : 
                        <InfoIcon color="info" />
                    }
                  </ListItemIcon>
                  <ListItemText primary={message} />
                </ListItem>
                {index < feedbackItems.length - 1 && <Divider />}
              </React.Fragment>
            );
          })}
        </List>
      </Paper>
    );
  };

  // Provide exercise-specific tips based on the selected exercise
  const getExerciseTips = () => {
    const tipsByExercise: {[key: string]: string[]} = {
      'pushup': [
        "Keep your core tight throughout the movement",
        "Maintain a straight line from head to heels",
        "Lower your chest until your elbows form a 90-degree angle",
        "Breathe out as you push up, breathe in as you lower down",
        "Keep your elbows closer to your body to reduce shoulder strain"
      ],
      'squat': [
        "Keep your weight in your heels",
        "Drive your knees outward as you descend",
        "Maintain a neutral spine throughout the movement",
        "Aim to lower until your thighs are parallel to the ground",
        "Keep your chest up and shoulders back"
      ],
      'plank': [
        "Engage your core by pulling your belly button toward your spine",
        "Keep your body in a straight line from head to heels",
        "Don't let your hips sag or pike up",
        "Focus on quality over duration",
        "Breathe normally throughout the hold"
      ]
    };
    
    // Return tips for the selected exercise, or generic tips if none available
    return tipsByExercise[selectedExercise] || [
      "Focus on proper form over speed or reps",
      "Record yourself regularly to track improvements",
      "Breathe in a controlled manner throughout the exercise",
      "Start with easier variations and progress gradually",
      "If you feel pain (not muscle fatigue), stop and reassess your form"
    ];
  };

  const formatExerciseName = (name: string) => {
    return name
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        <FitnessCenterIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
        Exercise Form Analysis
      </Typography>
      
      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <Typography variant="h6" gutterBottom>
          Upload a video of your exercise for form analysis
        </Typography>
        
        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel id="exercise-type-label">Exercise Type</InputLabel>
          <Select
            labelId="exercise-type-label"
            id="exercise-type"
            value={selectedExercise}
            onChange={handleExerciseChange}
            label="Exercise Type"
            disabled={loadingExercises}
          >
            {loadingExercises ? (
              <MenuItem value="">Loading exercise types...</MenuItem>
            ) : (
              exerciseTypes.map((exercise) => (
                <MenuItem key={exercise} value={exercise}>
                  {formatExerciseName(exercise)}
                </MenuItem>
              ))
            )}
          </Select>
        </FormControl>
        
        <Box sx={{ mb: 3 }}>
          <input
            accept="video/*"
            style={{ display: 'none' }}
            id="video-upload"
            type="file"
            onChange={handleFileChange}
          />
          <label htmlFor="video-upload">
            <Button 
              variant="contained" 
              component="span"
              startIcon={<VideoLibraryIcon />}
            >
              Select Video
            </Button>
          </label>
          
          {selectedFile && (
            <Typography variant="body2" sx={{ mt: 1 }}>
              Selected: {selectedFile.name}
            </Typography>
          )}
        </Box>
        
        {filePreview && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              Preview:
            </Typography>
            <video 
              width="100%" 
              height="auto" 
              controls 
              src={filePreview}
              style={{ maxHeight: '300px' }}
            />
          </Box>
        )}
        
        <Button
          variant="contained"
          color="primary"
          onClick={handleSubmit}
          disabled={!selectedFile || isLoading || !selectedExercise}
          sx={{ mt: 2 }}
        >
          {isLoading ? <CircularProgress size={24} sx={{ mr: 1 }} /> : null}
          {isLoading ? 'Analyzing...' : `Analyze ${formatExerciseName(selectedExercise)}`}
        </Button>
      </Paper>
      
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}
      
      {analysis && (
        <Box sx={{ mt: 4 }}>
          <Typography variant="h5" gutterBottom>
            Analysis Results
          </Typography>
          
          {renderMetrics(analysis)}
          {renderFeedback(analysis)}
          
          {annotatedVideoUrl && (
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                Annotated Video
              </Typography>
              <video 
                width="100%" 
                height="auto" 
                controls 
                src={annotatedVideoUrl}
                style={{ maxHeight: '500px' }}
              />
            </Box>
          )}
          
          <Paper elevation={3} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              {formatExerciseName(selectedExercise)} Improvement Tips
            </Typography>
            <List>
              {getExerciseTips().map((tip, index) => (
                <React.Fragment key={index}>
                  <ListItem>
                    <ListItemIcon><CheckCircleIcon color="success" /></ListItemIcon>
                    <ListItemText primary={tip} />
                  </ListItem>
                  {index < getExerciseTips().length - 1 && <Divider />}
                </React.Fragment>
              ))}
            </List>
          </Paper>
        </Box>
      )}
    </Box>
  );
};

export default FormAnalysisComponent;