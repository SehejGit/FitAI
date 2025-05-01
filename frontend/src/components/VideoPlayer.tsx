import * as React from 'react';
import { useState, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Paper,
  Button,
  GridLegacy,
  Card,
  CardContent,
  CardMedia,
  Divider,
  CircularProgress,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';

// Correct icon imports from @mui/icons-material
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import FileUploadIcon from '@mui/icons-material/FileUpload';
import PlayCircleIcon from '@mui/icons-material/PlayCircle';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import FeedbackIcon from '@mui/icons-material/Feedback';

// Mock function to get video URL for an exercise
const getExerciseVideoUrl = (exerciseName: string): string => {
  // In a real app, this would map to actual exercise videos
  // For now, we'll use placeholder videos
  const placeholderVideos: {[key: string]: string} = {
    "default": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4",
    "Push-ups": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4",
    "Squats": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4",
    "Planks": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4",
    "Burpees": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4",
    "Bench press": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4"
  };
  
  return placeholderVideos[exerciseName] || placeholderVideos.default;
};

// Mock function to analyze video
const analyzeExerciseForm = (videoBlob: Blob, exerciseName: string): Promise<any> => {
  // In a real app, this would call an AI service to analyze the form
  return new Promise((resolve) => {
    // Simulate API call delay
    setTimeout(() => {
      // Return mock analysis results
      resolve({
        score: Math.floor(Math.random() * 40) + 60, // Random score between 60-100
        feedback: [
          "Good overall form and technique",
          "Keep your back straight throughout the movement",
          "Try to maintain a consistent pace",
          "Focus on full range of motion"
        ]
      });
    }, 2000);
  });
};

interface VideoPlayerProps {
  // You could pass workout data here if needed
}

const VideoPlayer: React.FC<VideoPlayerProps> = () => {
  const { day, exercise } = useParams<{ day: string; exercise: string }>();
  const navigate = useNavigate();
  const [uploadedVideo, setUploadedVideo] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const decodedExercise = exercise ? decodeURIComponent(exercise) : '';
  const videoUrl = getExerciseVideoUrl(decodedExercise);
  
  const handleUploadClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };
  
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('video/')) {
      const videoUrl = URL.createObjectURL(file);
      setUploadedVideo(videoUrl);
      setAnalysisResults(null); // Clear previous results
    }
  };
  
  const handleAnalyzeVideo = async () => {
    if (!uploadedVideo) return;
    
    setIsAnalyzing(true);
    
    try {
      // In a real app, you would fetch the video blob and send it to your API
      const response = await fetch(uploadedVideo);
      const blob = await response.blob();
      
      const results = await analyzeExerciseForm(blob, decodedExercise);
      setAnalysisResults(results);
    } catch (error) {
      console.error('Error analyzing video:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };
  
  const handleGoBack = () => {
    navigate(-1); // Go back to previous page
  };
  
  return (
    <Box sx={{ p: 4 }}>
      <Paper elevation={3} sx={{ p: 4, borderRadius: 2 }}>
        <Box sx={{ mb: 4, display: 'flex', alignItems: 'center' }}>
          <Button 
            startIcon={<ArrowBackIcon />} 
            onClick={handleGoBack}
            sx={{ mr: 2 }}
          >
            Back to Workout
          </Button>
          <Typography variant="h5" component="h1">
            {decodedExercise} - Day {day}
          </Typography>
        </Box>
        
        <GridLegacy container spacing={4}>
          {/* Example Video */}
          <GridLegacy item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Example Form
                </Typography>
              </CardContent>
              <CardMedia
                component="video"
                controls
                src={videoUrl}
                sx={{ width: '100%', height: '300px' }}
              />
              <CardContent>
                <Typography variant="body2" color="text.secondary">
                  Watch this video to learn the proper technique for {decodedExercise}.
                </Typography>
              </CardContent>
            </Card>
          </GridLegacy>
          
          {/* User Video Upload/Analysis */}
          <GridLegacy item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Record & Analyze Your Form
                </Typography>
                
                <Box sx={{ mb: 2 }}>
                  <input
                    type="file"
                    ref={fileInputRef}
                    accept="video/*"
                    style={{ display: 'none' }}
                    onChange={handleFileChange}
                  />
                  <Button
                    variant="contained"
                    color="primary"
                    startIcon={<FileUploadIcon />}
                    onClick={handleUploadClick}
                    fullWidth
                  >
                    Upload Video
                  </Button>
                </Box>
                
                {uploadedVideo && (
                  <>
                    <CardMedia
                      component="video"
                      controls
                      src={uploadedVideo}
                      sx={{ width: '100%', height: '300px', mb: 2 }}
                    />
                    
                    <Button
                      variant="contained"
                      color="secondary"
                      startIcon={<PlayCircleIcon />}
                      onClick={handleAnalyzeVideo}
                      disabled={isAnalyzing}
                      fullWidth
                    >
                      Analyze My Form
                    </Button>
                    
                    {isAnalyzing && (
                      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
                        <CircularProgress />
                      </Box>
                    )}
                  </>
                )}
              </CardContent>
            </Card>
            
            {/* Analysis Results */}
            {analysisResults && (
              <Card sx={{ mt: 2 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Analysis Results
                  </Typography>
                  
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Typography variant="body1" sx={{ mr: 2 }}>
                      Form Score: 
                    </Typography>
                    <Typography 
                      variant="h5" 
                      color={analysisResults.score >= 80 ? "success.main" : "warning.main"}
                    >
                      {analysisResults.score}%
                    </Typography>
                  </Box>
                  
                  <Divider sx={{ mb: 2 }} />
                  
                  <Typography variant="subtitle1" gutterBottom>
                    Feedback:
                  </Typography>
                  
                  <List>
                    {analysisResults.feedback.map((item: string, index: number) => (
                      <ListItem key={index} sx={{ py: 0.5 }}>
                        <ListItemIcon>
                          <FeedbackIcon color="primary" />
                        </ListItemIcon>
                        <ListItemText primary={item} />
                      </ListItem>
                    ))}
                  </List>
                  
                  {analysisResults.score >= 80 ? (
                    <Alert 
                      icon={<CheckCircleIcon />} 
                      severity="success"
                      sx={{ mt: 2 }}
                    >
                      Great job! Your form looks excellent.
                    </Alert>
                  ) : (
                    <Alert 
                      icon={<ErrorIcon />} 
                      severity="warning"
                      sx={{ mt: 2 }}
                    >
                      Keep practicing! Review the feedback to improve your form.
                    </Alert>
                  )}
                </CardContent>
              </Card>
            )}
          </GridLegacy>
        </GridLegacy>
      </Paper>
    </Box>
  );
};

export default VideoPlayer;