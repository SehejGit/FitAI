import * as React from 'react';
import { useState, useRef, useEffect } from 'react';
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

// Icon imports
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import FileUploadIcon from '@mui/icons-material/FileUpload';
import PlayCircleIcon from '@mui/icons-material/PlayCircle';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import FeedbackIcon from '@mui/icons-material/Feedback';
import VideoLibraryIcon from '@mui/icons-material/VideoLibrary';

// Import the video analysis module
import { analyzeExerciseForm } from '../utils/videoAnalysis';

// Function to get example video URL for an exercise
const getExerciseVideoUrl = (exerciseName: string): string => {
  // In a real app, this would map to actual exercise videos
  // For now, we'll use placeholder videos
  const placeholderVideos: {[key: string]: string} = {
    "default": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4",
    "Push-ups": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4",
    "Pike push-ups": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4",
    "Squats": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4",
    "Planks": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4",
    "Burpees": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4",
    "Bench press": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4"
  };
  
  return placeholderVideos[exerciseName] || placeholderVideos.default;
};

interface VideoPlayerProps {
  // You could pass workout data here if needed
}

interface AnalysisResults {
  score: number;
  feedback: string[];
  annotatedVideoUrl?: string;
}

const VideoPlayer: React.FC<VideoPlayerProps> = () => {
  const { day, exercise } = useParams<{ day: string; exercise: string }>();
  const navigate = useNavigate();
  const [uploadedVideo, setUploadedVideo] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResults | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [showAnnotatedVideo, setShowAnnotatedVideo] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoFileRef = useRef<File | null>(null);
  
  const decodedExercise = exercise ? decodeURIComponent(exercise) : '';
  const videoUrl = getExerciseVideoUrl(decodedExercise);
  
  // Reset state when exercise changes
  useEffect(() => {
    setUploadedVideo(null);
    setAnalysisResults(null);
    setAnalysisError(null);
    setShowAnnotatedVideo(false);
    videoFileRef.current = null;
  }, [exercise]);
  
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
      videoFileRef.current = file;
      setAnalysisResults(null);
      setAnalysisError(null);
      setShowAnnotatedVideo(false);
    }
  };
  
  const handleAnalyzeVideo = async () => {
    if (!uploadedVideo || !videoFileRef.current) return;
    
    setIsAnalyzing(true);
    setAnalysisError(null);
    
    try {
      // Convert File to Blob for analysis
      const blob = await videoFileRef.current.arrayBuffer().then(buffer => new Blob([buffer], { type: videoFileRef.current?.type }));
      
      // Call the analysis function from our module
      const results = await analyzeExerciseForm(blob, decodedExercise);
      setAnalysisResults(results);
      
      // Automatically show the annotated video when analysis is complete
      if (results.annotatedVideoUrl) {
        setShowAnnotatedVideo(true);
      }
    } catch (error) {
      console.error('Error analyzing video:', error);
      setAnalysisError(typeof error === 'string' ? error : (error as Error).message || 'Failed to analyze video');
    } finally {
      setIsAnalyzing(false);
    }
  };
  
  const handleToggleVideo = () => {
    setShowAnnotatedVideo(!showAnnotatedVideo);
  };
  
  const handleGoBack = () => {
    navigate(-1); // Go back to previous page
  };
  
  // Calculate color based on score
  const getScoreColor = (score: number) => {
    if (score >= 80) return "success.main";
    if (score >= 60) return "warning.main";
    return "error.main";
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
                      src={showAnnotatedVideo && analysisResults?.annotatedVideoUrl ? analysisResults.annotatedVideoUrl : uploadedVideo}
                      sx={{ width: '100%', height: '300px', mb: 2 }}
                    />
                    
                    {analysisResults?.annotatedVideoUrl && (
                      <Box sx={{ mb: 2, display: 'flex', justifyContent: 'center' }}>
                        <Button
                          variant="contained"
                          color={showAnnotatedVideo ? "success" : "primary"}
                          startIcon={<VideoLibraryIcon />}
                          onClick={handleToggleVideo}
                          sx={{ borderRadius: 28 }}
                        >
                          {showAnnotatedVideo ? "Show Original Video" : "Show Annotated Video"}
                        </Button>
                      </Box>
                    )}
                    
                    {!analysisResults && !isAnalyzing && (
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
                    )}
                    
                    {isAnalyzing && (
                      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mt: 2 }}>
                        <CircularProgress sx={{ mb: 2 }} />
                        <Typography variant="body2" color="text.secondary">
                          Analyzing your form... This may take a minute.
                        </Typography>
                      </Box>
                    )}
                    
                    {analysisError && (
                      <Alert severity="error" sx={{ mt: 2 }}>
                        {analysisError}
                      </Alert>
                    )}
                  </>
                )}
              </CardContent>
            </Card>
            
            
          </GridLegacy>
        </GridLegacy>
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
                      color={getScoreColor(analysisResults.score)}
                    >
                      {analysisResults.score}%
                    </Typography>
                  </Box>
                  
                  <Divider sx={{ mb: 2 }} />
                  
                  <Typography variant="subtitle1" gutterBottom>
                    Feedback:
                  </Typography>
                  
                  <List>
                    {analysisResults.feedback.map((item, index) => (
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
                  ) : analysisResults.score >= 60 ? (
                    <Alert 
                      icon={<ErrorIcon />} 
                      severity="warning"
                      sx={{ mt: 2 }}
                    >
                      Good effort! Review the feedback to improve your form.
                    </Alert>
                  ) : (
                    <Alert 
                      icon={<ErrorIcon />} 
                      severity="error"
                      sx={{ mt: 2 }}
                    >
                      Keep practicing! There are some key form issues to address.
                    </Alert>
                  )}
                </CardContent>
              </Card>
            )}
      </Paper>
    </Box>
  );
};

export default VideoPlayer;