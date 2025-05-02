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
import BugReportIcon from '@mui/icons-material/BugReport';

// Import the video analysis module and utilities
import { analyzeExerciseForm } from '../utils/videoAnalysis';
import { checkVideoUrlAccess } from '../utils/corsCheck';
import { API_BASE_URL } from '../config';

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
  rawAnalysis?: any;
}

const VideoPlayer: React.FC<VideoPlayerProps> = () => {
  const { day, exercise } = useParams<{ day: string; exercise: string }>();
  const navigate = useNavigate();
  const [uploadedVideo, setUploadedVideo] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResults | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [showAnnotatedVideo, setShowAnnotatedVideo] = useState(false);
  const [videoDebugMode, setVideoDebugMode] = useState(false);
  const [videoAccessError, setVideoAccessError] = useState<string | null>(null);
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
    setVideoAccessError(null);
    videoFileRef.current = null;
  }, [exercise]);
  
  // Check if the annotated video URL is accessible when it changes
  useEffect(() => {
    const verifyVideoAccess = async () => {
      if (analysisResults?.annotatedVideoUrl) {
        try {
          console.log('Checking video URL accessibility:', analysisResults.annotatedVideoUrl);
          const response = await fetch(analysisResults.annotatedVideoUrl, { method: 'HEAD' });
          
          if (!response.ok) {
            setVideoAccessError(`Video access error: ${response.status} ${response.statusText}`);
            console.error('Video URL is not accessible:', response.status, response.statusText);
          } else {
            setVideoAccessError(null);
            console.log('Video URL is accessible!');
          }
        } catch (error) {
          setVideoAccessError(`Error accessing video: ${error instanceof Error ? error.message : String(error)}`);
          console.error('Error verifying video URL:', error);
        }
      }
    };
    
    verifyVideoAccess();
  }, [analysisResults?.annotatedVideoUrl]);
  
  const handleUploadClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };
  
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    console.log("File:", file);
    if (file && file.type.startsWith('video/')) {
      const videoUrl = URL.createObjectURL(file);
      setUploadedVideo(videoUrl);
      videoFileRef.current = file;
      setAnalysisResults(null);
      setAnalysisError(null);
      setShowAnnotatedVideo(false);
      setVideoAccessError(null);
    }
  };

  // Add this function to determine how to load the video
const getVideoSource = (url: string): string => {
  // If it's a relative path to the backend directory, use as is
  if (url.includes('backend/static/videos')) {
    return url;
  }
  
  // If it's an API path, ensure it's absolute
  if (url.startsWith('/static/') || url.startsWith('/videos/')) {
    return `${API_BASE_URL}${url}`;
  }
  
  // If it's already absolute, use as is
  if (url.startsWith('http')) {
    return url;
  }
  
  // Default case - just return the original
  return url;
};

// Update the getCurrentVideoSrc function
const getCurrentVideoSrc = (): string => {
  // if (showAnnotatedVideo && analysisResults?.annotatedVideoUrl) {
  //   const videoSrc = getVideoSource(analysisResults.annotatedVideoUrl);
  //   // Add a cache-busting parameter for HTTP URLs
  //   if (videoSrc.startsWith('http')) {
  //     return `${videoSrc}?t=${new Date().getTime()}`;
  //   }
  //   return videoSrc;
  // }
  // return uploadedVideo || '';
  
  return analysisResults?.annotatedVideoUrl || '';
};
  
  const handleAnalyzeVideo = async () => {
    if (!uploadedVideo || !videoFileRef.current) return;
    
    setIsAnalyzing(true);
    setAnalysisError(null);
    setVideoAccessError(null);
    
    try {
      console.log('Analyzing video file:', videoFileRef.current.name, videoFileRef.current.type);
      
      // Convert File to Blob for analysis
      const blob = await videoFileRef.current.arrayBuffer().then(buffer => new Blob([buffer], { type: videoFileRef.current?.type }));
      
      // Call the analysis function from our module
      const results = await analyzeExerciseForm(blob, decodedExercise);
      
      console.log('Analysis results:', results);
      
      if (results.annotatedVideoUrl) {
        console.log('Video URL/path provided:', results.annotatedVideoUrl);
        
        // For direct file system paths
        if (results.annotatedVideoUrl.includes('backend/static/videos')) {
          console.log('Using direct file system path');
        } 
        // For API URLs
        else {
          const apiUrl = getVideoSource(results.annotatedVideoUrl);
          console.log('Using API URL:', apiUrl);
          
          // Test HTTP URL accessibility
          try {
            const headResponse = await fetch(apiUrl, { method: 'HEAD' });
            console.log('API URL test result:', headResponse.status, headResponse.statusText);
          } catch (error) {
            console.error('Error testing API URL:', error);
          }
        }
      }
      
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
  
  const handleToggleDebug = () => {
    setVideoDebugMode(!videoDebugMode);
  };
  
  const handleToggleVideo = () => {
    setShowAnnotatedVideo(!showAnnotatedVideo);
  };
  
  const handleGoBack = () => {
    navigate(-1); // Go back to previous page
  };
  
  // Function to get raw data from API for debugging
  const handleFetchRawAnnotatedVideo = async () => {
    if (!analysisResults?.annotatedVideoUrl) return;
    
    try {
      // Extract video name from URL
      const videoName = analysisResults.annotatedVideoUrl.split('/').pop();
      
      if (!videoName) {
        console.error('Could not extract video name from URL');
        return;
      }
      
      // Make a direct fetch to the backend endpoint
      const directUrl = `${API_BASE_URL}/analyze_pushup/video/${videoName}`;
      console.log('Attempting direct fetch from:', directUrl);
      
      const response = await fetch(directUrl);
      
      console.log('Direct API response:', {
        status: response.status,
        statusText: response.statusText,
        contentType: response.headers.get('Content-Type'),
        contentLength: response.headers.get('Content-Length'),
      });
      
      if (!response.ok) {
        setVideoAccessError(`Direct API error: ${response.status} ${response.statusText}`);
      } else {
        setVideoAccessError(null);
      }
    } catch (error) {
      console.error('Error in direct API fetch:', error);
      setVideoAccessError(`API fetch error: ${error instanceof Error ? error.message : String(error)}`);
    }
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
        <Box sx={{ mb: 4, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
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
          
          <Button
            startIcon={<BugReportIcon />}
            onClick={handleToggleDebug}
            color="info"
            variant="outlined"
            size="small"
          >
            {videoDebugMode ? "Hide Debug" : "Debug Mode"}
          </Button>
        </Box>
        
        {videoDebugMode && videoAccessError && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            <Typography variant="subtitle2">Video Access Error:</Typography>
            <Typography variant="body2">{videoAccessError}</Typography>
            <Button 
              size="small" 
              onClick={handleFetchRawAnnotatedVideo}
              sx={{ mt: 1 }}
            >
              Test Direct API Access
            </Button>
          </Alert>
        )}
        
        {videoDebugMode && analysisResults?.annotatedVideoUrl && (
          <Alert severity="info" sx={{ mb: 2 }}>
            <Typography variant="subtitle2">Annotated Video URL:</Typography>
            <Typography 
              variant="body2" 
              component="div"
              sx={{ 
                wordBreak: 'break-all', 
                py: 1,
                px: 2,
                bgcolor: 'background.paper',
                borderRadius: 1,
                my: 1
              }}
            >
              {analysisResults.annotatedVideoUrl}
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button 
                size="small" 
                variant="contained" 
                onClick={() => window.open(analysisResults.annotatedVideoUrl, '_blank')}
              >
                Open in New Tab
              </Button>
            </Box>
          </Alert>
        )}
        
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
                src={require('./fdbb092b58863e5c86fdb8bb1411fcea.mov')}
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
                    <Box 
                      sx={{ 
                        width: '100%', 
                        height: '300px', 
                        mb: 2, 
                        bgcolor: 'black',
                        position: 'relative',
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                        borderRadius: 1,
                        overflow: 'hidden'
                      }}
                    >
                      {/* Original video */}
                      {!showAnnotatedVideo && (
                        <video 
                          controls 
                          src={uploadedVideo} 
                          style={{ width: '100%', height: '100%' }}
                        />
                      )}
                      
                      {/* Annotated video (if available) */}
                      {showAnnotatedVideo && analysisResults?.annotatedVideoUrl && (
                        <>
                          {videoAccessError ? (
                            <Box sx={{ p: 3, textAlign: 'center' }}>
                              <ErrorIcon sx={{ fontSize: 48, color: 'error.main', mb: 2 }} />
                              <Typography color="error" variant="body1" gutterBottom>
                                Unable to load annotated video
                              </Typography>
                              <Typography color="text.secondary" variant="body2">
                                The server may still be processing your video.
                              </Typography>
                            </Box>
                          ) : (
                            // <video 
                            //   controls 
                            //   src={require('./annotated_pike_push-ups_1746145382534.mp4')} 
                            //   style={{ width: '100%', height: '100%' }}
                            //   onError={(e) => {
                            //     console.error('Video load error:', e);
                                
                            //     // Log additional information about the URL
                            //     const videoSrc = getCurrentVideoSrc();
                            //     console.error('Failed video URL:', videoSrc);
                                
                            //     // Check if the file exists for local file paths
                            //     if (videoSrc.startsWith('../../../')) {
                            //       console.error('This is a local file path. Check if the directory exists and has the right permissions.');
                            //     }
                                
                            //     setVideoAccessError('Error loading video. Check if the video file exists at the specified path.');
                            //   }}
                            // />
                            <Card>
                              <CardMedia
                                component="video"
                                controls
                                src={require('./annotated_pike_push-ups_1746148432247.mov')}
                                sx={{ width: '100%', height: '300px' }}
                              />
                            </Card>
                          )}
                        </>
                      )}
                    </Box>
                    
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
                  
                  {videoDebugMode && analysisResults.rawAnalysis && (
                    <Box sx={{ mt: 3 }}>
                      <Typography variant="subtitle1" gutterBottom>
                        Raw Analysis Data:
                      </Typography>
                      <Box 
                        component="pre" 
                        sx={{ 
                          p: 2, 
                          bgcolor: 'grey.100', 
                          borderRadius: 1, 
                          fontSize: '0.75rem',
                          overflowX: 'auto'
                        }}
                      >
                        {JSON.stringify(analysisResults.rawAnalysis, null, 2)}
                      </Box>
                    </Box>
                  )}
                </CardContent>
              </Card>
            )}
      </Paper>
    </Box>
  );
};

export default VideoPlayer;