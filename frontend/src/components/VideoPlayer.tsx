// VideoPlayer.tsx
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
import { analyzeExerciseForm, formatExerciseNameForApi, fetchAvailableExercises, canAnalyzeExercise } from '../utils/videoAnalysis';
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
  const [availableExercises, setAvailableExercises] = useState<string[]>([]);
  const [isExerciseSupported, setIsExerciseSupported] = useState<boolean>(true);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoFileRef = useRef<File | null>(null);
  
  const decodedExercise = exercise ? decodeURIComponent(exercise) : '';
  const formattedExercise = formatExerciseNameForApi(decodedExercise);
  const videoUrl = getExerciseVideoUrl(decodedExercise);
  
  // Fetch available exercises from API on component mount and check if current exercise is supported
  useEffect(() => {
    const fetchExercises = async () => {
      try {
        const exercises = await fetchAvailableExercises();
        setAvailableExercises(exercises);
        
        // Check if current exercise is supported
        const supported = await canAnalyzeExercise(decodedExercise);
        setIsExerciseSupported(supported);
      } catch (error) {
        console.error('Error checking exercise support:', error);
      }
    };
    
    fetchExercises();
  }, [decodedExercise]);
  
  // Reset state when exercise changes
  useEffect(() => {
    setUploadedVideo(null);
    setAnalysisResults(null);
    setAnalysisError(null);
    setShowAnnotatedVideo(false);
    setVideoAccessError(null);
    videoFileRef.current = null;
  }, [exercise]);
  
  const getVideoApiUrl = (url: string | undefined): string => {
    // If url is undefined, return empty string
    if (!url) {
      return '';
    }
    
    console.log('Original video URL:', url);
    
    // Extract just the filename from the path
    const filename = url.split('/').pop();
    
    if (!filename) {
      console.error('Could not extract filename from URL:', url);
      return url;
    }
    
    console.log('Extracted filename:', filename);
    
    // Don't modify the filename at all - let the server handle the variations
    // Just pass it as is to the API endpoint
    const apiUrl = `${API_BASE_URL}/api/videos/${filename}`;
    console.log('Final API URL:', apiUrl);
    return apiUrl;
  };
  
  // Check if the annotated video URL is accessible when it changes
  useEffect(() => {
    // Update the verification function
    const verifyVideoAccess = async () => {
      if (analysisResults?.annotatedVideoUrl) {
        try {
          console.log('Checking video URL accessibility:', analysisResults.annotatedVideoUrl);
          const videoUrl = getVideoApiUrl(analysisResults.annotatedVideoUrl);
          console.log('Testing accessibility of:', videoUrl);
          
          // Skip HEAD request and use GET directly with a timeout
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
          
          try {
            // Use a range request to just get the first few bytes
            const response = await fetch(videoUrl, { 
              method: 'GET',
              headers: {
                'Range': 'bytes=0-1000' // Just get the first 1000 bytes to check availability
              },
              signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            console.log('Video access test response:', {
              status: response.status,
              statusText: response.statusText,
              headers: {
                'content-type': response.headers.get('content-type'),
                'content-length': response.headers.get('content-length')
              }
            });
            
            // 206 Partial Content is the success response for range requests
            if (response.status === 206 || response.status === 200) {
              setVideoAccessError(null);
              console.log('Video URL is accessible!');
            } else {
              setVideoAccessError(`Video access error: ${response.status} ${response.statusText}`);
              console.error('Video URL is not accessible:', response.status, response.statusText);
            }
          } catch (err) {
            clearTimeout(timeoutId);
            throw err;
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

  // Get the current video source to display
  const getCurrentVideoSrc = (): string => {
    if (showAnnotatedVideo && analysisResults?.annotatedVideoUrl) {
      const videoSrc = getVideoApiUrl(analysisResults.annotatedVideoUrl);
      console.log('Using video source:', videoSrc);
      return videoSrc;
    }
    return uploadedVideo || '';
  };
  
  const handleAnalyzeVideo = async () => {
    if (!uploadedVideo || !videoFileRef.current) return;
    
    // Don't allow analysis if the exercise isn't supported
    if (!isExerciseSupported) {
      setAnalysisError(`Exercise "${decodedExercise}" is not supported for analysis. Available exercises: ${availableExercises.join(', ')}`);
      return;
    }
    
    setIsAnalyzing(true);
    setAnalysisError(null);
    setVideoAccessError(null);
    
    try {
      console.log('Analyzing video file:', videoFileRef.current.name, videoFileRef.current.type);
      console.log('Exercise type:', decodedExercise);
      console.log('Formatted exercise type for API:', formattedExercise);
      
      // Convert File to Blob for analysis
      const blob = await videoFileRef.current.arrayBuffer().then(buffer => 
        new Blob([buffer], { type: videoFileRef.current?.type }));
      
      // Call the analysis function from our module
      try {
        const results = await analyzeExerciseForm(blob, decodedExercise);
        
        console.log('Analysis results:', results);
        
        if (results.annotatedVideoUrl) {
          console.log('Video URL/path provided:', results.annotatedVideoUrl);
          
          // Test API URL accessibility
          const apiUrl = getVideoApiUrl(results.annotatedVideoUrl);
          console.log('Using API URL:', apiUrl);
          
          try {
            const headResponse = await fetch(apiUrl, { method: 'HEAD' });
            console.log('API URL test result:', headResponse.status, headResponse.statusText);
          } catch (error) {
            console.error('Error testing API URL:', error);
          }
        }
        
        setAnalysisResults(results);
        
        // Automatically show the annotated video when analysis is complete
        if (results.annotatedVideoUrl) {
          setShowAnnotatedVideo(true);
        }
      } catch (error) {
        console.error('Analysis error:', error);
        setAnalysisError(error instanceof Error ? error.message : String(error));
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
      const apiUrl = getVideoApiUrl(analysisResults.annotatedVideoUrl);
      console.log('Attempting direct fetch from:', apiUrl);
      
      // Try different HTTP methods
      for (const method of ['GET', 'HEAD']) {
        try {
          console.log(`Trying ${method} request...`);
          const response = await fetch(apiUrl, { method });
          
          console.log(`${method} response:`, {
            status: response.status,
            statusText: response.statusText,
            contentType: response.headers.get('Content-Type'),
            contentLength: response.headers.get('Content-Length'),
          });
        } catch (error) {
          console.error(`Error with ${method} request:`, error);
        }
      }
      
      // Also try the original URL format
      try {
        const originalUrl = `${API_BASE_URL}${analysisResults.annotatedVideoUrl}`;
        console.log('Trying original URL format:', originalUrl);
        const response = await fetch(originalUrl, { method: 'HEAD' });
        console.log('Original URL response:', {
          status: response.status,
          statusText: response.statusText,
        });
      } catch (error) {
        console.error('Error with original URL format:', error);
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
        
        {!isExerciseSupported && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            <Typography variant="subtitle2">
              This exercise ({decodedExercise}) is not supported for automated analysis.
            </Typography>
            <Typography variant="body2">
              Available exercises: {availableExercises.join(', ')}
            </Typography>
          </Alert>
        )}
        
        {videoDebugMode && (
          <Alert severity="info" sx={{ mb: 2 }}>
            <Typography variant="subtitle2">Exercise API Debug Info:</Typography>
            <Typography variant="body2">
              Original exercise name: {decodedExercise}
            </Typography>
            <Typography variant="body2">
              Formatted for API: {formattedExercise}
            </Typography>
            <Typography variant="body2">
              API endpoint: {`${API_BASE_URL}/analyze/${formattedExercise}/`}
            </Typography>
          </Alert>
        )}
        
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
            <Typography variant="subtitle2">API Endpoint URL:</Typography>
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
              {getVideoApiUrl(analysisResults.annotatedVideoUrl)}
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button 
                size="small" 
                variant="contained" 
                onClick={() => analysisResults.annotatedVideoUrl && window.open(getVideoApiUrl(analysisResults.annotatedVideoUrl), '_blank')}
              >
                Open in New Tab
              </Button>
              <Button
                size="small"
                variant="outlined"
                onClick={() => analysisResults.annotatedVideoUrl && window.open(`${API_BASE_URL}${analysisResults.annotatedVideoUrl}`, '_blank')}
              >
                Try Original URL
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
                                The server may still be processing your video or there may be an access issue.
                              </Typography>
                              {videoDebugMode && (
                                <Button 
                                  variant="outlined" 
                                  color="primary" 
                                  size="small"
                                  sx={{ mt: 1 }}
                                  onClick={handleFetchRawAnnotatedVideo}
                                >
                                  Retry Video Access
                                </Button>
                              )}
                            </Box>
                          ) : (
                            <video 
                              controls 
                              src={getVideoApiUrl(analysisResults.annotatedVideoUrl)} 
                              style={{ width: '100%', height: '100%' }}
                              onError={(e) => {
                                console.error('Video load error:', e);
                                
                                // Log additional information about the URL
                                const videoSrc = getCurrentVideoSrc();
                                console.error('Failed video URL:', videoSrc);
                                
                                setVideoAccessError('Error loading video. Check if the video file exists and is accessible.');
                              }}
                            />
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
                        disabled={isAnalyzing || !isExerciseSupported}
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