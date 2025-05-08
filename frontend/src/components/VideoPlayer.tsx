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

const getExerciseVideoUrl = (exerciseName: string): string => {
  // normalize "Mountain climbers" → "mountain_climbers", "push-ups" → "push_ups"
  const formatted = exerciseName
    .toLowerCase()
    .replace(/[\s-]+/g, '_');
  // match your bucket's uppercase extension
  const ext = '.MOV';
  return `https://storage.googleapis.com/fitai-videos/${formatted}${ext}`;
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
  const [videoRetryCount, setVideoRetryCount] = useState(0);
  const [isRetryingVideo, setIsRetryingVideo] = useState(false);
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
    setVideoRetryCount(0);
    setIsRetryingVideo(false);
    videoFileRef.current = null;
  }, [exercise]);
  
  const getVideoApiUrl = (url: string | undefined): string => {
    // If url is undefined, return empty string
    if (!url) {
      console.warn('getVideoApiUrl called with undefined URL');
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
    
    // Check if the URL looks correct
    if (!filename.includes('annotated_')) {
      console.warn('Filename does not contain "annotated_":', filename);
    }
    
    // Create the API URL
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
      setVideoRetryCount(0);
      setIsRetryingVideo(false);
    }
  };

  // Get the current video source to display
  const getCurrentVideoSrc = (): string => {
    console.log('Getting current video src, showAnnotatedVideo:', showAnnotatedVideo);
    console.log('analysisResults?.annotatedVideoUrl:', analysisResults?.annotatedVideoUrl);
    
    if (showAnnotatedVideo && analysisResults?.annotatedVideoUrl) {
      const videoSrc = getVideoApiUrl(analysisResults.annotatedVideoUrl);
      console.log('Returning annotated video source:', videoSrc);
      return videoSrc;
    }
    console.log('Returning uploaded video source:', uploadedVideo);
    return uploadedVideo || '';
  };
  
  const handleAnalyzeVideo = async () => {
    if (!uploadedVideo || !videoFileRef.current) return;
    
    // Don't allow analysis if the exercise isn't supported
    // if (!isExerciseSupported) {
    //   setAnalysisError(`Exercise "${decodedExercise}" is not supported for analysis. Available exercises: ${availableExercises.join(', ')}`);
    //   return;
    // }
    
    setIsAnalyzing(true);
    setAnalysisError(null);
    setVideoAccessError(null);
    setVideoRetryCount(0);
    setIsRetryingVideo(false);
    
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
    console.log('Toggling video - current state:', showAnnotatedVideo);
    console.log('Analysis results URL exists:', !!analysisResults?.annotatedVideoUrl);
    setShowAnnotatedVideo(!showAnnotatedVideo);
    // Reset retry count when toggling
    setVideoRetryCount(0);
    setVideoAccessError(null);
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
  
  // Add retry mechanism for failed video loads
  const handleVideoRetry = async () => {
    if (videoRetryCount >= 3) {
      setVideoAccessError('Failed to load video after multiple attempts. Please check if the video exists on the server.');
      return;
    }
    
    setIsRetryingVideo(true);
    setVideoAccessError(null);
    setVideoRetryCount(prev => prev + 1);
    
    // Wait a moment before retrying
    setTimeout(() => {
      setIsRetryingVideo(false);
      // Force video reload by changing the src slightly
      const videoElement = document.querySelector('video[key="annotated"]') as HTMLVideoElement;
      if (videoElement && analysisResults?.annotatedVideoUrl) {
        const newSrc = `${getVideoApiUrl(analysisResults.annotatedVideoUrl)}?retry=${videoRetryCount}`;
        videoElement.src = newSrc;
        videoElement.load();
      }
    }, 1000);
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
                src={getExerciseVideoUrl(decodedExercise)}
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
                      {/* Show original video or annotated video based on toggle */}
                      {!showAnnotatedVideo ? (
                        <video 
                          key="original"
                          controls 
                          src={uploadedVideo} 
                          style={{ width: '100%', height: '100%' }}
                          onError={(e) => {
                            console.error('Original video error:', e);
                          }}
                        />
                      ) : analysisResults?.annotatedVideoUrl ? (
                        <>
                          {videoAccessError ? (
                            <Box sx={{ p: 3, textAlign: 'center' }}>
                              <ErrorIcon sx={{ fontSize: 48, color: 'error.main', mb: 2 }} />
                              <Typography color="error" variant="body1" gutterBottom>
                                {videoAccessError}
                              </Typography>
                              {videoRetryCount < 3 && (
                                <Button 
                                  variant="contained" 
                                  color="primary" 
                                  size="small"
                                  sx={{ mt: 1 }}
                                  onClick={handleVideoRetry}
                                  disabled={isRetryingVideo}
                                >
                                  {isRetryingVideo ? <CircularProgress size={20} /> : 'Retry Loading Video'}
                                </Button>
                              )}
                              {videoDebugMode && (
                                <Button 
                                  variant="outlined" 
                                  color="primary" 
                                  size="small"
                                  sx={{ mt: 1, ml: 1 }}
                                  onClick={handleFetchRawAnnotatedVideo}
                                >
                                  Test Direct API Access
                                </Button>
                              )}
                            </Box>
                          ) : (
                            <video 
                              key="annotated"
                              controls 
                              src={getCurrentVideoSrc()} 
                              style={{ width: '100%', height: '100%' }}
                              crossOrigin="anonymous"
                              onError={(e) => {
                                console.error('Annotated video error:', e);
                                console.log('Failed annotated video URL:', e.currentTarget.src);
                                
                                // Check if we have specific error details
                                const videoElement = e.currentTarget as HTMLVideoElement;
                                let errorMessage = 'Error loading video. Please try again.';
                                
                                if (videoElement.error) {
                                  // Map error codes to more helpful messages
                                  switch (videoElement.error.code) {
                                    case videoElement.error.MEDIA_ERR_ABORTED:
                                      errorMessage = 'Video loading was aborted.';
                                      break;
                                    case videoElement.error.MEDIA_ERR_NETWORK:
                                      errorMessage = 'Network error while loading the video.';
                                      break;
                                    case videoElement.error.MEDIA_ERR_DECODE:
                                      errorMessage = 'Error decoding the video file.';
                                      break;
                                    case videoElement.error.MEDIA_ERR_SRC_NOT_SUPPORTED:
                                      errorMessage = 'Video format not supported.';
                                      break;
                                    default:
                                      errorMessage = `Video error: ${videoElement.error.message || 'Unknown error'}`;
                                  }
                                }
                                
                                console.log('Video error details:', {
                                  code: videoElement.error?.code,
                                  message: videoElement.error?.message,
                                  src: videoElement.src
                                });
                                
                                // Wait a moment before showing error in case video is still loading
                                setTimeout(() => {
                                  setVideoAccessError(errorMessage);
                                }, 1000);
                              }}
                              onLoadStart={() => console.log('Video load started')}
                              onCanPlay={() => console.log('Video can play')}
                              onCanPlayThrough={() => {
                                console.log('Video can play through');
                                // Clear any existing error when video starts playing
                                setVideoAccessError(null);
                              }}
                            />
                          )}
                        </>
                      ) : null}
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