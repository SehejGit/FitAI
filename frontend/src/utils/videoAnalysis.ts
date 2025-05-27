import { API_BASE_URL, VIDEO_ANALYSIS_CONFIG } from '../config';

// Define the interface for analysis results
interface AnalysisResult {
  score: number;
  feedback: string[];
  rawAnalysis: any;
  annotatedVideoUrl?: string; // Make this property optional
}

// Store the cache of available exercises
let availableExercisesCache: string[] = [];

/**
 * Fetch available exercises from the API
 */
export const fetchAvailableExercises = async (): Promise<string[]> => {
  try {
    // Check if we already have the exercises cached
    if (availableExercisesCache.length > 0) {
      console.log('Using cached exercises:', availableExercisesCache);
      return availableExercisesCache;
    }
    
    // Fetch from the API
    console.log('Fetching from API:', `${API_BASE_URL}/exercises/`);
    const response = await fetch(`${API_BASE_URL}/exercises/`);
    
    if (!response.ok) {
      console.error('API response not OK:', response.status, response.statusText);
      throw new Error(`Failed to fetch available exercises: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('API Response:', data);
    
    const exercises = data.available_exercises || [];
    
    // Cache the result
    availableExercisesCache = exercises;
    console.log('Available exercises from API:', exercises);
    
    return exercises;
  } catch (error) {
    console.error('Error fetching available exercises:', error);
    console.log('Falling back to hardcoded list');
    // Return an empty array or fallback to hardcoded list
    return VIDEO_ANALYSIS_CONFIG.supportedExercises.map(ex => formatExerciseNameForApi(ex));
  }
};

/**
 * Format exercise name for API endpoint
 * Convert to lowercase and replace spaces with underscores
 */
export const formatExerciseNameForApi = (exerciseName: string): string => {
  return exerciseName.toLowerCase().replace(/\s+/g, '_').replace(/-/g, '_');
};

/**
 * Validate that we can analyze this exercise type
 */
export const canAnalyzeExercise = async (exerciseName: string): Promise<boolean> => {
  // Format the exercise name first
  const formattedName = formatExerciseNameForApi(exerciseName);
  
  // Get available exercises from API
  const availableExercises = await fetchAvailableExercises();
  
  // Check if this exercise is in the list
  return availableExercises.includes(formattedName);
};

/**
 * Analyze a video by sending it to the FastAPI backend
 * 
 * @param videoBlob - The video file blob to analyze
 * @param exerciseName - The type of exercise
 * @returns Analysis results from the server
 */
export const analyzeExerciseForm = async (videoBlob: Blob, exerciseName: string): Promise<AnalysisResult> => {
  // Format the exercise name for API use
  const formattedExerciseName = formatExerciseNameForApi(exerciseName);
  
  // Check if we support this exercise type
  const isSupported = await canAnalyzeExercise(exerciseName);
  if (!isSupported) {
    throw new Error(`Analysis for "${exerciseName}" is not currently supported by the server.`);
  }
  
  // Check file size
  if (videoBlob.size > VIDEO_ANALYSIS_CONFIG.maxFileSize) {
    throw new Error(`Video file too large. Maximum size is ${VIDEO_ANALYSIS_CONFIG.maxFileSize / (1024 * 1024)}MB`);
  }
  
  try {
    // Create a FormData object to hold the video file
    const formData = new FormData();
    
    // Add the video file to the FormData
    // Generate a filename with exercise name and timestamp
    const timestamp = new Date().getTime();
    const fileName = `${formattedExerciseName}_${timestamp}.mp4`;
    
    formData.append('file', videoBlob, fileName);
    
    // Set return_video parameter to get annotated video back
    const returnVideo = true;
    
    // Use the dynamic endpoint with the formatted exercise name
    const apiEndpoint = `${API_BASE_URL}/analyze/${formattedExerciseName}/?return_video=${returnVideo}`;
    console.log(`Calling API endpoint: ${apiEndpoint}`);
    
    // Make the API request
    const response = await fetch(apiEndpoint, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      // Try to get more detailed error info
      let errorDetail = 'Failed to analyze video';
      try {
        const errorData = await response.json();
        errorDetail = errorData.detail || errorDetail;
      } catch (e) {
        // If we can't parse the JSON, try to get the text
        try {
          errorDetail = await response.text() || errorDetail;
        } catch (_) {
          // If we can't get the text either, just use the status
          errorDetail = `HTTP ${response.status}: ${response.statusText}`;
        }
      }
      
      throw new Error(errorDetail);
    }
    
    // Parse the JSON response
    const analysisData = await response.json();
    console.log('Backend response:', analysisData);
    
    // Format the response to match our frontend expectations
    const result: AnalysisResult = {
      score: calculateFormScore(analysisData.analysis, formattedExerciseName),
      feedback: generateFeedback(analysisData.analysis),
      rawAnalysis: analysisData.analysis // Also return the raw analysis data for debugging
    };
    
    // Handle the annotated video URL and path
    if (analysisData.annotated_video_url) {
      // Use the annotated video URL from the API response
      let videoUrl = analysisData.annotated_video_url;
      
      // If it's not already absolute, make it absolute
      if (!videoUrl.startsWith('http')) {
        const separator = videoUrl.startsWith('/') ? '' : '/';
        videoUrl = `${API_BASE_URL}${separator}${videoUrl}`;
      }
      
      result.annotatedVideoUrl = videoUrl;
      console.log('Using annotated video URL:', result.annotatedVideoUrl);
    }
    
    return result;
  } catch (error) {
    console.error('Error analyzing video:', error);
    throw error;
  }
};

/**
 * Calculate an overall form score based on the analysis data
 * 
 * @param analysis - The analysis data from the backend
 * @param exerciseType - The type of exercise (formatted for API)
 * @returns A score from 0-100
 */
const calculateFormScore = (analysis: any, exerciseType: string): number => {
  if (analysis.error) {
    return 0; // Return a zero score if there was an error
  }
  
  // Generic score calculation based on exercise type
  switch (exerciseType) {
    case 'pushup':
    case 'push_up':
    case 'push_ups':
      return calculatePushupScore(analysis);
    
    case 'squat':
    case 'squats':
      return calculateSquatScore(analysis);
      
    case 'plank':
    case 'planks':
      return calculatePlankScore(analysis);
      
    case 'bench_press':
      return calculateBenchPressScore(analysis);
      
    // Add more exercise types as needed
      
    default:
      // Generic score calculation for unknown exercise types
      return calculateGenericScore(analysis);
  }
};

/**
 * Calculate score for pushup exercises
 */
const calculatePushupScore = (analysis: any): number => {
  // If no pushups detected, return low score
  if (analysis.pushup_count === 0) {
    return 30;
  }
  
  const {
    form_analysis: {
      body_alignment_score,
      elbow_angle_at_bottom,
      elbow_angle_at_top
    }
  } = analysis;
  
  // Calculate scores for different aspects (adjust weights as needed)
  let elbowBottomScore = 100;
  if (elbow_angle_at_bottom > 90) {
    // Deduct points if not going deep enough
    // 90 degrees is ideal, deduct more points the further from 90
    elbowBottomScore = Math.max(0, 100 - (elbow_angle_at_bottom - 90) * 2);
  }
  
  let elbowTopScore = 100;
  if (elbow_angle_at_top < 150) {
    // Deduct points if not extending fully
    // 180 degrees would be completely straight, but 150+ is good
    elbowTopScore = Math.max(0, 100 - (150 - elbow_angle_at_top) * 2);
  }
  
  // Combine the scores with appropriate weighting
  const weightedScore = (
    (body_alignment_score * 0.4) + 
    (elbowBottomScore * 0.3) + 
    (elbowTopScore * 0.3)
  );
  
  // Round to nearest integer
  return Math.round(weightedScore);
};

/**
 * Calculate score for squat exercises
 */
const calculateSquatScore = (analysis: any): number => {
  // If no squats detected, return low score
  if (analysis.squat_count === 0) {
    return 30;
  }
  
  const {
    form_analysis: {
      knee_angle_at_bottom,
      knee_angle_at_top,
      back_alignment_score = 100, // Default if not provided
      knee_tracking_score = 100   // Default if not provided
    }
  } = analysis;
  
  // Calculate scores for different aspects
  let kneeBottomScore = 100;
  if (knee_angle_at_bottom > 110) {
    // Deduct points if not squatting deep enough
    // Ideal is around 90-100 degrees
    kneeBottomScore = Math.max(0, 100 - (knee_angle_at_bottom - 110) * 3);
  } else if (knee_angle_at_bottom < 70) {
    // Too deep can also be problematic
    kneeBottomScore = Math.max(0, 100 - (70 - knee_angle_at_bottom) * 3);
  }
  
  // Combine the scores with appropriate weighting
  const weightedScore = (
    (back_alignment_score * 0.3) + 
    (kneeBottomScore * 0.4) + 
    (knee_tracking_score * 0.3)
  );
  
  // Round to nearest integer
  return Math.round(weightedScore);
};

/**
 * Calculate score for plank exercises
 */
const calculatePlankScore = (analysis: any): number => {
  const {
    plank_duration = 0,
    form_analysis: {
      body_alignment_score = 0,
      hip_position_score = 0
    }
  } = analysis;
  
  // Duration component (max score at 60+ seconds)
  const durationScore = Math.min(100, (plank_duration / 60) * 100);
  
  // Combine the scores with appropriate weighting
  const weightedScore = (
    (body_alignment_score * 0.5) + 
    (hip_position_score * 0.3) + 
    (durationScore * 0.2)
  );
  
  // Round to nearest integer
  return Math.round(weightedScore);
};

/**
 * Calculate score for bench press exercises
 */
const calculateBenchPressScore = (analysis: any): number => {
  // If no reps detected, return low score
  if (analysis.press_count === 0) {
    return 30;
  }
  
  const {
    form_analysis: {
      elbow_angle_at_bottom,
      elbow_angle_at_top,
      arm_symmetry_score = 100,
      bar_path_score = 100
    }
  } = analysis;
  
  // Calculate scores for different aspects
  let elbowBottomScore = 100;
  if (elbow_angle_at_bottom > 90) {
    // Deduct points if not lowering enough
    elbowBottomScore = Math.max(0, 100 - (elbow_angle_at_bottom - 90) * 2);
  }
  
  let elbowTopScore = 100;
  if (elbow_angle_at_top < 160) {
    // Deduct points if not extending fully
    elbowTopScore = Math.max(0, 100 - (160 - elbow_angle_at_top) * 2);
  }
  
  // Combine the scores with appropriate weighting
  const weightedScore = (
    (elbowBottomScore * 0.3) + 
    (elbowTopScore * 0.2) + 
    (arm_symmetry_score * 0.25) +
    (bar_path_score * 0.25)
  );
  
  // Round to nearest integer
  return Math.round(weightedScore);
};

/**
 * Generic score calculation for exercises without specific scoring
 */
const calculateGenericScore = (analysis: any): number => {
  // Check if there's a count property with 'count' in the name (e.g., rep_count, squat_count)
  const countProps = Object.keys(analysis).filter(key => key.toLowerCase().includes('count'));
  if (countProps.length > 0 && analysis[countProps[0]] === 0) {
    return 30; // Low score if no reps detected
  }
  
  // Look for form_analysis in the response
  if (analysis.form_analysis) {
    // Get all score properties (properties that end with '_score')
    const scoreProps = Object.keys(analysis.form_analysis)
      .filter(key => key.endsWith('_score'))
      .map(key => analysis.form_analysis[key]);
    
    if (scoreProps.length > 0) {
      // Average all the scores
      const averageScore = scoreProps.reduce((sum, score) => sum + score, 0) / scoreProps.length;
      return Math.round(averageScore);
    }
  }
  
  // If we can't calculate a score, use the feedback length as a proxy (more feedback = lower score)
  const feedbackCount = analysis.feedback?.length || 0;
  return Math.max(0, 100 - (feedbackCount * 15));
};

/**
 * Generate feedback array based on analysis data
 * 
 * @param analysis - The analysis data from the backend
 * @returns Array of feedback strings
 */
const generateFeedback = (analysis: any): string[] => {
  if (analysis.error) {
    return [analysis.error];
  }
  
  // Check for a count property (pushup_count, squat_count, etc.)
  const countProps = Object.keys(analysis).filter(key => key.toLowerCase().includes('count'));
  if (countProps.length > 0 && analysis[countProps[0]] === 0) {
    return ['No repetitions detected. Make sure your full body is visible in the video.'];
  }
  
  // Return the feedback directly from the backend if available
  if (analysis.feedback && Array.isArray(analysis.feedback)) {
    return analysis.feedback;
  }
  
  // Fallback generic feedback if none provided
  return ['Analysis completed. No specific feedback provided.'];
};