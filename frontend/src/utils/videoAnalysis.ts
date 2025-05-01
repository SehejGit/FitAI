// videoAnalysis.ts - A separate module for handling video analysis
import { API_BASE_URL, VIDEO_ANALYSIS_CONFIG } from '../config';

/**
 * Validate that we can analyze this exercise type
 */
export const canAnalyzeExercise = (exerciseName: string): boolean => {
  // Convert to lowercase and normalize
  const normalizedName = exerciseName.toLowerCase().trim();
  
  // Check if this is a supported exercise
  for (const supported of VIDEO_ANALYSIS_CONFIG.supportedExercises) {
    if (normalizedName.includes(supported.toLowerCase())) {
      return true;
    }
  }
  
  return false;
};

/**
 * Analyze a video by sending it to the FastAPI backend
 * 
 * @param videoBlob - The video file blob to analyze
 * @param exerciseName - The type of exercise
 * @returns Analysis results from the server
 */
export const analyzeExerciseForm = async (videoBlob: Blob, exerciseName: string): Promise<any> => {
  // Check if we support this exercise type
  if (!canAnalyzeExercise(exerciseName)) {
    throw new Error(`Analysis for "${exerciseName}" is not currently supported. Supported exercises: ${VIDEO_ANALYSIS_CONFIG.supportedExercises.join(', ')}`);
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
    const fileName = `${exerciseName.toLowerCase().replace(/\s+/g, '_')}_${timestamp}.mp4`;
    
    formData.append('file', videoBlob, fileName);
    
    // Set return_video parameter to get annotated video back
    const returnVideo = true;
    
    // Make the API request
    const response = await fetch(`${API_BASE_URL}/analyze_pushup/?return_video=${returnVideo}`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to analyze video');
    }
    
    // Parse the JSON response
    const analysisData = await response.json();
    
    // Format the response to match our frontend expectations
    return {
      score: calculateFormScore(analysisData.analysis),
      feedback: generateFeedback(analysisData.analysis),
      annotatedVideoUrl: analysisData.annotated_video_url ? 
        `${API_BASE_URL}${analysisData.annotated_video_url}` : undefined,
      rawAnalysis: analysisData.analysis // Also return the raw analysis data for debugging
    };
  } catch (error) {
    console.error('Error analyzing video:', error);
    throw error;
  }
};

/**
 * Calculate an overall form score based on the analysis data
 * 
 * @param analysis - The analysis data from the backend
 * @returns A score from 0-100
 */
const calculateFormScore = (analysis: any): number => {
  if (analysis.error) {
    return 0; // Return a zero score if there was an error
  }
  
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
 * Generate feedback array based on analysis data
 * 
 * @param analysis - The analysis data from the backend
 * @returns Array of feedback strings
 */
const generateFeedback = (analysis: any): string[] => {
  if (analysis.error) {
    return [analysis.error];
  }
  
  if (analysis.pushup_count === 0) {
    return ['No pushups detected. Make sure your full body is visible in the video.'];
  }
  
  // Return the feedback directly from the backend
  return analysis.feedback;
};