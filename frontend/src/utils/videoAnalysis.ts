// videoAnalysis.ts - A separate module for handling video analysis
import { API_BASE_URL, VIDEO_ANALYSIS_CONFIG } from '../config';

// Define the interface for analysis results
interface AnalysisResult {
  score: number;
  feedback: string[];
  rawAnalysis: any;
  annotatedVideoUrl?: string; // Make this property optional
}

/**
 * Validate that we can analyze this exercise type
 */
export const canAnalyzeExercise = (exerciseName: string): boolean => {
  // For now, we'll just assume all pushup-like exercises can be analyzed with the pushup endpoint
  const normalizedName = exerciseName.toLowerCase().trim();
  return normalizedName.includes('push') || VIDEO_ANALYSIS_CONFIG.supportedExercises.some(
    ex => normalizedName.includes(ex.toLowerCase())
  );
};

/**
 * Analyze a video by sending it to the FastAPI backend
 * 
 * @param videoBlob - The video file blob to analyze
 * @param exerciseName - The type of exercise
 * @returns Analysis results from the server
 */
export const analyzeExerciseForm = async (videoBlob: Blob, exerciseName: string): Promise<AnalysisResult> => {
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
    console.log('Backend response:', analysisData);
    
    // Format the response to match our frontend expectations
    const result: AnalysisResult = {
      score: calculateFormScore(analysisData.analysis),
      feedback: generateFeedback(analysisData.analysis),
      rawAnalysis: analysisData.analysis // Also return the raw analysis data for debugging
    };
    
    // Handle the annotated video URL and path
    if (analysisData.annotated_video_path) {
      // Use the direct file system path first
      result.annotatedVideoUrl = analysisData.annotated_video_path;
      console.log('Using direct file system path for video:', result.annotatedVideoUrl);
    } else if (analysisData.annotated_video_url) {
      // Fall back to the API URL if file system path is not available
      let videoUrl = analysisData.annotated_video_url;
      
      // If it's not already absolute, make it absolute
      if (!videoUrl.startsWith('http')) {
        const separator = videoUrl.startsWith('/') ? '' : '/';
        videoUrl = `${API_BASE_URL}${separator}${videoUrl}`;
      }
      
      result.annotatedVideoUrl = videoUrl;
      console.log('Using API URL for video:', result.annotatedVideoUrl);
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