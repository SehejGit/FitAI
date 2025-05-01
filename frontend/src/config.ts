// Configuration file for environment variables and app settings

// API Configuration 
// Default to localhost:8000 if no environment variable is set
export const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Video analysis settings
export const VIDEO_ANALYSIS_CONFIG = {
  supportedExercises: ['Push-ups', 'Pike push-ups'], // Exercises the backend can analyze
  maxFileSize: 50 * 1024 * 1024, // 50MB max file size
  supportedFormats: ['mp4', 'mov', 'webm'],
  
  // Endpoint mapping for different exercise types
  endpoints: {
    default: '/analyze_pushup/',
    'push-ups': '/analyze_pushup/',
    'pike push-ups': '/analyze_pushup/',
  }
};

// Export all config values as a default export
export default {
  API_BASE_URL,
  VIDEO_ANALYSIS_CONFIG,
};