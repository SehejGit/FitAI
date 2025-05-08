// config.ts
// Configuration file for environment variables and app settings

// API Configuration 
// Default to localhost:8000 if no environment variable is set
// export const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Changed to refer to the deployed backend
export const API_BASE_URL = process.env.REACT_APP_API_URL || '';  

// export const DISPLAY_TO_API_MAPPING = {
//   'Push-ups': 'push_ups'  // Match exactly what your backend expects
// };

// Video analysis settings
export const VIDEO_ANALYSIS_CONFIG = {
  supportedExercises: ['pushups', 'squats', 'planks', 'bench_press'], // Exercises the backend can analyze
  maxFileSize: 50 * 1024 * 1024, // 50MB max file size
  supportedFormats: ['mp4', 'mov', 'webm'],
  
  // Endpoint mapping for different exercise types
  endpoints: {
    default: '/analyze/push_ups/',
    'push_ups': '/analyze/push_ups/',
    'pushups': '/analyze/push_ups/',
    'pike push-ups': '/analyze/push_ups/',
  }
};

// Export all config values as a default export
export default {
  API_BASE_URL,
  VIDEO_ANALYSIS_CONFIG,
};