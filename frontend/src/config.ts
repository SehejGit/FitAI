// Configuration file for environment variables and app settings

// API Configuration 
// Default to localhost:8000 if no environment variable is set
// export const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Changed to refer to the deployed backend
export const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://api.allorigins.win/raw?url=https://34-8-10-114.nip.io';

export const DISPLAY_TO_API_MAPPING = {
  'Push-ups': 'pushups'
};

// Video analysis settings
export const VIDEO_ANALYSIS_CONFIG = {
  supportedExercises: ['pushups', 'squats', 'planks', 'bench_press'], // Exercises the backend can analyze
  maxFileSize: 50 * 1024 * 1024, // 50MB max file size
  supportedFormats: ['mp4', 'mov', 'webm'],
  
  // Endpoint mapping for different exercise types
  endpoints: {
    default: '/analyze_pushup/',
    'pushups': '/analyze_pushup/',
    'pike push-ups': '/analyze_pushup/',
  }
};

// Export all config values as a default export
export default {
  API_BASE_URL,
  VIDEO_ANALYSIS_CONFIG,
};