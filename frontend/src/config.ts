// Configuration file for environment variables and app settings

// API Configuration
export const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Video analysis settings
export const VIDEO_ANALYSIS_CONFIG = {
  supportedExercises: ['Push-ups'], // Currently only pushups are supported
  maxFileSize: 50 * 1024 * 1024, // 50MB max file size
  supportedFormats: ['mp4', 'mov', 'webm'],
};

export default {
  API_BASE_URL,
  VIDEO_ANALYSIS_CONFIG,
};