// Utility to check CORS issues and help with debugging backend connection

import { API_BASE_URL } from '../config';

/**
 * Check if the API is reachable and CORS is properly configured
 * @returns Promise resolving to true if API is accessible, false otherwise
 */
export const checkApiConnection = async (): Promise<boolean> => {
  try {
    // Make an OPTIONS request to check CORS headers
    const response = await fetch(API_BASE_URL, {
      method: 'OPTIONS',
    });
    
    // Log the response headers for debugging
    console.log('API Connection Check:');
    console.log('Status:', response.status);
    console.log('CORS Headers:', {
      'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
      'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
      'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers'),
    });
    
    return response.ok;
  } catch (error) {
    console.error('API Connection Error:', error);
    return false;
  }
};

/**
 * Check if a URL is accessible and can be loaded as a video source
 * @param url URL to check
 * @returns Promise resolving to true if URL is accessible, false otherwise
 */
export const checkVideoUrlAccess = async (url: string): Promise<boolean> => {
  try {
    // Make a HEAD request to check if the video is accessible
    const response = await fetch(url, {
      method: 'HEAD',
    });
    
    console.log('Video URL Access Check:');
    console.log('URL:', url);
    console.log('Status:', response.status);
    console.log('Content-Type:', response.headers.get('Content-Type'));
    
    // Fix TypeScript error by using optional chaining with nullish coalescing
    const contentType = response.headers.get('Content-Type') || '';
    return response.ok && contentType.includes('video');
  } catch (error) {
    console.error('Video URL Access Error:', error);
    return false;
  }
};

/**
 * Get a proxy URL for a video if CORS is an issue
 * This is a workaround for local development only
 * @param originalUrl The original video URL
 * @returns A proxied URL that should work with CORS
 */
export const getProxiedVideoUrl = (originalUrl: string): string => {
  // If the URL is already absolute and from the same origin, return it
  if (originalUrl.startsWith(API_BASE_URL)) {
    return originalUrl;
  }
  
  // If it's a relative URL, make it absolute
  if (originalUrl.startsWith('/')) {
    return `${API_BASE_URL}${originalUrl}`;
  }
  
  // If it's already absolute but from a different origin, use a CORS proxy
  // For real production, you'd need a proper CORS setup on your backend
  if (originalUrl.startsWith('http')) {
    // In development, you could use a CORS proxy like cors-anywhere
    // IMPORTANT: Don't use public proxies in production code!
    // For demo/dev purposes only:
    return originalUrl;
  }
  
  // Default case, just return the original
  return originalUrl;
};