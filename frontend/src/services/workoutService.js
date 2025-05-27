// src/services/workoutService.js

/**
 * Service to handle API calls to the Python backend
 */
const API_URL = process.env.REACT_APP_API_URL || 'https://api.allorigins.win/raw?url=https://34-8-10-114.nip.io';

/**
 * Generate a workout plan by calling the Python backend
 * @param {Object} formData - The user form data
 * @returns {Promise} - Promise with the workout plan data
 */
export const generateWorkoutPlan = async (formData) => {
  try {
    const response = await fetch(`${API_URL}/api/generate-workout`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(formData),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error generating workout plan:', error);
    throw error;
  }
};

/**
 * Save a workout plan to the backend
 * @param {Object} workout - The workout plan data
 * @returns {Promise} - Promise with the save result
 */
export const saveWorkoutPlan = async (workout) => {
  try {
    console.log('Saving workout data:', workout);
    const response = await fetch(`${API_URL}/api/save-workout`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(workout),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const result = await response.json();
    console.log('Save workout response:', result);
    return result;
  } catch (error) {
    console.error('Error saving workout plan:', error);
    throw error;
  }
};

/**
 * Get all saved workout plans
 * @returns {Promise} - Promise with the saved workout plans
 */
export const getSavedWorkouts = async () => {
  try {
    console.log('Fetching saved workouts from API');
    const response = await fetch(`${API_URL}/api/saved-workouts`);

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const result = await response.json();
    console.log('Fetched saved workouts:', result);
    return result;
  } catch (error) {
    console.error('Error fetching saved workouts:', error);
    throw error;
  }
};

/**
 * Get a specific workout plan by ID
 * @param {number} workoutId - The workout ID to fetch
 * @returns {Promise} - Promise with the workout plan data
 */
export const getWorkout = async (workoutId) => {
  try {
    const response = await fetch(`${API_URL}/api/workout/${workoutId}`);

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`Error fetching workout with ID ${workoutId}:`, error);
    throw error;
  }
};

/**
 * Log a completed workout
 * @param {Object} logData - The workout log data
 * @returns {Promise} - Promise with the log result
 */
export const logWorkoutCompletion = async (logData) => {
  try {
    const response = await fetch(`${API_URL}/api/log-workout`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(logData),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error logging workout completion:', error);
    throw error;
  }
};

/**
 * Get all workout logs
 * @returns {Promise} - Promise with all workout logs
 */
export const getWorkoutLogs = async () => {
  try {
    const response = await fetch(`${API_URL}/api/workout-logs`);

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching workout logs:', error);
    throw error;
  }
};

/**
 * Download a workout plan as PDF
 * @param {number} workoutId - The workout ID to download
 * @returns {void} - Triggers a download in the browser
 */
export const downloadWorkoutPlan = (workoutId) => {
  window.location.href = `${API_URL}/api/download-workout/${workoutId}`;
};