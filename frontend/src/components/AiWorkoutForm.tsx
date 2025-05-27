import React, { useState } from 'react';
import { Button, CircularProgress, Alert, Box } from '@mui/material';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import { API_BASE_URL } from '../config';

interface UserInfo {
  name: string;
  age: number;
  gender: string;
  currentWeight: number;
  goalWeight: number;
  height: number;
  fitnessGoal: string;
  fitnessLevel: string;
  daysPerWeek: number;
  timePerSession: number;
  injuries: string;
  preferences: string;
  equipment: string[];
}
interface AiWorkoutFormProps {
  userInfo: UserInfo;
  onGenerateWorkout: (workoutPlan: any, aiInsights: any) => void;
}

const AiWorkoutForm: React.FC<AiWorkoutFormProps> = ({ userInfo, onGenerateWorkout }) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAiGenerate = async () => {
    setIsGenerating(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/generate-ai-workout`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userInfo),
      });

      if (!response.ok) {
        throw new Error('Failed to generate AI workout plan');
      }

      const data = await response.json();
      
      if (data.success) {
        onGenerateWorkout(data.workout_plan, data.ai_insights);
      } else {
        throw new Error('AI generation failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate AI workout');
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <Box sx={{ mt: 3 }}>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      
      <Button
        variant="contained"
        color="secondary"
        startIcon={isGenerating ? <CircularProgress size={20} color="inherit" /> : <AutoFixHighIcon />}
        onClick={handleAiGenerate}
        disabled={isGenerating}
        fullWidth
        sx={{ py: 1.5 }}
      >
        {isGenerating ? 'Generating AI Workout...' : 'Generate AI-Enhanced Workout'}
      </Button>
    </Box>
  );
};

export default AiWorkoutForm;

