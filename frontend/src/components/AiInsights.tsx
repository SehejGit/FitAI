// components/AiInsights.tsx
import React from 'react';
import { 
  Paper, 
  Typography, 
  Box, 
  List, 
  ListItem, 
  ListItemIcon, 
  ListItemText,
  Divider 
} from '@mui/material';
import PsychologyIcon from '@mui/icons-material/Psychology';
import FitnessCenterIcon from '@mui/icons-material/FitnessCenter';
import RecoveryIcon from '@mui/icons-material/SelfImprovement';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';

interface AiInsightsProps {
  insights: {
    exercise_recommendations?: string[];
    rep_schemes?: Record<string, any>;
    modifications?: string[];
    recovery_tips?: string[];
    progression_tips?: string[];
  };
}

const AiInsights: React.FC<AiInsightsProps> = ({ insights }) => {
  if (!insights || Object.keys(insights).length === 0) {
    return null;
  }

  return (
    <Paper sx={{ p: 3, mt: 3, bgcolor: 'primary.light', color: 'primary.contrastText' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <PsychologyIcon sx={{ mr: 1 }} />
        <Typography variant="h6">AI Fitness Insights</Typography>
      </Box>
      
      {insights.modifications && insights.modifications.length > 0 && (
        <>
          <Typography variant="subtitle1" sx={{ mt: 2, mb: 1, fontWeight: 'bold' }}>
            <FitnessCenterIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
            Exercise Modifications
          </Typography>
          <List dense>
            {insights.modifications.map((mod, index) => (
              <ListItem key={index}>
                <ListItemText primary={mod} />
              </ListItem>
            ))}
          </List>
        </>
      )}
      
      {insights.recovery_tips && insights.recovery_tips.length > 0 && (
        <>
          <Divider sx={{ my: 2 }} />
          <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 'bold' }}>
            <RecoveryIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
            Recovery Recommendations
          </Typography>
          <List dense>
            {insights.recovery_tips.map((tip, index) => (
              <ListItem key={index}>
                <ListItemText primary={tip} />
              </ListItem>
            ))}
          </List>
        </>
      )}
      
      {insights.progression_tips && insights.progression_tips.length > 0 && (
        <>
          <Divider sx={{ my: 2 }} />
          <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 'bold' }}>
            <TrendingUpIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
            Progression Strategy
          </Typography>
          <List dense>
            {insights.progression_tips.map((tip, index) => (
              <ListItem key={index}>
                <ListItemText primary={tip} />
              </ListItem>
            ))}
          </List>
        </>
      )}
    </Paper>
  );
};

export default AiInsights;