import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';

const NavigationBar = () => {
  return (
    <AppBar position="static" color="primary" sx={{ marginBottom: 2 }}>
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          💪 Fitness Buddy
        </Typography>
        <Box>
          <Button 
            color="inherit" 
            component={RouterLink} 
            to="/"
          >
            Home
          </Button>
          <Button 
            color="inherit" 
            component={RouterLink} 
            to="/saved-workouts"
          >
            My Workouts
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default NavigationBar;