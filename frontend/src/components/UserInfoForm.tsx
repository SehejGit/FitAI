// components/UserInfoForm.tsx
import * as React from 'react';
import { useState } from 'react';
import {
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Box,
  Slider,
  Typography,
  Button,
  Paper,
  Chip,
  FormHelperText,
  SelectChangeEvent,
  OutlinedInput
} from '@mui/material';
// Import GridLegacy instead of Grid
import { GridLegacy } from '@mui/material';

interface UserInfoFormProps {
  onGenerateWorkout: (formData: UserInfo) => void;
}

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

const UserInfoForm: React.FC<UserInfoFormProps> = ({ onGenerateWorkout }) => {
  const [formData, setFormData] = useState<UserInfo>({
    name: '',
    age: 30,
    gender: 'Male',
    currentWeight: 150,
    goalWeight: 150,
    height: 68,
    fitnessGoal: 'General Fitness',
    fitnessLevel: 'Beginner',
    daysPerWeek: 3,
    timePerSession: 30,
    injuries: '',
    preferences: '',
    equipment: []
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSelectChange = (e: SelectChangeEvent<string>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value
    }));
  };

  const handleNumberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: parseInt(value, 10)
    }));
  };

  const handleSliderChange = (name: string) => (e: Event, newValue: number | number[]) => {
    setFormData((prev) => ({
      ...prev,
      [name]: newValue
    }));
  };

  const handleEquipmentChange = (event: SelectChangeEvent<string[]>) => {
    const {
      target: { value },
    } = event;
    
    setFormData({
      ...formData,
      equipment: typeof value === 'string' ? value.split(',') : value,
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onGenerateWorkout(formData);
  };

  const equipmentOptions = [
    "None (bodyweight only)",
    "Dumbbells",
    "Resistance bands",
    "Kettlebells",
    "Pull-up bar",
    "Bench",
    "Full gym access"
  ];

  return (
    <Paper elevation={3} sx={{ p: 6, mb: 5 }}>
      <form onSubmit={handleSubmit}>
        <GridLegacy container spacing={6} sx={{ pt: 2 }}>
          {/* Left Column - Personal Information */}
          <GridLegacy xs={12} md={6} sx={{ pl: 3, pr: 2}}>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', justifyContent: 'center' }}>
              Personal Information
            </Typography>
            
            <TextField
              fullWidth
              label="Name"
              name="name"
              value={formData.name}
              onChange={handleChange}
              margin="normal"
              sx={{ mb: 3 }}
            />
            
            <TextField
              fullWidth
              label="Age"
              name="age"
              type="number"
              value={formData.age}
              onChange={handleNumberChange}
              inputProps={{ min: 16, max: 90 }}
              margin="normal"
              sx={{ mb: 3 }}
            />
            
            <FormControl fullWidth margin="normal" sx={{ mb: 3 }}>
              <InputLabel>Gender</InputLabel>
              <Select
                name="gender"
                value={formData.gender}
                onChange={handleSelectChange}
                label="Gender"
              >
                <MenuItem value="Male">Male</MenuItem>
                <MenuItem value="Female">Female</MenuItem>
                <MenuItem value="Other">Other</MenuItem>
              </Select>
            </FormControl>
            
            <TextField
              fullWidth
              label="Current Weight (lbs)"
              name="currentWeight"
              type="number"
              value={formData.currentWeight}
              onChange={handleNumberChange}
              inputProps={{ min: 50, max: 500 }}
              margin="normal"
              sx={{ mb: 3 }}
            />
            
            <TextField
              fullWidth
              label="Goal Weight (lbs)"
              name="goalWeight"
              type="number"
              value={formData.goalWeight}
              onChange={handleNumberChange}
              inputProps={{ min: 50, max: 500 }}
              margin="normal"
              sx={{ mb: 3 }}
            />
            
            <TextField
              fullWidth
              label="Height (inches)"
              name="height"
              type="number"
              value={formData.height}
              onChange={handleNumberChange}
              inputProps={{ min: 36, max: 96 }}
              margin="normal"
              sx={{ mb: 3 }}
            />
          </GridLegacy>
          
          {/* Right Column - Fitness Goals & Preferences */}
          <GridLegacy xs={12} md={6} sx={{ pl: 2}}>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', justifyContent: 'center' }}>
              Fitness Goals & Preferences
            </Typography>
            
            <FormControl fullWidth margin="normal" sx={{ mb: 3 }}>
              <InputLabel>Primary Fitness Goal</InputLabel>
              <Select
                name="fitnessGoal"
                value={formData.fitnessGoal}
                onChange={handleSelectChange}
                label="Primary Fitness Goal"
              >
                <MenuItem value="Weight Loss">Weight Loss</MenuItem>
                <MenuItem value="Muscle Gain">Muscle Gain</MenuItem>
                <MenuItem value="Endurance">Endurance</MenuItem>
                <MenuItem value="General Fitness">General Fitness</MenuItem>
                <MenuItem value="Strength">Strength</MenuItem>
              </Select>
            </FormControl>
            
            <FormControl fullWidth margin="normal" sx={{ mb: 3 }}>
              <InputLabel>Fitness Level</InputLabel>
              <Select
                name="fitnessLevel"
                value={formData.fitnessLevel}
                onChange={handleSelectChange}
                label="Fitness Level"
              >
                <MenuItem value="Beginner">Beginner</MenuItem>
                <MenuItem value="Intermediate">Intermediate</MenuItem>
                <MenuItem value="Advanced">Advanced</MenuItem>
              </Select>
            </FormControl>
            
            <Box sx={{ mt: 4, mb: 2 }}>
              <Typography gutterBottom sx={{ mb: 2 }}>
                Days Available Per Week
              </Typography>
              <Slider
                value={formData.daysPerWeek}
                onChange={handleSliderChange('daysPerWeek')}
                step={1}
                marks={[
                  { value: 2, label: '2' },
                  { value: 3, label: '3' },
                  { value: 4, label: '4' },
                  { value: 5, label: '5' },
                  { value: 6, label: '6' },
                ]}
                min={2}
                max={6}
                valueLabelDisplay="auto"
              />
            </Box>
            
            <Box sx={{ mt: 4, mb: 2 }}>
              <Typography gutterBottom sx={{ mb: 2 }}>
                Time Available Per Session (minutes)
              </Typography>
              <Slider
                value={formData.timePerSession}
                onChange={handleSliderChange('timePerSession')}
                step={null}
                marks={[
                  { value: 30, label: '30' },
                  { value: 45, label: '45' },
                  { value: 60, label: '60' },
                  { value: 90, label: '90' },
                ]}
                min={30}
                max={90}
                valueLabelDisplay="auto"
              />
            </Box>
            
            <TextField
              fullWidth
              label="Any Injuries or Limitations?"
              name="injuries"
              value={formData.injuries}
              onChange={handleChange}
              multiline
              rows={2}
              placeholder="E.g., knee pain, shoulder issues..."
              margin="normal"
              sx={{ mb: 3 }}
            />
            
            <TextField
              fullWidth
              label="Personal Preferences"
              name="preferences"
              value={formData.preferences}
              onChange={handleChange}
              multiline
              rows={2}
              placeholder="E.g., prefer cardio, hate lunges..."
              margin="normal"
              sx={{ mb: 3 }}
            />
          </GridLegacy>
          
          {/* Equipment Selection - Full Width */}
          <GridLegacy xs={12}>
            <Typography variant="h6" gutterBottom sx={{ ml: 5 }}>
              Available Equipment
            </Typography>
            
            <FormControl fullWidth sx={{ ml: 3 }}>
              <InputLabel>Select available equipment</InputLabel>
              <Select
                multiple
                name="equipment"
                value={formData.equipment}
                onChange={handleEquipmentChange}
                input={<OutlinedInput label="Select available equipment" />}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip key={value} label={value} />
                    ))}
                  </Box>
                )}
              >
                {equipmentOptions.map((option) => (
                  <MenuItem key={option} value={option}>
                    {option}
                  </MenuItem>
                ))}
              </Select>
              <FormHelperText>Select all that apply, or none for bodyweight only</FormHelperText>
            </FormControl>
          </GridLegacy>
          
          {/* Submit Button - Full Width */}
          <GridLegacy xs={12} sx={{ mt: 2, textAlign: 'center' }}>
            <Button
              type="submit"
              variant="contained"
              color="primary"
              size="large"
              sx={{ py: 1.5, px: 4 }}
            >
              Generate Workout Plan
            </Button>
          </GridLegacy>
        </GridLegacy>
      </form>
    </Paper>
  );
};

export default UserInfoForm;