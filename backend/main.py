# main.py

import os
import shutil
import io
import json
import datetime
import inspect
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request, Path, Body
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn

# Import workout generation functions from your existing MVP
from mvp import determine_exercises, create_workout_schedule, generate_pdf

# Import exercise analysis modules
import analyze_module

app = FastAPI(title="Fitness Buddy API")

# Get all analysis functions automatically
analysis_functions = {}
for name, func in inspect.getmembers(analyze_module, inspect.isfunction):
    if name.startswith("analyze_"):
        exercise_name = name[8:]
        analysis_functions[exercise_name] = func

# Create a directory for storing videos if it doesn't exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "static", "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)

# Mount static files directory - only need to mount it once
app.mount("/videos", StaticFiles(directory=VIDEOS_DIR), name="videos")

# Configure CORS - CRITICAL for React frontend to work correctly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"Request path: {request.url.path}, method: {request.method}")
    response = await call_next(request)
    print(f"Response status: {response.status_code}")
    return response

# Data models for workout-related endpoints
class UserInfo(BaseModel):
    name: Optional[str] = None
    age: int
    gender: str
    currentWeight: float
    goalWeight: float
    height: float
    fitnessGoal: str
    fitnessLevel: str
    daysPerWeek: int
    timePerSession: int
    injuries: Optional[str] = ""
    preferences: Optional[str] = ""
    equipment: List[str]

class Exercise(BaseModel):
    name: str
    sets: int
    reps: str
    rest: str

class WorkoutDay(BaseModel):
    day: str
    name: str
    exercises: List[Exercise]

class WorkoutPlan(BaseModel):
    id: Optional[int] = None
    name: str
    date_created: str
    workout_plan: List[Dict[str, Any]]
    user_data: Dict[str, Any]
    time_per_session: int

class WorkoutLog(BaseModel):
    workout_name: str
    date: Optional[str] = None
    day: Optional[str] = None
    exercise: Optional[str] = None
    reps: Optional[int] = None
    notes: str = ""

# In-memory storage for demonstration purposes
# In production, use a proper database
SAVED_WORKOUTS = []
WORKOUT_LOGS = {}

# Video analysis endpoints
@app.post("/analyze/{exercise_type}/")
async def analyze_exercise_endpoint(
    exercise_type: str = Path(..., description="Type of exercise to analyze"),
    file: UploadFile = File(..., description="MP4 video of the exercise"),
    return_video: bool = Query(False, description="Also return the annotated video")
):
    # Check if the requested exercise type exists
    if exercise_type not in analysis_functions:
        raise HTTPException(
            status_code=404, 
            detail=f"Exercise type '{exercise_type}' not found. Available types: {list(analysis_functions.keys())}"
        )
    
    # Get the appropriate analysis function
    analysis_function = analysis_functions[exercise_type]
    
    # 1) Save the upload to disk with proper filename handling
    safe_filename = file.filename.replace(" ", "_").lower()
    input_path = os.path.join(VIDEOS_DIR, safe_filename)
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2) Build output path - store directly in VIDEOS_DIR
    if return_video:
        # Ensure output has .mov extension
        base_filename = safe_filename
        if base_filename.lower().endswith('.mp4'):
            base_filename = base_filename[:-4]  # Remove .mp4
        elif base_filename.lower().endswith('.mov'):
            base_filename = base_filename[:-4]  # Remove .mov
        
        # Use .mov extension for output
        output_filename = f"annotated_{exercise_type}_{base_filename}.mov"
        output_path = os.path.join(VIDEOS_DIR, output_filename)
    else:
        output_filename = None
        output_path = None
    
    # 3) Run analysis
    try:
        result = analysis_function(input_path, output_path)
        print(f"Analysis complete for {exercise_type}. Output path: {output_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    # 4) Return JSON + video URL if requested
    payload = {"analysis": result}
    if return_video and output_path and os.path.exists(output_path):
        # Use a relative URL path that will be served by the /videos static route
        video_url = f"/videos/{output_filename}"
        payload["annotated_video_url"] = video_url
        print(f"Video URL provided: {video_url}")
    elif return_video:
        # If video was requested but doesn't exist
        payload["error"] = "Annotated video could not be generated"
        print("Error: Annotated video requested but not generated")
    
    return JSONResponse(content=payload)

@app.get("/api/videos/{filename}")
@app.head("/api/videos/{filename}")  # Add explicit support for HEAD requests
async def get_video_by_filename(filename: str, request: Request):
    """Serve a video by filename - supports both the /api/videos and /videos paths"""
    # Add method to logs
    print(f"Request method: {request.method}")
    
    path = os.path.join(VIDEOS_DIR, filename)
    
    # Debug logging
    print(f"Attempting to serve video: {path}")
    print(f"File exists: {os.path.isfile(path)}")
    
    if not os.path.isfile(path):
        # Try to find the file with different prefixes/patterns
        possible_alternatives = [
            # If filename is ann-example.mp4, try annotated_example.mp4
            os.path.join(VIDEOS_DIR, f"annotated_{filename[4:]}") if filename.startswith("ann-") else None,
            
            # If filename is annotated_example.mp4, try ann-example.mp4
            os.path.join(VIDEOS_DIR, f"ann-{filename[10:]}") if filename.startswith("annotated_") else None,
            
            # Try without any prefix
            os.path.join(VIDEOS_DIR, filename.replace("annotated_", "").replace("ann-", "")),
            
            # Try with annotated_ prefix if it doesn't have one
            os.path.join(VIDEOS_DIR, f"annotated_{filename}") if not filename.startswith("annotated_") else None,
        ]
        
        # Filter out None values and check each alternative
        for alt_path in filter(None, possible_alternatives):
            print(f"Checking alternative path: {alt_path}")
            if os.path.isfile(alt_path):
                path = alt_path
                print(f"Found alternative path: {path}")
                break
        else:  # This else belongs to the for loop (runs if no break occurs)
            print(f"No alternative paths found for: {filename}")
            raise HTTPException(status_code=404, detail="Video not found")
    
    # Set the appropriate content type based on the file extension
    if path.endswith(".mp4"):
        media_type = "video/mp4"
    elif path.endswith(".mov"):
        media_type = "video/quicktime"
    else:
        media_type = "application/octet-stream"
    
    print(f"Serving file: {path} with media type: {media_type}")
    
    return FileResponse(path, media_type=media_type)

@app.options("/api/videos/{filename}")
async def options_video(filename: str):
    """Handle preflight requests for video endpoints"""
    response = Response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    print(f"Handled OPTIONS request for: {filename}")
    return response

# Debug endpoints for videos
@app.get("/debug/videos")
def list_videos():
    """List all videos in the videos directory"""
    if not os.path.exists(VIDEOS_DIR):
        return {"error": "Videos directory does not exist"}
    
    videos = [f for f in os.listdir(VIDEOS_DIR) if f.endswith(('.mp4', '.mov'))]
    return {
        "videos_dir": VIDEOS_DIR,
        "videos": videos,
        "count": len(videos),
        "directory_exists": os.path.exists(VIDEOS_DIR),
        "directory_writable": os.access(VIDEOS_DIR, os.W_OK)
    }

@app.get("/debug/video/{video_name}")
def debug_video(video_name: str):
    """Debug endpoint to check if a video exists"""
    path = os.path.join(VIDEOS_DIR, video_name)
    exists = os.path.isfile(path)
    
    response = {
        "video_name": video_name,
        "full_path": path,
        "exists": exists,
    }
    
    if exists:
        response["size"] = os.path.getsize(path)
        response["access_url"] = f"/videos/{video_name}"
    
    return response

# Endpoint to list available exercise types
@app.get("/exercises/")
async def list_exercises():
    return {"available_exercises": list(analysis_functions.keys())}

# Workout generation endpoints
@app.post("/api/generate-workout")
async def generate_workout(user_info: UserInfo):
    """Generate a workout plan based on user data"""
    try:
        # Ensure equipment is a list
        if not user_info.equipment or len(user_info.equipment) == 0:
            user_info.equipment = ["None (bodyweight only)"]
        
        # Generate the workout plan using functions from mvp.py
        selected_exercises = determine_exercises(
            user_info.equipment,
            user_info.fitnessGoal,
            user_info.fitnessLevel,
            user_info.injuries or ""
        )
        
        workout_plan = create_workout_schedule(
            selected_exercises,
            user_info.daysPerWeek,
            user_info.timePerSession,
            user_info.fitnessLevel
        )
        
        # Return the generated workout plan
        print(f"Workout plan generated successfully for {user_info.name or 'user'}")
        return {
            "success": True,
            "workout_plan": workout_plan
        }
    
    except Exception as e:
        print(f"Error generating workout plan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Modify the save_workout function in main.py to include better logging
@app.post("/api/save-workout")
async def save_workout(workout: WorkoutPlan):
    """Save a workout plan"""
    try:
        # Add a unique ID for the workout
        workout_id = len(SAVED_WORKOUTS) + 1
        workout_data = workout.dict()
        workout_data["id"] = workout_id
        
        # Add detailed logging
        print(f"Saving workout plan '{workout.name}' with ID {workout_id}")
        print(f"Workout data: {workout_data}")
        
        # Save the workout plan (in memory for this example)
        SAVED_WORKOUTS.append(workout_data)
        
        # Initialize workout logs for this plan
        WORKOUT_LOGS[workout.name] = []
        
        # Print the current state of SAVED_WORKOUTS for debugging
        print(f"Current SAVED_WORKOUTS: {SAVED_WORKOUTS}")
        
        return {
            "success": True,
            "message": f"Workout plan '{workout.name}' saved successfully!",
            "workout_id": workout_id
        }
    
    except Exception as e:
        print(f"Error saving workout plan: {str(e)}")
        # Print the stack trace for better debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Modify the get_saved_workouts function to return just the array
@app.get("/api/saved-workouts")
async def get_saved_workouts():
    """Get all saved workout plans"""
    try:
        print(f"Returning {len(SAVED_WORKOUTS)} saved workout plans")
        # For debugging
        print(f"SAVED_WORKOUTS contents: {SAVED_WORKOUTS}")
        
        # Return SAVED_WORKOUTS directly without wrapping it in an object
        # This matches what the frontend expects
        return SAVED_WORKOUTS
    
    except Exception as e:
        print(f"Error fetching saved workouts: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/workout/{workout_id}")
async def get_workout(workout_id: int):
    """Get a specific saved workout plan"""
    try:
        # Find the workout with the given ID
        workout = next((w for w in SAVED_WORKOUTS if w["id"] == workout_id), None)
        
        if not workout:
            print(f"Workout with ID {workout_id} not found")
            raise HTTPException(status_code=404, detail=f"Workout with ID {workout_id} not found")
        
        print(f"Retrieved workout plan with ID {workout_id}")
        return {
            "success": True,
            "workout": workout
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching workout: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/log-workout")
async def log_workout(log_data: WorkoutLog):
    """Log a completed workout"""
    try:
        workout_name = log_data.workout_name
        if not workout_name:
            raise HTTPException(status_code=400, detail="Workout name is required")
        
        # Ensure we have a log entry for this workout
        if workout_name not in WORKOUT_LOGS:
            WORKOUT_LOGS[workout_name] = []
        
        # Use current date if not provided
        if not log_data.date:
            log_data.date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Add log entry
        WORKOUT_LOGS[workout_name].append(log_data.dict())
        
        print(f"Workout logged for '{workout_name}' on {log_data.date}")
        return {
            "success": True,
            "message": "Workout logged successfully!"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error logging workout: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/workout-logs")
async def get_workout_logs():
    """Get all workout logs"""
    try:
        total_logs = sum(len(logs) for logs in WORKOUT_LOGS.values())
        print(f"Returning {total_logs} workout logs across {len(WORKOUT_LOGS)} workout plans")
        return {
            "success": True,
            "logs": WORKOUT_LOGS
        }
    
    except Exception as e:
        print(f"Error fetching workout logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download-workout/{workout_id}")
async def download_workout(workout_id: int):
    """Generate and return a PDF of the workout plan"""
    try:
        # Find the workout with the given ID
        workout = next((w for w in SAVED_WORKOUTS if w["id"] == workout_id), None)
        
        if not workout:
            print(f"Workout with ID {workout_id} not found for PDF generation")
            raise HTTPException(status_code=404, detail=f"Workout with ID {workout_id} not found")
        
        # Generate PDF using function from mvp.py
        pdf_bytes = generate_pdf(
            workout_plan=workout["workout_plan"],
            user_data=workout["user_data"],
            time_per_session=workout["time_per_session"]
        )
        
        print(f"PDF generated for workout plan with ID {workout_id}")
        # Return the PDF
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="workout_plan_{workout_id}.pdf"'
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# API status endpoint
@app.get("/api_status")
def api_status():
    """
    Check if API is running and configured correctly
    """
    # Check if videos directory exists and is writable
    videos_dir_exists = os.path.isdir(VIDEOS_DIR)
    videos_dir_writable = os.access(VIDEOS_DIR, os.W_OK)
    
    # List all files in the videos directory
    video_files = []
    if videos_dir_exists:
        video_files = [f for f in os.listdir(VIDEOS_DIR) if f.endswith(('.mp4', '.mov'))]
    
    return {
        "status": "running",
        "videos_directory": {
            "path": VIDEOS_DIR,
            "exists": videos_dir_exists,
            "writable": videos_dir_writable,
            "file_count": len(video_files),
            "example_files": video_files[:5] if video_files else []
        },
        "exercise_analysis": {
            "available_exercise_types": list(analysis_functions.keys()),
            "count": len(analysis_functions)
        },
        "workout_generator": {
            "saved_workouts_count": len(SAVED_WORKOUTS),
            "workout_logs_count": len(WORKOUT_LOGS)
        }
    }

# Simple endpoint for CORS testing
@app.options("/")
@app.get("/")
def root():
    return {"message": "Fitness Buddy API is running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)