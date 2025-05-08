# backend/main.py

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
        exercise_name = name[8:].lower()
        exercise_name = exercise_name.replace(" ", "_").replace("-", "_")  # Fixed: assign result back
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
    allow_origins=["*", "https://fitai-459007.web.app"],  # Add your frontend URL explicitly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # Important for preflight requests
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

    return_video = False
    try:
        # Log the incoming request
        print(f"=== ANALYZE REQUEST START: {exercise_type} ===")
        print(f"File name: {file.filename}, Content-Type: {file.content_type}")
        print(f"Return video: {return_video}")
        
        # Step 1: Check if the requested exercise type exists
        exercise_normalized = exercise_type.replace(" ", "_").lower()
        exercise_normalized = exercise_normalized.replace("-", "_")
        print(f"Normalized exercise name: {exercise_normalized}")
        
        if exercise_normalized not in analysis_functions:
            print(f"ERROR: Exercise type not found: {exercise_normalized}")
            print(f"Available types: {list(analysis_functions.keys())}")
            raise HTTPException(
                status_code=404,
                detail=f"Exercise type '{exercise_normalized}' not found. Available types: {list(analysis_functions.keys())}"
            )
        
        # Step 2: Get the appropriate analysis function
        analysis_function = analysis_functions[exercise_normalized]
        print(f"Using analysis function: {analysis_function.__name__}")
        
        # Step 3: Save the upload to disk
        safe_filename = file.filename.replace(" ", "_").lower()
        safe_filename = safe_filename.replace("-", "_")
        input_path = os.path.join(VIDEOS_DIR, safe_filename)
        
        print(f"Saving uploaded file to: {input_path}")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"File saved successfully. Size: {os.path.getsize(input_path)} bytes")

        # Step 4: Build output path
        if return_video:
            base_filename = safe_filename
            if base_filename.lower().endswith('.mp4'):
                base_filename = base_filename[:-4]
            elif base_filename.lower().endswith('.mov'):
                base_filename = base_filename[:-4]
            
            output_filename = f"annotated_{exercise_type}_{base_filename}.mov"
            output_path = os.path.join(VIDEOS_DIR, output_filename)
            print(f"Output video will be saved to: {output_path}")
        else:
            output_filename = None
            output_path = None
            print("No output video requested")
        
        # Step 5: Run analysis - THIS IS WHERE THE CRASH LIKELY HAPPENS
        try:
            # FOR DEBUGGING: catch any errors during analysis
            print(f"Starting analysis with {analysis_function.__name__}...")
            
            # Use a try-except block to catch any errors in the analysis function
            try:
                result = analysis_function(input_path, output_path)
                print(f"Analysis completed successfully")
            except Exception as analysis_error:
                print(f"ANALYSIS FUNCTION ERROR: {str(analysis_error)}")
                import traceback
                traceback.print_exc()
                
                # Fall back to simple analysis for debugging
                print("Falling back to simplified analysis...")
                result = {
                    "error": f"Original analysis failed: {str(analysis_error)}",
                    "fallback_result": {
                        "pushup_count": 0,
                        "form_analysis": {
                            "elbow_angle_at_bottom": 0,
                            "elbow_angle_at_top": 0,
                            "body_alignment_score": 0
                        },
                        "feedback": [
                            f"Error analyzing video: {str(analysis_error)}"
                        ]
                    }
                }
        except Exception as e:
            print(f"ERROR DURING ANALYSIS SETUP: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

        # Step 6: Return JSON + video URL
        payload = {"analysis": result}
        
        if return_video and output_path and os.path.exists(output_path):
            video_url = f"/videos/{output_filename}"
            payload["annotated_video_url"] = video_url
            print(f"Video URL provided: {video_url}")
        elif return_video:
            # If video was requested but doesn't exist
            payload["warning"] = "Annotated video could not be generated"
            print("Warning: Annotated video requested but not generated")
        
        print("=== ANALYZE REQUEST COMPLETE ===")
        return JSONResponse(content=payload)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"UNHANDLED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
    

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

@app.options("/{full_path:path}")
async def options_route(full_path: str):
    """Handle OPTIONS requests for all endpoints"""
    response = Response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.get("/debug/video_test/")
async def debug_video_test():
    """Test video processing capabilities"""
    try:
        # Create a simple test video
        import cv2
        import numpy as np
        import os
        from datetime import datetime
        
        # Create a test video file
        test_video_path = os.path.join(VIDEOS_DIR, "test_video.mp4")
        
        # Create a small video with a few frames
        width, height = 640, 480
        fps = 30
        seconds = 2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(test_video_path, fourcc, fps, (width, height))
        
        # Create some frames
        for i in range(fps * seconds):
            # Create a black frame
            img = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add some text
            cv2.putText(img, f"Test Frame {i}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            cv2.putText(img, timestamp, (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add a moving rectangle
            x = int((i / (fps * seconds)) * width)
            cv2.rectangle(img, (x, 200), (x + 50, 250), (0, 255, 0), -1)
            
            # Write the frame
            out.write(img)
        
        # Release the video writer
        out.release()
        
        # Now try to analyze this test video
        result = {
            "video_creation": "success",
            "video_path": test_video_path,
            "video_exists": os.path.exists(test_video_path),
            "video_size": os.path.getsize(test_video_path) if os.path.exists(test_video_path) else 0
        }
        
        # Try to open and read the video
        try:
            cap = cv2.VideoCapture(test_video_path)
            if not cap.isOpened():
                result["video_reading"] = "failed"
            else:
                result["video_reading"] = "success"
                result["video_properties"] = {
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": int(cap.get(cv2.CAP_PROP_FPS)),
                    "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                }
                
                # Read first frame
                ret, frame = cap.read()
                if ret:
                    result["first_frame_read"] = "success"
                    result["first_frame_shape"] = frame.shape
                else:
                    result["first_frame_read"] = "failed"
                
                cap.release()
        except Exception as e:
            result["video_reading_error"] = str(e)
        
        # Try initializing MediaPipe
        try:
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
            result["mediapipe_init"] = "success"
            
            # Try processing the first frame with MediaPipe
            if "first_frame_read" in result and result["first_frame_read"] == "success":
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(frame_rgb)
                result["mediapipe_processing"] = "success" if pose_results else "no results"
        except Exception as e:
            result["mediapipe_error"] = str(e)
        
        return result
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

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

# Add this new debug endpoint here
@app.get("/debug/analyze_functions")
def debug_analyze_functions():
    """Debug endpoint to check exercise function mappings"""
    return {
        "available_functions": [name for name in analysis_functions.keys()],
        "analyze_module_functions": [name for name, _ in inspect.getmembers(analyze_module, inspect.isfunction) 
                                   if name.startswith("analyze_")],
        "mapping_example": {
            "push_ups": "push_ups" in analysis_functions,
            "pushups": "pushups" in analysis_functions,
            "push-ups": "push-ups" in analysis_functions
        }
    }

# Endpoint to list available exercise types
@app.get("/exercises/")
async def list_exercises():
    return {"available_exercises": list(analysis_functions.keys())}

@app.post("/test_upload/")
async def test_upload(file: UploadFile = File(...)):
    """Test endpoint for file uploads without analysis"""
    try:
        print(f"Test upload received. Filename: {file.filename}, Content-Type: {file.content_type}")
        
        # Read a small part of the file to verify it's accessible
        content = await file.read(1024)  # Read first 1KB
        file_size = len(content)
        
        # Save the file to disk
        safe_filename = f"test_{file.filename.replace(' ', '_')}"
        save_path = os.path.join(VIDEOS_DIR, safe_filename)
        
        with open(save_path, "wb") as f:
            f.write(content)
            # Read and write the rest of the file
            while content := await file.read(1024):
                f.write(content)
                file_size += len(content)
        
        # Check if file was saved
        saved_size = os.path.getsize(save_path) if os.path.exists(save_path) else 0
        
        return {
            "success": True,
            "filename": file.filename,
            "content_type": file.content_type,
            "size_read": file_size,
            "saved_path": save_path,
            "saved_size": saved_size,
            "file_exists": os.path.exists(save_path),
            "videos_dir": VIDEOS_DIR
        }
    except Exception as e:
        print(f"Error in test upload: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
    
@app.post("/test_post/")
async def test_post(request: Request):
    """Test endpoint for simple POST requests"""
    try:
        # Try to read the request body
        body = await request.body()
        
        # Get headers
        headers = dict(request.headers.items())
        
        return {
            "success": True,
            "method": request.method,
            "body_length": len(body),
            "content_type": headers.get("content-type", "Not specified"),
            "headers": headers
        }
    except Exception as e:
        print(f"Error in test POST: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
    
@app.get("/debug/mediapipe_test/")
async def test_mediapipe():
    """Test if MediaPipe and OpenCV are working"""
    try:
        # Try importing libraries
        import mediapipe as mp
        import cv2
        import numpy as np
        
        # Create a simple test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Try initializing MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        
        # Try processing the test image
        results = pose.process(test_img)
        
        return {
            "success": True,
            "mediapipe_version": mp.__version__,
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
            "pose_results": "Empty (expected for blank image)",
            "test_image_shape": test_img.shape
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }

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

@app.post("/api/save-workout")
async def save_workout(workout: WorkoutPlan):
    """Save a workout plan"""
    try:
        # Add a unique ID for the workout
        workout_id = len(SAVED_WORKOUTS) + 1
        workout_data = workout.dict()
        workout_data["id"] = workout_id
        
        # Save the workout plan (in memory for this example)
        SAVED_WORKOUTS.append(workout_data)
        
        # Initialize workout logs for this plan
        WORKOUT_LOGS[workout.name] = []
        
        print(f"Workout plan '{workout.name}' saved with ID {workout_id}")
        return {
            "success": True,
            "message": f"Workout plan '{workout.name}' saved successfully!",
            "workout_id": workout_id
        }
    
    except Exception as e:
        print(f"Error saving workout plan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/saved-workouts")
async def get_saved_workouts():
    """Get all saved workout plans"""
    try:
        print(f"Returning {len(SAVED_WORKOUTS)} saved workout plans")
        return {
            "success": True,
            "workouts": SAVED_WORKOUTS
        }
    
    except Exception as e:
        print(f"Error fetching saved workouts: {str(e)}")
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

@app.post("/debug/analyze_push_ups/")
async def debug_analyze_push_ups(file: UploadFile = File(...), return_video: bool = Query(False)):
    """Debugging endpoint for push-ups analysis that uses simplified function"""
    try:
        print(f"Debug push-ups analysis requested. Filename: {file.filename}")
        
        # Save the file
        safe_filename = file.filename.replace(" ", "_").lower()
        safe_filename = safe_filename.replace("-", "_")
        input_path = os.path.join(VIDEOS_DIR, safe_filename)
        
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Call the simplified function
        result = analyze_module.analyze_push_ups_simple(input_path, None)
        
        # Return the result
        return {"analysis": result, "debug": True}
    except Exception as e:
        print(f"Error in debug push-ups analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@app.post("/simple_push_ups/")
async def simple_push_ups_analysis(file: UploadFile = File(...)):
    """Simplified version of push-ups analysis for debugging"""
    try:
        print(f"Simple push-ups analysis request received. Filename: {file.filename}")
        
        # Save the file
        safe_filename = file.filename.replace(" ", "_").lower()
        input_path = os.path.join(VIDEOS_DIR, safe_filename)
        
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Return a mock analysis without actually running the analysis code
        mock_result = {
            "pushup_count": 5,
            "form_analysis": {
                "elbow_angle_at_bottom": 90.5,
                "elbow_angle_at_top": 165.3,
                "body_alignment_score": 85.2,
                "frames_analyzed": 120
            },
            "feedback": [
                "Great form! Your pushups have good depth and body alignment."
            ]
        }
        
        return {"analysis": mock_result, "test_mode": True}
    except Exception as e:
        print(f"Error in simple push-ups analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

# Simple endpoint for CORS testing
@app.options("/")
@app.get("/")
def root():
    return {"message": "Fitness Buddy API is running"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)