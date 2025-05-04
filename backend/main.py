# main.py

import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request, Path
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import inspect

import analyze_module

app = FastAPI(title="Pushup Analysis API")

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

@app.get("/analyze_pushup/video/{video_name}")
def get_video(video_name: str):
    path = os.path.join(VIDEOS_DIR, video_name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path, media_type="video/mp4")

# Debug endpoint to check video files
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

# Simple endpoint for CORS testing
@app.options("/")
@app.get("/")
def root():
    return {"message": "Pushup Analysis API is running"}

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
        }
    }

# Endpoint to list available exercise types
@app.get("/exercises/")
async def list_exercises():
    return {"available_exercises": list(analysis_functions.keys())}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)