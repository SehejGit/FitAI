# main.py

import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from analyze_module import analyze_pushup

app = FastAPI(title="Pushup Analysis API")

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

@app.post("/analyze_pushup/")
async def analyze_pushup_endpoint(
    file: UploadFile = File(..., description="MP4 video of pushups"),
    return_video: bool = Query(False, description="Also return the annotated video")
):
    # 1) Save the upload to disk with proper filename handling
    safe_filename = file.filename.replace(" ", "_").lower()
    input_path = os.path.join(VIDEOS_DIR, safe_filename)
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2) Build output path - store directly in VIDEOS_DIR
    output_filename = f"annotated_{safe_filename}" if return_video else None
    output_path = os.path.join(VIDEOS_DIR, output_filename) if output_filename else None
    
    # 3) Run analysis
    try:
        result = analyze_pushup(input_path, output_path)
        print(f"Analysis complete. Output path: {output_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    # 4) Return JSON + video URL if requested
    payload = {"analysis": result}
    if return_video and output_path and os.path.exists(output_path):
        # Use a relative URL path that will be served by the /videos static route
        video_url = f"/videos/{output_filename}"
        payload["annotated_video_url"] = video_url
        payload["annotated_video_path"] = f"../../../backend/static/videos/{output_filename}"
        print(f"Video URL provided: {video_url}")
    elif return_video:
        # If video was requested but doesn't exist
        payload["error"] = "Annotated video could not be generated"
        print("Error: Annotated video requested but not generated")
    
    return JSONResponse(content=payload)

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)