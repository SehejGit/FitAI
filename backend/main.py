# main.py

import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from analyze_module import analyze_pushup

app = FastAPI(title="Pushup Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze_pushup/")
async def analyze_pushup_endpoint(
    file: UploadFile = File(..., description="MP4 video of pushups"),
    return_video: bool = Query(False, description="Also return the annotated video")
):
    # 1) Save the upload to disk
    input_path = f"/tmp/{file.filename}"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2) Build optional output path
    output_path = f"/tmp/annotated_{file.filename}" if return_video else None

    # 3) Run analysis
    try:
        result = analyze_pushup(input_path, output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")

    # 4) Return JSON + video URL if requested
    payload = {"analysis": result}
    if return_video:
        payload["annotated_video_url"] = f"/analyze_pushup/video/{os.path.basename(output_path)}"
    return JSONResponse(content=payload)

@app.get("/analyze_pushup/video/{video_name}")
def get_video(video_name: str):
    path = f"/tmp/{video_name}"
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path, media_type="video/mp4")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)