# FitAI
MLOps Final Project

A web application that analyzes exercise form from uploaded videos using computer vision.

## Features

- Upload videos of pushups or bicep curls
- Automatic exercise rep counting
- Form analysis with metrics
- Personalized feedback on technique
- Annotated output video with pose estimation

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenCV
- MediaPipe
- Flask

### Setup

1. Clone this repository:
   ```
   git clone <repository-url>
   cd exercise-analyzer
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Navigate to frontend

    ```
    cd FitAI/frontend
    ```

4. Install frontend dependencies:
   ```
   npm install
   ```

5. Start the frontend server:
    ```
    npm start
    ```

6. Open a web browser and navigate to:
   ```
   http://localhost:5000
   ```

### Docker Installation (Alternative)

1. Build the Docker image:
   ```
   docker build -t exercise-analyzer .
   ```

2. Run the container:
   ```
   docker run -p 5000:5000 exercise-analyzer
   ```

3. Access the application at `http://localhost:5000`

## Usage

1. Select the exercise type (pushup or bicep curl)
2. Upload a video file of your exercise
3. Click "Analyze" and wait for processing
4. View results including:
   - Rep count
   - Form metrics
   - Personalized feedback
   - Annotated video playback

## Tips for Best Results

- Record in a well-lit area with a plain background
- Position the camera to see your full body from the side
- Wear fitted clothing for better pose detection
- Ensure your entire body is visible throughout the exercise

## Project Structure

- `app.py`: Main Flask application
- `analyzer/`: Analysis code modules
  - `pushup_analyzer.py`: Pushup analysis logic
  - `curl_analyzer.py`: Bicep curl analysis logic
- `templates/`: HTML templates
- `static/`: CSS, JavaScript, and other static files
- `uploads/`: Directory for uploaded and processed videos

## License

[Include license information here]

## Acknowledgments

- MediaPipe for pose estimation
- OpenCV for video processing
