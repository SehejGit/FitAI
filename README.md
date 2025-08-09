# FitAI - Exercise Form Analysis Platform

> A web application that leverages computer vision and machine learning to provide real-time exercise form analysis, rep counting, and personalized feedback from uploaded videos.

## 🎯 Overview

FitAI is an end-to-end fitness analysis platform that combines computer vision, pose estimation, and machine learning to deliver automated exercise form assessment. The system processes uploaded videos through a sophisticated analysis pipeline, providing users with detailed metrics, rep counting, and actionable feedback to improve their workout technique.

**Key Differentiators:**
- Real-time pose estimation and biomechanical analysis
- Multi-exercise support across different muscle groups and equipment
- Production-grade deployment with Kubernetes and containerization
- Comprehensive quality assessment framework with quantitative metrics

## ✨ Core Features

### 🔍 **Intelligent Exercise Analysis**
- **Automated Rep Counting**: Precise repetition detection using motion analysis
- **Form Assessment**: Biomechanical evaluation with scoring metrics
- **Multi-Exercise Support**: Upper body, lower body, full body, dumbbell, core, and gym equipment exercises
- **Real-time Feedback**: Personalized technique recommendations based on analysis

### 📊 **Advanced Metrics & Visualization**
- **Pose Estimation Overlay**: Annotated output videos with skeletal tracking
- **Performance Metrics**: Quantitative assessment of form quality, range of motion, and tempo
- **Quality Scoring**: Comprehensive evaluation framework for exercise execution

### 🚀 **Production-Ready Infrastructure**
- **Scalable Architecture**: Containerized deployment with Docker and Kubernetes
- **Cloud-Native**: HTTPS-enabled deployment with nip.io domain configuration
- **Full-Stack Implementation**: React frontend with Flask backend API
- **Performance Optimized**: Efficient video processing and real-time analysis

## 🛠️ Technical Architecture

### **Backend Stack**
- **Flask**: RESTful API development and video processing endpoints
- **OpenCV**: Video analysis and computer vision operations
- **MediaPipe**: Advanced pose estimation and landmark detection
- **NumPy**: Numerical computations for biomechanical analysis

### **Frontend Stack**
- **React**: Interactive user interface and video upload functionality
- **Modern UI/UX**: Responsive design with real-time progress indicators

### **Infrastructure**
- **Docker**: Containerized application deployment
- **Kubernetes**: Orchestrated scaling and management
- **HTTPS Configuration**: Secure deployment with SSL/TLS encryption

## 📁 Project Structure
   ```bash
   fitai/
   ├── app.py                     # Main Flask application and API endpoints
   ├── analyzer/                  # Core analysis modules
   │   ├── pushup_analyzer.py     # Pushup form analysis logic
   │   ├── curl_analyzer.py       # Bicep curl analysis logic
   │   └── base_analyzer.py       # Shared analysis framework
   ├── frontend/                  # React application
   │   ├── src/                   # React source code
   │   └── public/                # Static assets
   ├── k8s/                       # Kubernetes deployment configurations
   ├── static/                    # Backend static files
   ├── templates/                 # HTML templates
   ├── uploads/                   # Video upload and processing directory
   ├── Dockerfile                 # Container configuration
   ├── requirements.txt           # Python dependencies
   └── deploy.sh                  # Deployment automation script
```
## 🚀 Quick Start

### **Option 1: Local Development**

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd fitai
   pip install -r requirements.txt

2. Start Backend
```bash
python app.py
# Backend runs on http://localhost:5000
```

3. Start Frontend
```bash
cd frontend
npm install
npm start
# Frontend runs on http://localhost:3000
```

Option 2: Docker Deployment
```bash
# Build and run container
docker build -t fitai .
docker run -p 5000:5000 fitai
# Access application at http://localhost:5000
```
Option 3: Kubernetes Production
```bash
# Deploy to Kubernetes cluster
kubectl apply -f k8s/
# Configured for HTTPS with nip.io domain
```
## Acknowledgments

- MediaPipe for pose estimation
- OpenCV for video processing
