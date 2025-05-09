#!/bin/bash
set -e  # Exit immediately if a command fails

echo "Starting deployment process..."

# 1. Deploy Frontend
echo "Building and deploying frontend..."
cd frontend
npm run build --production
firebase deploy

# 2. Deploy Backend
echo "Building and deploying backend..."
cd ../backend

# Authenticate to Google Cloud
echo "Authenticating to Google Cloud..."
gcloud auth configure-docker

# Build and push directly to GCR
echo "Building and pushing Docker image..."
docker build --no-cache --platform linux/amd64 -t gcr.io/fitai-459007/fitai-backend:forceupdate .

docker push gcr.io/fitai-459007/fitai-backend:forceupdate

# deployment
gcloud run deploy fitai-backend \
  --image gcr.io/fitai-459007/fitai-backend:forceupdate \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 4 \
  --timeout 1800 \
  --concurrency 80 \
  --set-secrets="OPENAI_API_KEY=openai-api-key-tyler:latest" \
  --set-env-vars="CLOUD_RUN=true" \
  --allow-unauthenticated

#   --set-env-vars="CLOUD_RUN=true" \

echo "Deployment successful!"