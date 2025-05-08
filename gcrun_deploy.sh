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
docker build --platform linux/amd64 -t gcr.io/fitai-459007/fitai-backend:cloudrun .

docker push gcr.io/fitai-459007/fitai-backend:cloudrun

# deployment
gcloud run deploy fitai-backend \
  --image gcr.io/fitai-459007/fitai-backend:cloudrun \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 4 \
  --timeout 1800 \
  --concurrency 80 \
  --set-env-vars="OPENAI_API_KEY=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" \
  --allow-unauthenticated

echo "Deployment successful!"