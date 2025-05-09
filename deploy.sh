#!/bin/bash
set -e  # Exit immediately if a command fails

echo "Starting deployment process..."

# 1. Deploy Frontend
echo "Building and deploying frontend..."
cd frontend
npm run build --production
firebase deploy

# 2. Deploy Backend
# echo "Building and deploying backend..."
# cd ../backend

# Authenticate to Google Cloud
# echo "Authenticating to Google Cloud..."
# gcloud auth configure-docker

# Build and push directly to GCR
# echo "Building and pushing Docker image..."
# docker buildx build --platform linux/amd64 --push -t gcr.io/fitai-459007/fitai-backend:latest .

# Update the Kubernetes deployment with the new image
# echo "Updating Kubernetes deployment..."
# kubectl rollout restart deployment fitai-backend

# Wait for rollout to complete
echo "Waiting for deployment to complete..."
# kubectl rollout status deployment fitai-backend --timeout=5m

# if [ $? -ne 0 ]; then
#   echo "Deployment failed! Rolling back..."
#   kubectl rollout undo deployment fitai-backend
#   echo "Rolled back to previous version."
#   exit 1
# fi

echo "Deployment successful!"