#!/bin/bash

set -e

IMAGE_NAME="timemanager-image"
CONTAINER_NAME="timemanager-app"

# Set machine:
if [[ "$2" ]]; then
  MACHINE="$2"
else
  if command -v nvidia-smi &> /dev/null; then
      MACHINE="gpu"
  else
      MACHINE="cpu"
  fi
fi

# Set correct path to Dockerfile:
DOCKERFILE="runtime/docker/$MACHINE/Dockerfile"
if [[ ! -f "$DOCKERFILE" ]]; then
  echo "Dockerfile not found for machine type '$MACHINE' at $DOCKERFILE"
  exit 1
fi
if [[ "$1" == "--build" ]]; then
  docker build -f "$DOCKERFILE" --no-cache -t "$IMAGE_NAME" .
elif [[ "$1" == "--update" ]]; then
  docker build -f "$DOCKERFILE" -t "$IMAGE_NAME" .
fi

# Remove any stale containers:
docker rm -f $CONTAINER_NAME 2>/dev/null || true

# Create and start the new container:
GPU_FLAG=""
if [[ "$MACHINE" == "gpu" ]]; then
  GPU_FLAG="--gpus all"
fi
docker run -d \
  --name "$CONTAINER_NAME" \
  $GPU_FLAG \
  --restart unless-stopped \
  -v /var/log/$CONTAINER_NAME:/app/logs \
  -v timemanager-data:/app/files \
  -v /home/ubuntu/ai-processing/.env:/app/.env \
  -v /home/ubuntu/.deepface:/home/ubuntu/.deepface \
  --log-opt max-size=25m \
  --log-opt max-file=4 \
  "$IMAGE_NAME"

echo -e "$CONTAINER_NAME container is up and running...\n"
echo -e "Logs are located at /var/log/$CONTAINER_NAME/app.log"
