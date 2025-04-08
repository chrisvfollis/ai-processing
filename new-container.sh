#!/bin/bash

set -e

IMAGE_NAME="timemanager-image"
CONTAINER_NAME="timemanager-app"

# [Optional] pass --build to force rebuild:
if [[ "$1" == "--build" ]]; then
  docker build --no-cache -t $IMAGE_NAME .
elif [[ "$1" == "--update" ]]; then
  docker build -t $IMAGE_NAME .
fi

# Remove any stale containers:
docker rm -f $CONTAINER_NAME 2>/dev/null || true

# Create and start the container:
docker run -d \
  --name $CONTAINER_NAME \
  --gpus all \
  --restart unless-stopped \
  -v /var/log/$CONTAINER_NAME:/app/logs \
  -v timemanager-data:/app/files \
  --log-opt max-size=25m \
  --log-opt max-file=4 \
  $IMAGE_NAME

echo -e "$CONTAINER_NAME container is up and running...\n"
echo -e "Logs are located at /var/log/$CONTAINER_NAME/app.log"
