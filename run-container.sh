#!/bin/bash

set -e

IMAGE_NAME="timemanager-image"
CONTAINER_NAME="timemanager-app"

docker rm -f $CONTAINER_NAME 2>/dev/null || true

docker build -t $IMAGE_NAME .

docker run -d \
  --name $CONTAINER_NAME \
  --gpus all \
  --restart unless-stopped \
  -v /var/log/$CONTAINER_NAME:/app/logs \
  $IMAGE_NAME

echo "Container '$CONTAINER_NAME' is up and running"
echo "Logs: /var/log/$CONTAINER_NAME/app.log"
