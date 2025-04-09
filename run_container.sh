#!/bin/bash

set -e

IMAGE_NAME="timemanager-image"
CONTAINER_NAME="timemanager-app"

# Check container status:
STATUS=$(docker inspect -f '{{.State.Status}}' $CONTAINER_NAME 2>/dev/null || echo "not_found")
IS_RESTARTING=$(docker inspect -f '{{.State.Restarting}}' $CONTAINER_NAME 2>/dev/null || echo "false")

if [[ "$STATUS" == "exited" || "$STATUS" == "paused" ]]; then
  echo "Starting up $CONTAINER_NAME container..."
  docker start $CONTAINER_NAME
elif [[ "$STATUS" == "running" ]]; then
  echo "$CONTAINER_NAME is already running. Restarting container..."
  docker restart $CONTAINER_NAME
elif [[ "$IS_RESTARTING" == "true" ]]; then
  echo "$CONTAINER_NAME is already restarting, it may be in a crash loop."
  exit 1
elif [[ "$STATUS" == "not_found" ]]; then
  echo "$CONTAINER_NAME container does not exist"
  exit 1
fi

echo -e "'$CONTAINER_NAME' container is up and running...\n"
echo -e "Logs are located at /var/log/$CONTAINER_NAME/app.log"
