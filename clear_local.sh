#!/bin/bash

if [ -z "$1" ]; then
  echo "Missing argument: <config_filename>"
  exit 1
fi

CONFIG_FILE="$1"

RESULT=$(python3 -c "
from admin_utils import ec2_public_dns
pem_path, remote_user, public_dns = ec2_public_dns('$CONFIG_FILE')
print(f'{pem_path} {remote_user} {public_dns}')
")

PEM_PATH=$(echo $RESULT | awk '{print $1}')
REMOTE_USER=$(echo $RESULT | awk '{print $2}')
PUBLIC_DNS=$(echo $RESULT | awk '{print $3}')

if [ -z "$PUBLIC_DNS" ]; then
  echo "Failed to retrieve public DNS. Is the instance running?"
  exit 1
fi

echo "Connecting to $PUBLIC_DNS..."

ssh -i "$PEM_PATH" "$REMOTE_USER@$PUBLIC_DNS"