#!/bin/bash

if [ -z "$1" ]; then
  echo "Missing argument: <instance_nickname>"
  exit 1
fi

INSTANCE_NICKNAME="$1"

RESULT=$(python3 -c "
import os
from utilities import admin_utils

instance_info = admin_utils.get_instance_info(nickname='$INSTANCE_NICKNAME')

key_path = os.path.join('files/keys/', instance_info['key_filename'])
remote_user = instance_info['remote_user']
public_dns = admin_utils.get_ec2_public_dns(instance_info)

print(f'{key_path} {remote_user} {public_dns}')
")

KEY_PATH=$(echo $RESULT | awk '{print $1}')
REMOTE_USER=$(echo $RESULT | awk '{print $2}')
PUBLIC_DNS=$(echo $RESULT | awk '{print $3}')

if [ -z "$PUBLIC_DNS" ]; then
  echo "Failed to retrieve public DNS. Is the instance running?"
  exit 1
fi

echo "Connecting to $PUBLIC_DNS..."

ssh -i "$KEY_PATH" "$REMOTE_USER@$PUBLIC_DNS"