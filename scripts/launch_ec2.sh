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

key_path = os.path.abspath(os.path.join('files/keys/', instance_info['key_filename']))
remote_user = instance_info['remote_user']
public_dns = admin_utils.get_ec2_public_dns(instance_info)

print(f'{key_path} {remote_user} {public_dns}')
")

KEY_PATH=$(echo "$RESULT" | awk '{print $1}')
REMOTE_USER=$(echo "$RESULT" | awk '{print $2}')
PUBLIC_DNS=$(echo "$RESULT" | awk '{print $3}')

if [ -z "$PUBLIC_DNS" ]; then
  echo "Failed to retrieve public DNS. Is the instance running?"
  exit 1
fi

SSH_CONFIG="$HOME/.ssh/config"
HOST_ALIAS="$INSTANCE_NICKNAME"

# Backup the config just in case
cp "$SSH_CONFIG" "$SSH_CONFIG.bak" 2>/dev/null

# Remove any old block for this host alias
awk -v alias="$HOST_ALIAS" '
    BEGIN { found = 0 }
    {
        if ($1 == "Host" && $2 == alias) { found = 1; next }
        if (found && $1 == "Host") { found = 0 }
        if (!found) print
    }
' "$SSH_CONFIG" > "${SSH_CONFIG}.tmp" && mv "${SSH_CONFIG}.tmp" "$SSH_CONFIG"

# Append new block
cat <<EOF >> "$SSH_CONFIG"

Host $HOST_ALIAS
    HostName $PUBLIC_DNS
    User $REMOTE_USER
    IdentityFile $KEY_PATH
    IdentitiesOnly yes
EOF

echo "SSH config updated with alias '$HOST_ALIAS'."

echo "Connecting to $PUBLIC_DNS..."
ssh "$HOST_ALIAS"
