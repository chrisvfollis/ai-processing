#!/bin/bash

if [ -z "$1" ]; then
  echo "Missing argument: <edge_nickname>"
  exit 1
fi

EDGE_NICKNAME="$1"

RESULT=$(python3 -c "
import os
from utilities import admin_utils

instance_info = admin_utils.get_edge_computer_info(nickname='$EDGE_NICKNAME')

remote_user = instance_info['remote_user']
ip_address = instance_info['ip_address']

print(f'{remote_user} {ip_address}')
")

REMOTE_USER=$(echo "$RESULT" | awk '{print $1}')
IP_ADDRESS=$(echo "$RESULT" | awk '{print $2}')

if [ -z "$IP_ADDRESS" ]; then
  echo "Failed to retrieve IP address. Is the edge device reachable?"
  exit 1
fi

SSH_CONFIG="$HOME/.ssh/config"
HOST_ALIAS="$EDGE_NICKNAME"

# Backup the config just in case
touch "$SSH_CONFIG"
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
    HostName $IP_ADDRESS
    User $REMOTE_USER
EOF

echo "SSH config updated with alias '$HOST_ALIAS'."

echo "Connecting to $IP_ADDRESS..."
ssh "$HOST_ALIAS"
