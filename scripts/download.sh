#!/bin/bash

if [ -z "$1" ]; then
  echo "Missing argument: <instance_nickname>"
  exit 1
fi

INSTANCE_NICKNAME="$1"

if [[ "$2" ]]; then
    REMOTE_DIR="$2"
else
    REMOTE_DIR="output/"
fi


RESULT=$(python3 -c "
from utilities import admin_utils

admin_utils.auto_scp(action='download', nickname='$INSTANCE_NICKNAME', remote_dir='$REMOTE_DIR')
")
