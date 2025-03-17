#!/bin/bash

if [ -z "$1" ]; then
  echo "Missing argument: <instance_nickname>"
  exit 1
fi

INSTANCE_NICKNAME="$1"

RESULT=$(python3 -c "
from utilities import admin_utils

admin_utils.auto_scp(action='download', nickname='$INSTANCE_NICKNAME')
")
