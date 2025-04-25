#!/bin/bash

if [ -z "$1" ]; then
  echo "Missing argument: <instance_nickname>"
  exit 1
else
  INSTANCE_NICKNAME="$1"
fi

REMOTE_DIR="files/"
RECURSIVE=false

EXTRA_ARGS="$*"

# parse any extra arguments:
shift
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --remote-dir)
      REMOTE_DIR="$2"
      shift 2  # consume key and value
      ;;
    -r)
      RECURSIVE=true
      shift
      ;;
    *)  # ignore unknown args
      shift
      ;;
  esac
done

python3 -c "
from utilities import admin_utils
admin_utils.auto_scp(
    action='download',
    nickname='${INSTANCE_NICKNAME}',
    remote_dir='${REMOTE_DIR}',
    recursive=${RECURSIVE}
)
"
