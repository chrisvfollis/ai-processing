#!/bin/bash
set -euo pipefail

ARGS="$*"

USE_JOURNAL=true
HIDE_MEMORY=true
REVERSE=true
PROGRESS_ONLY=false

[[ "$ARGS" == *"-j"* ]] && USE_JOURNAL=true
[[ "$ARGS" == *"--memory-traces"* ]] && HIDE_MEMORY=false
[[ "$ARGS" == *"--forward"* ]] && REVERSE=false
[[ "$ARGS" == *"--progress"* ]] && PROGRESS_ONLY=true

# common pager: -R keeps ANSI color, -S avoids hard wrapping
PAGER_CMD="less -R -S"

if $USE_JOURNAL; then
  CMD="sudo journalctl -u ai-process.service --no-pager -o cat"
  $REVERSE && CMD="$CMD --reverse"
else
  LOG_FILE="files/logs/app.log"
  if $REVERSE; then
    CMD="tac $LOG_FILE"
  else
    CMD="cat $LOG_FILE"
  fi
fi

if $HIDE_MEMORY; then
  CMD="$CMD | grep -v alloc | grep -v rss"
fi

if $PROGRESS_ONLY; then
  OUTPUT="$(eval "$CMD")"
  CURRENT_QUEUE_PATTERN=$(echo "$OUTPUT" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}' | tail -n 1)

  if [[ -z "$CURRENT_QUEUE_PATTERN" ]]; then
    echo "No queue pattern found."
    exit 1
  fi

  START_LINE_NUM=$(echo "$OUTPUT" | grep -n "$CURRENT_QUEUE_PATTERN" | grep 'Running' | head -n 1 | cut -d: -f1)

  if [[ -z "$START_LINE_NUM" ]]; then
    echo "No 'Running' line found for pattern $CURRENT_QUEUE_PATTERN."
    exit 1
  fi

  echo "$OUTPUT" | tail -n +"$START_LINE_NUM" | grep -E 'Running|Elapsed|([0-9]{1,3}%)' | $PAGER_CMD
else
  eval "$CMD" | $PAGER_CMD
fi
