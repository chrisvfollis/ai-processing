#!/bin/bash

ARGS="$*"

USE_JOURNAL=false
SHOW_ALLOC=false
REVERSE=true

[[ "$ARGS" == *"-j"* ]] && USE_JOURNAL=true

[[ "$ARGS" == *"--memory-traces"* ]] && SHOW_ALLOC=true
[[ "$ARGS" == *"--forward"* ]] && REVERSE=false

if $USE_JOURNAL; then
    CMD=(sudo journalctl -u ai-process.service)
    $REVERSE && CMD+=("--reverse")
    if $SHOW_ALLOC; then
        "${CMD[@]}" | less
    else
        "${CMD[@]}" | grep -v alloc | grep -v rss | less
    fi
else
    LOG_FILE="files/logs/app.log"
    if $REVERSE; then
        LOG_CMD=(tac "$LOG_FILE")
    else
        LOG_CMD=(cat "$LOG_FILE")
    fi
    if $SHOW_ALLOC; then
        "${LOG_CMD[@]}" | less
    else
        "${LOG_CMD[@]}" | grep -v alloc | grep -v rss | less
    fi
fi
