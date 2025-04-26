#!/bin/bash

ARGS="$*"

USE_JOURNAL=false
HIDE_MEMORY=true
REVERSE=true
PROGRESS_ONLY=false

[[ "$ARGS" == *"-j"* ]] && USE_JOURNAL=true
[[ "$ARGS" == *"--memory-traces"* ]] && HIDE_MEMORY=false
[[ "$ARGS" == *"--forward"* ]] && REVERSE=false
[[ "$ARGS" == *"--progress"* ]] && PROGRESS_ONLY=true

if $USE_JOURNAL; then
    CMD=(sudo journalctl -u ai-process.service)
    $REVERSE && CMD+=("--reverse")
    if $HIDE_MEMORY; then
        OUTPUT=$("${CMD[@]}" | grep -v alloc | grep -v rss)
    else
        OUTPUT=$("${CMD[@]}")
    fi
else
    LOG_FILE="files/logs/app.log"
    if $REVERSE; then
        LOG_CMD=(tac "$LOG_FILE")
    else
        LOG_CMD=(cat "$LOG_FILE")
    fi
    if $HIDE_MEMORY; then
        OUTPUT=$("${LOG_CMD[@]}" | grep -v alloc | grep -v rss)
    else
        OUTPUT=$("${LOG_CMD[@]}")
    fi
fi

if $PROGRESS_ONLY; then
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

    TAIL_OUTPUT=$(echo "$OUTPUT" | tail -n +"$START_LINE_NUM")

    echo "$TAIL_OUTPUT" | grep -E 'Running|Elapsed|([0-9]{1,3}%)' | less
else
    echo "$OUTPUT" | less
fi
