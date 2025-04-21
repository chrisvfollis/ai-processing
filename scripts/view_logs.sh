#!/bin/bash

ARGS="$*"

CMD=(sudo journalctl -u ai-process.service)

if [[ "$ARGS" != *"--forward"* ]]; then
    CMD+=("--reverse")
fi

if [[ "$ARGS" != *"--memory-traces"* ]]; then
    "${CMD[@]}" | grep -v alloc | less
else
    "${CMD[@]}"
fi
