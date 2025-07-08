#!/bin/bash

if [ -z "$1" ]; then
    echo "Missing argument: <search term>"
    exit 1
fi

SEARCH_TERM="$1"

grep -rn . \
    --binary-files=without-match \
    --exclude-dir='.git' \
    --exclude-dir='__pycache__' \
    --exclude='*.zip' \
    --exclude='*.bin' \
    --exclude='*.pkl' \
    --exclude='*.dat' \
    --exclude='*.tar' --exclude='*.tar-250' \
    --exclude='*.pth' --exclude='*.pt' \
    --exclude='*.engine' \
    --exclude='*.onnx' \
    -e "$SEARCH_TERM"
