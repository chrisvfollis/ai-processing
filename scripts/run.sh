#!/bin/bash

EXTRA_ARGS="$*"
echo "EXTRA_ARGS=\"$EXTRA_ARGS\"" | sudo tee /etc/default/ai-process

sudo systemctl daemon-reload
sudo systemctl restart ai-process.service
