#!/bin/bash

EXTRA_ARGS="$*"
echo "EXTRA_ARGS=\"$EXTRA_ARGS\"" | sudo tee /etc/default/osnet-training

sudo systemctl daemon-reload
sudo systemctl restart osnet-training.service
