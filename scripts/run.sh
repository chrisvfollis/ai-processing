#!/bin/bash

if [[ $1 == '--retain-footage' ]]; then
    echo 'EXTRA_ARGS="--retain-footage"' | sudo tee /etc/default/ai-process
else
    echo 'EXTRA_ARGS=""' | sudo tee /etc/default/ai-process
fi

sudo systemctl daemon-reload

sudo systemctl restart ai-process.service
