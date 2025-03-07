#!/bin/bash

if [[ "$1" == "--cleanup" ]]; then
    sudo systemctl stop ai-process.service
    ./cleanup.sh
else
    sudo systemctl stop ai-process.service
fi