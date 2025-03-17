#!/bin/bash

if [[ "$1" == "--cleanup" ]]; then
    sudo systemctl stop ai-process.service
    sudo ./cleanup.sh
else
    sudo systemctl stop ai-process.service
fi