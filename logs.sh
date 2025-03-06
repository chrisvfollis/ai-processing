#!/bin/bash

if [[ "$1" == "--forward" ]]; then
    sudo journalctl -u ai-process.service
else
    sudo journalctl -u ai-process.service --reverse
fi
