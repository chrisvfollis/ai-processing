#!/bin/bash

set -e && set -a
source .env.public
set +a

find . scripts -name "*.sh" -exec chmod +x {} +

# Add shell scripts to PATH:
EXPORT_S_PATH="export PATH=$SCRIPTS_PATH"

if ! grep -Fxq "$EXPORT_S_PATH" ~/.bashrc && ! grep -Fxq "$EXPORT_S_PATH" ~/.zshrc; then
    echo "$EXPORT_S_PATH" >> ~/.bashrc
    echo "$EXPORT_S_PATH" >> ~/.zshrc
    echo "Updated PATH to include project scripts"
fi

# Set machine:
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, setting up with GPU support..."
    MACHINE="gpu"
else
    echo "CUDA not detected, setting up with CPU support only..."
    MACHINE="cpu"
fi

if [[ $1 == "--use-pipenv" ]]; then
    cp "runtime/native/$MACHINE/Pipfile" Pipfile
    pipenv install
    pipenv run pip install -e .
    echo "Finished installing environment. To activate, run pipenv shell"

elif [[ $1 == "--use-docker" ]]; then
    ./deploy.sh --build
fi
