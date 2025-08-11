#!/bin/bash

set -e && set -a
source .env.public
set +a

find . scripts -name "*.sh" -exec chmod +x {} +

# Add shell scripts to PATH:
EXPORT_S_PATH="export PATH=$PATH:$SCRIPTS_PATH"

if ! grep -Fxq "$EXPORT_S_PATH" ~/.bashrc && ! grep -Fxq "$EXPORT_S_PATH" ~/.zshrc; then
    echo "$EXPORT_S_PATH" >> ~/.bashrc
    echo "$EXPORT_S_PATH" >> ~/.zshrc
    echo "Appended relevant scripts to PATH "
fi

# Set machine:
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, setting up with GPU support..."
    MACHINE="gpu"
else
    echo "CUDA not detected, setting up with CPU support only..."
    MACHINE="cpu"
fi

# Install:
if [[ $1 == "--use-pipenv" ]]; then
    cp "execution/runtime/native/$MACHINE/Pipfile" Pipfile
    pipenv install
    pipenv run pip install -e .
    printf "\n"
    echo "Finished installing environment. Next steps:"
    printf "\n"
    echo "   1. Source the updated shell configuration file:"
    echo "      source ~/.bashrc "
    echo "      # or "
    echo "      source ~/.zshrc"
    printf "\n"
    echo "   2. Activate the virtual environment with \`pipenv shell\`"
    printf "\n"

elif [[ $1 == "--use-docker" ]]; then
    ./deploy.sh --build
fi
