#!/bin/bash

set -e

# Add shell scripts to path
if ! grep -q 'export PATH="$PWD/scripts:$PATH"' ~/.bashrc && ! grep -q 'export PATH="$PWD/scripts:$PATH"' ~/.zshrc; then
    echo 'export PATH="$PWD/scripts:$PATH"' >> ~/.bashrc
    echo 'export PATH="$PWD/scripts:$PATH"' >> ~/.zshrc
    echo "Updated PATH to include project scripts."
fi


# Detect CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected! Writing Pipfile for CUDA..."
    cat > Pipfile <<EOL
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[[source]]
url = "https://download.pytorch.org/whl/cu121"
verify_ssl = true
name = "pytorch-cu121"

# ----------------------------------------------------------------------------

[packages]

# ------------------------
# Deep Learning Frameworks:

torch = {version = "*", index = "pytorch-cu121"}
onnx2torch = "*"
tensorflow = "==2.15.0"
keras = "==2.15.0"

# ------------------------------------------
# Image/Video Processing and Computer Vision:

opencv-contrib-python = "*"
pillow = "*"
torchvision = {version = "*", index = "pytorch-cu121"}
torchreid = "*"
deepface = {git = "https://github.com/serengil/deepface.git", ref = "master"}
facenet-pytorch = "*"

# -----------------
# Data Manipulation:

numpy = "*"
pandas = "*"
scipy = "*"
openpyxl = "*"
xlsxwriter = "*"
h5py = "*"

# -------------------
# Networks, APIs, etc:

requests = "*"
boto3 = "*"
python-dotenv = "*"

# ---------------
# Other Libraries:

shapely = "*"
tqdm = "4.43.0"
psutil = "*"

# ----------------------------------------------------------------------------

[dev-packages]

# ----------------------------------------------------------------------------

[requires]
python_version = "3.10"
EOL

else
    echo "No CUDA detected. Writing Pipfile for CPU-only PyTorch..."
    cat > Pipfile <<EOL
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[[source]]
url = "https://download.pytorch.org/whl/cpu"
verify_ssl = true
name = "pytorch-cpu"

# ----------------------------------------------------------------------------

[packages]

# ------------------------
# Deep Learning Frameworks:

torch = {version = "*", index = "pytorch-cpu"}
onnx2torch = "*"
tensorflow = "==2.15.0"
keras = "==2.15.0"

# ------------------------------------------
# Image/Video Processing and Computer Vision:

opencv-contrib-python = "*"
pillow = "*"
torchvision = {version = "*", index = "pytorch-cpu"}
torchreid = "*"
deepface = {git = "https://github.com/serengil/deepface.git", ref = "master"}
facenet-pytorch = "*"

# -----------------
# Data Manipulation:

numpy = "*"
pandas = "*"
scipy = "*"
openpyxl = "*"
xlsxwriter = "*"
h5py = "*"

# -------------------
# Networks, APIs, etc:

requests = "*"
boto3 = "*"
python-dotenv = "*"

# ---------------
# Other Libraries:

shapely = "*"
tqdm = "4.43.0"
psutil = "*"

# ----------------------------------------------------------------------------

[dev-packages]

# ----------------------------------------------------------------------------

[requires]
python_version = "3.10"
EOL
fi

pipenv install

echo -e "\nIvakt Timemanager — Successfully installed dependencies"
echo -e "Continue the installation by running one of the following commands: \n"
echo -e "    For Bash users: pipenv shell && source ~/.bashrc && pip install -e .\n"
echo -e "    For Zsh users: pipenv shell && source ~/.zshrc && pip install -e .\n"
