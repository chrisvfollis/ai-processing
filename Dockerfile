FROM ubuntu:22.04

ENV PYTHON_VERSION=3.10

# Install base tools, add deadsnakes PPA, install python & pip
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    libssl-dev \
    libffi-dev \
    curl \
    git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set python and pip symlinks
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
 && [ -e /usr/bin/pip ] || ln -s /usr/bin/pip3 /usr/bin/pip \
 && pip install --upgrade pip

WORKDIR /app

# Add the CUDA keyring and NVIDIA repository
RUN apt-get update && \
    apt-get install -y curl gnupg && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o cuda-keyring.deb && \
    dpkg -i cuda-keyring.deb && \
    rm cuda-keyring.deb && \
    apt-get update

# Install additional system packages
COPY installed-packages.txt .
RUN apt-get update && xargs -a installed-packages.txt apt-get install -y && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt dev-requirements.txt ./
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt -r dev-requirements.txt

# Copy source code last to optimize layer caching
COPY . .
RUN pip install --no-cache-dir -e .

ENV PATH="/app/scripts:${PATH}"

CMD ["python3", "execution/main.py"]
