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
 && ln -s /usr/bin/pip3 /usr/bin/pip \
 && pip install --upgrade pip

WORKDIR /app

# Install additional system packages
COPY installed-packages.txt .
RUN apt-get update && xargs -a installed-packages.txt apt-get install -y && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt dev-requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r dev-requirements.txt

# Copy source code last to optimize layer caching
COPY . .
RUN pip install --no-cache-dir -e .

ENV PATH="/app/scripts:${PATH}"

CMD ["python3", "execution/main.py"]
