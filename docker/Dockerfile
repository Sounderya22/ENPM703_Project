FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    build-essential \
    curl wget git \
    python3 python3-pip python3-venv python3-dev python3-tk \
    libgl1-mesa-glx libglu1-mesa libglfw3 libosmesa6 libxi6 \
    libxrandr2 libxxf86vm1 libxcursor1 libxinerama1 \
    zlib1g xz-utils bzip2 libffi8 libexpat1 \
    libreadline8 libsqlite3-0 openssl tzdata \
    xvfb x11-xserver-utils x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Python setup
RUN ln -sf /usr/bin/python3 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python packages
WORKDIR /home/ubuntu/marl_workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

# Default command
CMD ["bash"]
