FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    build-essential \
    curl wget git unzip \
    python3.10 python3-pip python3-dev python3-tk \
    libgl1-mesa-glx libglu1-mesa libglfw3 libosmesa6 libxi6 \
    libxrandr2 libxxf86vm1 libxcursor1 libxinerama1 \
    zlib1g xz-utils bzip2 libffi8 libexpat1 \
    libreadline8 libsqlite3-0 openssl tzdata \
    patchelf xvfb x11-xserver-utils x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Set working directory
WORKDIR /home/marl/workspace

# Default shell
SHELL ["/bin/bash", "-c"]

# Entry point
CMD ["bash"]
