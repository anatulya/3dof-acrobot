FROM ubuntu:22.04

# Set noninteractive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install essential packages
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libx11-dev \
    libxext-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libwayland-dev \
    libxkbcommon-dev \
    libglfw3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*
    
# Clone and build MuJoCo (from source)
WORKDIR /opt
RUN git clone https://github.com/deepmind/mujoco.git && \
    cd mujoco && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# Set environment variables
ENV MUJOCO_PATH=/opt/mujoco
ENV LD_LIBRARY_PATH=$MUJOCO_PATH/build/lib:$LD_LIBRARY_PATH
