#!/bin/bash

# Container name
CONTAINER_NAME="diffusion_dynamics_container"

DOCKER_IMAGE_NAME="diffusion_dynamics_image"

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Path to diffusion_dynamics/ folder
SRC_DIR="$(realpath "$SCRIPT_DIR")"

# Ensure src/ exists, else create it
mkdir -p "$SRC_DIR"

xhost +local:

# Run the container with GPU support and shared network
docker run --gpus all -it --rm --shm-size=8g \
    --name "$CONTAINER_NAME" \
    --network host \
    --ipc=host \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$SRC_DIR:/workspace" \
    -v ~/.vscode-server:/root/.vscode-server \
    -p 8888:8888 \
    $DOCKER_IMAGE_NAME bash

xhost -local:
