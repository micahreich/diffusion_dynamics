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

# Build up the docker run args
docker_run_args=()
docker_run_args+=(--name "$CONTAINER_NAME")
docker_run_args+=(--network host)
docker_run_args+=(--ipc=host)
docker_run_args+=(--privileged)
docker_run_args+=(-v "$SRC_DIR:/workspace")
docker_run_args+=(-v ~/.vscode-server:/root/.vscode-server)
docker_run_args+=(-p 8888:8888)
docker_run_args+=("$DOCKER_IMAGE_NAME")
docker_run_args+=(bash)

if [ -n "$DISPLAY" ]; then
    docker_run_args+=(-e "DISPLAY=$DISPLAY")
    docker_run_args+=(-v /tmp/.X11-unix:/tmp/.X11-unix)
    
    xhost +local:
fi

# Run the container with GPU support and shared network
docker run --gpus all -it --rm --shm-size=8g \
    "${docker_run_args[@]}"

if [ -n "$DISPLAY" ]; then
    xhost -local:
fi
