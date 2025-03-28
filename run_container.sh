#!/bin/bash

# Container name
IMAGE_NAME="diffusion_dynamics"
CONTAINER_NAME="diffusion_dynamics"

# Build the Docker image with the current user's UID and GID
echo "Building Docker image '${IMAGE_NAME}'..."
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --network=host -t ${IMAGE_NAME} .
if [ $? -ne 0 ]; then
    echo "Docker build failed!"
    exit 1
fi

# Get the full path of the current directory to mount as a volume
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_DIR="$(realpath "$SCRIPT_DIR")"

echo "Using host directory ${HOST_DIR} as the shared workspace."

# Run the container with GPU support enabled (if available)
echo "Running container..."

# Build up the docker run args
docker_run_args=()

if [ -n "$DISPLAY" ]; then
    docker_run_args+=(-e "DISPLAY=$DISPLAY")
    docker_run_args+=(-v /tmp/.X11-unix:/tmp/.X11-unix)
    
    xhost +local:
fi

docker_run_args+=(--name "$CONTAINER_NAME")
docker_run_args+=(--network host)
docker_run_args+=(--ipc=host)
docker_run_args+=(--privileged)
docker_run_args+=(-v "${HOST_DIR}:/home/dev/workspace")
docker_run_args+=(-v ~/.vscode-server:/root/.vscode-server)
docker_run_args+=(-p 8888:8888)
docker_run_args+=("${IMAGE_NAME}")
docker_run_args+=(bash)

# Run the container with GPU support and shared network
docker run --gpus all -it --rm --shm-size=8g \
    "${docker_run_args[@]}"

if [ -n "$DISPLAY" ]; then
    xhost -local:
fi
