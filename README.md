# diffusion_dynamics

## Installing Docker

Install Docker with:
```bash
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh && rm get-docker.sh
```

Complete the [post-setup instructions for Linux](https://docs.docker.com/engine/install/linux-postinstall/).
Then, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to make your GPU visible to the container.

## Building the Docker Container

From the root of the repository, run:
```bash
docker build \
--network=host \
--build-arg HOST_USER=$(whoami) --build-arg HOST_UID=$(id -u) --build-arg HOST_GID=$(id -g) \
-t diffusion_dynamics_image -f Dockerfile .
```

## Running the Docker Container

From the root of the repository, run:
```bash
./run_container.sh
```

## Formatting

Run `black` from the repository root with:
```bash
black --config pyproject.toml .
```