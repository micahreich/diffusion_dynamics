# diffusion_dynamics

## Building the Docker Container

From the root of the repository, run:
```bash
docker build --network=host -t diffusion_dynamics_image -f Dockerfile .
```

## Running the Docker Container

From the root of the repository, run:
```bash
./run_container.sh
```