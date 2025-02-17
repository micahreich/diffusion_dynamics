# Use PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
ARG HOST_UID
ARG HOST_GID
ARG HOST_USER

# Set non-interactive mode for installations
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install necessary dependencies
RUN apt-get update && apt-get install -y \
    sudo \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    net-tools \
    iputils-ping \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    jupyterlab \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    scipy \
    tqdm \
    ipywidgets \
    tensorboard \
    jax \
    black \
    && pip cache purge

RUN pip install \
    diffusers \
    smalldiffusion \
    torch_ema

# Set up current user
RUN groupadd -g ${HOST_GID} ${HOST_USER} && \
    useradd -m -u ${HOST_UID} -g ${HOST_USER} ${HOST_USER} && \
    usermod -aG sudo ${HOST_USER} && \
    echo "${HOST_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set up Jupyter Notebook config
RUN mkdir -p /root/.jupyter && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py

# Set working directory
WORKDIR /workspace

# Install the package in editable mode
COPY . /workspace/
RUN pip install -e .

# Expose Jupyter notebook port
EXPOSE 8888

USER ${HOST_USER}

# Set default command to keep the container running
CMD ["bash"]

# To build the container after changes, run:
# $ docker build --network=host -t diffusion_dynamics_image -f Dockerfile .
