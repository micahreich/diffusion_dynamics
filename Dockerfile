# Use PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Build arguments for user and group IDs; defaults to 1000.
ARG USER_ID=1000
ARG GROUP_ID=1000

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
    yapf \
    pre-commit \
    && pip cache purge

RUN pip install \
    diffusers \
    smalldiffusion \
    torch_ema

# Create a non-root user named "dev" with the provided UID/GID.
RUN groupadd -g ${GROUP_ID} dev && \
    useradd -m -u ${USER_ID} -g dev -s /bin/bash dev && \
    echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN mkdir -p /home/dev/workspace && chown -R dev:dev /home/dev/workspace

# Set up Jupyter Notebook config
RUN mkdir -p /root/.jupyter && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py

EXPOSE 8888
    
# Set the working directory.
WORKDIR /home/dev/workspace

# Install the package in editable mode
COPY . /home/dev/workspace
RUN pip install -e .

USER dev

ENV PATH="/home/dev/.local/bin:${PATH}"
RUN git config --global --add safe.directory /home/dev/workspace

# Set default command to keep the container running
CMD ["bash"]
