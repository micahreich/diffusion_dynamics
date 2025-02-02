# Use PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set non-interactive mode for installations
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install necessary dependencies
RUN apt-get update && apt-get install -y \
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
    && pip cache purge

RUN pip install \
    diffusers

# Set up Jupyter Notebook config
RUN mkdir -p /root/.jupyter && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py

# Set working directory
WORKDIR /workspace

# Expose Jupyter notebook port
EXPOSE 8888

# Set default command to keep the container running
CMD ["bash"]

# To build the container after changes, run:
# $ docker build --network=host -t diffusion_dynamics_image -f Dockerfile .
