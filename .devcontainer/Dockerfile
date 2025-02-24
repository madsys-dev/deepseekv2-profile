FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

FROM nvidia/cuda:12.6.0-base-ubuntu22.04

# Add NVIDIA runtime components
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Install system packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support - matching your system version
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Optional: Add any other packages you need
RUN pip3 install numpy pandas matplotlib jupyter transformers
