FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

# Set environment variable to avoid interactive timezone prompt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    nano \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Set working directory
WORKDIR /workspace

# Copy only dependency files first for caching
COPY pyproject.toml ./

# Install Python dependencies with pip (manually listed from pyproject.toml)
RUN pip install --no-cache-dir huanzhi-utils scikit-learn torch setuptools deepspeed==0.16.3 accelerate peft wandb rich duckduckgo-search 'trl @ git+https://github.com/huggingface/trl.git@fc4dae2' 'liger-kernel>=0.5.2' vllm==0.7.3 'brave-search>=0.1.8'

# Install flash-attn separately as in README
RUN pip install --no-cache-dir flash-attn --no-build-isolation
RUN pip install loguru

# Copy the rest of the code
COPY . .

# Set entrypoint (can be overridden by docker-compose)
# Copy and setup entrypoint script
RUN chmod +x entrypoint.sh

# Set entrypoint
ENTRYPOINT ["sh", "entrypoint.sh"] 