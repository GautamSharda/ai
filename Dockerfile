FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Update system packages and install necessary tools including SSH server
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y nano git-lfs openssh-server && \
    rm -rf /var/lib/apt/lists/*

# Configure SSH server
RUN mkdir /var/run/sshd && \
    # Allow root login with key-based authentication
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config && \
    # Disable password authentication (key-only)
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

# Set up authorized keys with your public key
RUN mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh && \
    echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEZ13uHZJf4V3SMCw22qZKMTm4FSivjsFM53jy14hX1j gautamsharda001@gmail.com" > /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys

# Configure git user settings
RUN git config --global user.email "gautamsharda001@gmail.com" && \
    git config --global user.name "Gautam Sharda"

# Ensure curl and unzip exist
RUN if ! command -v curl >/dev/null 2>&1 || ! command -v unzip >/dev/null 2>&1; then \
      apt-get update && apt-get install -y curl unzip && rm -rf /var/lib/apt/lists/*; \
    fi

# Install nvm + Node
ENV NVM_DIR=/root/.nvm
RUN mkdir -p "$NVM_DIR" && \
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash && \
    bash -lc "source $NVM_DIR/nvm.sh && nvm install 18 && nvm alias default 18"

ENV PATH="$NVM_DIR/versions/node/$(bash -lc 'source $NVM_DIR/nvm.sh >/dev/null 2>&1 && nvm version default')/bin:$PATH"

RUN bash -lc "source $NVM_DIR/nvm.sh && node -v && npm -v"

# Install Claude Code globally
RUN bash -lc "source $NVM_DIR/nvm.sh && npm install -g @anthropic-ai/claude-code"

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Clone the AI repository
WORKDIR /workspace
RUN git clone https://github.com/GautamSharda/ai.git

# Create virtual environment
WORKDIR /workspace/ai
# RUN uv venv .venv && \
#    uv pip install kaggle

# Unzip ARC-AGI competition data
RUN cd arc-agi/arc-agi-2025 && unzip arc-prize-2025.zip && \
    cd ../arc-agi-2024 && unzip arc-prize-2024.zip

# We will want to use the HF HUB in the network volume
RUN export HF_HOME=/ai_network_volume/huggingface_cache

# Start SSH server and then keep container running
CMD service ssh start && tail -f /dev/null