FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Ensure curl and unzip exist (many RunPod images already include curl)
RUN if ! command -v curl >/dev/null 2>&1 || ! command -v unzip >/dev/null 2>&1; then \
      apt-get update && apt-get install -y curl unzip && rm -rf /var/lib/apt/lists/*; \
    fi

# Install nvm + Node (use stable v18 instead of latest LTS)
ENV NVM_DIR=/root/.nvm
RUN mkdir -p "$NVM_DIR" \
 && curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash \
 && bash -lc "source $NVM_DIR/nvm.sh && nvm install 18 && nvm alias default 18"

# Put node/npm on PATH for non-interactive shells
ENV PATH="$NVM_DIR/versions/node/$(bash -lc 'source $NVM_DIR/nvm.sh >/dev/null 2>&1 && nvm version default')/bin:$PATH"

# Quick sanity check
RUN bash -lc "source $NVM_DIR/nvm.sh && node -v && npm -v"

# Install Claude Code globally
RUN bash -lc "source $NVM_DIR/nvm.sh && npm install -g @anthropic-ai/claude-code"

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Clone the AI repository
WORKDIR /workspace
RUN git clone https://github.com/GautamSharda/ai.git && \
    cd ai

# Create virtual environment and install kaggle
WORKDIR /workspace/ai
RUN uv venv .venv && \
    uv pip install kaggle

# Unzip ARC-AGI competition data
RUN cd arc-agi/arc-agi-2025 && unzip arc-prize-2025.zip && \
    cd ../arc-agi-2024 && unzip arc-prize-2024.zip

# Activate virtual environment by default
ENV VIRTUAL_ENV="/workspace/ai/.venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

CMD ["/bin/bash"]
