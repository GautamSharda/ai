FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Ensure curl exists (many RunPod images already include it)
RUN if ! command -v curl >/dev/null 2>&1; then \
      apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*; \
    fi

# Install nvm + Node (LTS by default)
ENV NVM_DIR=/root/.nvm
RUN mkdir -p "$NVM_DIR" \
 && curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash \
 && bash -lc "source $NVM_DIR/nvm.sh && nvm install --lts && nvm alias default 'lts/*'"

# Put node/npm on PATH for non-interactive shells
ENV PATH="$NVM_DIR/versions/node/$(bash -lc 'source $NVM_DIR/nvm.sh >/dev/null 2>&1 && nvm version default')/bin:$PATH"

# Quick sanity check
RUN node -v && npm -v

# Install Claude Code globally
RUN npm install -g @anthropic-ai/claude-code

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Create virtual environment and install kaggle
WORKDIR /workspace
RUN uv venv && \
    uv pip install kaggle

# Activate virtual environment by default
ENV VIRTUAL_ENV="/workspace/.venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

CMD ["/bin/bash"]
