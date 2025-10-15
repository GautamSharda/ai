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
RUN bash -lc "source $NVM_DIR/nvm.sh && npm install -g @anthropic-ai/claude-code && npm install -g @openai/codex"

# =======================
# All Python via uv (pinned Torch; no upgrades)
# =======================
# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# Make uv + venv the default Python toolchain for the rest of the image
ENV PATH="/root/.local/bin:/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV=/opt/venv

# Create venv
RUN uv venv /opt/venv

# Install PyTorch stack from cu121 (matches CUDA 12.4.x) and pin via constraints
RUN . /opt/venv/bin/activate && \
    uv pip install --index-url https://download.pytorch.org/whl/cu121 \
        "torch==2.4.0" "torchvision==0.19.0" "torchaudio==2.4.0" && \
    printf "torch==2.4.0\ntorchvision==0.19.0\ntorchaudio==2.4.0\n" > /tmp/torch-constraints.txt

# Install the rest of your packages into the same venv (torch* intentionally omitted)
RUN . /opt/venv/bin/activate && \
    uv pip install --constraint /tmp/torch-constraints.txt \
        aiohappyeyeballs==2.6.1 aiohttp==3.12.15 aiosignal==1.4.0 annotated-types==0.7.0 anyio==4.6.0 \
        argon2-cffi==23.1.0 argon2-cffi-bindings==21.2.0 arrow==1.3.0 astor==0.8.1 asttokens==2.4.1 async-lru==2.0.4 \
        attrs==24.2.0 babel==2.16.0 beautifulsoup4==4.12.3 blake3==1.0.7 bleach==6.1.0 cachetools==6.2.0 cbor2==5.7.0 \
        certifi==2024.8.30 cffi==1.17.1 charset-normalizer==3.3.2 click==8.3.0 cloudpickle==3.1.1 comm==0.2.2 \
        compressed-tensors==0.11.0 cupy-cuda12x==13.6.0 debugpy==1.8.5 decorator==5.1.1 defusedxml==0.7.1 depyf==0.19.0 \
        dill==0.4.0 diskcache==5.6.3 dnspython==2.8.0 einops==0.8.1 email-validator==2.3.0 entrypoints==0.4 executing==2.1.0 \
        fastapi==0.118.0 fastapi-cli==0.0.13 fastapi-cloud-cli==0.3.0 fastjsonschema==2.20.0 fastrlock==0.8.3 filelock==3.19.1 \
        flashinfer-python==0.3.1.post1 fqdn==1.5.1 frozendict==2.4.6 frozenlist==1.7.0 fsspec==2024.2.0 gguf==0.17.1 \
        h11==0.14.0 hf-xet==1.1.10 httpcore==1.0.5 httptools==0.6.4 httpx==0.27.2 huggingface-hub==0.35.3 idna==3.10 \
        interegular==0.3.3 ipykernel==6.29.5 ipython==8.27.0 ipywidgets==8.1.5 isoduration==20.11.0 jedi==0.19.1 \
        jinja2==3.1.6 jiter==0.11.0 json5==0.9.25 jsonpointer==3.0.0 jsonschema==4.23.0 jupyterlab==4.2.5 \
        jupyterlab-server==2.27.3 jupyterlab-widgets==3.0.13 lark==1.2.2 llguidance==0.7.30 llvmlite==0.44.0 \
        lm-format-enforcer==0.11.3 lxml==5.3.0 markdown-it-py==4.0.0 markupsafe==2.1.5 matplotlib-inline==0.1.7 \
        mistral-common==1.8.5 msgspec==0.19.0 multidict==6.6.4 nbconvert==7.16.4 nbformat==5.10.4 networkx==3.2.1 \
        ninja==1.13.0 numba==0.61.2 numpy==2.2.6 openai==2.1.0 opencv-python-headless==4.12.0.88 outlines-core==0.2.11 \
        packaging==25.0 pillow==11.3.0 pip==24.2 platformdirs==4.3.6 prometheus-client==0.21.0 propcache==0.3.2 \
        protobuf==6.32.1 psutil==6.0.0 pydantic==2.11.9 pydantic-core==2.33.2 pydantic-extra-types==2.10.5 \
        pygments==2.18.0 pynvml==13.0.1 python-dateutil==2.9.0.post0 python-dotenv==1.1.1 pyyaml==6.0.2 pyzmq==27.1.0 \
        ray==2.49.2 regex==2025.9.18 requests==2.32.3 rich==14.1.0 safetensors==0.6.2 scipy==1.16.2 sentencepiece==0.2.1 \
        sentry-sdk==2.39.0 setuptools==75.1.0 shellingham==1.5.4 sniffio==1.3.1 soundfile==0.13.1 soupsieve==2.6 soxr==1.0.0 \
        starlette==0.48.0 sympy==1.14.0 tabulate==0.9.0 tiktoken==0.11.0 tokenizers==0.22.1 tornado==6.4.1 tqdm==4.67.1 \
        transformers==4.57.0 typer==0.19.2 typing-extensions==4.15.0 urllib3==2.2.3 uvicorn==0.37.0 uvloop==0.21.0 \
        vllm==0.11.0 watchfiles==1.1.0 wcwidth==0.2.13 websocket-client==1.8.0 websockets==15.0.1 wheel==0.44.0 \
        xformers==0.0.32.post1 xgrammar==0.1.25 yarl==1.20.1

# Make venv visible in ALL shells (bash -l, su -, VS Code Remote); add symlinks
RUN printf 'export VIRTUAL_ENV=/opt/venv\nexport PATH=/opt/venv/bin:$PATH\n' \
  > /etc/profile.d/uv-venv.sh && \
  ln -sf /opt/venv/bin/python /usr/local/bin/python && \
  ln -sf /opt/venv/bin/pip /usr/local/bin/pip && \
  ln -sf /opt/venv/bin/vllm /usr/local/bin/vllm

# Clone the AI repository
WORKDIR /workspace
RUN git clone https://github.com/GautamSharda/ai.git

# Create virtual environment
WORKDIR /workspace/ai

# Unzip ARC-AGI competition data
RUN cd arc-agi/arc-agi-2025 && unzip arc-prize-2025.zip && \
    cd ../arc-agi-2024 && unzip arc-prize-2024.zip

# Use the HF HUB in the network volume (persist at runtime)
ENV HF_HOME=/ai_network_volume/huggingface_cache

# Start SSH server and then keep container running
CMD service ssh start && tail -f /dev/null
