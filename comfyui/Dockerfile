# Lightweight CPU ComfyUI container with the comfyui-anyserve custom nodes preinstalled.
# Heavy LLM, YOLO, Triton, or diffusion work runs on remote GPU workers over AnyServe.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ComfyUI
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git .

RUN uv pip install --system --no-cache \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio \
    && uv pip install --system --no-cache -r requirements.txt

COPY . /ComfyUI/custom_nodes/comfyui-anyserve
RUN uv pip install --system --no-cache -r /ComfyUI/custom_nodes/comfyui-anyserve/requirements.txt

EXPOSE 8188
CMD ["python3", "main.py", "--cpu", "--listen", "0.0.0.0", "--port", "8188"]
