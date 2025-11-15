# RunPod Serverless Dockerfile for Qwen-Image with ComfyUI
# Optimized for RTX 5090 with 140GB+ system memory

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV HF_HOME=/runpod-volume/.cache/huggingface
ENV COMFYUI_PATH=/app/ComfyUI

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git

WORKDIR /app/ComfyUI

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir packaging

# Clone custom nodes
WORKDIR /app/ComfyUI/custom_nodes

RUN git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts && \
    git clone https://github.com/rgthree/rgthree-comfy && \
    git clone https://github.com/Jonseed/ComfyUI-Detail-Daemon && \
    git clone https://github.com/ClownsharkBatwing/RES4LYF && \
    git clone https://github.com/digitaljohn/comfyui-propost && \
    git clone https://github.com/gseth/ControlAltAI-Nodes && \
    git clone https://github.com/WASasquatch/was-node-suite-comfyui && \
    git clone https://github.com/M1kep/ComfyLiterals

# Install custom node requirements
RUN for dir in */ ; do \
        if [ -d "$dir" ]; then \
            if [ -f "$dir/requirements.txt" ]; then \
                pip install --no-cache-dir -r "$dir/requirements.txt" || true; \
            fi; \
            if [ -f "$dir/install.py" ]; then \
                cd "$dir" && python install.py && cd ..; \
            fi; \
        fi; \
    done

# Install RunPod SDK for serverless
RUN pip install --no-cache-dir runpod

WORKDIR /app/ComfyUI

# Create model directories
RUN mkdir -p models/diffusion_models models/loras models/vae models/text_encoders

# Download models (these will be cached in the docker image)
# This speeds up cold starts significantly
RUN wget --progress=bar:force:noscroll -O models/diffusion_models/qwen_image_bf16.safetensors \
    https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_bf16.safetensors && \
    wget --progress=bar:force:noscroll -O models/loras/Qwen-Image-Lightning-8steps-V2.0.safetensors \
    https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V2.0.safetensors && \
    wget --progress=bar:force:noscroll -O models/vae/qwen_image_vae.safetensors \
    https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors && \
    wget --progress=bar:force:noscroll -O models/text_encoders/qwen_2.5_vl_7b.safetensors \
    https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors

# Copy handler and workflow
WORKDIR /app
COPY handler.py .
COPY qwen_sfw_workflow_api.json workflow.json

# Add Python path
ENV PYTHONPATH=/app/ComfyUI:$PYTHONPATH

# Run the handler
CMD ["python", "-u", "handler.py"]
