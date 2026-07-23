# syntax=docker/dockerfile:1.7

# Matches the CUDA major/minor used by the ds environment. The devel image is
# required by packages that compile CUDA extensions at install or runtime.
ARG CUDA_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_VERSION=2.12.0
ARG TORCHVISION_VERSION=0.27.0
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG EXPECTED_TORCH_CUDA=13.0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/ds \
    PATH=/opt/ds/bin:${PATH} \
    HF_HOME=/root/.cache/huggingface \
    TORCH_HOME=/root/.cache/torch \
    LD_LIBRARY_PATH=/opt/ds/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH} \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        git-lfs \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        nodejs \
        npm \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        tini \
        tmux \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system \
    && python3 -m venv "${VIRTUAL_ENV}"

RUN python -m pip install \
        pip==26.0.1 \
        setuptools==81.0.0 \
        wheel==0.46.3 \
    && python -m pip install \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        --index-url "${PYTORCH_INDEX_URL}"

COPY docker/requirements-ds.lock /tmp/requirements-ds.lock
RUN python -m pip install -r /tmp/requirements-ds.lock \
    && python -m pip check \
    && python -c "import cv2, diffusers, torch, torchvision, transformers; assert torch.version.cuda == '${EXPECTED_TORCH_CUDA}'; assert diffusers.__version__ == '0.35.2'; assert transformers.__version__ == '5.10.2'; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# The Hugging Face MambaVision remote code currently targets Transformers 4.x.
# Set this only for evaluation images that need that model.
ARG EVAL_TRANSFORMERS_4=0
RUN if [ "${EVAL_TRANSFORMERS_4}" = "1" ]; then \
        python -m pip install --upgrade "transformers>=4.50,<5" \
        && python -m pip check; \
    fi

WORKDIR /workspace/Adversarial-Scenario
COPY . .

# Vim uses a vendored Mamba 1.1.1 CUDA extension. Keep it optional because it
# makes the image architecture-specific and adds several minutes to the build.
ARG INSTALL_VIM_MAMBA=0
RUN if [ "${INSTALL_VIM_MAMBA}" = "1" ]; then \
        MAMBA_FORCE_BUILD=TRUE MAMBA_CUDA_ARCH_LIST=89 MAX_JOBS=4 \
        python -m pip install --no-build-isolation --no-deps ./third_party/Vim/mamba-1p1p1; \
    fi
RUN if [ "${INSTALL_VIM_MAMBA}" = "1" ]; then \
        CAUSAL_CONV1D_FORCE_BUILD=TRUE CAUSAL_CONV1D_CUDA_ARCH_LIST=89 MAX_JOBS=4 \
        python -m pip install --no-build-isolation --no-deps ./third_party/Vim/causal-conv1d; \
    fi
RUN if [ "${INSTALL_VIM_MAMBA}" = "1" ]; then \
        python -c "import torch, causal_conv1d_cuda, selective_scan_cuda; print('Vim CUDA extensions ready')"; \
    fi

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
