#!/usr/bin/env bash
set -euo pipefail

repo_root=/workspace/Adversarial-Scenario
causal_src="${repo_root}/third_party/Vim/causal-conv1d"
causal_build=/tmp/causal-conv1d

cp -a "${causal_src}" "${causal_build}"
cd "${causal_build}"
CAUSAL_CONV1D_FORCE_BUILD=TRUE \
CAUSAL_CONV1D_CUDA_ARCH_LIST=89 \
MAX_JOBS=4 \
python -m pip install --no-build-isolation --no-deps .

python -c "import torch, causal_conv1d_cuda, selective_scan_cuda; print('Vim CUDA extensions ready')"

cd "${repo_root}"
exec python .asa_flux2_eval_driver_019f6576.py "$@"
