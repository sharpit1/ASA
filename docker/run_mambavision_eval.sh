#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade "transformers>=4.50,<5"
python -m pip check
python -c "import transformers; print('transformers', transformers.__version__)"

cd /workspace/Adversarial-Scenario
exec python .asa_flux2_eval_driver_019f6576.py "$@"
