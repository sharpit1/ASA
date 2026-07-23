# Isolated VLM robustness attack

This directory contains the supported, reduced attack pipeline only.

## Included

- Attack modes: `vlm`, `and`
- Generators: FLUX.2 Klein, Qwen2 Image Edit, Bernini
- Three generator-specific runners and runtimes
- Shared black-box attack core and minimal VLM helpers
- In-memory candidate-image handoff without strip serialization
- 32 supported YAML presets
- Static/unit verification tests

Successful classifier-input images are saved immediately at 224x224. The
pipeline does not persist source images and does not rerender a successful
candidate.

## Deliberately excluded

- `vlm_attack_legacy.py`
- inversion and inversion prompts
- fixed-prompt, IC, standalone CWOR/GS, and intermediate-image flows
- duplicate datasets, checkpoints, caches, outputs, and third-party packages

The launcher uses the parent ASA checkout for shared, large dependencies:
`data/`, `eval/`, `torch_nets/`, `third_party/`, and `.cache/`. Override that
location with `ASA_PROJECT_ROOT` when this folder is moved elsewhere.

## Usage

From the ASA project root:

```bash
./isolated_vlm_attack/run_vlm_attack.sh --help
./isolated_vlm_attack/run_vlm_attack.sh \
  --config configs/flux2_vlm_attack_nips.yaml
```

`--config` paths are resolved relative to this isolated directory. Dataset and
output paths remain relative to `ASA_PROJECT_ROOT`.

## Code-only verification

The checks below do not load a generator or execute an attack:

```bash
python -m unittest discover -s isolated_vlm_attack/tests -p 'test_attack_refactor.py' -v
bash -n isolated_vlm_attack/run_vlm_attack.sh
RUN_VLM_ATTACK_DRY_RUN=1 ./isolated_vlm_attack/run_vlm_attack.sh
```
