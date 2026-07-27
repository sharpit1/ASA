# Isolated VLM robustness attack

This directory contains the supported, reduced attack pipeline only.

## Included

- Attack modes: `vlm`, `and`
- Generators: FLUX.2 Klein, Qwen2 Image Edit, Bernini
- Three generator-specific runners and runtimes
- Shared black-box attack core and minimal VLM helpers
- In-memory candidate-image handoff without strip serialization
- 33 supported YAML presets
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

Qwen presets enable experimental single-GPU prompt batching with
`qwen_batch_size: 3`. The reference image is encoded once for the batch,
individual prompt embeddings are padded and combined, and the denoising loop
runs all candidate prompts together. This path requires
`QwenImageEditPlusPipeline`, does not run when `cpu_offload: true`, and
automatically retries the complete candidate set sequentially when
`qwen_batch_fallback: true`. Set `qwen_batch_size: 1` to restore the upstream
sequential behavior.

To attack only images that the selected victim classifies correctly before any
edit, enable the clean filter:

```bash
./isolated_vlm_attack/run_vlm_attack.sh \
  --config configs/flux2_vlm_attack_nips.yaml \
  --attack_only_clean_correct true
```

The same option can be set as `attack_only_clean_correct: true` in YAML or as
`ATTACK_ONLY_CLEAN_CORRECT=1`. `start_index`, `end_index`, `max_samples`, and
`sample_indices` continue to refer to the original dataset indices; incorrect
clean predictions inside that selection are skipped. Filter predictions are
recorded in `run_summary.json` and do not consume `max_victim_queries`.

For the joint `no_explicit_class_guidance` ablation, use:

```bash
./isolated_vlm_attack/run_vlm_attack.sh \
  --config configs/flux2_and_attack_nips_no_explicit_class_guidance.yaml
```

That preset enables `class_ablation: true`. It replaces explicit class
placeholders with the neutral phrase `the subject` and removes the
`Every candidate must be semantically related to the target class` instruction
from the candidate-generation LLM prompt. The reference image, classifier label,
victim objective, and all other attack settings remain unchanged, so this is an
explicit-text ablation rather than a fully class-free attack.

## Code-only verification

The checks below do not load a generator or execute an attack:

```bash
python -m unittest discover -s isolated_vlm_attack/tests -p 'test_attack_refactor.py' -v
bash -n isolated_vlm_attack/run_vlm_attack.sh
RUN_VLM_ATTACK_DRY_RUN=1 ./isolated_vlm_attack/run_vlm_attack.sh
```
