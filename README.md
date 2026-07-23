# ASA source-to-transfer attack pipeline

This repository runs the ASA black-box image attack on a source classifier and,
only after the attack finishes successfully, evaluates the generated images on
the source and transfer classifiers.

The Git checkout contains the code and `data/nips2017` only. Generator weights,
classifier weights, Hugging Face caches, checkpoints, adversarial images, NPZ
bundles, and result files are downloaded or generated on the remote host and
are ignored by Git.

## One-shot FaaS run

Use a Linux GPU image with Bash and Python. Configure the Hugging Face token as
a FaaS secret; never write it into a config file.

```bash
export HF_TOKEN='configured-by-the-faas-secret-manager'
INSTALL_DEPS=1 ./run_faas_source_transfer.sh
```

The default job:

1. attacks `resnet50` on all 1,000 NIPS2017 images with
   `configs/flux2_and_attack_nips.yaml`;
2. downloads generator/classifier weights into ignored cache directories;
3. builds `adversarial_examples.npz` in the ignored run directory;
4. evaluates source ASR and transfer ASR on
   `resnet50,wrn50,inception_v3,convnext,vgg19,vit,swin,deit`;
5. prints all progress and a final TSV/JSON metric summary to standard output;
6. mirrors standard output to `results/<run-name>.txt`.

FaaS consoles that expose only text output can therefore show the full result
without downloading image artifacts.

Common environment overrides:

```bash
SOURCE_MODEL=resnet50 \
TRANSFER_MODELS=resnet50,convnext,swin,deit \
MAX_SAMPLES=100 \
RUN_NAME=asa_resnet50_nips100 \
INSTALL_DEPS=1 \
./run_faas_source_transfer.sh
```

Available settings:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CONFIG` | `configs/flux2_and_attack_nips.yaml` | Attack YAML |
| `SOURCE_MODEL` | `resnet50` | Classifier queried during the attack |
| `TRANSFER_MODELS` | source plus seven downloadable models | Evaluation models |
| `MAX_SAMPLES` | `1000` | Number of NIPS2017 samples |
| `RUN_NAME` | timestamped name | Unique output name |
| `OUTPUT_ROOT` | `outputs` | Ignored attack artifact root |
| `RESULT_DIR` | `results` | Ignored TXT result root |
| `INSTALL_DEPS` | `0` | Install `requirements-faas.txt` when set to `1` |
| `HF_CACHE_ROOT` | `.cache/huggingface` | Ignored Hugging Face cache |
| `NPZ_BATCH_SIZE` | `64` | NPZ conversion batch size |

Run a configuration-only check without loading models:

```bash
FAAS_DRY_RUN=1 ./run_faas_source_transfer.sh
```

## Manual attack

The lower-level launcher supports FLUX.2, Qwen Image Edit, and Bernini runners:

```bash
HF_TOKEN=... CONFIG=configs/flux2_and_attack_nips.yaml ./run_vlm_attack.sh
```

Configuration priority is:

```text
CLI arguments > environment variables > YAML run_args/top-level values > defaults
```

Generated attack images are fixed at the classifier resolution of 224×224.
NPZ construction preserves all sample positions, prefers
`images/vlm_final_selection.png`, falls back to `images/final_selected.png`,
and records query-budget failures in `victim_query_exhausted`.

## Container

The provided Dockerfile targets CUDA 13.0:

```bash
docker compose build
docker compose run --rm asa
```

The Docker build includes `data/nips2017` but excludes all model weights and
runtime outputs. Optional Bernini, Vim, DINOv3, and defense-model checkouts must
be installed separately when those paths are selected.
