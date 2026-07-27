#!/usr/bin/env bash
set -euo pipefail

ISOLATED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ASA_PROJECT_ROOT:-$(cd "$ISOLATED_DIR/.." && pwd)}"
ROOT_DIR="$ISOLATED_DIR"
export ASA_PROJECT_ROOT="$PROJECT_ROOT"
cd "$PROJECT_ROOT"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "ERROR: python interpreter not found." >&2
  exit 1
fi

# The default path is intentionally a supported FLUX.2 Klein configuration.
CONFIG_DEFAULT="configs/flux2_vlm_attack_nips.yaml"
CONFIG_PATH="${CONFIG:-$CONFIG_DEFAULT}"
HF_TOKEN_ENV_NAME="${HF_TOKEN_ENV_NAME:-}"
MODEL_PATH_EXPLICIT=0
if [[ -n "${MODEL_PATH:-}" ]]; then
  MODEL_PATH_EXPLICIT=1
fi

set_from_config() {
  local var_name="$1"
  local var_value="$2"
  if [[ -z "${!var_name:-}" ]]; then
    printf -v "$var_name" '%s' "$var_value"
  fi
}

require_cli_value() {
  local option="$1"
  local index="$2"
  local length="$3"
  if [[ "$index" -ge "$length" ]]; then
    echo "ERROR: $option requires a value." >&2
    exit 1
  fi
}

# Consume launcher-owned options. All other arguments are forwarded to the
# selected runner after the normalized launcher arguments, so CLI remains the
# highest-priority input source.
RAW_ARGS=("$@")
FORWARD_ARGS=()
i=0
while [[ $i -lt ${#RAW_ARGS[@]} ]]; do
  arg="${RAW_ARGS[$i]}"
  case "$arg" in
    --config)
      i=$((i + 1))
      require_cli_value --config "$i" "${#RAW_ARGS[@]}"
      CONFIG_PATH="${RAW_ARGS[$i]}"
      ;;
    --config=*)
      CONFIG_PATH="${arg#--config=}"
      ;;
    --runner|--runner_variant)
      i=$((i + 1))
      require_cli_value "$arg" "$i" "${#RAW_ARGS[@]}"
      RUNNER_VARIANT="${RAW_ARGS[$i]}"
      ;;
    --runner=*|--runner_variant=*)
      RUNNER_VARIANT="${arg#*=}"
      ;;
    --attack_mode)
      i=$((i + 1))
      require_cli_value --attack_mode "$i" "${#RAW_ARGS[@]}"
      ATTACK_MODE="${RAW_ARGS[$i]}"
      ;;
    --attack_mode=*)
      ATTACK_MODE="${arg#--attack_mode=}"
      ;;
    --class_ablation)
      i=$((i + 1))
      require_cli_value --class_ablation "$i" "${#RAW_ARGS[@]}"
      CLASS_ABLATION="${RAW_ARGS[$i]}"
      ;;
    --class_ablation=*)
      CLASS_ABLATION="${arg#--class_ablation=}"
      ;;
    --run_name)
      i=$((i + 1))
      require_cli_value --run_name "$i" "${#RAW_ARGS[@]}"
      RUN_NAME="${RAW_ARGS[$i]}"
      ;;
    --run_name=*)
      RUN_NAME="${arg#--run_name=}"
      ;;
    --model_path)
      i=$((i + 1))
      require_cli_value --model_path "$i" "${#RAW_ARGS[@]}"
      MODEL_PATH="${RAW_ARGS[$i]}"
      MODEL_PATH_EXPLICIT=1
      ;;
    --model_path=*)
      MODEL_PATH="${arg#--model_path=}"
      MODEL_PATH_EXPLICIT=1
      ;;
    --saved_image_size)
      i=$((i + 1))
      require_cli_value --saved_image_size "$i" "${#RAW_ARGS[@]}"
      SAVED_IMAGE_SIZE="${RAW_ARGS[$i]}"
      ;;
    --saved_image_size=*)
      SAVED_IMAGE_SIZE="${arg#--saved_image_size=}"
      ;;
    --inversion_prompt|--inversion_prompt=*|--run_mode|--run_mode=*|--fixed_prompt|--fixed_prompt=*|\
    --latent_nudging_scalar|--latent_nudging_scalar=*|--script_path|--script_path=*|\
    --cwor_*|--flux2_strategy_cwor_*|--gcg_early_stop_on_cwor_success_only|--gcg_early_stop_on_cwor_success_only=*|\
    --save_intermediate|--save_intermediate=*|--save_intermediate_interval|--save_intermediate_interval=*|\
    --save_candidate_strips|--save_candidate_strips=*|--capture_classifier_tile_image|--capture_classifier_tile_image=*|\
    --gcg_save_intermediate|--gcg_save_intermediate=*|--gcg_save_intermediate_interval|--gcg_save_intermediate_interval=*|\
    --gcg_save_candidate_strips|--gcg_save_candidate_strips=*|--gcg_capture_classifier_tile_image|--gcg_capture_classifier_tile_image=*)
      echo "ERROR: removed launcher option is not supported: $arg" >&2
      exit 1
      ;;
    *)
      FORWARD_ARGS+=("$arg")
      ;;
  esac
  i=$((i + 1))
done
set -- "${FORWARD_ARGS[@]}"

if [[ -n "$CONFIG_PATH" && "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$ROOT_DIR/$CONFIG_PATH"
fi
if [[ -n "$CONFIG_PATH" && ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: config file not found: $CONFIG_PATH" >&2
  exit 1
fi
if [[ -n "$CONFIG_PATH" ]]; then
  CONFIG_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"
fi
USING_DEFAULT_CONFIG=0
if [[ "$CONFIG_PATH" == "$ROOT_DIR/$CONFIG_DEFAULT" ]]; then
  USING_DEFAULT_CONFIG=1
fi

# A value may live either at the YAML top level or under run_args. run_args
# wins inside the file; an already-defined environment/CLI variable wins over
# both through set_from_config.
if [[ -n "$CONFIG_PATH" ]]; then
  CONFIG_ENV_OUTPUT="$(
    "$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
import sys
import yaml

path = sys.argv[1]
cfg = yaml.safe_load(open(path, encoding="utf-8")) or {}
if not isinstance(cfg, dict):
    raise SystemExit("config root must be a mapping")
run_args = cfg.get("run_args", {}) or {}
if not isinstance(run_args, dict):
    raise SystemExit("config run_args must be a mapping")

forbidden_keys = {
    "inversion_prompt",
    "run_mode",
    "fixed_prompt",
    "gcg_save_intermediate",
    "gcg_save_intermediate_interval",
    "gcg_save_candidate_strips",
    "gcg_capture_classifier_tile_image",
    "save_intermediate",
    "save_intermediate_interval",
    "save_candidate_strips",
    "capture_classifier_tile_image",
    "gcg_early_stop_on_cwor_success_only",
    "latent_nudging_scalar",
}
for section_name, section in (("top level", cfg), ("run_args", run_args)):
    for key in section:
        key_text = str(key)
        if key_text == "saved_image_size":
            try:
                size = int(section[key])
            except (TypeError, ValueError):
                raise SystemExit(f"saved_image_size must be 224 ({section_name})")
            if size != 224:
                raise SystemExit(f"saved_image_size is fixed at 224 ({section_name}: {size})")
            continue
        if (
            key_text in forbidden_keys
            or key_text.startswith("cwor_")
            or key_text.startswith("flux2_strategy_cwor_")
        ):
            raise SystemExit(f"removed config key is not supported ({section_name}): {key_text}")

mapping = {
    # Launcher and environment.
    "script_path": "RUNNER_SCRIPT",
    "runner": "RUNNER_VARIANT",
    "runner_variant": "RUNNER_VARIANT",
    "cuda_visible_devices": "CUDA_VISIBLE_DEVICES",
    "hf_token": "HF_TOKEN",
    "hf_token_file": "HF_TOKEN_FILE",
    "hf_token_env": "HF_TOKEN_ENV_NAME",
    "hf_cache_root": "HF_CACHE_ROOT",
    # Dataset, attack and classifier.
    "dataset_root": "DATASET_ROOT",
    "dataset_name": "DATASET_NAME",
    "output_root": "OUTPUT_ROOT",
    "run_name": "RUN_NAME",
    "start_index": "START_INDEX",
    "end_index": "END_INDEX",
    "max_samples": "MAX_SAMPLES",
    "sample_indices": "SAMPLE_INDICES",
    "sample_indices_file": "SAMPLE_INDICES_FILE",
    "attack_only_clean_correct": "ATTACK_ONLY_CLEAN_CORRECT",
    "clean_correct_skip": "CLEAN_CORRECT_SKIP",
    "clean_correct_count": "CLEAN_CORRECT_COUNT",
    "image_size": "IMAGE_SIZE",
    "saved_image_size": "SAVED_IMAGE_SIZE",
    "batchsize": "BATCHSIZE",
    "victim_model": "VICTIM_MODEL",
    "device": "DEVICE",
    "classifier_objective": "CLASSIFIER_OBJECTIVE",
    "manual_seed": "MANUAL_SEED",
    "model_path": "MODEL_PATH",
    "prompt": "PROMPT",
    "gcg_word": "GCG_WORD",
    "gcg_occurrence": "GCG_OCCURRENCE",
    "gcg_steps": "GCG_STEPS",
    "gcg_batch_size": "GCG_BATCH_SIZE",
    "max_victim_queries": "MAX_VICTIM_QUERIES",
    "attack_mode": "ATTACK_MODE",
    # Generator-neutral render settings.
    "height": "HEIGHT",
    "width": "WIDTH",
    "num_inference_steps": "NUM_INFERENCE_STEPS",
    "max_sequence_length": "MAX_SEQUENCE_LENGTH",
    "guidance_scale": "GUIDANCE_SCALE",
    "cpu_offload": "CPU_OFFLOAD",
    # Candidate generation and VLM/LLM controls.
    "gcg_scene_vocab_size": "GCG_SCENE_VOCAB_SIZE",
    "gcg_scene_vocab_prompts_per_strategy": "GCG_SCENE_VOCAB_PROMPTS_PER_STRATEGY",
    "gcg_scene_vocab_enabled_strategies": "GCG_SCENE_VOCAB_ENABLED_STRATEGIES",
    "gcg_slot_candidate_max_words": "GCG_SLOT_CANDIDATE_MAX_WORDS",
    "class_ablation": "CLASS_ABLATION",
    "gcg_candidate_source": "GCG_CANDIDATE_SOURCE",
    "gcg_scene_vocab_feedback": "GCG_SCENE_VOCAB_FEEDBACK",
    "gcg_scene_feedback_limit": "GCG_SCENE_FEEDBACK_LIMIT",
    "scene_vlm_question": "SCENE_VLM_QUESTION",
    "scene_fallback": "SCENE_FALLBACK",
    "scene_vlm_backend": "SCENE_VLM_BACKEND",
    "scene_vlm_model_id": "SCENE_VLM_MODEL_ID",
    "scene_vlm_device": "SCENE_VLM_DEVICE",
    "scene_vlm_max_new_tokens": "SCENE_VLM_MAX_NEW_TOKENS",
    "scene_vlm_thinking": "SCENE_VLM_THINKING",
    "scene_vlm_do_sample": "SCENE_VLM_DO_SAMPLE",
    "gcg_scene_llm_backend": "GCG_SCENE_LLM_BACKEND",
    "gcg_scene_llm_model_id": "GCG_SCENE_LLM_MODEL_ID",
    "gcg_scene_llm_device": "GCG_SCENE_LLM_DEVICE",
    "gcg_scene_llm_max_new_tokens": "GCG_SCENE_LLM_MAX_NEW_TOKENS",
    "gcg_scene_llm_thinking": "GCG_SCENE_LLM_THINKING",
    "gcg_scene_llm_do_sample": "GCG_SCENE_LLM_DO_SAMPLE",
    "gcg_early_stop_on_attack_success": "GCG_EARLY_STOP_ON_ATTACK_SUCCESS",
    "gcg_eval_naturalness_on_attack_success": "GCG_EVAL_NATURALNESS_ON_ATTACK_SUCCESS",
    "gcg_eval_naturalness_llm_thinking": "GCG_EVAL_NATURALNESS_LLM_THINKING",
    # Qwen Image Edit.
    "qwen_true_cfg_scale": "QWEN_TRUE_CFG_SCALE",
    "qwen_negative_prompt": "QWEN_NEGATIVE_PROMPT",
    "qwen_num_images_per_prompt": "QWEN_NUM_IMAGES_PER_PROMPT",
    "qwen_batch_size": "QWEN_BATCH_SIZE",
    "qwen_batch_fallback": "QWEN_BATCH_FALLBACK",
    # Bernini.
    "bernini_root": "BERNINI_ROOT",
    "bernini_config": "BERNINI_CONFIG",
    "bernini_high_noise_ckpt": "BERNINI_HIGH_NOISE_CKPT",
    "bernini_low_noise_ckpt": "BERNINI_LOW_NOISE_CKPT",
    "bernini_task_type": "BERNINI_TASK_TYPE",
    "bernini_guidance_mode": "BERNINI_GUIDANCE_MODE",
    "bernini_num_frames": "BERNINI_NUM_FRAMES",
    "bernini_max_image_size": "BERNINI_MAX_IMAGE_SIZE",
    "bernini_use_unipc": "BERNINI_USE_UNIPC",
    "bernini_use_src_tgt_id": "BERNINI_USE_SRC_TGT_ID",
    "bernini_neg_prompt": "BERNINI_NEG_PROMPT",
    "bernini_system_prompt": "BERNINI_SYSTEM_PROMPT",
    "bernini_omega_V": "BERNINI_OMEGA_V",
    "bernini_omega_I": "BERNINI_OMEGA_I",
    "bernini_omega_TI": "BERNINI_OMEGA_TI",
    "bernini_omega_scale": "BERNINI_OMEGA_SCALE",
    "bernini_flow_shift": "BERNINI_FLOW_SHIFT",
    "bernini_fps": "BERNINI_FPS",
    "bernini_eta": "BERNINI_ETA",
    "bernini_norm_threshold": "BERNINI_NORM_THRESHOLD",
    "bernini_momentum": "BERNINI_MOMENTUM",
    "bernini_batch_render": "BERNINI_BATCH_RENDER",
    "bernini_batch_fallback": "BERNINI_BATCH_FALLBACK",
    "bernini_batch_size": "BERNINI_BATCH_SIZE",
    # Tracking credentials and metadata.
    "wandb_enable": "WANDB_ENABLE",
    "wandb_project": "WANDB_PROJECT",
    "wandb_entity": "WANDB_ENTITY",
    "wandb_group": "WANDB_GROUP",
    "wandb_tags": "WANDB_TAGS",
    "wandb_mode": "WANDB_MODE",
    "wandb_log_every": "WANDB_LOG_EVERY",
    "wandb_api_key": "WANDB_API_KEY",
    "wandb_api_key_file": "WANDB_API_KEY_FILE",
}

def normalize(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value).replace("\r", " ").replace("\n", " ")

for source_key, env_key in mapping.items():
    value = run_args[source_key] if source_key in run_args else cfg.get(source_key)
    value = normalize(value)
    if value is not None:
        print(f"{env_key}\t{value}")
PY
  )"
  while IFS=$'\t' read -r key value; do
    key="${key//$'\r'/}"
    value="${value//$'\r'/}"
    [[ -z "$key" ]] && continue
    set_from_config "$key" "$value"
  done <<< "$CONFIG_ENV_OUTPUT"
fi

# Normalize the runner selector, then enforce an exact script whitelist. A
# script_path containing a directory is deliberately rejected.
RUNNER_VARIANT="${RUNNER_VARIANT:-${RUNNER_KIND:-}}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-}"
case "${RUNNER_VARIANT,,}" in
  "") ;;
  flux2|flux2_klein) RUNNER_SCRIPT="flux2_attack_runner.py" ;;
  qwen2|qwen2_image_edit|qwen-image-edit) RUNNER_SCRIPT="qwen2_attack_runner.py" ;;
  bernini) RUNNER_SCRIPT="bernini_attack_runner.py" ;;
  *)
    echo "ERROR: runner must be one of: flux2, qwen2, bernini (got '$RUNNER_VARIANT')." >&2
    exit 1
    ;;
esac
RUNNER_SCRIPT="${RUNNER_SCRIPT:-flux2_attack_runner.py}"
case "$RUNNER_SCRIPT" in
  flux2_attack_runner.py|qwen2_attack_runner.py|bernini_attack_runner.py) ;;
  *)
    echo "ERROR: runner script is not allowed: $RUNNER_SCRIPT" >&2
    echo "Allowed: flux2_attack_runner.py, qwen2_attack_runner.py, bernini_attack_runner.py" >&2
    exit 1
    ;;
esac

DATASET_ROOT="${DATASET_ROOT:-data/nips2017}"
DATASET_NAME="${DATASET_NAME:-nips2017}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
RUN_NAME="${RUN_NAME:-}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
SAMPLE_INDICES="${SAMPLE_INDICES:-}"
SAMPLE_INDICES_FILE="${SAMPLE_INDICES_FILE:-}"
ATTACK_ONLY_CLEAN_CORRECT="${ATTACK_ONLY_CLEAN_CORRECT:-0}"
CLEAN_CORRECT_SKIP="${CLEAN_CORRECT_SKIP:-}"
CLEAN_CORRECT_COUNT="${CLEAN_CORRECT_COUNT:-}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
BATCHSIZE="${BATCHSIZE:-1}"
VICTIM_MODEL="${VICTIM_MODEL:-resnet50}"
DEVICE="${DEVICE:-cuda}"
CLASSIFIER_OBJECTIVE="${CLASSIFIER_OBJECTIVE:-ce_max}"
MANUAL_SEED="${MANUAL_SEED:-}"

case "${ATTACK_ONLY_CLEAN_CORRECT,,}" in
  1|true|yes|on) ATTACK_ONLY_CLEAN_CORRECT="1" ;;
  0|false|no|off|'') ATTACK_ONLY_CLEAN_CORRECT="0" ;;
  *)
    echo "ERROR: ATTACK_ONLY_CLEAN_CORRECT must be boolean-like (got '$ATTACK_ONLY_CLEAN_CORRECT')." >&2
    exit 1
    ;;
esac

ATTACK_MODE="${ATTACK_MODE:-vlm}"
case "${ATTACK_MODE,,}" in
  vlm|and) ATTACK_MODE="${ATTACK_MODE,,}" ;;
  *)
    echo "ERROR: attack_mode must be vlm or and (got '$ATTACK_MODE')." >&2
    exit 1
    ;;
esac

SAVED_IMAGE_SIZE="${SAVED_IMAGE_SIZE:-224}"
if [[ "$SAVED_IMAGE_SIZE" != "224" ]]; then
  echo "ERROR: saved_image_size is fixed at 224 (got '$SAVED_IMAGE_SIZE')." >&2
  exit 1
fi

PROMPT="${PROMPT:-a photo of <class> in the background}"
GCG_WORD="${GCG_WORD:-background}"
GCG_OCCURRENCE="${GCG_OCCURRENCE:-0}"
GCG_STEPS="${GCG_STEPS:-10}"
GCG_BATCH_SIZE="${GCG_BATCH_SIZE:-64}"
MAX_VICTIM_QUERIES="${MAX_VICTIM_QUERIES:-100}"
HEIGHT="${HEIGHT:-}"
WIDTH="${WIDTH:-}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-}"
CPU_OFFLOAD="${CPU_OFFLOAD:-}"

case "$RUNNER_SCRIPT" in
  flux2_attack_runner.py)
    MODEL_PATH="${MODEL_PATH:-black-forest-labs/FLUX.2-klein-9B}"
    model_token="${MODEL_PATH,,}"
    case "$model_token" in
      black-forest-labs/flux.2-klein-4b|black-forest-labs/flux.2-klein-9b|black-forest-labs/flux.2-klein-4b-kv|black-forest-labs/flux.2-klein-9b-kv) ;;
      *)
        echo "ERROR: flux2 runner requires a registered FLUX.2 Klein model_path (got '$MODEL_PATH')." >&2
        exit 1
        ;;
    esac
    RUNNER_LABEL="flux2"
    ;;
  qwen2_attack_runner.py)
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen-Image-Edit-2511}"
    model_token="${MODEL_PATH,,}"
    if [[ "$model_token" != "qwen/qwen-image-edit-2511" ]]; then
      if [[ "$USING_DEFAULT_CONFIG" == "1" && "$MODEL_PATH_EXPLICIT" == "0" ]]; then
        MODEL_PATH="Qwen/Qwen-Image-Edit-2511"
      else
        echo "ERROR: qwen2 runner requires a Qwen Image Edit model_path (got '$MODEL_PATH')." >&2
        exit 1
      fi
    fi
    RUNNER_LABEL="qwen2"
    ;;
  bernini_attack_runner.py)
    MODEL_PATH="${MODEL_PATH:-bernini}"
    if [[ "${MODEL_PATH,,}" != "bernini" ]]; then
      if [[ "$USING_DEFAULT_CONFIG" == "1" && "$MODEL_PATH_EXPLICIT" == "0" ]]; then
        MODEL_PATH="bernini"
      else
        echo "ERROR: Bernini uses model_path=bernini; select the checkpoint with bernini_config." >&2
        exit 1
      fi
    fi
    RUNNER_LABEL="bernini"
    ;;
esac

if [[ -z "$RUN_NAME" || "${RUN_NAME,,}" == "auto" ]]; then
  RUN_NAME="${RUNNER_LABEL}_${ATTACK_MODE}_$(date +%Y%m%d_%H%M%S)"
fi

GCG_SCENE_VOCAB_SIZE="${GCG_SCENE_VOCAB_SIZE:-100}"
GCG_SCENE_VOCAB_PROMPTS_PER_STRATEGY="${GCG_SCENE_VOCAB_PROMPTS_PER_STRATEGY:-0}"
GCG_SCENE_VOCAB_ENABLED_STRATEGIES="${GCG_SCENE_VOCAB_ENABLED_STRATEGIES:-all}"
GCG_SLOT_CANDIDATE_MAX_WORDS="${GCG_SLOT_CANDIDATE_MAX_WORDS:-5}"
CLASS_ABLATION="${CLASS_ABLATION:-0}"
GCG_SCENE_VOCAB_FEEDBACK="${GCG_SCENE_VOCAB_FEEDBACK:-1}"
GCG_SCENE_FEEDBACK_LIMIT="${GCG_SCENE_FEEDBACK_LIMIT:-1000}"
GCG_CANDIDATE_SOURCE="${GCG_CANDIDATE_SOURCE:-}"
GCG_EVAL_NATURALNESS_ON_ATTACK_SUCCESS="${GCG_EVAL_NATURALNESS_ON_ATTACK_SUCCESS:-0}"
GCG_EVAL_NATURALNESS_LLM_THINKING="${GCG_EVAL_NATURALNESS_LLM_THINKING:-0}"
case "${CLASS_ABLATION,,}" in
  1|true|yes|on) CLASS_ABLATION="1" ;;
  0|false|no|off|'') CLASS_ABLATION="0" ;;
  *)
    echo "ERROR: CLASS_ABLATION must be boolean-like (got '$CLASS_ABLATION')." >&2
    exit 1
    ;;
esac
if [[ "$ATTACK_MODE" == "and" ]]; then
  GCG_CANDIDATE_SOURCE="gemma_scene_vocab"
  if [[ "$GCG_SCENE_VOCAB_PROMPTS_PER_STRATEGY" == "0" ]]; then
    GCG_SCENE_VOCAB_PROMPTS_PER_STRATEGY="1"
  fi
elif [[ -z "$GCG_CANDIDATE_SOURCE" ]]; then
  if [[ "$RUNNER_SCRIPT" == "qwen2_attack_runner.py" ]]; then
    GCG_CANDIDATE_SOURCE="gemma_scene_vocab"
  else
    GCG_CANDIDATE_SOURCE="vlm_query"
  fi
fi

SCENE_VLM_QUESTION="${SCENE_VLM_QUESTION:-What is the background scene in this image? Answer in 1 word.}"
SCENE_FALLBACK="${SCENE_FALLBACK:-outdoor}"
WANDB_ENABLE="${WANDB_ENABLE:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-vlm-attack}"
WANDB_MODE="${WANDB_MODE:-auto}"
WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-1}"

for numeric_name in IMAGE_SIZE BATCHSIZE GCG_STEPS GCG_BATCH_SIZE MAX_VICTIM_QUERIES GCG_SCENE_VOCAB_SIZE GCG_SCENE_VOCAB_PROMPTS_PER_STRATEGY GCG_SLOT_CANDIDATE_MAX_WORDS GCG_SCENE_FEEDBACK_LIMIT; do
  numeric_value="${!numeric_name}"
  if ! [[ "$numeric_value" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $numeric_name must be a non-negative integer (got '$numeric_value')." >&2
    exit 1
  fi
done
if [[ "$IMAGE_SIZE" -lt 1 || "$BATCHSIZE" -lt 1 || "$GCG_STEPS" -lt 1 || "$GCG_BATCH_SIZE" -lt 1 || "$MAX_VICTIM_QUERIES" -lt 1 ]]; then
  echo "ERROR: image_size, batchsize, gcg_steps, gcg_batch_size and max_victim_queries must be >= 1." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
CUDA_VISIBLE_DEVICES="$(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr '; ' ',,' | tr -s ',' | sed 's/^,//; s/,$//')"
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
  if ! [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES must be comma-separated GPU indices (got '$CUDA_VISIBLE_DEVICES')." >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES
fi

if [[ -n "$HF_TOKEN_ENV_NAME" ]]; then
  if ! [[ "$HF_TOKEN_ENV_NAME" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    echo "ERROR: invalid hf_token_env variable name: $HF_TOKEN_ENV_NAME" >&2
    exit 1
  fi
  if [[ -z "${HF_TOKEN:-}" && -v "$HF_TOKEN_ENV_NAME" ]]; then
    HF_TOKEN="${!HF_TOKEN_ENV_NAME}"
  fi
fi
if [[ -z "${HF_TOKEN:-}" && -n "${HF_TOKEN_FILE:-}" ]]; then
  token_file="$HF_TOKEN_FILE"
  [[ "$token_file" != /* ]] && token_file="$ROOT_DIR/$token_file"
  if [[ ! -f "$token_file" ]]; then
    echo "ERROR: HF token file not found: $token_file" >&2
    exit 1
  fi
  HF_TOKEN="$(tr -d '\r\n' < "$token_file")"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

if [[ -z "${WANDB_API_KEY:-}" && -n "${WANDB_API_KEY_FILE:-}" ]]; then
  wandb_key_file="$WANDB_API_KEY_FILE"
  [[ "$wandb_key_file" != /* ]] && wandb_key_file="$ROOT_DIR/$wandb_key_file"
  if [[ ! -f "$wandb_key_file" ]]; then
    echo "ERROR: W&B API key file not found: $wandb_key_file" >&2
    exit 1
  fi
  WANDB_API_KEY="$(tr -d '\r\n' < "$wandb_key_file")"
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
fi

HF_CACHE_ROOT="${HF_CACHE_ROOT:-$PROJECT_ROOT/.cache/huggingface}"
HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_CACHE_ROOT/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_CACHE_ROOT/transformers}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
export HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$ISOLATED_DIR:$PROJECT_ROOT:$PYTHONPATH"
else
  export PYTHONPATH="$ISOLATED_DIR:$PROJECT_ROOT"
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./isolated_vlm_attack/run_vlm_attack.sh [--config FILE] [--runner flux2|qwen2|bernini] [runner arguments]

Supported attack modes:
  --attack_mode vlm|and

Configuration priority:
  CLI arguments > environment variables > YAML run_args/top-level values > defaults

Core environment overrides:
  CONFIG, RUNNER_VARIANT, CUDA_VISIBLE_DEVICES
  DATASET_ROOT, DATASET_NAME, OUTPUT_ROOT, RUN_NAME
  START_INDEX, END_INDEX, MAX_SAMPLES, SAMPLE_INDICES, SAMPLE_INDICES_FILE
  ATTACK_ONLY_CLEAN_CORRECT, CLEAN_CORRECT_SKIP, CLEAN_CORRECT_COUNT
  IMAGE_SIZE, BATCHSIZE, VICTIM_MODEL, DEVICE, CLASSIFIER_OBJECTIVE
  ATTACK_MODE, PROMPT, GCG_WORD, GCG_OCCURRENCE, GCG_STEPS
  GCG_BATCH_SIZE, MAX_VICTIM_QUERIES, MODEL_PATH, CLASS_ABLATION
  GCG_EVAL_NATURALNESS_ON_ATTACK_SUCCESS, GCG_EVAL_NATURALNESS_LLM_THINKING
  HEIGHT, WIDTH, NUM_INFERENCE_STEPS, MAX_SEQUENCE_LENGTH
  GUIDANCE_SCALE, CPU_OFFLOAD
  HF_TOKEN, HF_TOKEN_FILE, HF_TOKEN_ENV_NAME, HF_CACHE_ROOT
  WANDB_ENABLE, WANDB_PROJECT, WANDB_ENTITY, WANDB_GROUP, WANDB_TAGS
  WANDB_MODE, WANDB_LOG_EVERY, WANDB_API_KEY, WANDB_API_KEY_FILE

Generated attack images are stored at 224x224. The launcher does not expose
legacy inversion, fixed-prompt, IC, standalone CWOR/GS, or intermediate-image options.
EOF
  exit 0
fi

CMD=(
  "$PYTHON_BIN" "$ISOLATED_DIR/$RUNNER_SCRIPT"
  --dataset_root "$DATASET_ROOT"
  --dataset_name "$DATASET_NAME"
  --output_root "$OUTPUT_ROOT"
  --run_name "$RUN_NAME"
  --start_index "$START_INDEX"
  --attack_only_clean_correct "$ATTACK_ONLY_CLEAN_CORRECT"
  --image_size "$IMAGE_SIZE"
  --batchsize "$BATCHSIZE"
  --victim_model "$VICTIM_MODEL"
  --device "$DEVICE"
  --classifier_objective "$CLASSIFIER_OBJECTIVE"
  --attack_mode "$ATTACK_MODE"
  --model_path "$MODEL_PATH"
  --prompt "$PROMPT"
  --gcg_word "$GCG_WORD"
  --gcg_occurrence "$GCG_OCCURRENCE"
  --gcg_steps "$GCG_STEPS"
  --gcg_batch_size "$GCG_BATCH_SIZE"
  --max_victim_queries "$MAX_VICTIM_QUERIES"
  --gcg_scene_vocab_size "$GCG_SCENE_VOCAB_SIZE"
  --gcg_scene_vocab_prompts_per_strategy "$GCG_SCENE_VOCAB_PROMPTS_PER_STRATEGY"
  --gcg_scene_vocab_enabled_strategies "$GCG_SCENE_VOCAB_ENABLED_STRATEGIES"
  --gcg_slot_candidate_max_words "$GCG_SLOT_CANDIDATE_MAX_WORDS"
  --class_ablation "$CLASS_ABLATION"
  --gcg_candidate_source "$GCG_CANDIDATE_SOURCE"
  --gcg_scene_feedback_limit "$GCG_SCENE_FEEDBACK_LIMIT"
  --scene_vlm_question "$SCENE_VLM_QUESTION"
  --scene_fallback "$SCENE_FALLBACK"
  --saved_image_size 224
  --wandb_enable "$WANDB_ENABLE"
  --wandb_project "$WANDB_PROJECT"
  --wandb_mode "$WANDB_MODE"
  --wandb_log_every "$WANDB_LOG_EVERY"
)

if [[ -n "$END_INDEX" ]]; then
  CMD+=(--end_index "$END_INDEX")
else
  CMD+=(--max_samples "$MAX_SAMPLES")
fi
[[ -n "$SAMPLE_INDICES" ]] && CMD+=(--sample_indices "$SAMPLE_INDICES")
[[ -n "$SAMPLE_INDICES_FILE" ]] && CMD+=(--sample_indices_file "$SAMPLE_INDICES_FILE")
[[ -n "$MANUAL_SEED" ]] && CMD+=(--manual_seed "$MANUAL_SEED")
[[ -n "$HEIGHT" ]] && CMD+=(--height "$HEIGHT")
[[ -n "$WIDTH" ]] && CMD+=(--width "$WIDTH")
[[ -n "$NUM_INFERENCE_STEPS" ]] && CMD+=(--num_inference_steps "$NUM_INFERENCE_STEPS")
[[ -n "$MAX_SEQUENCE_LENGTH" ]] && CMD+=(--max_sequence_length "$MAX_SEQUENCE_LENGTH")
[[ -n "$GUIDANCE_SCALE" ]] && CMD+=(--guidance_scale "$GUIDANCE_SCALE")
[[ -n "$CPU_OFFLOAD" ]] && CMD+=(--cpu_offload "$CPU_OFFLOAD")

case "${GCG_SCENE_VOCAB_FEEDBACK,,}" in
  1|true|yes|on) CMD+=(--gcg_scene_vocab_feedback 1) ;;
  0|false|no|off|'') CMD+=(--gcg_scene_vocab_feedback 0) ;;
  *)
    echo "ERROR: GCG_SCENE_VOCAB_FEEDBACK must be boolean-like (got '$GCG_SCENE_VOCAB_FEEDBACK')." >&2
    exit 1
    ;;
esac

append_optional() {
  local var_name="$1"
  local option="$2"
  local value="${!var_name:-}"
  if [[ -n "$value" ]]; then
    CMD+=("$option" "$value")
  fi
  return 0
}

append_optional SCENE_VLM_BACKEND --scene_vlm_backend
append_optional SCENE_VLM_MODEL_ID --scene_vlm_model_id
append_optional SCENE_VLM_DEVICE --scene_vlm_device
append_optional SCENE_VLM_MAX_NEW_TOKENS --scene_vlm_max_new_tokens
append_optional SCENE_VLM_THINKING --scene_vlm_thinking
append_optional SCENE_VLM_DO_SAMPLE --scene_vlm_do_sample
append_optional GCG_SCENE_LLM_BACKEND --gcg_scene_llm_backend
append_optional GCG_SCENE_LLM_MODEL_ID --gcg_scene_llm_model_id
append_optional GCG_SCENE_LLM_DEVICE --gcg_scene_llm_device
append_optional GCG_SCENE_LLM_MAX_NEW_TOKENS --gcg_scene_llm_max_new_tokens
append_optional GCG_SCENE_LLM_THINKING --gcg_scene_llm_thinking
append_optional GCG_SCENE_LLM_DO_SAMPLE --gcg_scene_llm_do_sample
append_optional GCG_EARLY_STOP_ON_ATTACK_SUCCESS --gcg_early_stop_on_attack_success
append_optional GCG_EVAL_NATURALNESS_ON_ATTACK_SUCCESS --gcg_eval_naturalness_on_attack_success
append_optional GCG_EVAL_NATURALNESS_LLM_THINKING --gcg_eval_naturalness_llm_thinking
append_optional WANDB_ENTITY --wandb_entity
append_optional WANDB_GROUP --wandb_group
append_optional WANDB_TAGS --wandb_tags

case "$RUNNER_SCRIPT" in
  qwen2_attack_runner.py)
    append_optional CLEAN_CORRECT_SKIP --clean_correct_skip
    append_optional CLEAN_CORRECT_COUNT --clean_correct_count
    append_optional QWEN_TRUE_CFG_SCALE --qwen_true_cfg_scale
    append_optional QWEN_NEGATIVE_PROMPT --qwen_negative_prompt
    append_optional QWEN_NUM_IMAGES_PER_PROMPT --qwen_num_images_per_prompt
    append_optional QWEN_BATCH_SIZE --qwen_batch_size
    append_optional QWEN_BATCH_FALLBACK --qwen_batch_fallback
    ;;
  bernini_attack_runner.py)
    append_optional BERNINI_ROOT --bernini_root
    append_optional BERNINI_CONFIG --bernini_config
    append_optional BERNINI_HIGH_NOISE_CKPT --bernini_high_noise_ckpt
    append_optional BERNINI_LOW_NOISE_CKPT --bernini_low_noise_ckpt
    append_optional BERNINI_TASK_TYPE --bernini_task_type
    append_optional BERNINI_GUIDANCE_MODE --bernini_guidance_mode
    append_optional BERNINI_NUM_FRAMES --bernini_num_frames
    append_optional BERNINI_MAX_IMAGE_SIZE --bernini_max_image_size
    append_optional BERNINI_USE_UNIPC --bernini_use_unipc
    append_optional BERNINI_USE_SRC_TGT_ID --bernini_use_src_tgt_id
    append_optional BERNINI_NEG_PROMPT --bernini_neg_prompt
    append_optional BERNINI_SYSTEM_PROMPT --bernini_system_prompt
    append_optional BERNINI_OMEGA_V --bernini_omega_V
    append_optional BERNINI_OMEGA_I --bernini_omega_I
    append_optional BERNINI_OMEGA_TI --bernini_omega_TI
    append_optional BERNINI_OMEGA_SCALE --bernini_omega_scale
    append_optional BERNINI_FLOW_SHIFT --bernini_flow_shift
    append_optional BERNINI_FPS --bernini_fps
    append_optional BERNINI_ETA --bernini_eta
    if [[ -n "${BERNINI_NORM_THRESHOLD:-}" ]]; then
      read -r -a bernini_threshold_values <<< "${BERNINI_NORM_THRESHOLD//,/ }"
      CMD+=(--bernini_norm_threshold "${bernini_threshold_values[@]}")
    fi
    append_optional BERNINI_MOMENTUM --bernini_momentum
    append_optional BERNINI_BATCH_RENDER --bernini_batch_render
    append_optional BERNINI_BATCH_FALLBACK --bernini_batch_fallback
    append_optional BERNINI_BATCH_SIZE --bernini_batch_size
    ;;
esac

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

echo "[run_vlm_attack] runner=$RUNNER_SCRIPT model=$MODEL_PATH attack_mode=$ATTACK_MODE"
echo "[run_vlm_attack] dataset=$DATASET_ROOT victim=$VICTIM_MODEL run_name=$RUN_NAME"
echo "[run_vlm_attack] config=${CONFIG_PATH:-none} saved_image_size=224"

case "${RUN_VLM_ATTACK_DRY_RUN:-0}" in
  1|true|yes|on)
    DRY_RUN_CMD=("${CMD[@]}")
    mask_next=0
    for ((i = 0; i < ${#DRY_RUN_CMD[@]}; i++)); do
      if [[ "$mask_next" == "1" ]]; then
        DRY_RUN_CMD[$i]="<redacted>"
        mask_next=0
        continue
      fi
      case "${DRY_RUN_CMD[$i]}" in
        --hf_token|--wandb_api_key) mask_next=1 ;;
        --hf_token=*|--wandb_api_key=*)
          DRY_RUN_CMD[$i]="${DRY_RUN_CMD[$i]%%=*}=<redacted>"
          ;;
      esac
    done
    printf '[run_vlm_attack] dry_run command:'
    printf ' %q' "${DRY_RUN_CMD[@]}"
    printf '\n'
    exit 0
    ;;
esac

exec "${CMD[@]}"
