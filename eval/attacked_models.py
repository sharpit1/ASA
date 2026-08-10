import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pytorch_fid.fid_score as fid_score
import timm
import torch
import torch.nn as nn
import torchvision.models as models
from art.estimators.classification import PyTorchClassifier
from timm import create_model
from transformers import AutoModelForImageClassification

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from torch_nets import (
    tf2torch_adv_inception_v3,
    tf2torch_ens_adv_inc_res_v2,
)


warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_VIM_ROOT_CANDIDATES = (
    _REPO_ROOT / "third_party" / "Vim",
    _REPO_ROOT.parent / "Natural-Color-Fool" / "third_party" / "Vim",
)


def _resolve_vim_root():
    for candidate in _VIM_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate
    return _VIM_ROOT_CANDIDATES[0]


_VIM_ROOT = _resolve_vim_root()
_VIM_MODEL_ROOT = _VIM_ROOT / "vim"
_VIM_MAMBA_ROOT = _VIM_ROOT / "mamba-1p1p1"
_VIM_CAUSAL_CONV_ROOT = _VIM_ROOT / "causal-conv1d"
_CKPT_ROOT = _REPO_ROOT / "ckpt"
_TORCH_HUB_ROOT = _CKPT_ROOT / "torch_hub"
torch.hub.set_dir(str(_TORCH_HUB_ROOT))

_DINOV3_ROOT_ENV = "DINOV3_REPO_DIR"
_DINOV3_WEIGHT_ENVS = ("DINOV3_VIT7B16_LC_WEIGHTS", "DINOV3_WEIGHTS")
_DINOV3_BACKBONE_WEIGHT_ENVS = (
    "DINOV3_VIT7B16_LC_BACKBONE_WEIGHTS",
    "DINOV3_VIT7B16_BACKBONE_WEIGHTS",
    "DINOV3_BACKBONE_WEIGHTS",
)
_DINOV2_ROOT_ENV = "DINOV2_REPO_DIR"
_DINOV1_ROOT_ENV = "DINOV1_REPO_DIR"
_DINOV1_LINEAR_WEIGHT_ENV = "DINOV1_VITB16_LINEAR_WEIGHTS"
_DINOV1_FEATURE_DIM_ENV = "DINOV1_VITB16_FEATURE_DIM"
_DINOV1_VITBASE16_LINEAR_URL = (
    "https://dl.fbaipublicfiles.com/dino/"
    "dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth"
)
_DINOV1_VITBASE16_LINEAR_FILENAME = "dino_vitbase16_linearweights.pth"
_DINOV3_LC_WEIGHT_CANDIDATES = (
    "dinov3_vit7b16_lc.pth",
    "dinov3_vit7b16_lc.pt",
    "dinov3_vit7b16_lc.ckpt",
    "dinov3_vit7b16_lc.safetensors",
    "dinov3_vit7b16_linear_head.pth",
    "dinov3_vit7b16_lc_head.pth",
)
_DINOV3_BACKBONE_WEIGHT_CANDIDATES = (
    "dinov3_vit7b16_backbone.pth",
    "dinov3_vit7b16_backbone.pt",
    "dinov3_vit7b16_backbone.ckpt",
    "dinov3_vit7b16_backbone.safetensors",
    "dinov3_vit7b16_pretrain.pth",
    "dinov3_vit7b16.pth",
)
_DINOV1_LINEAR_WEIGHT_CANDIDATES = (
    "dino/dino_vitbase16_linearweights.pth",
    "dinov1_vitb16_linear.pth",
    "dinov1_vitb16_linear.pt",
    "dinov1_vitb16_linear.ckpt",
    "dino_vitb16_linear.pth",
)


class _LogitsOnlyWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        if hasattr(output, "logits"):
            return output.logits
        if isinstance(output, dict):
            if "logits" in output:
                return output["logits"]
            if len(output) == 1:
                return next(iter(output.values()))
        if isinstance(output, (tuple, list)) and output:
            return output[0]
        return output


class _BackboneLinearClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        *,
        n_last_blocks: int = 0,
        avgpool_patchtokens: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.n_last_blocks = int(n_last_blocks)
        self.avgpool_patchtokens = bool(avgpool_patchtokens)

    def forward(self, x):
        if self.n_last_blocks > 0 and hasattr(self.backbone, "get_intermediate_layers"):
            intermediate = self.backbone.get_intermediate_layers(x, self.n_last_blocks)
            features = torch.cat([layer[:, 0] for layer in intermediate], dim=-1)
            if self.avgpool_patchtokens:
                patch_avg = torch.mean(intermediate[-1][:, 1:], dim=1)
                features = torch.cat((features, patch_avg), dim=-1)
            return self.classifier(features)

        features = self.backbone(x)
        if hasattr(features, "logits"):
            features = features.logits
        elif isinstance(features, dict):
            for key in ("x_norm_clstoken", "cls_token", "features"):
                if key in features:
                    features = features[key]
                    break
            else:
                features = next(iter(features.values()))
        elif isinstance(features, (tuple, list)) and features:
            features = features[0]
        if features.ndim > 2:
            features = features.flatten(1)
        return self.classifier(features)


@contextmanager
def _prepend_import_paths(*paths: Path):
    import sys

    original_sys_path = list(sys.path)
    try:
        for path in reversed(paths):
            path_str = str(path.resolve())
            if path_str in sys.path:
                sys.path.remove(path_str)
            sys.path.insert(0, path_str)
        yield
    finally:
        sys.path[:] = original_sys_path


def _resolve_vim_checkpoint(filename):
    candidate_dirs = [
        _VIM_MODEL_ROOT / "ckpts",
        _VIM_MODEL_ROOT / "ckpt",
    ]
    for directory in candidate_dirs:
        path = directory / filename
        if path.exists():
            return path
    return candidate_dirs[0] / filename


def _is_url(value: str) -> bool:
    parsed = urlparse(str(value))
    return bool(parsed.scheme and parsed.netloc)


def _resolve_path_or_url(value: str) -> str:
    value = str(value).strip()
    if not value or _is_url(value):
        return value
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return str(path)


def _first_env_value(names):
    for env_name in names:
        value = os_environ_get(env_name)
        if value:
            return _resolve_path_or_url(value)
    return None


def _first_existing_ckpt(filenames):
    for filename in filenames:
        path = _CKPT_ROOT / filename
        if path.is_file():
            return str(path)
    return None


def _format_ckpt_candidates(filenames):
    return ", ".join(f"ckpt/{filename}" for filename in filenames)


def _torch_hub_load(repo_or_dir: str, model_name: str, *, local_repo_env: str = "", **kwargs):
    local_repo = os_environ_get(local_repo_env) if local_repo_env else ""
    if local_repo:
        repo_dir = Path(_resolve_path_or_url(local_repo))
        if not repo_dir.is_dir():
            raise FileNotFoundError(f"{local_repo_env} does not exist or is not a directory: {repo_dir}")
        return torch.hub.load(str(repo_dir), model_name, source="local", **kwargs)
    return torch.hub.load(repo_or_dir, model_name, trust_repo=True, **kwargs)


def os_environ_get(name: str) -> str:
    return str(os.environ.get(name, "")).strip()


def _resolve_dinov3_repo_dir() -> Path:
    env_value = os_environ_get(_DINOV3_ROOT_ENV)
    if env_value:
        repo_dir = Path(_resolve_path_or_url(env_value))
        if repo_dir.is_dir():
            return repo_dir
        raise FileNotFoundError(
            f"{_DINOV3_ROOT_ENV} does not exist or is not a directory: {repo_dir}"
        )

    candidates = (
        _REPO_ROOT / "third_party" / "dinov3",
        _REPO_ROOT / "third_party" / "DINOv3",
        _REPO_ROOT.parent / "dinov3",
        _REPO_ROOT.parent / "DINOv3",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"DINOv3 local repo not found. Set {_DINOV3_ROOT_ENV} or place it under one of: {searched}"
    )


def get_dinov3_vit7b16_lc_model():
    weights = _first_env_value(_DINOV3_WEIGHT_ENVS) or _first_existing_ckpt(
        _DINOV3_LC_WEIGHT_CANDIDATES
    )
    backbone_weights = _first_env_value(_DINOV3_BACKBONE_WEIGHT_ENVS) or _first_existing_ckpt(
        _DINOV3_BACKBONE_WEIGHT_CANDIDATES
    )
    hub_kwargs = {}
    if weights:
        hub_kwargs["weights"] = weights
    if backbone_weights:
        hub_kwargs["backbone_weights"] = backbone_weights
    model = torch.hub.load(
        str(_resolve_dinov3_repo_dir()),
        "dinov3_vit7b16_lc",
        source="local",
        **hub_kwargs,
    )
    return _LogitsOnlyWrapper(model)


def get_dinov2_model(*, with_registers: bool = False):
    hub_name = "dinov2_vitb14_reg_lc" if with_registers else "dinov2_vitb14_lc"
    model = _torch_hub_load(
        "facebookresearch/dinov2",
        hub_name,
        local_repo_env=_DINOV2_ROOT_ENV,
    )
    return _LogitsOnlyWrapper(model)


def _linear_state_dict_from_checkpoint(checkpoint, weight_source: str):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "classifier"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported DINOv1 linear classifier checkpoint: {weight_source}")

    state_dict = {}
    for key, value in checkpoint.items():
        normalized_key = str(key)
        for prefix in ("module.", "linear.", "classifier."):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix):]
        state_dict[normalized_key] = value
    return state_dict


def _load_linear_classifier_state_dict(classifier: nn.Module, state_dict) -> None:
    load_target = classifier
    if not any(key.startswith("linear.") for key in state_dict) and hasattr(classifier, "linear"):
        load_target = classifier.linear
    load_target.load_state_dict(state_dict, strict=True)


def _load_linear_classifier_weights(classifier: nn.Module, weight_path: str) -> None:
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    state_dict = _linear_state_dict_from_checkpoint(checkpoint, weight_path)
    _load_linear_classifier_state_dict(classifier, state_dict)


def _download_dino_vitbase16_linear_weights():
    model_dir = _CKPT_ROOT / "dino"
    model_dir.mkdir(parents=True, exist_ok=True)
    return torch.hub.load_state_dict_from_url(
        _DINOV1_VITBASE16_LINEAR_URL,
        model_dir=str(model_dir),
        file_name=_DINOV1_VITBASE16_LINEAR_FILENAME,
        map_location="cpu",
    )


def get_dinov1_vitb16_model():
    linear_weights = os_environ_get(_DINOV1_LINEAR_WEIGHT_ENV)
    linear_weights = _resolve_path_or_url(linear_weights) if linear_weights else None
    linear_weights = linear_weights or _first_existing_ckpt(_DINOV1_LINEAR_WEIGHT_CANDIDATES)
    feature_dim = int(os_environ_get(_DINOV1_FEATURE_DIM_ENV) or "1536")
    backbone = _torch_hub_load(
        "facebookresearch/dino:main",
        "dino_vitb16",
        local_repo_env=_DINOV1_ROOT_ENV,
    )
    classifier = nn.Linear(feature_dim, 1000)
    if linear_weights:
        _load_linear_classifier_weights(classifier, linear_weights)
    else:
        checkpoint = _download_dino_vitbase16_linear_weights()
        state_dict = _linear_state_dict_from_checkpoint(
            checkpoint,
            f"{_DINOV1_VITBASE16_LINEAR_URL} -> ckpt/dino/{_DINOV1_VITBASE16_LINEAR_FILENAME}",
        )
        _load_linear_classifier_state_dict(classifier, state_dict)
    return _BackboneLinearClassifier(
        backbone,
        classifier,
        n_last_blocks=1,
        avgpool_patchtokens=True,
    )


def get_vim_model(type="small", *, use_local_mamba: bool = False):
    if not _VIM_MODEL_ROOT.exists():
        searched_paths = ", ".join(str(path) for path in _VIM_ROOT_CANDIDATES)
        raise FileNotFoundError(
            "VisionMamba source is missing. Checked: {}".format(searched_paths)
        )

    import_paths = [_VIM_MODEL_ROOT]
    if use_local_mamba:
        import_paths[:0] = [_VIM_MAMBA_ROOT, _VIM_CAUSAL_CONV_ROOT]
    with _prepend_import_paths(*import_paths):
        import models_mamba

    if type == "small":
        model_name = "vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"
    elif type == "tiny":
        model_name = "vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"
    else:
        raise ValueError("Unknown Vim type: {}".format(type))

    print("Creating Vim")
    model = create_model(
        model_name=model_name,
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size=224,
    )

    if type == "small":
        path = _resolve_vim_checkpoint("vim_s_midclstok_80p5acc.pth")
    elif type == "tiny":
        path = _resolve_vim_checkpoint("vim_t_midclstok_76p1acc.pth")
    else:
        raise ValueError("Unknown Vim type: {}".format(type))

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)
    return model


def model_selection(name):
    if name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif name == "wrn50":
        model = models.wide_resnet50_2(pretrained=True)
    elif name == "inception_v3":
        model = models.inception_v3(pretrained=True)
    elif name == "convnext":
        model = models.convnext_base(pretrained=True)
    elif name == "vgg19":
        model = models.vgg19(pretrained=True)
    elif name == "vit":
        model = models.vit_b_16(pretrained=True)
    elif name == "swin":
        model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
    elif name in {"deit", "deit-b"}:
        model = create_model("deit_base_patch16_224", pretrained=True)
    elif name == "vim-small":
        model = get_vim_model("small", use_local_mamba=True)
    elif name == "vim-tiny":
        model = get_vim_model("tiny")
    elif name == "mambavision":
        model = _LogitsOnlyWrapper(
            AutoModelForImageClassification.from_pretrained(
                "nvidia/MambaVision-B-1K",
                trust_remote_code=True,
                torch_dtype="auto",
            )
        )
    elif name in {"dinov3_vit7b16_lc", "dino-v3-vit7b16-lc"}:
        model = get_dinov3_vit7b16_lc_model()
    elif name in {"dinov2_vitb14", "dinov2_vitb14_lc", "dino-v2-vitb14"}:
        model = get_dinov2_model(with_registers=False)
    elif name in {
        "dinov2_vitb14_reg",
        "dinov2_vitb14_reg_lc",
        "dino-v2-vitb14-reg",
    }:
        model = get_dinov2_model(with_registers=True)
    elif name in {
        "dinov1_vitb16",
        "dinov1_vitbase16",
        "dino_vitb16",
        "dino_vitbase16",
        "dino_vitbase16_lc",
        "dino-v1-vitb16",
    }:
        model = get_dinov1_vitb16_model()

    elif name == 'adv_inc':
        net = tf2torch_adv_inception_v3
        model_path = str(_REPO_ROOT / "pretrained_models" / f"{name}.npy")
        model = net.KitModel(model_path)
    elif name == 'adv_res':
        net = tf2torch_ens_adv_inc_res_v2
        model_path = str(_REPO_ROOT / "pretrained_models" / f"{name}.npy")
        model = net.KitModel(model_path)

    
    
    else:
        raise NotImplementedError("No such model!")
    return model.to(DEVICE)


if __name__ == "__main__":
    model = model_selection("vim-tiny")
    print(model)
