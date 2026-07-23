# defense_wrapper.py

import csv
import importlib
import io
import math
import os
import random
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms.functional as TF
from PIL import Image


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_HGD_SOURCE_ENV = "GUIDED_DENOISE_ROOT"
_HGD_CHECKPOINT_ENV = "HGD_CHECKPOINT_DIR"
_HGD_ALIASES = {"hgd", "guided-denoise", "guided_denoise"}
_NIPS_R3_SOURCE_ENV = "NIPS_R3_ROOT"
_NIPS_R3_CHECKPOINT_ENV = "NIPS_R3_CHECKPOINT_DIR"
_NIPS_R3_PYTHON_ENV = "NIPS_R3_PYTHON"
_NIPS_R3_TORCH_WEIGHT_ENV = "NIPS_R3_TORCH_WEIGHT_DIR"
_NIPS_R3_ALIASES = {"nips-r3", "nips_r3", "mmd"}
_NIPS_R3_TF1_ALIASES = {"nips-r3-tf1", "nips_r3_tf1", "mmd-tf1", "mmd_tf1"}
_NIPS_R3_CHECKPOINT_PREFIXES = (
    "ens_adv_inception_resnet_v2.ckpt",
    "adv_inception_v3.ckpt",
    "resnet_v2_152.ckpt",
    "vgg_16.ckpt",
)
_NIPS_R3_TORCH_WEIGHT_FILENAMES = {
    "adv_inc": "adv_inc.npy",
    "adv_res": "adv_res.npy",
}
_HGD_MODEL_SPECS = {
    "res": {
        "module": "res152_wide",
        "checkpoint": "denoise_res_015.ckpt",
        "normalization": "torch",
    },
    "inres": {
        "module": "inres",
        "checkpoint": "denoise_inres_014.ckpt",
        "normalization": "tf",
    },
    "incepv3": {
        "module": "v3",
        "checkpoint": "denoise_incepv3_012.ckpt",
        "normalization": "tf",
    },
    "rex": {
        "module": "resnext101",
        "checkpoint": "denoise_rex_001.ckpt",
        "normalization": "torch",
    },
}
HGD_MODEL_NAMES = tuple(_HGD_MODEL_SPECS.keys())


def is_hgd_defense_name(name: Optional[str]) -> bool:
    return isinstance(name, str) and name.lower() in _HGD_ALIASES


def is_nips_r3_defense_name(name: Optional[str]) -> bool:
    return isinstance(name, str) and name.lower() in _NIPS_R3_ALIASES


def is_nips_r3_tf1_defense_name(name: Optional[str]) -> bool:
    return isinstance(name, str) and name.lower() in _NIPS_R3_TF1_ALIASES


@contextmanager
def _prepend_sys_path(path: Path):
    original_sys_path = list(sys.path)
    path_str = str(path.resolve())
    try:
        if path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)
        yield
    finally:
        sys.path[:] = original_sys_path


def _candidate_hgd_source_dirs(source_dir: Optional[Union[str, Path]]) -> list[Path]:
    candidates = []
    if source_dir is not None:
        candidates.append(Path(source_dir).expanduser())

    env_source_dir = os.environ.get(_HGD_SOURCE_ENV)
    if env_source_dir:
        candidates.append(Path(env_source_dir).expanduser())

    candidates.extend(
        [
            _REPO_ROOT / "third_party" / "Guided-Denoise" / "nips_deploy",
            _REPO_ROOT / "third_party" / "Guided-Denoise",
        ]
    )
    return candidates


def _resolve_hgd_source_dir(source_dir: Optional[Union[str, Path]] = None) -> Path:
    candidates = _candidate_hgd_source_dirs(source_dir)
    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        checked.append(candidate)
        if (candidate / "nips_deploy").is_dir():
            candidate = candidate / "nips_deploy"
            checked.append(candidate)
        if (candidate / "defense.py").is_file() and (candidate / "res152_wide.py").is_file():
            return candidate

    checked_text = "\n  - ".join(str(path) for path in checked)
    raise FileNotFoundError(
        "HGD source files were not found. Set --hgd-source-dir to "
        "Guided-Denoise/nips_deploy or set {}. Checked:\n  - {}".format(
            _HGD_SOURCE_ENV,
            checked_text,
        )
    )


def _candidate_hgd_checkpoint_dirs(
    checkpoint_dir: Optional[Union[str, Path]],
    source_dir: Path,
) -> list[Path]:
    candidates = []
    if checkpoint_dir is not None:
        candidates.append(Path(checkpoint_dir).expanduser())

    env_checkpoint_dir = os.environ.get(_HGD_CHECKPOINT_ENV)
    if env_checkpoint_dir:
        candidates.append(Path(env_checkpoint_dir).expanduser())

    candidates.extend(
        [
            source_dir,
            source_dir.parent / "checkpoints",
            _REPO_ROOT / "pretrained_models" / "hgd",
        ]
    )
    return candidates


def _resolve_hgd_checkpoint_dir(
    checkpoint_dir: Optional[Union[str, Path]],
    source_dir: Path,
    model_names: Sequence[str],
) -> Path:
    required = [_HGD_MODEL_SPECS[name]["checkpoint"] for name in model_names]
    candidates = _candidate_hgd_checkpoint_dirs(checkpoint_dir, source_dir)
    for candidate in candidates:
        candidate = candidate.resolve()
        if all((candidate / filename).is_file() for filename in required):
            return candidate

    missing_by_dir = []
    for candidate in candidates:
        candidate = candidate.resolve()
        missing = [filename for filename in required if not (candidate / filename).is_file()]
        missing_by_dir.append("{}: {}".format(candidate, ", ".join(missing)))

    raise FileNotFoundError(
        "Missing HGD checkpoint file(s). Download the nips_deploy weights from "
        "https://github.com/lfz/Guided-Denoise and pass --hgd-checkpoint-dir, "
        "or set {}. Checked:\n  - {}".format(
            _HGD_CHECKPOINT_ENV,
            "\n  - ".join(missing_by_dir),
        )
    )


def _candidate_nips_r3_source_dirs(source_dir: Optional[Union[str, Path]]) -> list[Path]:
    candidates = []
    if source_dir is not None:
        candidates.append(Path(source_dir).expanduser())

    env_source_dir = os.environ.get(_NIPS_R3_SOURCE_ENV)
    if env_source_dir:
        candidates.append(Path(env_source_dir).expanduser())

    candidates.extend(
        [
            _REPO_ROOT / "third_party" / "nips-2017" / "mmd",
            _REPO_ROOT / "third_party" / "nips-r3",
        ]
    )
    return candidates


def _resolve_nips_r3_source_dir(source_dir: Optional[Union[str, Path]] = None) -> Path:
    candidates = _candidate_nips_r3_source_dirs(source_dir)
    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        checked.append(candidate)
        if (candidate / "mmd").is_dir():
            candidate = candidate / "mmd"
            checked.append(candidate)
        if (candidate / "defense_mmd.py").is_file() and (candidate / "inception_resnet_v2.py").is_file():
            return candidate

    checked_text = "\n  - ".join(str(path) for path in checked)
    raise FileNotFoundError(
        "NIPS-R3 source files were not found. Set --nips-r3-source-dir to "
        "anlthms/nips-2017/mmd or set {}. Checked:\n  - {}".format(
            _NIPS_R3_SOURCE_ENV,
            checked_text,
        )
    )


def _tf_checkpoint_exists(directory: Path, prefix: str) -> bool:
    direct_path = directory / prefix
    if direct_path.is_file():
        return True
    index_path = directory / "{}.index".format(prefix)
    return index_path.is_file() and any(directory.glob("{}.data-*".format(prefix)))


def _candidate_nips_r3_checkpoint_dirs(
    checkpoint_dir: Optional[Union[str, Path]],
    source_dir: Path,
) -> list[Path]:
    candidates = []
    if checkpoint_dir is not None:
        candidates.append(Path(checkpoint_dir).expanduser())

    env_checkpoint_dir = os.environ.get(_NIPS_R3_CHECKPOINT_ENV)
    if env_checkpoint_dir:
        candidates.append(Path(env_checkpoint_dir).expanduser())

    candidates.extend(
        [
            source_dir,
            source_dir.parent / "checkpoints",
            _REPO_ROOT / "pretrained_models" / "nips-r3",
            _REPO_ROOT / "pretrained_models" / "mmd",
        ]
    )
    return candidates


def _resolve_nips_r3_checkpoint_dir(
    checkpoint_dir: Optional[Union[str, Path]],
    source_dir: Path,
) -> Path:
    candidates = _candidate_nips_r3_checkpoint_dirs(checkpoint_dir, source_dir)
    for candidate in candidates:
        candidate = candidate.resolve()
        if all(_tf_checkpoint_exists(candidate, prefix) for prefix in _NIPS_R3_CHECKPOINT_PREFIXES):
            return candidate

    missing_by_dir = []
    for candidate in candidates:
        candidate = candidate.resolve()
        missing = [
            prefix
            for prefix in _NIPS_R3_CHECKPOINT_PREFIXES
            if not _tf_checkpoint_exists(candidate, prefix)
        ]
        missing_by_dir.append("{}: {}".format(candidate, ", ".join(missing)))

    raise FileNotFoundError(
        "Missing NIPS-R3 TensorFlow checkpoint file(s). The original mmd defense "
        "restores these checkpoint prefixes: {}. Pass --nips-r3-checkpoint-dir "
        "or set {}. Checked:\n  - {}".format(
            ", ".join(_NIPS_R3_CHECKPOINT_PREFIXES),
            _NIPS_R3_CHECKPOINT_ENV,
            "\n  - ".join(missing_by_dir),
        )
    )


def _candidate_nips_r3_torch_weight_dirs(weights_dir: Optional[Union[str, Path]]) -> list[Path]:
    candidates = []
    if weights_dir is not None:
        candidates.append(Path(weights_dir).expanduser())

    env_weights_dir = os.environ.get(_NIPS_R3_TORCH_WEIGHT_ENV)
    if env_weights_dir:
        candidates.append(Path(env_weights_dir).expanduser())

    candidates.extend(
        [
            _REPO_ROOT / "pretrained_models" / "nips-r3-torch",
            _REPO_ROOT / "pretrained_models",
        ]
    )
    return candidates


def _resolve_nips_r3_torch_weight_dir(weights_dir: Optional[Union[str, Path]] = None) -> Path:
    required = list(_NIPS_R3_TORCH_WEIGHT_FILENAMES.values())
    candidates = _candidate_nips_r3_torch_weight_dirs(weights_dir)
    for candidate in candidates:
        candidate = candidate.resolve()
        if all((candidate / filename).is_file() for filename in required):
            return candidate

    missing_by_dir = []
    for candidate in candidates:
        candidate = candidate.resolve()
        missing = [filename for filename in required if not (candidate / filename).is_file()]
        missing_by_dir.append("{}: {}".format(candidate, ", ".join(missing)))

    raise FileNotFoundError(
        "Missing NIPS-R3 torch weight file(s). Expected {}. Pass "
        "--nips-r3-torch-weights-dir or set {}. Checked:\n  - {}".format(
            ", ".join(required),
            _NIPS_R3_TORCH_WEIGHT_ENV,
            "\n  - ".join(missing_by_dir),
        )
    )


def _normalize_hgd_model_names(model_names: Optional[Union[str, Sequence[str]]]) -> tuple[str, ...]:
    if model_names is None:
        return HGD_MODEL_NAMES
    if isinstance(model_names, str):
        raw_names = model_names.replace(",", " ").split()
    else:
        raw_names = []
        for name in model_names:
            raw_names.extend(str(name).replace(",", " ").split())

    normalized = []
    for name in raw_names:
        lowered = name.lower()
        if lowered == "all":
            normalized.extend(HGD_MODEL_NAMES)
        elif lowered in _HGD_MODEL_SPECS:
            normalized.append(lowered)
        else:
            raise ValueError(
                "Unknown HGD submodel '{}'. Expected one of: {}".format(
                    name,
                    ", ".join(HGD_MODEL_NAMES),
                )
            )

    deduped = []
    for name in normalized:
        if name not in deduped:
            deduped.append(name)
    if not deduped:
        raise ValueError("At least one HGD submodel must be selected.")
    return tuple(deduped)


class HGDNIPSDefense(nn.Module):
    """
    NIPS 2017 HGD defense from lfz/Guided-Denoise.

    This is a model-level defense, not a preprocessing-only transform. It loads
    the original denoiser-equipped ImageNet classifiers and returns summed
    ensemble logits over the standard 1000 ImageNet classes.
    """
    def __init__(
        self,
        source_dir: Optional[Union[str, Path]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        model_names: Optional[Union[str, Sequence[str]]] = None,
    ):
        super().__init__()
        self.image_size = 299
        self.model_names = _normalize_hgd_model_names(model_names)
        self.source_dir = _resolve_hgd_source_dir(source_dir)
        self.checkpoint_dir = _resolve_hgd_checkpoint_dir(
            checkpoint_dir,
            self.source_dir,
            self.model_names,
        )
        self.models = nn.ModuleDict()
        self.model_normalizations = {}

        self.register_buffer(
            "torch_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "torch_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "tf_mean",
            torch.tensor((0.5, 0.5, 0.5), dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "tf_std",
            torch.tensor((0.5, 0.5, 0.5), dtype=torch.float32).view(1, 3, 1, 1),
        )

        with _prepend_sys_path(self.source_dir):
            importlib.invalidate_caches()
            for model_name in self.model_names:
                spec = _HGD_MODEL_SPECS[model_name]
                module = importlib.import_module(spec["module"])
                _, wrapped_model = module.get_model()
                checkpoint_path = self.checkpoint_dir / spec["checkpoint"]
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
                state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
                wrapped_model.load_state_dict(state_dict)
                self.models[model_name] = wrapped_model.net
                self.model_normalizations[model_name] = spec["normalization"]

        self.eval()

    def _normalize(self, x: torch.Tensor, normalization: str) -> torch.Tensor:
        if normalization == "torch":
            mean = self.torch_mean.to(device=x.device, dtype=x.dtype)
            std = self.torch_std.to(device=x.device, dtype=x.dtype)
        elif normalization == "tf":
            mean = self.tf_mean.to(device=x.device, dtype=x.dtype)
            std = self.tf_std.to(device=x.device, dtype=x.dtype)
        else:
            raise ValueError("Unknown HGD normalization: {}".format(normalization))
        return (x - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        logits = []
        for model_name in self.model_names:
            model_input = self._normalize(x, self.model_normalizations[model_name])
            logits.append(self.models[model_name](model_input, True)[-1])
        return torch.stack(logits, dim=0).sum(dim=0)


class NIPSR3TorchDefense(nn.Module):
    """
    Torch implementation of the NIPS 2017 R3/mmd ensemble path.

    This uses the converted adversarial Inception models already wired by
    eval/attacked_models.py, plus torchvision VGG16 and ResNet152 branches. Note that
    torchvision's ResNet152_Weights.IMAGENET1K_V2 is a weight recipe version,
    not the TF-Slim ResNet-v2-152 architecture used by the original script.
    """
    def __init__(
        self,
        weights_dir: Optional[Union[str, Path]] = None,
        seed: int = 0,
    ):
        super().__init__()
        self.image_size = 299
        self.vgg_image_size = 224
        self.num_classes = 1000
        self.rng = random.Random(seed)
        self.weights_dir = _resolve_nips_r3_torch_weight_dir(weights_dir)

        from torch_nets import tf2torch_adv_inception_v3, tf2torch_ens_adv_inc_res_v2

        self.adv_res = tf2torch_ens_adv_inc_res_v2.KitModel(
            str(self.weights_dir / _NIPS_R3_TORCH_WEIGHT_FILENAMES["adv_res"]),
            aux_logits=True,
        )
        self.adv_inc = tf2torch_adv_inception_v3.KitModel(
            str(self.weights_dir / _NIPS_R3_TORCH_WEIGHT_FILENAMES["adv_inc"]),
            aux_logits=True,
        )
        self.vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
        self.resnet = tv_models.resnet152(weights=tv_models.ResNet152_Weights.IMAGENET1K_V2)
        self.register_buffer(
            "torch_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "torch_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.eval()

    @staticmethod
    def _to_tf_range(x: torch.Tensor) -> torch.Tensor:
        return x * 2.0 - 1.0

    @staticmethod
    def _remove_background(logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] == 1001:
            return logits[:, 1:]
        if logits.shape[1] == 1000:
            return logits
        raise ValueError("Expected 1000 or 1001 logits, got {}".format(tuple(logits.shape)))

    def _main_aux_logits(self, output: Union[torch.Tensor, Sequence[torch.Tensor]]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(output, (list, tuple)):
            return self._remove_background(output[0]), self._remove_background(output[1])
        return self._remove_background(output), None

    def _jpeg_transcode(self, x: torch.Tensor, quality: int = 50) -> torch.Tensor:
        device = x.device
        dtype = x.dtype
        outputs = []
        for image in x.detach().cpu():
            pil_image = TF.to_pil_image(image.clamp(0.0, 1.0))
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            outputs.append(TF.to_tensor(Image.open(buffer).convert("RGB")))
        return torch.stack(outputs, dim=0).to(device=device, dtype=dtype)

    def _affine(
        self,
        image: torch.Tensor,
        angle: float = 0.0,
        translate: Optional[Sequence[int]] = None,
        scale: float = 1.0,
        shear: Optional[Sequence[float]] = None,
        fill: float = 0.0,
    ) -> torch.Tensor:
        translate = list(translate) if translate is not None else [0, 0]
        shear = list(shear) if shear is not None else [0.0, 0.0]
        return TF.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=TF.InterpolationMode.NEAREST,
            fill=[fill, fill, fill],
        )

    def _jitter(self, value: float) -> float:
        return value * (self.rng.randint(95, 105) / 100.0)

    def _ens_distort_one(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        shear = math.degrees(self._jitter(math.pi / 128.0))
        theta = math.degrees(self._jitter(-math.pi / 128.0))
        tx = int(self._jitter(width * 0.02))
        ty = int(self._jitter(height * 0.02))
        zx = self._jitter(0.95)
        zy = self._jitter(0.95)
        zoom = (zx + zy) / 2.0

        image = self._affine(image, shear=[shear, 0.0])
        image = self._affine(image, translate=[tx, ty])
        image = self._affine(image, scale=zoom)
        image = self._affine(image, angle=theta)
        return image

    def _inc_distort_one(self, image: torch.Tensor) -> torch.Tensor:
        image = image + torch.randn_like(image) * (4.0 / 255.0)
        image = image.clamp(-1.0, 1.0)
        angle = float(self.rng.randint(2, 3))
        if self.rng.randint(0, 1) == 0:
            angle = -angle
        zoom = self.rng.uniform(1.05, 1.1)
        return self._affine(image, angle=angle, scale=zoom)

    def _ens_distort(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self._ens_distort_one(image) for image in x], dim=0)

    def _inc_distort(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self._inc_distort_one(image) for image in x], dim=0)

    def _vgg_distort_one(self, image: torch.Tensor) -> torch.Tensor:
        angle = float(self.rng.randint(2, 3))
        if self.rng.randint(0, 1) == 0:
            angle = -angle
        zoom = self.rng.uniform(0.75, 0.8)
        image = self._affine(image, angle=angle, scale=zoom)
        return TF.center_crop(image, [self.vgg_image_size, self.vgg_image_size])

    def _vgg_distort(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self._vgg_distort_one(image) for image in x], dim=0)

    def _torch_normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.torch_mean.to(device=x.device, dtype=x.dtype)
        std = self.torch_std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().clamp(0.0, 1.0)
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        transcoded = self._jpeg_transcode(x)
        x_tf = self._to_tf_range(x)
        tcd_tf = self._to_tf_range(transcoded)
        ens_input = self._ens_distort(x_tf)
        inc_input = self._inc_distort(x_tf)
        vgg_input = self._torch_normalize(self._vgg_distort(transcoded))

        res_ens_logits, res_ens_aux = self._main_aux_logits(self.adv_res(ens_input))
        res_tcd_logits, res_tcd_aux = self._main_aux_logits(self.adv_res(tcd_tf))
        inc_logits, inc_aux = self._main_aux_logits(self.adv_inc(inc_input))
        inc_tcd_logits, inc_tcd_aux = self._main_aux_logits(self.adv_inc(tcd_tf))

        logits = (
            res_ens_logits * 0.7
            + res_tcd_logits * 0.1
            + inc_logits * 0.2
            + inc_tcd_logits * 0.1
            + self.vgg(vgg_input) * 0.1
            + self.resnet(vgg_input) * 0.05
        )
        aux_logits = (
            res_ens_aux * 0.7
            + res_tcd_aux * 0.1
            + inc_aux * 0.2
            + inc_tcd_aux * 0.1
        )
        logits = logits + aux_logits * 0.8
        return logits


class NIPSR3TF1Defense(nn.Module):
    """
    NIPS 2017 R3/mmd defense from anlthms/nips-2017.

    The upstream implementation is TensorFlow 1.x and only exposes the original
    competition-style run_defense.sh interface. This wrapper preserves that
    behavior for evaluation by writing each torch batch to PNG files, invoking
    defense_mmd.py, and converting the returned 1-based ImageNet labels into
    1000-way logits.
    """
    def __init__(
        self,
        source_dir: Optional[Union[str, Path]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        python_executable: Optional[str] = None,
        batch_size: int = 16,
    ):
        super().__init__()
        if batch_size < 1:
            raise ValueError("NIPS-R3 batch_size must be positive.")
        self.image_size = 299
        self.num_classes = 1000
        self.batch_size = int(batch_size)
        self.source_dir = _resolve_nips_r3_source_dir(source_dir)
        self.checkpoint_dir = _resolve_nips_r3_checkpoint_dir(checkpoint_dir, self.source_dir)
        self.python_executable = (
            python_executable
            or os.environ.get(_NIPS_R3_PYTHON_ENV)
            or sys.executable
        )
        self.eval()

    def _run_upstream_defense(self, input_dir: Path, output_file: Path) -> None:
        command = [
            self.python_executable,
            str(self.source_dir / "defense_mmd.py"),
            "--input_dir={}".format(input_dir),
            "--output_file={}".format(output_file),
            "--image_width={}".format(self.image_size),
            "--image_height={}".format(self.image_size),
            "--batch_size={}".format(self.batch_size),
        ]
        env = os.environ.copy()
        source_path = str(self.source_dir)
        if env.get("PYTHONPATH"):
            env["PYTHONPATH"] = source_path + os.pathsep + env["PYTHONPATH"]
        else:
            env["PYTHONPATH"] = source_path

        completed = subprocess.run(
            command,
            cwd=str(self.checkpoint_dir),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            stdout = completed.stdout[-4000:] if completed.stdout else ""
            stderr = completed.stderr[-4000:] if completed.stderr else ""
            raise RuntimeError(
                "NIPS-R3 TF1 upstream defense failed with exit code {}.\n"
                "Command: {}\nstdout:\n{}\nstderr:\n{}".format(
                    completed.returncode,
                    " ".join(command),
                    stdout,
                    stderr,
                )
            )

    def _read_labels(self, output_file: Path) -> dict[str, int]:
        labels = {}
        with output_file.open("r", newline="") as handle:
            for row in csv.reader(handle):
                if len(row) < 2:
                    continue
                filename = row[0].strip()
                label = int(row[1])
                if not (1 <= label <= self.num_classes):
                    raise ValueError("NIPS-R3 returned out-of-range label {} for {}".format(label, filename))
                labels[filename] = label - 1
        return labels

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        dtype = x.dtype
        x = x.detach().clamp(0.0, 1.0).cpu()
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        batch_size = x.shape[0]
        filenames = ["{:08d}.png".format(idx) for idx in range(batch_size)]
        with tempfile.TemporaryDirectory(prefix="nips_r3_") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            input_dir = temp_dir / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_file = temp_dir / "labels.csv"

            images = (x.detach().cpu().mul(255.0).round().clamp(0, 255).to(torch.uint8))
            for filename, image in zip(filenames, images):
                TF.to_pil_image(image).save(input_dir / filename)

            self._run_upstream_defense(input_dir, output_file)
            labels_by_name = self._read_labels(output_file)

        labels = []
        for filename in filenames:
            if filename not in labels_by_name:
                raise RuntimeError("NIPS-R3 did not return a label for {}".format(filename))
            labels.append(labels_by_name[filename])

        labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
        logits = torch.zeros((batch_size, self.num_classes), dtype=dtype, device=device)
        logits[torch.arange(batch_size, device=device), labels_tensor] = 1.0
        return logits


NIPSR3Defense = NIPSR3TorchDefense


# ============================================================
# 1. Basic normalization
# ============================================================

class ImageNetNormalize(nn.Module):
    """
    Input:
        x: torch.Tensor, shape [B, 3, H, W], range [0, 1]

    Output:
        normalized tensor for ImageNet classifiers
    """
    def __init__(
        self,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


# ============================================================
# 2. JPEG defense
# ============================================================

class JPEGDefense(nn.Module):
    """
    JPEG compression defense.

    Input:
        x: [B, 3, H, W], range [0, 1]

    Note:
        This operation is not differentiable.
        It is intended for evaluation-time defense.
    """
    def __init__(self, quality: int = 75):
        super().__init__()
        if not (1 <= quality <= 100):
            raise ValueError("JPEG quality must be in [1, 100].")
        self.quality = quality

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        x = x.detach().clamp(0.0, 1.0).cpu()

        outputs = []
        for img in x:
            pil_img = TF.to_pil_image(img)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=self.quality)
            buffer.seek(0)

            rec_img = Image.open(buffer).convert("RGB")
            rec_tensor = TF.to_tensor(rec_img)
            outputs.append(rec_tensor)

        return torch.stack(outputs, dim=0).to(device)


# ============================================================
# 3. Bit-depth reduction defense
# ============================================================

class BitDepthReduction(nn.Module):
    """
    Bit-depth reduction defense.

    Example:
        bits=4 means each channel is quantized into 16 levels.
    """
    def __init__(self, bits: int = 4):
        super().__init__()
        if bits < 1 or bits > 8:
            raise ValueError("bits must be between 1 and 8.")
        self.bits = bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)
        levels = 2 ** self.bits - 1
        return torch.round(x * levels) / levels


# ============================================================
# 4. Random resize and padding defense
# ============================================================

class RandomResizePadding(nn.Module):
    """
    Random resize and padding defense.

    This implementation is designed for ImageNet-style classifiers.

    Input:
        x: [B, 3, H, W], range [0, 1]

    For ResNet/MobileNet:
        final_size=224
        resize_min=224
        resize_max=256

    For Inception:
        final_size=299
        resize_min=299
        resize_max=330
    """
    def __init__(
        self,
        final_size: int = 224,
        resize_min: Optional[int] = None,
        resize_max: Optional[int] = None,
        padding_value: float = 0.0,
    ):
        super().__init__()
        self.final_size = final_size
        self.resize_min = resize_min if resize_min is not None else final_size
        self.resize_max = resize_max if resize_max is not None else int(final_size * 1.15)
        self.padding_value = padding_value

        if self.resize_min > self.resize_max:
            raise ValueError("resize_min must be <= resize_max.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)

        rnd = random.randint(self.resize_min, self.resize_max)

        x = F.interpolate(
            x,
            size=(rnd, rnd),
            mode="bilinear",
            align_corners=False,
        )

        pad_total = self.resize_max - rnd

        pad_top = random.randint(0, pad_total)
        pad_bottom = pad_total - pad_top
        pad_left = random.randint(0, pad_total)
        pad_right = pad_total - pad_left

        x = F.pad(
            x,
            pad=(pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=self.padding_value,
        )

        if x.shape[-1] != self.final_size or x.shape[-2] != self.final_size:
            x = F.interpolate(
                x,
                size=(self.final_size, self.final_size),
                mode="bilinear",
                align_corners=False,
            )

        return x


# ============================================================
# 5. Identity defense
# ============================================================

class IdentityDefense(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ============================================================
# 6. Defense factory
# ============================================================

def build_defense(
    name: Optional[str],
    image_size: int = 224,
    jpeg_quality: int = 75,
    bit_depth: int = 4,
) -> nn.Module:
    """
    Build a preprocessing defense.

    Args:
        name:
            None, "none", "identity",
            "jpeg",
            "bitred", "bit-red",
            "rp", "r&p", "random-resize-padding"
        image_size:
            classifier input size.
            Use 224 for ResNet/MobileNet/ViT-B-16 torchvision models.
            Use 299 for Inception-style models.
    """
    if name is None:
        return IdentityDefense()

    name = name.lower()

    if name in ["none", "identity", "clean"]:
        return IdentityDefense()

    if is_hgd_defense_name(name):
        raise ValueError(
            "HGD is a model-level defense. Use eval/eval_defense.py or eval/eval_asr_from_npz.py "
            "with --defense hgd instead of building it as a preprocessing transform."
        )

    if is_nips_r3_defense_name(name) or is_nips_r3_tf1_defense_name(name):
        raise ValueError(
            "NIPS-R3 is a model-level defense. Use eval/eval_defense.py or eval/eval_asr_from_npz.py "
            "with --defense nips-r3 instead of building it as a preprocessing transform."
        )

    if name in ["jpeg", "jpg"]:
        return JPEGDefense(quality=jpeg_quality)

    if name in ["bitred", "bit-red", "bit_depth", "bit-depth", "bit"]:
        return BitDepthReduction(bits=bit_depth)

    if name in ["rp", "r&p", "random-resize-padding", "random_resize_padding"]:
        return RandomResizePadding(
            final_size=image_size,
            resize_min=image_size,
            resize_max=int(image_size * 1.15),
        )

    raise ValueError(f"Unknown defense name: {name}")


# ============================================================
# 7. Unified wrapper
# ============================================================

class DefendedModel(nn.Module):
    """
    A unified wrapper:

        input image -> defense preprocessing -> normalization -> classifier

    Input:
        x: [B, 3, H, W], range [0, 1]

    Output:
        logits
    """
    def __init__(
        self,
        model: nn.Module,
        defense: Optional[Union[str, nn.Module]] = None,
        image_size: int = 224,
        normalize: bool = True,
        jpeg_quality: int = 75,
        bit_depth: int = 4,
    ):
        super().__init__()
        self.model = model
        self.image_size = image_size

        if isinstance(defense, str) or defense is None:
            self.defense = build_defense(
                name=defense,
                image_size=image_size,
                jpeg_quality=jpeg_quality,
                bit_depth=bit_depth,
            )
        elif isinstance(defense, nn.Module):
            self.defense = defense
        else:
            raise TypeError("defense must be None, str, or nn.Module.")

        self.normalize = ImageNetNormalize() if normalize else IdentityDefense()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)

        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        x = self.defense(x)
        x = x.clamp(0.0, 1.0)
        x = self.normalize(x)
        logits = self.model(x)

        return logits


# ============================================================
# 8. Evaluation helper
# ============================================================

@torch.no_grad()
def evaluate_attack_success_rate(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 32,
    targeted: bool = False,
) -> float:
    """
    Compute ASR.

    Untargeted:
        success = prediction != true label

    Targeted:
        success = prediction == target label

    Args:
        model:
            DefendedModel or ordinary classifier.
        images:
            [N, 3, H, W], range [0, 1]
        labels:
            [N]
            true labels for untargeted attack,
            target labels for targeted attack.
    """
    model.eval()

    total = 0
    success = 0

    for start in range(0, images.size(0), batch_size):
        end = start + batch_size
        x = images[start:end]
        y = labels[start:end].to(x.device)

        logits = model(x)
        pred = logits.argmax(dim=1)

        if targeted:
            success += (pred == y).sum().item()
        else:
            success += (pred != y).sum().item()

        total += y.numel()

    return success / max(total, 1)


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate adversarial examples with preprocessing defenses."
    )
    parser.add_argument("--npz", type=str, required=True, help="Path to adversarial_examples*.npz.")
    parser.add_argument(
        "--file-name",
        type=str,
        default=None,
        help="Run directory used for metadata/logs. Defaults to the npz parent directory.",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=["resnet50"],
        help="Victim model name(s), e.g. resnet50 swin convnext.",
    )
    parser.add_argument(
        "--defense",
        nargs="+",
        default=["none", "jpeg", "bitred", "rp"],
        help="Defense name(s): none, jpeg, bitred, rp, hgd, nips-r3, nips-r3-tf1.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--bit-depth", type=int, default=4)
    parser.add_argument(
        "--hgd-source-dir",
        type=str,
        default=None,
        help="Path to Guided-Denoise/nips_deploy. Defaults to third_party/Guided-Denoise/nips_deploy or GUIDED_DENOISE_ROOT.",
    )
    parser.add_argument(
        "--hgd-checkpoint-dir",
        type=str,
        default=None,
        help="Directory containing denoise_res_015.ckpt, denoise_inres_014.ckpt, denoise_incepv3_012.ckpt, and denoise_rex_001.ckpt.",
    )
    parser.add_argument(
        "--hgd-models",
        nargs="+",
        default=None,
        help="HGD ensemble members to use: all, res, inres, incepv3, rex. Defaults to all.",
    )
    parser.add_argument(
        "--nips-r3-torch-weights-dir",
        type=str,
        default=None,
        help="Directory containing adv_inc.npy and adv_res.npy for the default torch NIPS-R3 backend.",
    )
    parser.add_argument(
        "--nips-r3-source-dir",
        type=str,
        default=None,
        help="Path to anlthms/nips-2017/mmd for --defense nips-r3-tf1. Defaults to third_party/nips-2017/mmd or NIPS_R3_ROOT.",
    )
    parser.add_argument(
        "--nips-r3-checkpoint-dir",
        type=str,
        default=None,
        help="Directory containing ens_adv_inception_resnet_v2.ckpt, adv_inception_v3.ckpt, resnet_v2_152.ckpt, and vgg_16.ckpt for --defense nips-r3-tf1.",
    )
    parser.add_argument(
        "--nips-r3-python",
        type=str,
        default=None,
        help="Python executable for the TensorFlow 1.x upstream mmd defense. Defaults to NIPS_R3_PYTHON or this interpreter.",
    )
    parser.add_argument(
        "--nips-r3-batch-size",
        type=int,
        default=16,
        help="Internal batch size passed to upstream defense_mmd.py.",
    )
    parser.add_argument(
        "--nips-r3-predict-batch-size",
        type=int,
        default=1000,
        help="Outer ART/PyTorch batch size for NIPS-R3. Torch backend caps this at 4 to avoid OOM.",
    )
    parser.add_argument(
        "--clean-images-root",
        type=str,
        default="./data/nips2017/images/",
        help="Directory containing clean images for clean-correct filtering.",
    )
    parser.add_argument(
        "--ground-truth-csv",
        type=str,
        default="./data/nips2017/images.csv",
        help="NIPS-style CSV with ImageId, TrueLabel, and TargetClass columns.",
    )
    parser.add_argument(
        "--labels-txt",
        type=str,
        default=None,
        help="One label per line. When set, this replaces --ground-truth-csv and sample_names from the NPZ are used as clean image IDs.",
    )
    parser.add_argument(
        "--labels-zero-based",
        dest="labels_one_based",
        action="store_false",
        help="Interpret --labels-txt values as 0-based ImageNet labels. Default is 1-based.",
    )
    parser.add_argument("--quant", action="store_true", help="Keep compatibility with eval_asr_from_npz.")
    parser.add_argument("--no-re-logger", action="store_true", help="Log to the current logger instead of a new eval dir.")
    parser.set_defaults(use_npz_source_query=True, use_prompt_label_as_correct=None, labels_one_based=True)
    parser.add_argument("--use-npz-source-query", dest="use_npz_source_query", action="store_true")
    parser.add_argument("--no-use-npz-source-query", dest="use_npz_source_query", action="store_false")
    parser.add_argument("--use-prompt-label-as-correct", dest="use_prompt_label_as_correct", action="store_true")
    parser.add_argument("--no-prompt-label-as-correct", dest="use_prompt_label_as_correct", action="store_false")
    return parser.parse_args()


def main():
    from eval.eval_asr_from_npz import eval_asr_from_npz

    args = _parse_args()
    npz_path = Path(args.npz).expanduser()
    file_name = args.file_name if args.file_name is not None else str(npz_path.parent)
    use_prompt_label_as_correct = args.use_prompt_label_as_correct
    if use_prompt_label_as_correct is None:
        use_prompt_label_as_correct = "ASA" in str(file_name) or "ASA" in str(npz_path)

    for defense_name in args.defense:
        eval_asr_from_npz(
            file_name=file_name,
            re_logger=not args.no_re_logger,
            quant=args.quant,
            model_name=args.model,
            use_npz_source_query=args.use_npz_source_query,
            use_prompt_label_as_correct=use_prompt_label_as_correct,
            npz_path=npz_path,
            defense_name=defense_name,
            jpeg_quality=args.jpeg_quality,
            bit_depth=args.bit_depth,
            hgd_source_dir=args.hgd_source_dir,
            hgd_checkpoint_dir=args.hgd_checkpoint_dir,
            hgd_models=args.hgd_models,
            nips_r3_torch_weights_dir=args.nips_r3_torch_weights_dir,
            nips_r3_source_dir=args.nips_r3_source_dir,
            nips_r3_checkpoint_dir=args.nips_r3_checkpoint_dir,
            nips_r3_python=args.nips_r3_python,
            nips_r3_batch_size=args.nips_r3_batch_size,
            nips_r3_predict_batch_size=args.nips_r3_predict_batch_size,
            clean_images_root=args.clean_images_root,
            ground_truth_csv=args.ground_truth_csv,
            labels_txt=args.labels_txt,
            labels_one_based=args.labels_one_based,
        )


if __name__ == "__main__":
    main()
