from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.npz_loader import load_adv_images_from_npz


def _to_uint8_hwc(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images, dtype=np.float32)
    images = np.clip(images, 0.0, 1.0)
    images = np.round(images * 255.0).astype(np.uint8)
    return images


def _save_grid(images_hwc_uint8: np.ndarray, indices: list[int], out_path: Path, columns: int = 4) -> None:
    if len(indices) == 0:
        raise ValueError("No indices selected for visualization.")

    tile_h, tile_w = images_hwc_uint8.shape[1:3]
    label_h = 24
    rows = math.ceil(len(indices) / columns)
    canvas = Image.new("RGB", (columns * tile_w, rows * (tile_h + label_h)), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    for grid_idx, image_idx in enumerate(indices):
        row = grid_idx // columns
        col = grid_idx % columns
        x = col * tile_w
        y = row * (tile_h + label_h)
        tile = Image.fromarray(images_hwc_uint8[image_idx])
        canvas.paste(tile, (x, y))
        draw.text((x + 6, y + tile_h + 4), f"idx={image_idx}", fill=(235, 235, 235))

    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Save selected images from an adversarial_examples.npz file.")
    parser.add_argument("--npz", type=Path, required=True, help="Path to the .npz file.")
    parser.add_argument("--start", type=int, default=0, help="Start index, inclusive.")
    parser.add_argument("--end", type=int, default=10, help="End index, inclusive.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to save PNGs. Defaults to <npz parent>/npz_preview_<start>_<end>.",
    )
    args = parser.parse_args()

    if args.start < 0 or args.end < args.start:
        raise ValueError(f"Invalid range: start={args.start}, end={args.end}")

    images_nhwc = load_adv_images_from_npz(str(args.npz), output_layout="nhwc")
    total = images_nhwc.shape[0]
    if args.start >= total:
        raise ValueError(f"Start index {args.start} is out of range for {total} images.")

    end = min(args.end, total - 1)
    indices = list(range(args.start, end + 1))
    images_uint8 = _to_uint8_hwc(images_nhwc)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = args.npz.parent / f"npz_preview_{args.start:04d}_{end:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        Image.fromarray(images_uint8[idx]).save(out_dir / f"img_{idx:04d}.png")

    _save_grid(images_uint8, indices, out_dir / f"grid_{args.start:04d}_{end:04d}.png")

    print(f"saved {len(indices)} images to {out_dir}")
    print(f"grid: {out_dir / f'grid_{args.start:04d}_{end:04d}.png'}")


if __name__ == "__main__":
    main()
