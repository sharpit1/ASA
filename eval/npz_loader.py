import numpy as np


_PREFERRED_KEYS = (
    "adversarial_examples",
    "arr_0",
    "adv_images",
    "all_adv_images",
    "images",
)


def _is_image_batch(arr):
    return isinstance(arr, np.ndarray) and arr.ndim == 4 and np.issubdtype(arr.dtype, np.number)


def _pick_adv_array(npz_data):
    for key in _PREFERRED_KEYS:
        if key in npz_data.files:
            arr = np.asarray(npz_data[key])
            if _is_image_batch(arr):
                return arr

    for key in npz_data.files:
        arr = np.asarray(npz_data[key])
        if _is_image_batch(arr):
            return arr

    raise ValueError(
        f"No 4D numeric adversarial image array found in {npz_data.files}. "
        "Expected keys like 'adversarial_examples' or 'arr_0'."
    )


def _normalize_to_01(images):
    images = images.astype(np.float32, copy=False)
    min_val = float(images.min())
    max_val = float(images.max())

    # Support legacy data saved in [-1, 1].
    if min_val < -1e-6 and -1.0001 <= min_val and max_val <= 1.0001:
        images = (images + 1.0) / 2.0
    # Support 0-255 storage.
    elif max_val > 1.0001:
        images = images / 255.0

    return np.clip(images, 0.0, 1.0)


def quantize_to_8bit_01(images):
    images = np.asarray(images, dtype=np.float32)
    images = np.clip(images, 0.0, 1.0)
    return np.clip(np.round(images * 255.0), 0.0, 255.0) / 255.0


def is_8bit_quantized_01(images, tol=1e-3, max_samples=1_000_000):
    images = np.asarray(images, dtype=np.float32)
    if images.size == 0:
        return False

    min_val = float(images.min())
    max_val = float(images.max())
    return min_val >= -tol and max_val <= 1.0 + tol


def load_adv_images_from_npz(npz_file, output_layout="nchw", normalize=True):
    npz_data = np.load(npz_file, allow_pickle=False)
    try:
        if isinstance(npz_data, np.lib.npyio.NpzFile):
            images = _pick_adv_array(npz_data)
        else:
            images = np.asarray(npz_data)
    finally:
        if isinstance(npz_data, np.lib.npyio.NpzFile):
            npz_data.close()

    if images.ndim != 4:
        raise ValueError(f"Expected 4D image batch, but got shape {images.shape}.")

    is_nchw = images.shape[1] == 3
    is_nhwc = images.shape[-1] == 3
    if is_nchw and not is_nhwc:
        src_layout = "nchw"
    elif is_nhwc and not is_nchw:
        src_layout = "nhwc"
    elif is_nchw and is_nhwc:
        raise ValueError(f"Ambiguous image layout for shape {images.shape}.")
    else:
        raise ValueError(
            f"Unsupported image layout for shape {images.shape}. "
            "Expected NCHW or NHWC with 3 channels."
        )

    if normalize:
        images = _normalize_to_01(images)
    else:
        images = images.astype(np.float32, copy=False)

    if output_layout == src_layout:
        return images
    if output_layout == "nchw" and src_layout == "nhwc":
        return images.transpose(0, 3, 1, 2)
    if output_layout == "nhwc" and src_layout == "nchw":
        return images.transpose(0, 2, 3, 1)
    raise ValueError(f"Unsupported output_layout: {output_layout}. Use 'nchw' or 'nhwc'.")
