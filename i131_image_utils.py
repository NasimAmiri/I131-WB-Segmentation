"""Shared orientation and rendering utilities for paired planar I-131 images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageFont


LABEL_NAMES = {
    0: "background",
    1: "thyroid",
    2: "metastasis",
    3: "salivary",
    4: "GI",
    5: "bladder",
    6: "liver",
    7: "unknown",
    8: "contamination",
}

LABEL_COLORS = {
    1: (228, 58, 58),
    2: (72, 199, 82),
    3: (73, 97, 224),
    4: (246, 224, 73),
    5: (65, 213, 215),
    6: (145, 85, 25),
    7: (157, 119, 215),
    8: (233, 74, 223),
}

COMPARISON_COLORS = {
    "overlap": (255, 230, 0),
    "gt_only": (0, 210, 0),
    "pred_only": (230, 50, 50),
}


@dataclass(frozen=True)
class Transform:
    rotation_clockwise: int
    flip_lr: bool


def load_nifti(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    return np.asarray(image.get_fdata()), image


def select_view(array: np.ndarray, view: str) -> np.ndarray:
    data = np.squeeze(np.asarray(array))
    if data.ndim == 2:
        return data
    if data.ndim == 3 and data.shape[-1] == 2:
        return data[..., 0 if view.upper() == "ANT" else 1]
    raise ValueError(f"Expected a 2-D array or paired projections; received {data.shape}")


def robust_normalize(image: np.ndarray) -> np.ndarray:
    data = np.nan_to_num(np.asarray(image, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(data, [1.0, 99.5])
    if high <= low:
        low, high = float(data.min()), float(data.max())
    if high <= low:
        return np.zeros_like(data)
    return (np.clip(data, low, high) - low) / (high - low)


def rotate_clockwise(array: np.ndarray, degrees: int) -> np.ndarray:
    rotations = {0: 0, 90: 3, 180: 2, 270: 1}
    normalized = degrees % 360
    if normalized not in rotations:
        raise ValueError(f"Rotation must be a right angle; received {degrees}")
    return np.rot90(np.asarray(array), k=rotations[normalized])


def apply_transform(array: np.ndarray, transform: Transform) -> np.ndarray:
    output = rotate_clockwise(array, transform.rotation_clockwise)
    return np.fliplr(output) if transform.flip_lr else output


def _center_y(array: np.ndarray) -> float:
    normalized = robust_normalize(array)
    threshold = np.percentile(normalized, 65)
    weights = np.clip(normalized - threshold, 0, None)
    if float(weights.sum()) <= 1e-6:
        return 0.5
    y = np.arange(weights.shape[0], dtype=np.float32)[:, None]
    return float((y * weights).sum() / weights.sum() / max(weights.shape[0] - 1, 1))


def _label_center(mask: np.ndarray, labels: tuple[int, ...]) -> float | None:
    selected = np.isin(np.rint(mask).astype(np.int16), labels)
    if not np.any(selected):
        return None
    return float(np.where(selected)[0].mean() / max(selected.shape[0] - 1, 1))


def detect_transform(
    ant_image: np.ndarray,
    post_image: np.ndarray,
    ant_ground_truth: np.ndarray | None = None,
    post_ground_truth: np.ndarray | None = None,
    *,
    flip_lr: bool = True,
) -> Transform:
    """Choose an upright right-angle rotation, optionally applying the study laterality correction."""
    combined_image = robust_normalize(ant_image) + robust_normalize(post_image)
    combined_ground_truth = None
    if ant_ground_truth is not None and post_ground_truth is not None:
        combined_ground_truth = np.maximum(
            np.rint(ant_ground_truth).astype(np.int16),
            np.rint(post_ground_truth).astype(np.int16),
        )

    scored: list[tuple[float, int]] = []
    for rotation in (0, 90, 180, 270):
        image = rotate_clockwise(combined_image, rotation)
        height, width = image.shape
        score = 4.0 if height <= width else 0.0
        score += abs(_center_y(image) - 0.35)
        if combined_ground_truth is not None and np.any(combined_ground_truth):
            ground_truth = rotate_clockwise(combined_ground_truth, rotation)
            upper = _label_center(ground_truth, (1, 3))
            lower = _label_center(ground_truth, (4, 5, 6))
            if upper is not None:
                score += max(0.0, upper - 0.45) * 6.0
            if lower is not None:
                score += max(0.0, 0.55 - lower) * 6.0
            if upper is not None and lower is not None:
                score += max(0.0, 0.20 - (lower - upper)) * 10.0
        scored.append((score, rotation))
    return Transform(rotation_clockwise=min(scored)[1], flip_lr=flip_lr)


def resize_nearest(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    image = Image.fromarray(np.asarray(array, dtype=np.float32), mode="F")
    return np.asarray(image.resize((width, height), Image.Resampling.NEAREST))


def scan_rgb(scan: np.ndarray) -> np.ndarray:
    gray = 255 - np.rint(robust_normalize(scan) * 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    integer_mask = np.rint(mask).astype(np.int16)
    output = np.zeros((*integer_mask.shape, 3), dtype=np.uint8)
    for label, color in LABEL_COLORS.items():
        output[integer_mask == label] = color
    return output


def overlay_mask(scan: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    base = scan_rgb(scan)
    resized = resize_nearest(mask, base.shape[:2])
    colors = colorize_mask(resized)
    foreground = np.rint(resized).astype(np.int16) > 0
    output = base.copy()
    output[foreground] = (
        (1.0 - alpha) * output[foreground] + alpha * colors[foreground]
    ).astype(np.uint8)
    return output


def comparison_mask(ground_truth: np.ndarray, prediction: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    ground_truth = resize_nearest(ground_truth, shape)
    prediction = resize_nearest(prediction, shape)
    reference = np.rint(ground_truth).astype(np.int16) > 0
    predicted = np.rint(prediction).astype(np.int16) > 0
    output = np.full((*shape, 3), 255, dtype=np.uint8)
    output[reference & predicted] = COMPARISON_COLORS["overlap"]
    output[reference & ~predicted] = COMPARISON_COLORS["gt_only"]
    output[~reference & predicted] = COMPARISON_COLORS["pred_only"]
    return output


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
    )
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()
