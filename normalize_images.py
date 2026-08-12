#!/usr/bin/env python3
"""Percentile-normalize paired planar I-131 images."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def normalize_slice(array: np.ndarray, lower: float = 0.0, upper: float = 99.0) -> np.ndarray:
    """Clip one projection and scale it to [0, 1].

    The default lower bound is exactly zero, preserving the normalization used
    by the original training and inference scripts. A nonzero lower percentile
    is available only as an explicit opt-in.
    """
    data = np.asarray(array, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    samples = data[data > 0]
    if samples.size < 10:
        samples = data.reshape(-1)
    low_value = 0.0 if lower == 0.0 else float(np.percentile(samples, lower))
    high_value = float(np.percentile(samples, upper))
    clipped = np.clip(data, low_value, high_value)
    minimum, maximum = float(clipped.min()), float(clipped.max())
    if maximum <= minimum:
        return np.zeros_like(clipped, dtype=np.float32)
    return ((clipped - minimum) / (maximum - minimum)).astype(np.float32)


def normalize_array(array: np.ndarray, lower: float = 0.0, upper: float = 99.0) -> np.ndarray:
    """Normalize a 2-D image or each projection of a paired 3-D image."""
    data = np.squeeze(np.asarray(array))
    if data.ndim == 2:
        return normalize_slice(data, lower, upper)
    if data.ndim == 3:
        return np.stack(
            [normalize_slice(data[..., index], lower, upper) for index in range(data.shape[-1])],
            axis=-1,
        )
    raise ValueError(f"Expected a 2-D image or paired 3-D image; received shape {data.shape}")


def normalize_image(
    input_path: Path,
    output_path: Path,
    lower: float = 0.0,
    upper: float = 99.0,
) -> None:
    """Normalize one NIfTI image while preserving its spatial metadata."""
    image = nib.load(str(input_path))
    normalized = normalize_array(image.get_fdata(), lower, upper)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(normalized.astype(np.float32), image.affine, image.header),
        str(output_path),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing input NIfTI images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for normalized NIfTI images.")
    parser.add_argument("--lower-percentile", type=float, default=0.0)
    parser.add_argument("--upper-percentile", type=float, default=99.0)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing normalized files.")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if not 0 <= args.lower_percentile < args.upper_percentile <= 100:
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100")
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if args.input_dir.resolve() == args.output_dir.resolve():
        raise ValueError("Input and output directories must be different")

    files = sorted(path for path in args.input_dir.glob("*.nii.gz") if not path.name.endswith("-label.nii.gz"))
    if not files:
        raise FileNotFoundError(f"No NIfTI images found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for input_path in files:
        output_path = args.output_dir / input_path.name
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        normalize_image(input_path, output_path, args.lower_percentile, args.upper_percentile)
        written += 1

    print(f"Normalized: {written}; skipped existing: {skipped}; output: {args.output_dir}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
