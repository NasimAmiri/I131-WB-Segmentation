#!/usr/bin/env python3
"""Normalize paired images, run ensemble nnU-Net inference, and restack predictions."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from normalize_images import normalize_image


def case_name(path: Path) -> str:
    return path.name.removesuffix(".nii.gz")


def find_predict_command(dry_run: bool = False) -> list[str]:
    for candidate in ("nnUNetv2_predict", "nnUNetv2_predict_from_raw_data"):
        executable = shutil.which(candidate)
        if executable:
            return [executable]
    if dry_run:
        return ["nnUNetv2_predict"]
    try:
        __import__("nnunetv2")
    except ImportError as error:
        raise RuntimeError("nnU-Net v2 is not installed in the active environment") from error
    return [sys.executable, "-m", "nnunetv2.inference.predict_from_raw_data"]


def normalize_inputs(input_dir: Path, normalized_dir: Path, overwrite: bool) -> list[Path]:
    normalized_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for source in sorted(path for path in input_dir.glob("*.nii.gz") if not path.name.endswith("-label.nii.gz")):
        target = normalized_dir / source.name
        if overwrite or not target.exists():
            normalize_image(source, target)
        outputs.append(target)
    if not outputs:
        raise FileNotFoundError(f"No input NIfTI images found in {input_dir}")
    return outputs


def split_projections(images: list[Path], slices_dir: Path, overwrite: bool) -> list[dict[str, str]]:
    slices_dir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, str]] = []
    for image_path in images:
        image = nib.load(str(image_path))
        data = np.squeeze(image.get_fdata())
        if data.ndim != 3 or data.shape[-1] < 2:
            raise ValueError(f"Expected at least two projections in {image_path}; got shape {data.shape}")
        if data.shape[-1] > 2:
            print(
                f"Warning: {image_path.name} contains {data.shape[-1]} projections; "
                "using the first two as ANT and POST to preserve the original pipeline."
            )
        name = case_name(image_path)
        projections = {"ANT": data[..., 0], "POST": data[..., 1]}
        for view, array in projections.items():
            target = slices_dir / f"{name}_{view}_0000.nii.gz"
            if overwrite or not target.exists():
                nib.save(nib.Nifti1Image(array.astype(np.float32), image.affine, image.header), str(target))
        metadata.append({"case_name": name, "normalized_image": str(image_path.resolve())})
    return metadata


def prediction_command(args: argparse.Namespace, slices_dir: Path, predictions_dir: Path) -> list[str]:
    command = [
        *find_predict_command(args.dry_run),
        "-i", str(slices_dir),
        "-o", str(predictions_dir),
        "-d", str(args.dataset_id),
        "-c", args.configuration,
        "-f", *[str(fold) for fold in args.folds],
        "-chk", args.checkpoint,
    ]
    if args.trainer:
        command.extend(["-tr", args.trainer])
    if args.plans_name:
        command.extend(["-p", args.plans_name])
    return command


def stack_predictions(
    metadata: list[dict[str, str]],
    predictions_dir: Path,
    masks_dir: Path,
    overwrite: bool,
) -> None:
    masks_dir.mkdir(parents=True, exist_ok=True)
    for entry in metadata:
        name = entry["case_name"]
        ant_path = predictions_dir / f"{name}_ANT.nii.gz"
        post_path = predictions_dir / f"{name}_POST.nii.gz"
        if not ant_path.exists() or not post_path.exists():
            raise FileNotFoundError(f"Missing ANT or POST prediction for one input case")
        target = masks_dir / f"{name}_mask.nii.gz"
        if target.exists() and not overwrite:
            continue
        ant = np.squeeze(nib.load(str(ant_path)).get_fdata())
        post = np.squeeze(nib.load(str(post_path)).get_fdata())
        source = nib.load(entry["normalized_image"])
        stacked = np.stack((ant, post), axis=-1).astype(np.uint8)
        nib.save(nib.Nifti1Image(stacked, source.affine, source.header), str(target))


def run(args: argparse.Namespace) -> None:
    work_dir = args.work_dir.resolve()
    normalized_dir = work_dir / "normalized_images"
    slices_dir = work_dir / "inference_slices"
    predictions_dir = work_dir / "inference_predictions"
    masks_dir = args.output_masks.resolve() if args.output_masks else work_dir / "inference_stacked_masks"

    os.environ.update(
        nnUNet_raw=str(args.nnunet_raw.resolve()),
        nnUNet_preprocessed=str(args.nnunet_preprocessed.resolve()),
        nnUNet_results=str(args.nnunet_results.resolve()),
    )
    command = prediction_command(args, slices_dir, predictions_dir)
    if args.dry_run:
        print("Would prepare normalized images in:", normalized_dir)
        print("Would prepare projection slices in:", slices_dir)
        print("$ " + " ".join(command))
        print("Would stack masks in:", masks_dir)
        return

    input_images = sorted(path for path in args.input_dir.glob("*.nii.gz") if not path.name.endswith("-label.nii.gz"))
    if args.input_normalized:
        images = input_images
        if not images:
            raise FileNotFoundError(f"No normalized NIfTI images found in {args.input_dir}")
    else:
        images = normalize_inputs(args.input_dir, normalized_dir, args.overwrite)
    metadata = split_projections(images, slices_dir, args.overwrite)
    (work_dir / "inference_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    predictions_dir.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(command))
    subprocess.run(command, check=True)
    stack_predictions(metadata, predictions_dir, masks_dir, args.overwrite)
    print(f"Created {len(metadata)} stacked prediction masks in {masks_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-masks", type=Path, default=None)
    parser.add_argument("--input-normalized", action="store_true")
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    parser.add_argument("--nnunet-preprocessed", type=Path, required=True)
    parser.add_argument("--nnunet-results", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--folds", type=int, nargs="+", choices=range(5), default=list(range(5)))
    parser.add_argument("--checkpoint", default="checkpoint_best.pth")
    parser.add_argument("--trainer", default=None)
    parser.add_argument("--plans-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
