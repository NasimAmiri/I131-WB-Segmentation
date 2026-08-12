#!/usr/bin/env python3
"""Convert paired planar NIfTI studies into an nnU-Net v2 2-D dataset."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np


LABEL_PATTERN = re.compile(r'^\s*(\d+)\s+.*?"([^"]+)"\s*$')


def parse_labels_file(path: Path) -> dict[int, str]:
    """Parse an ITK-SNAP labels file into an integer-to-name mapping."""
    labels: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = LABEL_PATTERN.match(line)
        if match:
            index = int(match.group(1))
            name = match.group(2).strip().lower()
            labels[index] = "background" if index == 0 else ("unknown" if name == "unkown" else name)
    if not labels:
        raise ValueError(f"No labels could be parsed from {path}")
    labels[0] = "background"
    return labels


def image_case_name(path: Path) -> str:
    return path.name.removesuffix(".nii.gz")


def label_case_name(path: Path) -> str:
    return path.name.removesuffix("-label.nii.gz")


def build_grouped_splits(patient_frames: dict[str, list[str]], folds: int) -> list[dict[str, list[str]]]:
    """Build deterministic contiguous folds while keeping a patient's projections together."""
    if folds < 2:
        raise ValueError("At least two folds are required")
    patients = sorted(patient_frames)
    if len(patients) < folds:
        raise ValueError(f"Cannot create {folds} folds from {len(patients)} patients")

    base = len(patients) // folds
    split_sizes = [base] * (folds - 1) + [len(patients) - base * (folds - 1)]
    splits: list[dict[str, list[str]]] = []
    start = 0
    for size in split_sizes:
        validation_patients = set(patients[start : start + size])
        start += size
        train = [frame for patient in patients if patient not in validation_patients for frame in patient_frames[patient]]
        validation = [frame for patient in patients if patient in validation_patients for frame in patient_frames[patient]]
        splits.append({"train": train, "val": validation})
    return splits


def save_2d(array: np.ndarray, source: nib.Nifti1Image, path: Path, dtype: np.dtype) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(array, dtype=dtype), source.affine, source.header), str(path))


def prepare_dataset(args: argparse.Namespace) -> dict[str, object]:
    labels = parse_labels_file(args.labels_file)
    dataset_dir = args.nnunet_raw / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    existing = list(images_tr.glob("*.nii.gz")) + list(labels_tr.glob("*.nii.gz"))
    if existing and not args.overwrite:
        raise FileExistsError(f"Generated files already exist in {dataset_dir}; use --overwrite")
    if args.overwrite:
        for path in existing:
            path.unlink()
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    image_paths = {image_case_name(path): path for path in args.images_dir.glob("*.nii.gz") if not path.name.endswith("-label.nii.gz")}
    label_paths = {label_case_name(path): path for path in args.labels_dir.glob("*-label.nii.gz")}
    cases = sorted(image_paths.keys() & label_paths.keys())
    if not cases:
        raise FileNotFoundError("No matched image-label pairs were found")
    unmatched = sorted(image_paths.keys() ^ label_paths.keys())
    if unmatched:
        raise ValueError(f"Found {len(unmatched)} unmatched image or label filenames")

    patient_frames: dict[str, list[str]] = {}
    for case_name in cases:
        image = nib.load(str(image_paths[case_name]))
        label = nib.load(str(label_paths[case_name]))
        image_data = np.squeeze(image.get_fdata())
        label_data = np.squeeze(label.get_fdata())
        if image_data.shape != label_data.shape:
            raise ValueError(f"Shape mismatch for case {case_name}: {image_data.shape} vs {label_data.shape}")

        if image_data.ndim == 2:
            projections = [("slice00", image_data, label_data)]
        elif image_data.ndim == 3 and image_data.shape[-1] >= 2:
            if image_data.shape[-1] > 2:
                print(
                    f"Warning: {case_name} contains {image_data.shape[-1]} projections; "
                    "using the first two as ANT and POST to preserve the original pipeline."
                )
            projections = [
                ("ANT", image_data[..., 0], label_data[..., 0]),
                ("POST", image_data[..., 1], label_data[..., 1]),
            ]
        else:
            raise ValueError(f"Expected a 2-D image or at least two projections for {case_name}; got {image_data.shape}")

        patient_frames[case_name] = []
        for projection, image_slice, label_slice in projections:
            frame_name = f"{case_name}_{projection}"
            patient_frames[case_name].append(frame_name)
            save_2d(image_slice, image, images_tr / f"{frame_name}_0000.nii.gz", np.float32)
            save_2d(np.rint(label_slice), label, labels_tr / f"{frame_name}.nii.gz", np.uint8)

    splits = build_grouped_splits(patient_frames, args.folds)
    (dataset_dir / "splits_final.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    dataset_json = {
        "channel_names": {"0": "SPECT"},
        "labels": {name: index for index, name in sorted(labels.items())},
        "numTraining": sum(len(frames) for frames in patient_frames.values()),
        "file_ending": ".nii.gz",
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")
    return {
        "dataset_dir": dataset_dir,
        "patients": len(patient_frames),
        "projections": dataset_json["numTraining"],
        "folds": len(splits),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--labels-file", type=Path, required=True)
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--dataset-name", default="I131SPECT2D")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    result = prepare_dataset(parse_args())
    print(
        f"Prepared {result['patients']} patients / {result['projections']} projections "
        f"in {result['dataset_dir']} with {result['folds']} patient-grouped folds."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
