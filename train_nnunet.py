#!/usr/bin/env python3
"""Plan, preprocess, and train all requested nnU-Net v2 folds."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def run_command(command: list[str], dry_run: bool) -> None:
    print("$ " + " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def install_grouped_splits(args: argparse.Namespace, dry_run: bool) -> None:
    dataset = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    source = args.nnunet_raw / dataset / "splits_final.json"
    target = args.nnunet_preprocessed / dataset / "splits_final.json"
    if not source.exists() and not dry_run:
        raise FileNotFoundError(f"Patient-grouped split file not found: {source}")
    print(f"Install patient-grouped folds: {source} -> {target}")
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def train(args: argparse.Namespace) -> None:
    os.environ.update(
        nnUNet_raw=str(args.nnunet_raw.resolve()),
        nnUNet_preprocessed=str(args.nnunet_preprocessed.resolve()),
        nnUNet_results=str(args.nnunet_results.resolve()),
    )
    if not args.dry_run:
        for path in (args.nnunet_raw, args.nnunet_preprocessed, args.nnunet_results):
            path.mkdir(parents=True, exist_ok=True)

    if not args.skip_preprocessing:
        command = [
            "nnUNetv2_plan_and_preprocess",
            "-d",
            str(args.dataset_id),
            "-c",
            args.configuration,
            "--verify_dataset_integrity",
        ]
        if args.plans_name:
            command.extend(["-overwrite_plans_name", args.plans_name])
        run_command(command, args.dry_run)
    install_grouped_splits(args, args.dry_run)

    for fold in args.folds:
        command = ["nnUNetv2_train", str(args.dataset_id), args.configuration, str(fold)]
        if args.trainer:
            command.extend(["-tr", args.trainer])
        if args.plans_name:
            command.extend(["-p", args.plans_name])
        if args.continue_training:
            command.append("--c")
        run_command(command, args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    parser.add_argument("--nnunet-preprocessed", type=Path, required=True)
    parser.add_argument("--nnunet-results", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--dataset-name", default="I131SPECT2D")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--folds", type=int, nargs="+", choices=range(5), default=list(range(5)))
    parser.add_argument("--trainer", default=None)
    parser.add_argument("--plans-name", default=None)
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    train(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
