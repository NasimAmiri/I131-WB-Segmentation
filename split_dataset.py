#!/usr/bin/env python3
"""Split paired images and labels into training, external, and artifact cohorts."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


COHORTS = ("train", "external", "artifact_cases")


@dataclass(frozen=True)
class SplitOptions:
    spreadsheet: Path
    images_dir: Path
    labels_dir: Path
    output_root: Path
    patient_column: str
    quality_column: str
    scanner_column: str
    external_scanner: str
    mode: str = "copy"
    overwrite: bool = False


def normalize_case_name(value: object) -> str | None:
    """Convert a spreadsheet patient value into the corresponding filename stem."""
    if pd.isna(value):
        return None
    try:
        return f"{int(float(value)):05d}"
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or None


def classify_rows(frame: pd.DataFrame, options: SplitOptions) -> dict[str, list[str]]:
    """Apply the study cohort rules and return non-overlapping case-name lists."""
    required = {options.patient_column, options.quality_column, options.scanner_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Spreadsheet is missing required columns: {', '.join(missing)}")

    cohorts: dict[str, set[str]] = {name: set() for name in COHORTS}
    for _, row in frame.iterrows():
        case_name = normalize_case_name(row[options.patient_column])
        if not case_name:
            continue
        quality = str(row[options.quality_column]).strip().lower()
        scanner = str(row[options.scanner_column]).strip()
        if "artifact" in quality or "poor" in quality:
            cohort = "artifact_cases"
        elif scanner == options.external_scanner:
            cohort = "external"
        else:
            cohort = "train"
        cohorts[cohort].add(case_name)
    return {name: sorted(values) for name, values in cohorts.items()}


def transfer_pair(case_name: str, cohort: str, options: SplitOptions) -> list[Path]:
    """Copy or move one image-label pair and return any missing source paths."""
    destination = options.output_root / cohort
    image_source = options.images_dir / f"{case_name}.nii.gz"
    label_source = options.labels_dir / f"{case_name}-label.nii.gz"
    pairs = (
        (image_source, destination / "Images_norm" / image_source.name),
        (label_source, destination / "labels" / label_source.name),
    )
    missing: list[Path] = []
    for source, target in pairs:
        if not source.exists():
            missing.append(source)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not options.overwrite:
            raise FileExistsError(f"Destination exists; use --overwrite to replace it: {target}")
        if target.exists():
            target.unlink()
        if options.mode == "move":
            shutil.move(str(source), str(target))
        else:
            shutil.copy2(source, target)
    return missing


def execute_split(options: SplitOptions) -> dict[str, int]:
    frame = pd.read_excel(options.spreadsheet)
    cohorts = classify_rows(frame, options)
    missing: list[Path] = []
    for cohort, case_names in cohorts.items():
        for case_name in case_names:
            missing.extend(transfer_pair(case_name, cohort, options))
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:20])
        raise FileNotFoundError(f"Missing {len(missing)} expected source files:\n{preview}")
    return {name: len(case_names) for name, case_names in cohorts.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spreadsheet", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--patient-column", default="Patient ID")
    parser.add_argument("--quality-column", default="Image_quality")
    parser.add_argument("--scanner-column", default="ManufacturersModelName")
    parser.add_argument("--external-scanner", default="Discovery NM 630")
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move source files instead of copying them. Copying is the safe default.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    values = vars(args)
    values["mode"] = "move" if values.pop("move") else "copy"
    options = SplitOptions(**values)
    counts = execute_split(options)
    for cohort in COHORTS:
        print(f"{cohort}: {counts[cohort]}")
    print(f"Transfer mode: {options.mode}; output: {options.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
