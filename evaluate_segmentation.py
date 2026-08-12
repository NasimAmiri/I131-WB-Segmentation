"""Evaluate paired segmentation masks with ``segmentationmetrics``."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

try:
    import nibabel as nib
except ImportError:
    print("[ERROR] nibabel is not installed. Install requirements.txt first.")
    raise SystemExit(1)

try:
    import SimpleITK as sitk
except ImportError:
    print("[ERROR] SimpleITK is not installed. Install requirements.txt first.")
    raise SystemExit(1)

try:
    import segmentationmetrics as sm
except ImportError:
    print("[ERROR] segmentationmetrics is not installed. Install requirements.txt first.")
    raise SystemExit(1)


DEFAULT_LABELS = {
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

LABEL_LINE_RE = re.compile(r'^\s*(\d+)\s+.*?"([^"]+)"\s*$')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate segmentation predictions with segment_match_eval_updated "
            "by matching files with the same name in ground-truth and "
            "prediction directories."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Base directory that contains inference_slices and inference_predictions.",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=None,
        help="Ground-truth directory. Defaults to <root>/inference_slices.",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=None,
        help="Prediction directory. Defaults to <root>/inference_predictions.",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=Path("labels_itksnap.txt"),
        help="Shared ITK-SNAP labels file used when GT/prediction-specific files are not provided.",
    )
    parser.add_argument(
        "--gt-labels-file",
        type=Path,
        default=None,
        help="Ground-truth ITK-SNAP labels file. If omitted, it is auto-detected.",
    )
    parser.add_argument(
        "--pred-labels-file",
        type=Path,
        default=None,
        help="Prediction ITK-SNAP labels file. If omitted, it is auto-detected.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file. Defaults to <root>/evaluation_results.json.",
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=0.5,
        help="Average per-case Dice threshold used for failure-case reporting.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on the number of matched files to evaluate.",
    )
    parser.add_argument(
        "--empty-reference-predicted-behaviour",
        type=str,
        default="perfect",
        choices=["perfect", "zero", "exclude"],
        help="How segment_match_eval_updated handles empty-reference + empty-prediction labels.",
    )
    parser.add_argument(
        "--force-match",
        action="store_true",
        help="Resample prediction into reference space inside segment_match_eval_updated.",
    )
    return parser.parse_args()


def canonicalize_label_name(name: str) -> str:
    normalized = name.strip().lower().replace("clear label", "background")
    if normalized == "unkown":
        normalized = "unknown"
    return normalized


def load_labels(labels_file: Path) -> dict[int, str]:
    if not labels_file.exists():
        return DEFAULT_LABELS.copy()

    labels: dict[int, str] = {}
    for line in labels_file.read_text().splitlines():
        match = LABEL_LINE_RE.match(line)
        if not match:
            continue
        idx = int(match.group(1))
        name = canonicalize_label_name(match.group(2))
        labels[idx] = "background" if idx == 0 else name

    return labels or DEFAULT_LABELS.copy()


def to_integer_mask(data: np.ndarray, source_name: str) -> np.ndarray:
    rounded = np.rint(data)
    if not np.allclose(data, rounded, atol=1e-6):
        unique_preview = np.unique(data)[:12]
        raise ValueError(
            f"{source_name} is not a categorical segmentation mask. "
            f"Example values: {unique_preview}"
        )
    if np.any(rounded < 0):
        raise ValueError(f"{source_name} contains negative labels, which is invalid.")
    return rounded.astype(np.int16)


def load_nifti_array(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj)


def load_nifti(path: Path) -> nib.Nifti1Image:
    return nib.load(str(path))


def get_spacing_mm(image: nib.Nifti1Image, ndim: int) -> tuple[float, ...]:
    zooms = image.header.get_zooms()[:ndim]
    if not zooms:
        return tuple([1.0] * ndim)
    return tuple(float(z) for z in zooms)


def resolve_labels_file(
    root: Path,
    default_labels_file: Path,
    explicit_file: Path | None,
    purpose: str,
) -> Path:
    if explicit_file is not None:
        return explicit_file.resolve() if explicit_file.is_absolute() else (Path.cwd() / explicit_file).resolve()

    purpose_specific = root / f"labels_itksnap_{purpose}.txt"
    if purpose_specific.exists():
        return purpose_specific.resolve()

    root_candidate = root / default_labels_file
    if root_candidate.exists():
        return root_candidate.resolve()

    return (Path.cwd() / default_labels_file).resolve()


def infer_view(case_id: str) -> str:
    if case_id.endswith("_ANT"):
        return "ANT"
    if case_id.endswith("_POST"):
        return "POST"
    return "OTHER"


def summarize_metric_values(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "count": 0, "nonfinite_count": 0}

    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return {
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "std": float(np.std(finite)) if finite.size else float("nan"),
        "count": int(arr.size),
        "nonfinite_count": int(arr.size - finite.size),
    }


def build_metric_store(class_names: list[str], metric_names: list[str]) -> dict[str, dict[str, list[float]]]:
    return {
        class_name: {metric_name: [] for metric_name in metric_names}
        for class_name in class_names
    }


def build_support_store(class_names: list[str]) -> dict[str, dict[str, int]]:
    return {
        class_name: {
            "valid_count": 0,
            "empty_empty_count": 0,
            "gt_positive_count": 0,
            "pred_positive_count": 0,
        }
        for class_name in class_names
    }


def summarize_overall_metrics(
    class_names: list[str],
    metric_names: list[str],
    per_class_metrics: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, float | int]]:
    overall_metric_values = {
        metric_name: [value for class_name in class_names for value in per_class_metrics[class_name][metric_name]]
        for metric_name in metric_names
    }
    return {
        metric_name: summarize_metric_values(values)
        for metric_name, values in overall_metric_values.items()
    }


def summarize_per_class(
    class_names: list[str],
    per_class_metrics: dict[str, dict[str, list[float]]],
    support_counts: dict[str, dict[str, int]],
) -> dict[str, dict[str, float | str]]:
    summary: dict[str, dict[str, float | str]] = {}
    for class_name in class_names:
        class_summary = {"name": class_name, **support_counts[class_name]}
        for metric_name, metric_values in per_class_metrics[class_name].items():
            stats = summarize_metric_values(metric_values)
            class_summary[f"{metric_name}_mean"] = stats["mean"]
            class_summary[f"{metric_name}_std"] = stats["std"]
            class_summary[f"{metric_name}_count"] = stats["count"]
            class_summary[f"{metric_name}_nonfinite_count"] = stats["nonfinite_count"]
        summary[class_name] = class_summary
    return summary


def make_json_safe(value):
    if isinstance(value, dict):
        return {key: make_json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def match_space(moving: sitk.Image, reference: sitk.Image) -> sitk.Image:
    return sitk.Resample(
        moving,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        moving.GetPixelID(),
    )


def count_segmented_objects(mask: np.ndarray) -> tuple[int, np.ndarray]:
    labeled, num_objects = ndimage.label(mask.astype(np.uint8))
    return int(num_objects), labeled


def select_segment_number(image: sitk.Image, label: int) -> tuple[sitk.Image]:
    return (sitk.Cast(image == int(label), sitk.sitkUInt8),)


def area_difference_mm2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spacing_mm: tuple[float, ...],
) -> float:
    pixel_area = float(np.prod(spacing_mm))
    return float(abs(int(y_true.sum()) - int(y_pred.sum())) * pixel_area)


def segmentation_metrics_to_frame(metrics_obj) -> pd.DataFrame:
    if hasattr(metrics_obj, "get_df"):
        df = metrics_obj.get_df().T
        df.columns = df.iloc[0]
        return df.drop("Metric").reset_index(drop=True)

    data = {
        "Dice": float(getattr(metrics_obj, "dice", np.nan)),
        "Jaccard": float(getattr(metrics_obj, "jaccard", np.nan)),
        "Sensitivity": float(getattr(metrics_obj, "sensitivity", np.nan)),
        "Specificity": float(getattr(metrics_obj, "specificity", np.nan)),
        "Precision": float(getattr(metrics_obj, "precision", np.nan)),
        "Accuracy": float(getattr(metrics_obj, "accuracy", np.nan)),
        "Mean Surface Distance": float(getattr(metrics_obj, "mean_surface_distance", np.nan)),
        "Hausdorff Distance": float(getattr(metrics_obj, "hausdorff_distance", np.nan)),
        "True Volume (ml)": float(getattr(metrics_obj, "true_volume", np.nan)),
        "Predicted Volume (ml)": float(getattr(metrics_obj, "predicted_volume", np.nan)),
        "Volume Difference (ml)": float(getattr(metrics_obj, "volume_difference", np.nan)),
    }
    return pd.DataFrame([data])


def segment_match_eval_updated(
    reference,
    predicted,
    pixdim=(1, 1, 1),
    force_match=False,
    empty_refernce_predicted_behaviour="perfect",
    expected_label_values=None,
):
    def to_image_array(data):
        if isinstance(data, str):
            image = sitk.ReadImage(data)
            array = sitk.GetArrayFromImage(image)
            spacing = image.GetSpacing()
        elif isinstance(data, sitk.Image):
            image = data
            array = sitk.GetArrayFromImage(image)
            spacing = image.GetSpacing()
        elif isinstance(data, np.ndarray):
            array = data
            image = sitk.GetImageFromArray(array)
            image.SetSpacing(tuple(float(v) for v in pixdim))
            spacing = tuple(float(v) for v in pixdim)
        else:
            raise TypeError("Unsupported input type")
        return image, array, spacing

    ref_img, ref_arr, spacing = to_image_array(reference)
    pred_img, pred_arr, _ = to_image_array(predicted)

    if force_match:
        pred_img = match_space(pred_img, ref_img)
        pred_arr = sitk.GetArrayFromImage(pred_img)

    reference_labels = [int(l) for l in np.unique(ref_arr) if int(l) != 0]
    predicted_labels = [int(l) for l in np.unique(pred_arr) if int(l) != 0]
    labels_both = list(set(reference_labels + predicted_labels))
    labels = expected_label_values if expected_label_values is not None else labels_both

    if not labels:
        return pd.DataFrame()

    results = []
    for label in labels:
        ref_bin = ref_arr == label
        pred_bin = pred_arr == label
        empty_reference = int(ref_bin.sum()) == 0
        empty_predicted = int(pred_bin.sum()) == 0

        if empty_reference and empty_predicted:
            if empty_refernce_predicted_behaviour == "perfect":
                overlap_override = {
                    "Dice": 1.0,
                    "Jaccard": 1.0,
                    "Sensitivity": 1.0,
                    "Specificity": 1.0,
                    "Precision": 1.0,
                    "Accuracy": 1.0,
                }
                surface_override = {
                    "Mean Surface Distance": 0.0,
                    "Hausdorff Distance": 0.0,
                }
                area_diff_override = 0.0
            elif empty_refernce_predicted_behaviour == "zero":
                overlap_override = {
                    "Dice": 0.0,
                    "Jaccard": 0.0,
                }
                surface_override = {}
                area_diff_override = 0.0
            elif empty_refernce_predicted_behaviour == "exclude":
                continue
            else:
                raise ValueError("Invalid empty_refernce_predicted_behaviour")

            metrics = segmentation_metrics_to_frame(
                sm.SegmentationMetrics(pred_bin.astype(np.uint8), ref_bin.astype(np.uint8), spacing)
            )
            for metric_name, metric_value in overlap_override.items():
                metrics[metric_name] = metric_value
            for metric_name, metric_value in surface_override.items():
                metrics[metric_name] = metric_value
        else:
            metrics = segmentation_metrics_to_frame(
                sm.SegmentationMetrics(pred_bin.astype(np.uint8), ref_bin.astype(np.uint8), spacing)
            )

        metrics["label"] = int(label)
        metrics["ref_n_objects"] = count_segmented_objects(ref_bin)[0]
        metrics["pred_n_objects"] = count_segmented_objects(pred_bin)[0]
        metrics["area_difference_mm2"] = area_difference_mm2(ref_bin, pred_bin, spacing)
        if empty_reference and empty_predicted and empty_refernce_predicted_behaviour in {"perfect", "zero"}:
            metrics["area_difference_mm2"] = area_diff_override
        metrics["empty_reference"] = empty_reference
        metrics["empty_predicted"] = empty_predicted
        metrics["empty_refernce_predicted_behaviour"] = empty_refernce_predicted_behaviour
        metrics["expected_label_values"] = (
            [list(int(v) for v in expected_label_values)] if expected_label_values is not None else None
        )
        results.append(metrics)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def build_shared_label_masks(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    gt_name_to_idx: dict[str, int],
    pred_name_to_idx: dict[str, int],
    class_names: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    gt_eval = np.zeros_like(gt_mask, dtype=np.int16)
    pred_eval = np.zeros_like(pred_mask, dtype=np.int16)
    shared_idx_to_name: dict[int, str] = {}

    for shared_idx, class_name in enumerate(class_names, start=1):
        gt_eval[gt_mask == gt_name_to_idx[class_name]] = shared_idx
        pred_eval[pred_mask == pred_name_to_idx[class_name]] = shared_idx
        shared_idx_to_name[shared_idx] = class_name

    return gt_eval, pred_eval, shared_idx_to_name


def extract_metric_value(row: pd.Series, candidates: list[str]) -> float:
    for column_name in candidates:
        if column_name in row and pd.notna(row[column_name]):
            return float(row[column_name])
    return float("nan")


def validate_first_match(gt_file: Path, pred_file: Path) -> None:
    gt_data = load_nifti_array(gt_file)
    pred_data = load_nifti_array(pred_file)

    if gt_data.shape != pred_data.shape:
        raise ValueError(
            f"First matched pair has mismatched shape: {gt_file.name} -> "
            f"{gt_data.shape} vs {pred_data.shape}"
        )

    to_integer_mask(gt_data, f"ground truth '{gt_file}'")
    to_integer_mask(pred_data, f"prediction '{pred_file}'")


def main() -> bool:
    args = parse_args()

    root = args.root.resolve()
    gt_dir = (args.gt_dir or (root / "inference_slices")).resolve()
    pred_dir = (args.pred_dir or (root / "inference_predictions")).resolve()
    output_file = (args.output or (root / "evaluation_results.json")).resolve()
    gt_labels_file = resolve_labels_file(root, args.labels_file, args.gt_labels_file, "inference_slices")
    pred_labels_file = resolve_labels_file(root, args.labels_file, args.pred_labels_file, "inference_predictions")

    print("=" * 60)
    print("SEGMENTATION EVALUATION")
    print("=" * 60)
    print(f"[INFO] Root           : {root}")
    print(f"[INFO] Ground truth   : {gt_dir}")
    print(f"[INFO] Predictions    : {pred_dir}")
    print(f"[INFO] GT labels map  : {gt_labels_file}")
    print(f"[INFO] Pred labels map: {pred_labels_file}")
    print(f"[INFO] Empty behavior : {args.empty_reference_predicted_behaviour}")

    if not gt_dir.exists():
        print(f"[ERROR] Ground-truth directory not found: {gt_dir}")
        return False
    if not pred_dir.exists():
        print(f"[ERROR] Prediction directory not found: {pred_dir}")
        return False

    gt_files = {p.name: p for p in sorted(gt_dir.glob("*.nii.gz"))}
    pred_files = {p.name: p for p in sorted(pred_dir.glob("*.nii.gz"))}
    common_names = sorted(set(gt_files) & set(pred_files))
    gt_only = sorted(set(gt_files) - set(pred_files))
    pred_only = sorted(set(pred_files) - set(gt_files))

    print(f"[INFO] GT files        : {len(gt_files)}")
    print(f"[INFO] Prediction files: {len(pred_files)}")
    print(f"[INFO] Matched files   : {len(common_names)}")

    if not common_names:
        print("[ERROR] No matching .nii.gz files found between ground truth and predictions.")
        return False

    if args.max_files is not None:
        common_names = common_names[: args.max_files]
        print(f"[INFO] Evaluating only the first {len(common_names)} matched files (--max-files).")

    try:
        validate_first_match(gt_files[common_names[0]], pred_files[common_names[0]])
    except Exception as exc:
        print("[ERROR] The matched directories do not look like mask-vs-mask segmentation data.")
        print(f"        First pair checked: {common_names[0]}")
        print(f"        Details: {exc}")
        return False

    gt_labels = load_labels(gt_labels_file)
    pred_labels = load_labels(pred_labels_file)
    gt_name_to_idx = {name: idx for idx, name in gt_labels.items() if idx > 0}
    pred_name_to_idx = {name: idx for idx, name in pred_labels.items() if idx > 0}
    class_names = sorted(set(gt_name_to_idx) & set(pred_name_to_idx))
    metric_names = [
        "dice",
        "jaccard",
        "sensitivity",
        "specificity",
        "precision",
        "accuracy",
        "mean_surface_distance_mm",
        "hausdorff_distance_mm",
        "area_difference_mm2",
    ]
    per_class_metrics = build_metric_store(class_names, metric_names)
    per_view_metrics = {view: build_metric_store(class_names, metric_names) for view in ["ANT", "POST", "OTHER"]}
    per_class_support = build_support_store(class_names)
    per_view_support = {view: build_support_store(class_names) for view in ["ANT", "POST", "OTHER"]}

    case_metrics = []
    errors = []

    print(f"\nEvaluating {len(common_names)} matched files...")
    for idx, name in enumerate(common_names, 1):
        gt_file = gt_files[name]
        pred_file = pred_files[name]
        case_id = name.replace(".nii.gz", "")
        view = infer_view(case_id)

        try:
            gt_img = load_nifti(gt_file)
            pred_img = load_nifti(pred_file)
            gt_data = np.asarray(gt_img.dataobj)
            pred_data = np.asarray(pred_img.dataobj)

            if gt_data.shape != pred_data.shape:
                errors.append(f"{case_id}: shape mismatch {gt_data.shape} vs {pred_data.shape}")
                continue

            gt_mask = to_integer_mask(gt_data, f"ground truth '{gt_file}'")
            pred_mask = to_integer_mask(pred_data, f"prediction '{pred_file}'")
            spacing_mm = get_spacing_mm(gt_img, gt_mask.ndim)
            gt_eval, pred_eval, shared_idx_to_name = build_shared_label_masks(
                gt_mask,
                pred_mask,
                gt_name_to_idx,
                pred_name_to_idx,
                class_names,
            )

            df = segment_match_eval_updated(
                gt_eval,
                pred_eval,
                pixdim=spacing_mm,
                force_match=args.force_match,
                empty_refernce_predicted_behaviour=args.empty_reference_predicted_behaviour,
                expected_label_values=list(shared_idx_to_name.keys()),
            )

            case_metric_values = {metric_name: {} for metric_name in metric_names}
            if df.empty:
                for class_name in class_names:
                    for metric_name in metric_names:
                        case_metric_values[metric_name][class_name] = float("nan")
                case_metrics.append({"case_id": case_id, "view": view, **case_metric_values})
                continue

            for class_name in class_names:
                for metric_name in metric_names:
                    case_metric_values[metric_name][class_name] = float("nan")

            for _, row in df.iterrows():
                shared_idx = int(row["label"])
                class_name = shared_idx_to_name[shared_idx]
                empty_reference = bool(row.get("empty_reference", False))
                empty_predicted = bool(row.get("empty_predicted", False))

                if not empty_reference:
                    per_class_support[class_name]["gt_positive_count"] += 1
                    per_view_support[view][class_name]["gt_positive_count"] += 1
                if not empty_predicted:
                    per_class_support[class_name]["pred_positive_count"] += 1
                    per_view_support[view][class_name]["pred_positive_count"] += 1
                if empty_reference and empty_predicted:
                    per_class_support[class_name]["empty_empty_count"] += 1
                    per_view_support[view][class_name]["empty_empty_count"] += 1
                per_class_support[class_name]["valid_count"] += 1
                per_view_support[view][class_name]["valid_count"] += 1

                metric_values = {
                    "dice": extract_metric_value(row, ["Dice", "dice"]),
                    "jaccard": extract_metric_value(row, ["Jaccard", "jaccard"]),
                    "sensitivity": extract_metric_value(row, ["Sensitivity", "sensitivity"]),
                    "specificity": extract_metric_value(row, ["Specificity", "specificity"]),
                    "precision": extract_metric_value(row, ["Precision", "precision"]),
                    "accuracy": extract_metric_value(row, ["Accuracy", "accuracy"]),
                    "mean_surface_distance_mm": extract_metric_value(row, ["Mean Surface Distance", "Mean Surface Distance (mm)", "mean_surface_distance"]),
                    "hausdorff_distance_mm": extract_metric_value(row, ["Hausdorff Distance", "Hausdorff Distance (mm)", "hausdorff_distance"]),
                    "area_difference_mm2": extract_metric_value(row, ["area_difference_mm2"]),
                }

                for metric_name, metric_value in metric_values.items():
                    per_class_metrics[class_name][metric_name].append(metric_value)
                    per_view_metrics[view][class_name][metric_name].append(metric_value)
                    case_metric_values[metric_name][class_name] = metric_value

            case_metrics.append({"case_id": case_id, "view": view, **case_metric_values})
        except Exception as exc:
            errors.append(f"{case_id}: {exc}")

        if idx % 100 == 0 or idx == len(common_names):
            print(f"  Processed {idx}/{len(common_names)}")

    if not case_metrics:
        print("[ERROR] No valid cases were evaluated.")
        if errors:
            print("First errors:")
            for err in errors[:10]:
                print(f"  {err}")
        return False

    results_summary = summarize_per_class(class_names, per_class_metrics, per_class_support)
    overall_summary = summarize_overall_metrics(class_names, metric_names, per_class_metrics)
    per_view_summary = {}
    for view, view_metrics in per_view_metrics.items():
        if not any(view_metrics[class_name]["dice"] for class_name in class_names):
            continue
        per_view_summary[view] = {
            "overall_metric_summary": summarize_overall_metrics(class_names, metric_names, view_metrics),
            "per_class_metrics": summarize_per_class(class_names, view_metrics, per_view_support[view]),
            "slice_count": sum(len(view_metrics[class_name]["dice"]) for class_name in class_names) // max(len(class_names), 1),
        }

    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(
        f"{'Class':<14} {'Dice':<12} {'Jaccard':<12} {'Sens':<12} "
        f"{'Spec':<12} {'Acc':<12}"
    )
    print("-" * 78)
    for class_name in class_names:
        summary = results_summary[class_name]
        print(
            f"{class_name:<14} "
            f"{summary['dice_mean']:.3f}±{summary['dice_std']:.3f}  "
            f"{summary['jaccard_mean']:.3f}±{summary['jaccard_std']:.3f}  "
            f"{summary['sensitivity_mean']:.3f}±{summary['sensitivity_std']:.3f}  "
            f"{summary['specificity_mean']:.3f}±{summary['specificity_std']:.3f}  "
            f"{summary['accuracy_mean']:.3f}±{summary['accuracy_std']:.3f}"
        )
    print("\nBoundary and overlap extras (mean ± std):")
    print(f"{'Class':<14} {'Precision':<12} {'MSD mm':<14} {'HD mm':<14} {'AreaDiff mm2':<16}")
    print("-" * 84)
    for class_name in class_names:
        summary = results_summary[class_name]
        print(
            f"{class_name:<14} "
            f"{summary['precision_mean']:.3f}±{summary['precision_std']:.3f}  "
            f"{summary['mean_surface_distance_mm_mean']:.3f}±{summary['mean_surface_distance_mm_std']:.3f}  "
            f"{summary['hausdorff_distance_mm_mean']:.3f}±{summary['hausdorff_distance_mm_std']:.3f}  "
            f"{summary['area_difference_mm2_mean']:.3f}±{summary['area_difference_mm2_std']:.3f}"
        )

    print("\nOverall metrics:")
    for metric_name in metric_names:
        metric_label = metric_name.replace("_", " ")
        stats = overall_summary[metric_name]
        print(f"  {metric_label:<24}: {stats['mean']:.3f} ± {stats['std']:.3f}")

    if per_view_summary:
        print("\nANT/POST overall metrics:")
        print(f"{'View':<8} {'Dice':<12} {'Jaccard':<12} {'MSD mm':<14} {'HD mm':<14}")
        print("-" * 66)
        for view in ["ANT", "POST", "OTHER"]:
            if view not in per_view_summary:
                continue
            view_stats = per_view_summary[view]["overall_metric_summary"]
            print(
                f"{view:<8} "
                f"{view_stats['dice']['mean']:.3f}±{view_stats['dice']['std']:.3f}  "
                f"{view_stats['jaccard']['mean']:.3f}±{view_stats['jaccard']['std']:.3f}  "
                f"{view_stats['mean_surface_distance_mm']['mean']:.3f}±{view_stats['mean_surface_distance_mm']['std']:.3f}  "
                f"{view_stats['hausdorff_distance_mm']['mean']:.3f}±{view_stats['hausdorff_distance_mm']['std']:.3f}"
            )

    failure_cases = []
    for case in case_metrics:
        finite_dice = [float(value) for value in case["dice"].values() if np.isfinite(value)]
        if not finite_dice:
            continue
        avg_dice = float(np.mean(finite_dice))
        if avg_dice < args.failure_threshold:
            failure_cases.append((case["case_id"], avg_dice))
    failure_cases.sort(key=lambda item: item[1])

    print(f"\n{'-' * 60}")
    print("FAILURE CASE ANALYSIS")
    print("-" * 60)
    if failure_cases:
        print(f"Cases with average Dice < {args.failure_threshold}: {len(failure_cases)}")
        for case_id, avg_dice in failure_cases[:10]:
            print(f"  {case_id}: Dice = {avg_dice:.3f}")
    else:
        print(f"No cases with average Dice < {args.failure_threshold}")

    if errors:
        print(f"\n[WARNINGS] First {min(10, len(errors))} warnings:")
        for err in errors[:10]:
            print(f"  {err}")
    if gt_only:
        print(f"\n[WARNINGS] GT-only files: {len(gt_only)}")
        print(f"  Sample: {gt_only[:10]}")
    if pred_only:
        print(f"\n[WARNINGS] Prediction-only files: {len(pred_only)}")
        print(f"  Sample: {pred_only[:10]}")

    assessment = {
        "mean_dice": overall_summary["dice"]["mean"],
        "dice_std": overall_summary["dice"]["std"],
        "mean_jaccard": overall_summary["jaccard"]["mean"],
        "jaccard_std": overall_summary["jaccard"]["std"],
        "mean_sensitivity": overall_summary["sensitivity"]["mean"],
        "sensitivity_std": overall_summary["sensitivity"]["std"],
        "mean_specificity": overall_summary["specificity"]["mean"],
        "specificity_std": overall_summary["specificity"]["std"],
        "mean_precision": overall_summary["precision"]["mean"],
        "precision_std": overall_summary["precision"]["std"],
        "mean_accuracy": overall_summary["accuracy"]["mean"],
        "accuracy_std": overall_summary["accuracy"]["std"],
        "mean_surface_distance_mm": overall_summary["mean_surface_distance_mm"]["mean"],
        "mean_surface_distance_mm_std": overall_summary["mean_surface_distance_mm"]["std"],
        "hausdorff_distance_mm": overall_summary["hausdorff_distance_mm"]["mean"],
        "hausdorff_distance_mm_std": overall_summary["hausdorff_distance_mm"]["std"],
        "area_difference_mm2": overall_summary["area_difference_mm2"]["mean"],
        "area_difference_mm2_std": overall_summary["area_difference_mm2"]["std"],
        "failure_cases_count": len(failure_cases),
        "total_cases": len(case_metrics),
        "matched_file_count": len(common_names),
        "gt_only_count": len(gt_only),
        "pred_only_count": len(pred_only),
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as handle:
        json.dump(
            make_json_safe({
                "root": str(root),
                "ground_truth_dir": str(gt_dir),
                "prediction_dir": str(pred_dir),
                "gt_labels_file": str(gt_labels_file),
                "pred_labels_file": str(pred_labels_file),
                "empty_reference_predicted_behaviour": args.empty_reference_predicted_behaviour,
                "assessment": assessment,
                "overall_metric_summary": overall_summary,
                "per_view_summary": per_view_summary,
                "per_class_metrics": results_summary,
                "slice_metrics": case_metrics,
                "failure_cases": [
                    {"case_id": case_id, "avg_dice": avg_dice}
                    for case_id, avg_dice in failure_cases[:20]
                ],
                "errors": errors[:200],
                "gt_only_files": gt_only[:200],
                "pred_only_files": pred_only[:200],
            }),
            handle,
            indent=2,
        )

    print(f"\n[OK] Results saved to: {output_file}")
    print("\n[OK] EVALUATION COMPLETE")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
