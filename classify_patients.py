"""Compute case-level three-class classification metrics."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

try:
    import nibabel as nib
except ImportError:
    print("[ERROR] nibabel not installed. Please install: pip install nibabel")
    raise SystemExit(1)


CLASS_IDS = [0, 1, 2]
CLASS_ID_TO_NAME = {
    0: "normal",
    1: "remnant_bed",
    2: "metastasis",
}
CLASS_NAME_TO_ID = {
    "normal": 0,
    "remnant_bed": 1,
    "remendatat_bed": 1,
    "metastasis": 2,
}
INT_LIKE_RE = re.compile(r"^\d+(?:\.0+)?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read an Excel file and a stacked-mask directory, convert both to a "
            "3-class case label, and compute case-level classification metrics."
        )
    )
    parser.add_argument(
        "--excel",
        type=Path,
        required=True,
        help="Excel file containing Patient ID and Class_validated.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        required=True,
        help="Directory containing predicted stacked masks named like <case>_mask.nii.gz.",
    )
    parser.add_argument(
        "--patient-id-col",
        type=str,
        default="Patient ID",
        help="Excel column containing the patient identifier.",
    )
    parser.add_argument(
        "--class-col",
        type=str,
        default="Class_validated",
        help="Excel column containing the validated classification label.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <mask-dir>/classification_results.json.",
    )
    return parser.parse_args()


def normalize_patient_id(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return None
        if float(value).is_integer():
            return str(int(value))

    text = str(value).strip()
    if not text:
        return None
    if INT_LIKE_RE.fullmatch(text):
        return str(int(float(text)))
    if text.isdigit():
        return str(int(text))
    return text


def normalize_class_name(value: object) -> str:
    if value is None:
        return "normal"
    try:
        if pd.isna(value):
            return "normal"
    except TypeError:
        pass
    return str(value).strip().lower()


def excel_class_to_id(value: object) -> int:
    normalized = normalize_class_name(value)
    if normalized == "metastasis":
        return 2
    if normalized in {"remnant_bed", "remendatat_bed"}:
        return 1
    return 0


def mask_path_to_patient_id(mask_path: Path) -> str | None:
    name = mask_path.name
    if not name.endswith("_mask.nii.gz"):
        return None
    raw_id = name[: -len("_mask.nii.gz")]
    return normalize_patient_id(raw_id)


def load_predicted_case_class(mask_path: Path) -> int:
    data = np.asarray(nib.load(str(mask_path)).dataobj)
    rounded = np.rint(data)
    if not np.allclose(data, rounded, atol=1e-6):
        raise ValueError(f"Mask contains non-integer values: {mask_path}")
    labels = set(np.unique(rounded.astype(np.int16)).tolist())
    if 2 in labels:
        return 2
    if 1 in labels:
        return 1
    return 0


def safe_specificity(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> float:
    tn = int(np.logical_and(y_true_bin == 0, y_pred_bin == 0).sum())
    fp = int(np.logical_and(y_true_bin == 0, y_pred_bin == 1).sum())
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def safe_auc_binary(y_true_bin: np.ndarray, y_score: np.ndarray) -> float | None:
    if np.unique(y_true_bin).size < 2:
        return None
    try:
        return float(roc_auc_score(y_true_bin, y_score))
    except ValueError:
        return None


def safe_auc_multiclass(
    y_true: np.ndarray,
    y_score: np.ndarray,
    average: str,
    multi_class: str,
) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score, labels=CLASS_IDS, average=average, multi_class=multi_class))
    except ValueError:
        return None


def make_json_safe(value):
    if isinstance(value, dict):
        return {key: make_json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> bool:
    args = parse_args()
    excel_path = args.excel.resolve()
    mask_dir = args.mask_dir.resolve()
    output_path = (args.output.resolve() if args.output else (mask_dir / "classification_results.json").resolve())

    print("=" * 60)
    print("CASE-LEVEL 3-CLASS CLASSIFICATION")
    print("=" * 60)
    print(f"[INFO] Excel   : {excel_path}")
    print(f"[INFO] Masks   : {mask_dir}")
    print(f"[INFO] Output  : {output_path}")

    if not excel_path.exists():
        print(f"[ERROR] Excel file not found: {excel_path}")
        return False
    if not mask_dir.exists():
        print(f"[ERROR] Mask directory not found: {mask_dir}")
        return False

    df = pd.read_excel(excel_path)
    missing_cols = [col for col in [args.patient_id_col, args.class_col] if col not in df.columns]
    if missing_cols:
        print(f"[ERROR] Missing required Excel columns: {missing_cols}")
        return False

    df = df.copy()
    df["normalized_patient_id"] = df[args.patient_id_col].map(normalize_patient_id)
    df["excel_class_validated_raw"] = df[args.class_col].map(lambda x: None if pd.isna(x) else str(x).strip())
    df["true_class"] = df[args.class_col].map(excel_class_to_id)

    warnings: list[str] = []
    mask_predictions: dict[str, dict[str, object]] = {}

    mask_files = sorted(mask_dir.glob("*.nii.gz"))
    print(f"[INFO] Found {len(mask_files)} mask files")
    for mask_path in mask_files:
        patient_id = mask_path_to_patient_id(mask_path)
        if patient_id is None:
            warnings.append(f"Skipped mask with unexpected name: {mask_path.name}")
            continue
        if patient_id in mask_predictions:
            print(f"[ERROR] Duplicate normalized patient ID in masks: {patient_id}")
            return False
        try:
            pred_class = load_predicted_case_class(mask_path)
        except Exception as exc:
            warnings.append(f"Skipped mask {mask_path.name}: {exc}")
            continue
        mask_predictions[patient_id] = {
            "patient_id": patient_id,
            "mask_file": str(mask_path),
            "mask_name": mask_path.name,
            "pred_class": pred_class,
        }

    matched_df = df[df["normalized_patient_id"].isin(mask_predictions)].copy()
    duplicate_ids = matched_df["normalized_patient_id"][matched_df["normalized_patient_id"].duplicated(keep=False)]
    if not duplicate_ids.empty:
        duplicates = sorted(set(duplicate_ids.tolist()))
        print(f"[ERROR] Duplicate normalized Patient ID values found in matched Excel rows: {duplicates[:20]}")
        return False

    excel_ids = set(matched_df["normalized_patient_id"].dropna().tolist())
    unmatched_masks = sorted(set(mask_predictions) - excel_ids)
    for patient_id in unmatched_masks:
        warnings.append(f"Mask has no Excel match and was skipped: {patient_id} ({mask_predictions[patient_id]['mask_name']})")

    rows = []
    y_true: list[int] = []
    y_pred: list[int] = []
    for _, row in matched_df.sort_values("normalized_patient_id").iterrows():
        patient_id = row["normalized_patient_id"]
        if patient_id is None or patient_id not in mask_predictions:
            continue
        pred_entry = mask_predictions[patient_id]
        true_class = int(row["true_class"])
        pred_class = int(pred_entry["pred_class"])
        rows.append(
            {
                "patient_id": patient_id,
                "excel_class_validated_raw": row["excel_class_validated_raw"],
                "true_class": true_class,
                "true_class_name": CLASS_ID_TO_NAME[true_class],
                "pred_class": pred_class,
                "pred_class_name": CLASS_ID_TO_NAME[pred_class],
                "mask_file": pred_entry["mask_file"],
            }
        )
        y_true.append(true_class)
        y_pred.append(pred_class)

    ignored_excel_count = int(df["normalized_patient_id"].notna().sum()) - len(rows)

    if not rows:
        print("[ERROR] No matched patient cases found between Excel and mask directory.")
        return False

    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    y_score = np.eye(len(CLASS_IDS), dtype=float)[y_pred_arr]
    conf = confusion_matrix(y_true_arr, y_pred_arr, labels=CLASS_IDS)

    per_class_metrics: dict[str, dict[str, object]] = {}
    for class_id in CLASS_IDS:
        class_name = CLASS_ID_TO_NAME[class_id]
        y_true_bin = (y_true_arr == class_id).astype(int)
        y_pred_bin = (y_pred_arr == class_id).astype(int)

        sen = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
        pres = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
        f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
        spe = safe_specificity(y_true_bin, y_pred_bin)
        acc = float(accuracy_score(y_true_bin, y_pred_bin))
        bac = float((sen + spe) / 2.0)
        auc = safe_auc_binary(y_true_bin, y_score[:, class_id])

        per_class_metrics[class_name] = {
            "class_id": class_id,
            "acc": acc,
            "bac": bac,
            "sen": sen,
            "spe": spe,
            "f1": f1,
            "pres": pres,
            "auc": auc,
            "support_true": int(y_true_bin.sum()),
            "support_pred": int(y_pred_bin.sum()),
        }

    overall_metrics = {
        "acc": float(accuracy_score(y_true_arr, y_pred_arr)),
        "bac_macro": float(np.mean([per_class_metrics[CLASS_ID_TO_NAME[c]]["bac"] for c in CLASS_IDS])),
        "sen_macro": float(np.mean([per_class_metrics[CLASS_ID_TO_NAME[c]]["sen"] for c in CLASS_IDS])),
        "spe_macro": float(np.mean([per_class_metrics[CLASS_ID_TO_NAME[c]]["spe"] for c in CLASS_IDS])),
        "f1_macro": float(np.mean([per_class_metrics[CLASS_ID_TO_NAME[c]]["f1"] for c in CLASS_IDS])),
        "pres_macro": float(np.mean([per_class_metrics[CLASS_ID_TO_NAME[c]]["pres"] for c in CLASS_IDS])),
        "auc_ovr_macro": safe_auc_multiclass(y_true_arr, y_score, average="macro", multi_class="ovr"),
        "auc_ovr_weighted": safe_auc_multiclass(y_true_arr, y_score, average="weighted", multi_class="ovr"),
        "auc_ovo_macro": safe_auc_multiclass(y_true_arr, y_score, average="macro", multi_class="ovo"),
    }

    results = {
        "inputs": {
            "excel": str(excel_path),
            "mask_dir": str(mask_dir),
            "patient_id_col": args.patient_id_col,
            "class_col": args.class_col,
            "output": str(output_path),
        },
        "label_mapping": {
            "excel_to_target": {
                "metastasis": 2,
                "remnant_bed": 1,
                "remendatat_bed": 1,
                "others": 0,
            },
            "mask_to_target": {
                "contains_2": 2,
                "contains_1_without_2": 1,
                "otherwise": 0,
            },
            "class_id_to_name": CLASS_ID_TO_NAME,
        },
        "counts": {
            "excel_rows_total": int(len(df)),
            "mask_files_total": int(len(mask_predictions)),
            "matched_cases": int(len(rows)),
            "ignored_excel_rows_without_mask_match": int(ignored_excel_count),
            "unmatched_mask_files": int(len(unmatched_masks)),
        },
        "confusion_matrix": {
            "labels": CLASS_IDS,
            "label_names": [CLASS_ID_TO_NAME[c] for c in CLASS_IDS],
            "matrix": conf.tolist(),
        },
        "per_class_metrics": per_class_metrics,
        "overall_metrics": overall_metrics,
        "per_patient_rows": rows,
        "warnings": warnings,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(make_json_safe(results), indent=2))

    print("\nPer-class metrics:")
    print(f"{'Class':<14} {'ACC':<8} {'BAC':<8} {'SEN':<8} {'SPE':<8} {'F1':<8} {'PRES':<8} {'AUC':<8}")
    print("-" * 78)
    for class_id in CLASS_IDS:
        class_name = CLASS_ID_TO_NAME[class_id]
        metric = per_class_metrics[class_name]
        auc_text = "NA" if metric["auc"] is None else f"{metric['auc']:.3f}"
        print(
            f"{class_name:<14} {metric['acc']:.3f}    {metric['bac']:.3f}    {metric['sen']:.3f}    "
            f"{metric['spe']:.3f}    {metric['f1']:.3f}    {metric['pres']:.3f}    {auc_text:<8}"
        )

    print("\nOverall metrics:")
    for key, value in overall_metrics.items():
        text = "NA" if value is None else f"{value:.3f}"
        print(f"  {key:<18}: {text}")

    print("\nCounts:")
    print(f"  matched_cases          : {len(rows)}")
    print(f"  unmatched_mask_files   : {len(unmatched_masks)}")
    print(f"  ignored_excel_rows     : {ignored_excel_count}")
    print(f"\n[OK] Results saved to: {output_path}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
