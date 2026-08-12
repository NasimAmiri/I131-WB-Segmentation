"""
Read classification results JSON and generate:
  - summary CSV with 2-decimal classification metrics
  - confusion matrix heatmap with counts and row percentages
  - per-class metric bar chart
  - true-vs-predicted class distribution plot
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OVERALL_METRICS = [
    ("acc", "acc"),
    ("bac_macro", "bac"),
    ("sen_macro", "sen"),
    ("spe_macro", "spe"),
    ("f1_macro", "f1"),
    ("pres_macro", "pres"),
    ("auc_ovr_macro", "auc"),
]

PER_CLASS_METRICS = [
    ("acc", "acc"),
    ("bac", "bac"),
    ("sen", "sen"),
    ("spe", "spe"),
    ("f1", "f1"),
    ("pres", "pres"),
    ("auc", "auc"),
]

CSV_COLUMNS = [
    "scope",
    "class",
    "true_n",
    "pred_n",
    "acc",
    "bac",
    "sen",
    "spe",
    "f1",
    "pres",
    "auc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CSV and figures from classification_results.json."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Path to classification_results.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the input JSON parent directory.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="classification_report",
        help="Filename prefix for generated outputs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def format_metric(value: object) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(value):
        return "NA"
    return f"{value:.2f}"


def get_class_order(data: dict) -> list[str]:
    label_mapping = data.get("label_mapping", {}).get("class_id_to_name", {})
    if label_mapping:
        ordered = []
        for class_id in sorted(label_mapping, key=lambda x: int(x)):
            ordered.append(label_mapping[class_id])
        return ordered
    label_names = data.get("confusion_matrix", {}).get("label_names", [])
    if label_names:
        return list(label_names)
    per_class = data.get("per_class_metrics", {})
    return list(per_class.keys())


def derive_class_counts(data: dict, class_order: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    true_counts = {class_name: 0 for class_name in class_order}
    pred_counts = {class_name: 0 for class_name in class_order}

    for row in data.get("per_patient_rows", []):
        true_name = row.get("true_class_name")
        pred_name = row.get("pred_class_name")
        if true_name in true_counts:
            true_counts[true_name] += 1
        if pred_name in pred_counts:
            pred_counts[pred_name] += 1

    return true_counts, pred_counts


def build_summary_dataframe(data: dict) -> pd.DataFrame:
    class_order = get_class_order(data)
    true_counts, pred_counts = derive_class_counts(data, class_order)
    matched_cases = int(data.get("counts", {}).get("matched_cases", len(data.get("per_patient_rows", []))))

    rows: list[dict[str, object]] = []

    overall_row: dict[str, object] = {
        "scope": "overall",
        "class": "OVERALL",
        "true_n": matched_cases,
        "pred_n": matched_cases,
    }
    overall_metrics = data.get("overall_metrics", {})
    for source_key, csv_key in OVERALL_METRICS:
        overall_row[csv_key] = format_metric(overall_metrics.get(source_key))
    rows.append(overall_row)

    per_class_metrics = data.get("per_class_metrics", {})
    for class_name in class_order:
        metrics = per_class_metrics.get(class_name, {})
        row: dict[str, object] = {
            "scope": "class",
            "class": class_name,
            "true_n": int(true_counts.get(class_name, 0)),
            "pred_n": int(pred_counts.get(class_name, 0)),
        }
        for source_key, csv_key in PER_CLASS_METRICS:
            row[csv_key] = format_metric(metrics.get(source_key))
        rows.append(row)

    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def save_summary_csv(data: dict, output_path: Path) -> pd.DataFrame:
    df = build_summary_dataframe(data)
    df.to_csv(output_path, index=False)
    return df


def plot_confusion_matrix(data: dict, output_path: Path) -> None:
    cm = data.get("confusion_matrix", {})
    matrix = np.asarray(cm.get("matrix", []), dtype=float)
    labels = list(cm.get("label_names", []))
    row_sums = matrix.sum(axis=1, keepdims=True) if matrix.ndim == 2 else np.asarray([])
    matrix_percent = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    ) * 100.0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix_percent if matrix.size > 0 else matrix, cmap="Blues", vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion Matrix (Count / Row %)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticklabels(labels)

    if matrix.size > 0:
        threshold = matrix_percent.max() / 2.0 if matrix_percent.size > 0 else 0.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = int(matrix[i, j])
                percent = matrix_percent[i, j]
                color = "white" if percent > threshold else "black"
                ax.text(
                    j,
                    i,
                    f"{value}\n{percent:.1f}%",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_metrics(data: dict, output_path: Path) -> None:
    class_order = get_class_order(data)
    per_class_metrics = data.get("per_class_metrics", {})
    metric_names = [csv_key for _, csv_key in PER_CLASS_METRICS]

    x = np.arange(len(class_order))
    width = 0.11
    fig, ax = plt.subplots(figsize=(11, 6))

    for index, (source_key, csv_key) in enumerate(PER_CLASS_METRICS):
        values = []
        for class_name in class_order:
            metric_value = per_class_metrics.get(class_name, {}).get(source_key)
            if metric_value is None or not np.isfinite(float(metric_value)):
                values.append(np.nan)
            else:
                values.append(float(metric_value))
        offset = (index - (len(metric_names) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=csv_key.upper())

    ax.set_title("Per-Class Metrics")
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(class_order, rotation=20, ha="right")
    ax.legend(ncols=4, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(data: dict, output_path: Path) -> None:
    class_order = get_class_order(data)
    true_counts, pred_counts = derive_class_counts(data, class_order)

    x = np.arange(len(class_order))
    width = 0.35
    true_values = [true_counts[class_name] for class_name in class_order]
    pred_values = [pred_counts[class_name] for class_name in class_order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, true_values, width=width, label="True")
    ax.bar(x + width / 2, pred_values, width=width, label="Predicted")

    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(class_order, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for positions, values in [(x - width / 2, true_values), (x + width / 2, pred_values)]:
        for pos, value in zip(positions, values):
            ax.text(pos, value, str(int(value)), ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> bool:
    args = parse_args()
    input_json = args.input_json.resolve()
    output_dir = (args.output_dir.resolve() if args.output_dir else input_json.parent.resolve())
    prefix = args.prefix

    if not input_json.exists():
        print(f"[ERROR] Input JSON not found: {input_json}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_json(input_json)

    summary_csv = output_dir / f"{prefix}_summary.csv"
    cm_png = output_dir / f"{prefix}_confusion_matrix.png"
    per_class_png = output_dir / f"{prefix}_per_class_metrics.png"
    distribution_png = output_dir / f"{prefix}_class_distribution.png"

    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(f"[INFO] Input JSON : {input_json}")
    print(f"[INFO] Output dir : {output_dir}")
    print(f"[INFO] Prefix     : {prefix}")

    df = save_summary_csv(data, summary_csv)
    plot_confusion_matrix(data, cm_png)
    plot_per_class_metrics(data, per_class_png)
    plot_class_distribution(data, distribution_png)

    print(f"[OK] Summary CSV               : {summary_csv}")
    print(f"[OK] Confusion matrix image    : {cm_png}")
    print(f"[OK] Per-class metric image    : {per_class_png}")
    print(f"[OK] Class distribution image  : {distribution_png}")
    print("")
    print(df.to_string(index=False))
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
