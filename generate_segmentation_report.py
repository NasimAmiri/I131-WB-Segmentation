"""
Read segmentation evaluation JSON and generate:
  - summary CSV with mean±SD metrics for OVERALL / ANT / POST
  - per-class failure CSV
  - Dice box plot figure

This script is presentation-oriented: metric cells are written as mean±SD with
two decimals. Failure detection is based on slice_metrics in the JSON. Because
the JSON does not store explicit per-slice GT/prediction presence flags, the
"not_predicted" reason is inferred from the metric pattern produced by
segmentationmetrics for GT-positive / prediction-empty slices.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    ("dice", "dice", "Dice"),
    ("jaccard", "jaccard", "Jaccard"),
    ("sensitivity", "sensitivity", "Sensitivity"),
    ("specificity", "specificity", "Specificity"),
    ("precision", "precision", "Precision"),
    ("accuracy", "accuracy", "Accuracy"),
    ("mean_surface_distance_mm", "msd_mm", "MSD mm"),
    ("hausdorff_distance_mm", "hd_mm", "HD mm"),
    ("area_difference_mm2", "area_diff_mm2", "AreaDiff mm2"),
]

DISPLAY_CLASS_NAMES = {
    "bladder": "Bladder",
    "contamination": "Contamination",
    "gi": "GI",
    "liver": "Liver",
    "metastasis": "Metastasis",
    "salivary": "SG",
    "thyroid": "Thyroid",
    "unknown": "Unknown",
}

VIEW_COLORS = {
    "ANT": "#1f77b4",
    "POST": "#ff7f0e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CSV and Dice box plot reports from segmentation evaluation JSON."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Path to evaluation_results.json.",
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
        default="seg_report",
        help="Filename prefix for generated outputs.",
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=0.5,
        help="Dice threshold below which a class-slice pair is reported as a failure.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def format_mean_sd(mean_value, std_value) -> str:
    if mean_value is None or std_value is None:
        return "NA"
    if not np.isfinite(mean_value) or not np.isfinite(std_value):
        return "NA"
    return f"{mean_value:.2f}\u00b1{std_value:.2f}"


def build_class_row(view: str, class_name: str, class_metrics: dict) -> dict[str, object]:
    row: dict[str, object] = {
        "view": view,
        "class": class_name,
        "gt_n": class_metrics.get("gt_positive_count"),
        "pred_n": class_metrics.get("pred_positive_count"),
        "valid_n": class_metrics.get("valid_count"),
        "fn_n": 0,
        "fp_n": 0,
    }
    for metric_key, csv_key, _ in METRIC_COLUMNS:
        row[csv_key] = format_mean_sd(
            class_metrics.get(f"{metric_key}_mean"),
            class_metrics.get(f"{metric_key}_std"),
        )
    return row


def build_overall_row(view: str, overall_metric_summary: dict, per_class_metrics: dict, slice_count: int | None) -> dict[str, object]:
    gt_n = int(sum((metrics.get("gt_positive_count") or 0) for metrics in per_class_metrics.values()))
    pred_n = int(sum((metrics.get("pred_positive_count") or 0) for metrics in per_class_metrics.values()))
    row: dict[str, object] = {
        "view": view,
        "class": "OVERALL",
        "gt_n": gt_n,
        "pred_n": pred_n,
        "valid_n": slice_count,
        "fn_n": 0,
        "fp_n": 0,
    }
    for metric_key, csv_key, _ in METRIC_COLUMNS:
        stats = overall_metric_summary.get(metric_key, {})
        row[csv_key] = format_mean_sd(stats.get("mean"), stats.get("std"))
    return row


def build_summary_dataframe(data: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    event_counts = compute_confusion_event_counts(data)

    overall_per_class = data.get("per_class_metrics", {})
    overall_row = build_overall_row(
        "OVERALL",
        data.get("overall_metric_summary", {}),
        overall_per_class,
        data.get("assessment", {}).get("total_cases"),
    )
    overall_row["fn_n"] = int(sum(item["fn_n"] for item in event_counts.get("OVERALL", {}).values()))
    overall_row["fp_n"] = int(sum(item["fp_n"] for item in event_counts.get("OVERALL", {}).values()))
    rows.append(overall_row)
    for class_name, class_metrics in overall_per_class.items():
        row = build_class_row("OVERALL", class_name, class_metrics)
        row["fn_n"] = int(event_counts.get("OVERALL", {}).get(class_name, {}).get("fn_n", 0))
        row["fp_n"] = int(event_counts.get("OVERALL", {}).get(class_name, {}).get("fp_n", 0))
        rows.append(row)

    for view in ["ANT", "POST"]:
        view_summary = data.get("per_view_summary", {}).get(view)
        if not view_summary:
            continue
        overall_row = build_overall_row(
            view,
            view_summary.get("overall_metric_summary", {}),
            view_summary.get("per_class_metrics", {}),
            view_summary.get("slice_count"),
        )
        overall_row["fn_n"] = int(sum(item["fn_n"] for item in event_counts.get(view, {}).values()))
        overall_row["fp_n"] = int(sum(item["fp_n"] for item in event_counts.get(view, {}).values()))
        rows.append(overall_row)
        for class_name, class_metrics in view_summary.get("per_class_metrics", {}).items():
            row = build_class_row(view, class_name, class_metrics)
            row["fn_n"] = int(event_counts.get(view, {}).get(class_name, {}).get("fn_n", 0))
            row["fp_n"] = int(event_counts.get(view, {}).get(class_name, {}).get("fp_n", 0))
            rows.append(row)

    ordered_columns = ["view", "class", "gt_n", "pred_n", "valid_n", "fn_n", "fp_n"] + [
        csv_key for _, csv_key, _ in METRIC_COLUMNS
    ]
    return pd.DataFrame(rows, columns=ordered_columns)


def is_finite_number(value) -> bool:
    return value is not None and np.isfinite(value)


def infer_not_predicted(slice_entry: dict, class_name: str) -> bool:
    dice = slice_entry.get("dice", {}).get(class_name)
    sensitivity = slice_entry.get("sensitivity", {}).get(class_name)
    precision = slice_entry.get("precision", {}).get(class_name)
    specificity = slice_entry.get("specificity", {}).get(class_name)
    area_diff = slice_entry.get("area_difference_mm2", {}).get(class_name)

    if not all(is_finite_number(v) for v in [dice, sensitivity, precision, specificity, area_diff]):
        return False

    return (
        np.isclose(float(dice), 0.0)
        and np.isclose(float(sensitivity), 0.0)
        and np.isclose(float(precision), 0.0)
        and np.isclose(float(specificity), 1.0)
        and float(area_diff) > 0.0
    )


def infer_false_positive(slice_entry: dict, class_name: str) -> bool:
    dice = slice_entry.get("dice", {}).get(class_name)
    sensitivity = slice_entry.get("sensitivity", {}).get(class_name)
    precision = slice_entry.get("precision", {}).get(class_name)
    specificity = slice_entry.get("specificity", {}).get(class_name)
    area_diff = slice_entry.get("area_difference_mm2", {}).get(class_name)

    if not all(is_finite_number(v) for v in [dice, sensitivity, precision, specificity, area_diff]):
        return False

    return (
        np.isclose(float(dice), 0.0)
        and np.isclose(float(sensitivity), 0.0)
        and np.isclose(float(precision), 0.0)
        and float(specificity) < 1.0
        and float(area_diff) > 0.0
    )


def compute_confusion_event_counts(data: dict) -> dict[str, dict[str, dict[str, int]]]:
    counts: dict[str, dict[str, dict[str, int]]] = {"OVERALL": {}, "ANT": {}, "POST": {}}

    for slice_entry in data.get("slice_metrics", []):
        view = slice_entry.get("view")
        target_views = ["OVERALL"]
        if view in {"ANT", "POST"}:
            target_views.append(view)

        for class_name in slice_entry.get("dice", {}).keys():
            is_fn = infer_not_predicted(slice_entry, class_name)
            is_fp = infer_false_positive(slice_entry, class_name)
            if not is_fn and not is_fp:
                continue

            for target_view in target_views:
                view_counts = counts.setdefault(target_view, {})
                class_counts = view_counts.setdefault(class_name, {"fn_n": 0, "fp_n": 0})
                if is_fn:
                    class_counts["fn_n"] += 1
                if is_fp:
                    class_counts["fp_n"] += 1

    return counts


def build_failure_reason(is_not_predicted: bool, is_below_threshold: bool) -> str | None:
    if is_not_predicted and is_below_threshold:
        return "not_predicted_and_dice_below_threshold"
    if is_not_predicted:
        return "not_predicted"
    if is_below_threshold:
        return "dice_below_threshold"
    return None


def build_failure_dataframe(data: dict, failure_threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for slice_entry in data.get("slice_metrics", []):
        case_id = slice_entry.get("case_id")
        view = slice_entry.get("view")
        dice_map = slice_entry.get("dice", {})

        for class_name, dice_value in dice_map.items():
            if not is_finite_number(dice_value):
                continue

            is_not_predicted = infer_not_predicted(slice_entry, class_name)
            is_below_threshold = float(dice_value) < failure_threshold
            failure_reason = build_failure_reason(is_not_predicted, is_below_threshold)
            if failure_reason is None:
                continue

            rows.append(
                {
                    "class": class_name,
                    "case_id": case_id,
                    "view": view,
                    "dice": round(float(dice_value), 4),
                    "failure_reason": failure_reason,
                }
            )

    df = pd.DataFrame(rows, columns=["class", "case_id", "view", "dice", "failure_reason"])
    if not df.empty:
        df = df.sort_values(["class", "dice", "case_id"], ascending=[True, True, True]).reset_index(drop=True)
    return df


def build_dice_long_dataframe(data: dict, exclude_not_predicted: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for slice_entry in data.get("slice_metrics", []):
        case_id = slice_entry.get("case_id")
        view = slice_entry.get("view")
        for class_name, dice_value in slice_entry.get("dice", {}).items():
            if not is_finite_number(dice_value):
                continue
            if exclude_not_predicted and infer_not_predicted(slice_entry, class_name):
                continue
            if np.isclose(float(dice_value), 0.0):
                continue
            if np.isclose(float(dice_value), 1.0):
                continue
            rows.append(
                {
                    "case_id": case_id,
                    "view": view,
                    "class": class_name,
                    "dice": float(dice_value),
                }
            )
    return pd.DataFrame(rows)


def format_plot_class_name(class_name: str) -> str:
    if class_name in DISPLAY_CLASS_NAMES:
        return DISPLAY_CLASS_NAMES[class_name]
    return class_name.replace("_", " ").title()


def draw_grouped_boxplots(ax, dice_df: pd.DataFrame, title: str, ylabel: str = "Dice") -> None:
    if dice_df.empty:
        ax.text(0.5, 0.5, "No finite Dice values", ha="center", va="center")
        ax.set_axis_off()
        return

    class_names = sorted(dice_df["class"].unique().tolist())
    positions = np.arange(len(class_names), dtype=float)
    width = 0.32
    legend_handles: list[Patch] = []

    for view_index, view in enumerate(["ANT", "POST"]):
        view_color = VIEW_COLORS[view]
        view_positions = positions + (-width / 2 if view == "ANT" else width / 2)
        box_values: list[np.ndarray] = []
        box_positions: list[float] = []

        for pos, class_name in zip(positions, class_names):
            view_values = dice_df[
                (dice_df["class"] == class_name) & (dice_df["view"] == view)
            ]["dice"].to_numpy(dtype=float)
            view_values = view_values[np.isfinite(view_values)]
            if view_values.size == 0:
                continue
            box_values.append(view_values)
            box_positions.append(pos + (-width / 2 if view == "ANT" else width / 2))

        if box_values:
            bp = ax.boxplot(
                box_values,
                positions=box_positions,
                widths=width * 0.85,
                patch_artist=True,
                manage_ticks=False,
                medianprops={"color": "#222222", "linewidth": 1.4},
                whiskerprops={"color": "#555555", "linewidth": 1.0},
                capprops={"color": "#555555", "linewidth": 1.0},
                boxprops={"edgecolor": "#444444", "linewidth": 1.0},
                flierprops={
                    "marker": "o",
                    "markerfacecolor": view_color,
                    "markeredgecolor": view_color,
                    "markersize": 3,
                    "alpha": 0.45,
                },
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(view_color)
                patch.set_alpha(0.82)

        legend_handles.append(Patch(facecolor=view_color, edgecolor="#444444", label=view))

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [format_plot_class_name(name) for name in class_names],
        rotation=0,
        fontsize=15,
        fontweight="bold",
    )
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=18, pad=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=15, fontweight="bold")
    ax.tick_params(axis="y", labelsize=15)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    legend = ax.legend(
        handles=legend_handles,
        title="View",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=15,
        title_fontsize=16,
    )
    if legend is not None:
        legend.get_title().set_fontweight("bold")
        for text in legend.get_texts():
            text.set_fontweight("bold")


def save_dice_boxplot(data: dict, output_path: Path) -> None:
    dice_df = build_dice_long_dataframe(data, exclude_not_predicted=True)

    fig, ax = plt.subplots(1, 1, figsize=(16, 7), constrained_layout=True)

    if dice_df.empty:
        ax.text(0.5, 0.5, "No finite Dice values", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    draw_grouped_boxplots(ax, dice_df, "Dice by Organ and View")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> bool:
    args = parse_args()
    input_json = args.input_json.resolve()
    output_dir = (args.output_dir or input_json.parent).resolve()

    if not input_json.exists():
        print(f"[ERROR] Input JSON not found: {input_json}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_json(input_json)

    summary_csv = output_dir / f"{args.prefix}_summary.csv"
    failure_csv = output_dir / f"{args.prefix}_class_failures.csv"
    dice_plot = output_dir / f"{args.prefix}_dice_boxplot.png"

    summary_df = build_summary_dataframe(data)
    failure_df = build_failure_dataframe(data, args.failure_threshold)
    save_dice_boxplot(data, dice_plot)

    summary_df.to_csv(summary_csv, index=False)
    failure_df.to_csv(failure_csv, index=False)

    print(f"[OK] Summary CSV     : {summary_csv}")
    print(f"[OK] Failure CSV     : {failure_csv}")
    print(f"[OK] Dice box plot   : {dice_plot}")
    print(f"[INFO] Summary rows  : {len(summary_df)}")
    print(f"[INFO] Failure rows  : {len(failure_df)}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
