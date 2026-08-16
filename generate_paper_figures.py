#!/usr/bin/env python3
"""Generate manuscript Figures 2-8 from a private, path-based JSON configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from i131_image_utils import (
    apply_transform,
    comparison_mask,
    detect_transform,
    load_font,
    load_nifti,
    overlay_mask,
    scan_rgb,
    select_view,
)


WHITE = (255, 255, 255)
TEXT = (22, 22, 22)
PANEL_LABELS = ("(A)", "(B)", "(C)", "(D)")

# Compact spacing shared by the stage and paired-case layouts. These values
# follow the proportions of the accepted manuscript Figures 3-5 instead of
# the wider spacing inherited from the former Figure 6 helper.
PAIR_GAP = 10
GROUP_GAP = 16
COMPOSITE_GAP = 40


def read_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("figures"), dict):
        raise ValueError("Configuration must contain a 'figures' object")
    return data


def configured_path(value: str, config_dir: Path) -> Path:
    if not value or "<" in value or ">" in value:
        raise ValueError("Replace all placeholder paths in the private figure configuration")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_dir / path).resolve()


def trim_white(image: Image.Image, threshold: int = 8, padding: int = 8) -> Image.Image:
    rgb = image.convert("RGB")
    difference = ImageChops.difference(rgb, Image.new("RGB", rgb.size, WHITE)).convert("L")
    bounding_box = difference.point(lambda value: 255 if value > threshold else 0).getbbox()
    if bounding_box is None:
        return rgb
    left, top, right, bottom = bounding_box
    return rgb.crop(
        (
            max(0, left - padding),
            max(0, top - padding),
            min(rgb.width, right + padding),
            min(rgb.height, bottom + padding),
        )
    )


def make_composite(
    panel_paths: list[Path],
    gap: int = 100,
    label_padding: int = 45,
) -> Image.Image:
    if len(panel_paths) != 4:
        raise ValueError("Composite figures require exactly four panels in A-D order")
    panels = [trim_white(Image.open(path)) for path in panel_paths]
    cell_width = max(panel.width for panel in panels)
    image_height = max(panel.height for panel in panels)
    font = load_font(max(54, int(cell_width * 0.035)), bold=True)
    label_height = font.getbbox("(A)")[3] - font.getbbox("(A)")[1]
    label_band = label_height + label_padding
    cell_height = label_band + image_height
    canvas = Image.new("RGB", (2 * cell_width + gap, 2 * cell_height + gap), WHITE)
    draw = ImageDraw.Draw(canvas)
    positions = ((0, 0), (cell_width + gap, 0), (0, cell_height + gap), (cell_width + gap, cell_height + gap))
    for label, panel, (x, y) in zip(PANEL_LABELS, panels, positions):
        canvas.paste(panel, (x, y + label_band))
        draw.text((x + 18, y + 12), label, font=font, fill=TEXT)
    return canvas


def crop_array(array: np.ndarray, padding: int = 12, threshold: int = 245) -> np.ndarray:
    foreground = np.any(np.asarray(array) < threshold, axis=-1)
    if not np.any(foreground):
        return array
    y, x = np.where(foreground)
    return array[
        max(0, y.min() - padding) : min(array.shape[0], y.max() + padding + 1),
        max(0, x.min() - padding) : min(array.shape[1], x.max() + padding + 1),
    ]


def fit_array(array: np.ndarray, size: tuple[int, int], crop: bool = False) -> Image.Image:
    if crop:
        array = crop_array(array)
    image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
    width, height = size
    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, WHITE)
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def render_group(
    title: str,
    arrays: dict[str, np.ndarray],
    panel_width: int,
    panel_height: int,
    crop: bool,
) -> Image.Image:
    header_height = 62
    view_height = 30
    width = panel_width * 2 + PAIR_GAP
    canvas = Image.new("RGB", (width, header_height + view_height + panel_height), WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.text((width // 2, 4), title, font=load_font(25, True), fill=TEXT, anchor="mt")
    for index, view in enumerate(("ANT", "POST")):
        x = index * (panel_width + PAIR_GAP)
        draw.text((x + panel_width // 2, header_height), view, font=load_font(17, True), fill=TEXT, anchor="mt")
        canvas.paste(fit_array(arrays[view], (panel_width, panel_height), crop), (x, header_height + view_height))
    return canvas


def case_arrays(spec: dict[str, Any], config_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    image_volume, _ = load_nifti(configured_path(spec["image"], config_dir))
    ground_truth_volume, _ = load_nifti(configured_path(spec["ground_truth"], config_dir))
    predictions = {
        view: load_nifti(configured_path(spec["predictions"][view], config_dir))[0]
        for view in ("ANT", "POST")
    }
    transform = detect_transform(
        select_view(image_volume, "ANT"),
        select_view(image_volume, "POST"),
        select_view(ground_truth_volume, "ANT"),
        select_view(ground_truth_volume, "POST"),
        flip_lr=bool(spec.get("flip_lr", True)),
    )
    output: dict[str, dict[str, np.ndarray]] = {}
    for view in ("ANT", "POST"):
        scan = apply_transform(select_view(image_volume, view), transform)
        ground_truth = apply_transform(select_view(ground_truth_volume, view), transform)
        prediction = apply_transform(np.squeeze(predictions[view]), transform)
        output[view] = {
            "original": scan_rgb(scan),
            "ground_truth": overlay_mask(scan, ground_truth),
            "prediction": overlay_mask(scan, prediction),
            "comparison": comparison_mask(ground_truth, prediction, scan.shape),
        }
    return output


def render_case_figure(spec: dict[str, Any], config_dir: Path) -> Image.Image:
    arrays = case_arrays(spec, config_dir)
    groups = [
        render_group("Original Scan", {view: arrays[view]["original"] for view in arrays}, 160, 620, False),
        render_group("Ground Truth", {view: arrays[view]["ground_truth"] for view in arrays}, 160, 620, False),
        render_group("Prediction", {view: arrays[view]["prediction"] for view in arrays}, 160, 620, False),
        render_group("Overlay", {view: arrays[view]["comparison"] for view in arrays}, 240, 620, True),
    ]
    outer_margin = 12
    canvas = Image.new(
        "RGB",
        (
            sum(group.width for group in groups) + GROUP_GAP * (len(groups) - 1) + outer_margin * 2,
            max(group.height for group in groups) + 10,
        ),
        WHITE,
    )
    x = outer_margin
    for group in groups:
        canvas.paste(group, (x, 5))
        x += group.width + GROUP_GAP
    return canvas


def stage_arrays(spec: dict[str, Any], config_dir: Path) -> list[tuple[str, dict[str, dict[str, np.ndarray]]]]:
    image_volume, _ = load_nifti(configured_path(spec["image"], config_dir))
    stage_volumes = {
        stage: load_nifti(configured_path(spec[stage]["ground_truth"], config_dir))[0]
        for stage in ("stage_1", "stage_2")
    }
    transform = detect_transform(
        select_view(image_volume, "ANT"),
        select_view(image_volume, "POST"),
        select_view(stage_volumes["stage_2"], "ANT"),
        select_view(stage_volumes["stage_2"], "POST"),
        flip_lr=bool(spec.get("flip_lr", True)),
    )
    rows: list[tuple[str, dict[str, dict[str, np.ndarray]]]] = []
    for stage, label in (("stage_1", "Stage-1"), ("stage_2", "Stage-2")):
        predictions = {
            view: load_nifti(configured_path(spec[stage]["predictions"][view], config_dir))[0]
            for view in ("ANT", "POST")
        }
        row: dict[str, dict[str, np.ndarray]] = {}
        for view in ("ANT", "POST"):
            scan = apply_transform(select_view(image_volume, view), transform)
            ground_truth = apply_transform(select_view(stage_volumes[stage], view), transform)
            prediction = apply_transform(np.squeeze(predictions[view]), transform)
            row[view] = {
                "original": scan_rgb(scan),
                "ground_truth": overlay_mask(scan, ground_truth),
                "prediction": overlay_mask(scan, prediction),
                "comparison": comparison_mask(ground_truth, prediction, scan.shape),
            }
        rows.append((label, row))
    return rows


def render_stage_figure(spec: dict[str, Any], config_dir: Path) -> Image.Image:
    rows = stage_arrays(spec, config_dir)
    rendered_rows: list[Image.Image] = []
    for label, arrays in rows:
        groups = [
            render_group("Original Scan", {view: arrays[view]["original"] for view in arrays}, 135, 470, False),
            render_group("Ground Truth", {view: arrays[view]["ground_truth"] for view in arrays}, 135, 470, False),
            render_group("Prediction", {view: arrays[view]["prediction"] for view in arrays}, 135, 470, False),
            render_group("Overlay", {view: arrays[view]["comparison"] for view in arrays}, 200, 470, True),
        ]
        label_width = 120
        row = Image.new(
            "RGB",
            (
                label_width + sum(group.width for group in groups) + GROUP_GAP * (len(groups) - 1),
                max(group.height for group in groups),
            ),
            WHITE,
        )
        ImageDraw.Draw(row).text((10, row.height // 2), label, font=load_font(24, True), fill=TEXT, anchor="lm")
        x = label_width
        for group in groups:
            row.paste(group, (x, 0))
            x += group.width + GROUP_GAP
        rendered_rows.append(row)
    row_gap = 16
    canvas = Image.new(
        "RGB",
        (max(row.width for row in rendered_rows), sum(row.height for row in rendered_rows) + row_gap),
        WHITE,
    )
    y = 0
    for row in rendered_rows:
        canvas.paste(row, (0, y))
        y += row.height + row_gap
    return canvas


def save_figure(image: Image.Image, figure_number: int, output_dir: Path, dpi: int) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"figure_{figure_number:02d}.png"
    tiff = output_dir / f"figure_{figure_number:02d}.tiff"
    image.save(png, dpi=(dpi, dpi), optimize=True)
    image.save(tiff, dpi=(dpi, dpi), compression="tiff_lzw")
    return png, tiff


def generate_figures(
    config: dict[str, Any],
    config_path: Path,
    figure_numbers: list[int],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    outputs: list[Path] = []
    config_dir = config_path.resolve().parent
    for number in figure_numbers:
        spec = config["figures"].get(str(number))
        if spec is None:
            raise KeyError(f"Figure {number} is not defined in the configuration")
        kind = spec.get("kind")
        if kind == "composite":
            paths = [configured_path(value, config_dir) for value in spec["panels"]]
            image = (
                make_composite(paths, gap=COMPOSITE_GAP, label_padding=30)
                if number == 2
                else make_composite(paths)
            )
        elif kind == "stage_comparison" and number == 3:
            image = render_stage_figure(spec, config_dir)
        elif kind == "case_comparison" and number in {4, 5, 6, 7}:
            image = render_case_figure(spec, config_dir)
        else:
            raise ValueError(f"Unsupported kind '{kind}' for Figure {number}")
        png, tiff = save_figure(image, number, output_dir, dpi)
        outputs.extend((png, tiff))
        print(f"Saved {png} and {tiff}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Private JSON file containing input paths.")
    parser.add_argument("--figures", type=int, nargs="+", choices=range(2, 9), default=list(range(2, 9)))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generate_figures(read_config(args.config), args.config, args.figures, args.output_dir, args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
