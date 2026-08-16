# Multiclass I-131 Planar Scintigraphy Segmentation with nnU-Net

Publication code for the workflow described in **“Multiclass Uptake Segmentation and Segmentation-based Classification in Planar Iodine-131 Whole-Body Scintigraphy Using nnU-Net.”** The repository covers cohort preparation, intensity normalization, paired-projection nnU-Net training and inference, segmentation evaluation, segmentation-derived patient classification, and manuscript Figures 2–8.

## Requirements

- Python 3.11
- An environment compatible with nnU-Net v2.7.0
- A CUDA-capable PyTorch installation for GPU training, or a CPU build for code inspection and lightweight tests

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch before the remaining packages. For the tested CUDA 12.8 build:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Then install the remaining runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

Install test dependencies with `python -m pip install -r requirements-dev.txt`.

## Private input layout

The scripts use generic case stems and do not require a fixed project location:

```text
private_data/
├── images/
│   └── <case>.nii.gz
├── normalized_images/
│   └── <case>.nii.gz
├── labels/
│   └── <case>-label.nii.gz
└── metadata.xlsx
```

Each paired study is expected to have shape `(height, width, 2)`, with ANT in the first projection and POST in the second. A 2-D input is accepted by dataset preparation as a single projection. For compatibility with the original pipeline, inputs containing more than two projections emit a warning and use only the first two.

### Command contracts

| Command | Required private input | Main output |
|---|---|---|
| `split_dataset.py` | Spreadsheet plus paired image/label directories | Cohort folders containing copied pairs; sources are changed only with `--move` |
| `normalize_images.py` | Raw 2-D or paired NIfTI images | Float32 NIfTI images scaled to `[0, 1]` |
| `prepare_nnunet_dataset.py` | Normalized images, labels, and label definition | `Dataset<id>_<name>` with `imagesTr`, `labelsTr`, `dataset.json`, and grouped `splits_final.json` |
| `train_nnunet.py` | Prepared nnU-Net raw dataset and three nnU-Net roots | Preprocessed dataset and fold-specific model results |
| `run_inference.py` | New paired images and trained fold results | Per-view predictions plus one stacked paired mask per study |
| `evaluate_segmentation.py` | Filename-matched ground-truth and prediction slices | Evaluation JSON |
| `generate_segmentation_report.py` | Evaluation JSON | Summary/failure CSV files and Dice plot |
| `classify_patients.py` | Reviewed metadata spreadsheet and stacked masks | Three-class classification JSON |
| `generate_classification_report.py` | Classification JSON | Summary CSV and diagnostic plots |
| `generate_paper_figures.py` | Private JSON path configuration | Figures 2–8 as PNG and TIFF |

The public label example is [labels_itksnap.example.txt](labels_itksnap.example.txt):

| Value | Class |
|---:|---|
| 0 | background |
| 1 | thyroid/remnant bed |
| 2 | metastasis |
| 3 | salivary glands |
| 4 | gastrointestinal uptake |
| 5 | urinary bladder |
| 6 | liver uptake |
| 7 | unknown uptake |
| 8 | contamination |

## Reproduction workflow

### 1. Split cohorts

Splitting is non-destructive by default. It copies paired images and labels into `train`, `external`, and `artifact_cases` using image-quality and scanner metadata. Add `--move` only when destructive relocation is explicitly intended.

```bash
python split_dataset.py \
  --spreadsheet /path/to/private_data/metadata.xlsx \
  --images-dir /path/to/private_data/normalized_images \
  --labels-dir /path/to/private_data/labels \
  --output-root /path/to/work/cohorts
```

### 2. Normalize intensities

ANT and POST projections are clipped and min-max normalized independently.

```bash
python normalize_images.py \
  --input-dir /path/to/private_data/images \
  --output-dir /path/to/private_data/normalized_images
```

### 3. Prepare the 2-D nnU-Net dataset

```bash
python prepare_nnunet_dataset.py \
  --images-dir /path/to/work/cohorts/train/Images_norm \
  --labels-dir /path/to/work/cohorts/train/labels \
  --labels-file labels_itksnap.example.txt \
  --nnunet-raw /path/to/work/nnUNet_raw \
  --dataset-id 501 \
  --dataset-name I131SPECT2D
```

The script writes ANT and POST as separate 2-D nnU-Net samples. The five folds are deterministic and patient-grouped: both projections from a patient remain in the same fold. They preserve the original contiguous allocation, including assigning any remainder to fold 4, and are not described as class-stratified.

### 4. Plan, preprocess, and train

Inspect all commands without starting preprocessing or training:

```bash
python train_nnunet.py \
  --nnunet-raw /path/to/work/nnUNet_raw \
  --nnunet-preprocessed /path/to/work/nnUNet_preprocessed \
  --nnunet-results /path/to/work/nnUNet_results \
  --dry-run
```

Remove `--dry-run` to preprocess and train folds 0–4 sequentially. Use `--folds 0 1` to select a subset. After preprocessing, the script installs the prepared patient-grouped `splits_final.json` into the preprocessed dataset before training.

### 5. Ensemble inference

```bash
python run_inference.py \
  --input-dir /path/to/private_input_images \
  --work-dir /path/to/work/inference_run \
  --nnunet-raw /path/to/work/nnUNet_raw \
  --nnunet-preprocessed /path/to/work/nnUNet_preprocessed \
  --nnunet-results /path/to/work/nnUNet_results
```

The default uses folds 0–4 and `checkpoint_best.pth`, then reconstructs one paired mask per input study. Add `--input-normalized` when the supplied images were already normalized; this is required when reproducing the original inference workflow from an `Images_norm` directory. `--dry-run` prints the nnU-Net command without creating files.

### 6. Segmentation evaluation and reports

```bash
python evaluate_segmentation.py \
  --root /path/to/work/evaluation \
  --gt-dir /path/to/ground_truth_slices \
  --pred-dir /path/to/prediction_slices \
  --labels-file labels_itksnap.example.txt \
  --output /path/to/work/evaluation/evaluation_results.json

python generate_segmentation_report.py \
  --input-json /path/to/work/evaluation/evaluation_results.json \
  --output-dir /path/to/work/evaluation/report
```

### 7. Patient-level classification and reports

```bash
python classify_patients.py \
  --excel /path/to/private_data/metadata.xlsx \
  --mask-dir /path/to/work/inference_stacked_masks \
  --output /path/to/work/classification_results.json

python generate_classification_report.py \
  --input-json /path/to/work/classification_results.json \
  --output-dir /path/to/work/classification_report
```

The three patient classes are hierarchical: metastasis maps to class 2, `remnant_bed` map to class 1, and every other or missing clinical label maps to normal class 0.

## Figures 2–8

Figure 1 is a separately prepared workflow schematic and is not generated by this repository. Figures 2 and 8 combine four aggregate report panels; Figure 3 compares the two annotation/training stages; Figures 4–7 use a shared paired ANT/POST case layout.

Copy the placeholder configuration and fill it with private paths. 
```bash
cp figure_config.example.json local_figures.json
python generate_paper_figures.py \
  --config local_figures.json \
  --figures 2 3 4 5 6 7 8 \
  --output-dir /path/to/private_figure_outputs
```

Outputs are named `figure_02.png`/`figure_02.tiff` through `figure_08.png`/`figure_08.tiff` and carry 600-DPI metadata. Scans use inverse gray; masks use the label table above. Agreement overlays use yellow only for exact class agreement, green for missed ground truth, red for false-positive prediction, and purple when both masks are foreground but their classes differ. A purple mismatch is counted as both a false negative for the reference class and a false positive for the predicted class.

## Troubleshooting

- **nnU-Net command not found:** activate the environment in which `requirements.txt` was installed.
- **CUDA unavailable:** verify the installed PyTorch wheel and driver with `python -c "import torch; print(torch.cuda.is_available())"`.
- **Unexpected image shape:** paired images and labels must have the same spatial shape. The first two projections are interpreted as ANT and POST.
- **Existing outputs:** preprocessing scripts stop or skip by default; use `--overwrite` only after checking the target directory.


## Citation

Please cite the associated manuscript. Replace this placeholder with the journal citation and DOI after publication:

> Multiclass Uptake Segmentation and Image-based Classification in Planar Iodine-131 Whole-Body Scintigraphy Using nnU-Net.

## License

Released under the [MIT License](LICENSE).
