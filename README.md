# AFib-Multimodal

Code for the paper:  
**"Predicting Paroxysmal Atrial Fibrillation: A Machine Learning Approach Leveraging EHR and ECG Data"**

This repository contains the full pipeline from raw data ingestion through model training, hyperparameter search, result review, and figure generation. **No patient data is included.** To run the pipeline end-to-end you will need to supply your own data in the formats described below.

---

## Repository Structure

```
AFib-multimodal/
├── experiment.py            # Step 1 – Raw data pipeline + hyperparameter search
├── experiment_on_best.py    # Step 2a – Re-train best configs (no BMI filter)
├── experiment_on_best2.py   # Step 2b – Re-train best configs (BMI ≤ 35 filter)
├── review_results.py        # Step 3 – Rerun top models, generate curves & SHAP
├── plot_curves.py           # Step 4 – Generate final ROC / PR figures
├── best_configs.csv         # Best hyperparameter configurations (used by Step 2)
├── ECG_include.ipynb        # Step 5 – Figures and analyses for the paper
└── ecg_vit/
    ├── data.py              # ViT data pipeline (public WFDB ECG dataset)
    ├── train.py             # ViT training entry point
    ├── vit.py               # Vision Transformer model definition
    ├── plot.py              # 12-lead ECG plotting utility
    └── plot_attention.py    # Attention map visualization on ECG traces
```

---

## Data You Must Supply

The main pipeline requires two private data sources that are **not** included:

| File / Folder | Description |
|---|---|
| `../ECGs/*.XML` | Per-patient 12-lead ECG files in the Philips/GE XML format (one file per MRN) |
| `Afib_data_v2.csv` | Tabular EHR/clinical spreadsheet with one row per patient, including the AFib diagnosis label column |

After running **Step 1** once, three intermediate files are saved and reused in all later steps:

| File | Description |
|---|---|
| `ready_EHR.csv` | Processed and aligned EHR feature matrix |
| `ready_ECG.npy` | Stacked ECG arrays: raw, wavelet-denoised, filtered, FFT, and synthetic variants |
| `ready_Y.csv` | Binary AFib labels aligned to the above |

---

## Pipeline Walkthrough

### Step 1 – Raw data ingestion and hyperparameter search: `experiment.py`

Reads raw ECG XML files and `Afib_data_v2.csv`, aligns patients by MRN, applies wavelet denoising (`bior4.4`), computes FFT representations, and sweeps a grid of model hyperparameters via 5-fold cross-validation.

**Model:** A dual-input Keras model combining a 1D-CNN ± multi-head attention ECG encoder with a small MLP EHR encoder, fused and decoded to a binary AFib prediction.

Key arguments (positional, in order):

```
python experiment.py <com_arch> <n_comp> <num_heads> <ecg_app> <aug_p> <n_inc> <n_comp_ecg> <n_comp_ehr> <lr>
```

| Argument | Meaning |
|---|---|
| `com_arch` | ECG encoder architecture (0–6; 0–2 include attention) |
| `n_comp` | Compression dimension |
| `num_heads` | Attention heads (used when `com_arch` ∈ {0,1,2}) |
| `ecg_app` | Which ECG representation: 0=raw, 1=denoised, 2=denoised+filtered, 3=FFT, -1=shuffled labels (sanity check) |
| `aug_p` | Augmentation probability per transformation |
| `n_inc` | Augmentation multiplier (train-time and test-time) |
| `n_comp_ecg` | ECG embedding size (0 = ECG-only off, uses EHR only) |
| `n_comp_ehr` | EHR embedding size (0 = EHR-only off, uses ECG only) |
| `lr` | Initial learning rate |

Each run saves a random-ID result to `res/<ID>.csv` and `res/<ID>.npy`.

**Before running the sweep**, uncomment the three lines near the bottom of `experiment.py` that save `ready_EHR.csv`, `ready_ECG.npy`, and `ready_Y.csv`, run once, then comment them back out or just let the sweep proceed (data loading happens at script start each time).

---

### Step 2 – Re-train best configurations

Once you have a set of runs in `res/`, identify the best hyperparameter configurations and list them in `best_configs.csv` (the one committed here reflects the published results). Then run:

**Without BMI filter:**
```
python experiment_on_best.py <row_index>
```

**With BMI ≤ 35 filter** (matches the primary published analysis):
```
python experiment_on_best2.py <row_index>
```

`<row_index>` is the 0-based row in `best_configs.csv`. Results are saved to `res_best/` and `res_best2/` respectively, including per-fold predictions (`*_pred.csv`) needed for the review step.

---

### Step 3 – Result review, curve generation, and SHAP: `review_results.py`

Scans a results directory, selects the top-K runs by a chosen metric, **reruns** them on the standardized `ready_*` data, and saves:
- `roc_curve.csv` / `pr_curve.csv` per run
- `summary.csv` ranking all reruns
- SHAP EHR feature importance (if `shap` is installed)

```bash
python review_results.py \
    --res_dir res \
    --out_dir res_review \
    --metric auroc \
    --top_k 10 \
    --use_bmi_filter        # add this flag to match published BMI-filtered analysis
```

Key flags:

| Flag | Default | Meaning |
|---|---|---|
| `--res_dir` | `res` | Folder with `*.csv` + `*.npy` run pairs |
| `--out_dir` | `res_review` | Output folder |
| `--metric` | `auroc` | Ranking metric: `auroc`, `accuracy`, `f1`, `precision`, `recall`, `specificity` |
| `--top_k` | `10` | How many top runs to rerun |
| `--use_bmi_filter` | off | Apply BMI ≤ 35 patient filter |
| `--keep_only_best` | off | Delete all rerun folders except best AUROC and best AUPRC |

---

### Step 4 – Final ROC / PR figures: `plot_curves.py`

Reads `res_review/summary.csv`, picks the single best run, loads its curve CSV, and saves a publication-ready figure.

```bash
# ROC curve figure
python plot_curves.py --metric auroc --res_dir res_review

# PR curve figure
python plot_curves.py --metric auprc --res_dir res_review
```

Output: `res_review/best_roc_curves.png` and `res_review/best_pr_curves.png`.

---

### Step 5 – Paper figures and analyses: `ECG_include.ipynb`

Jupyter notebook used to generate the figures and statistical analyses that appear in the paper. It reads from `ready_EHR.csv`, `ready_ECG.npy`, `ready_Y.csv`, and the `res_review/` outputs produced in Steps 3–4. Open with Jupyter Lab or Jupyter Notebook and run cells sequentially.

---

## `ecg_vit/` – Vision Transformer ECG sub-model

A separate ViT-based ECG classifier trained on a **public** dataset (the [A Large Scale 12-Lead ECG Database for Arrhythmia Study](https://physionet.org/content/ecg-arrhythmia/1.0.0/), PhysioNet). This module is self-contained and does not require the private AFib dataset.

| File | Role |
|---|---|
| `ecg_vit/data.py` | Builds TensorFlow `Dataset` pipelines from WFDB `.hea` records; handles NaN replacement, train/val splitting |
| `ecg_vit/vit.py` | Full Vision Transformer: patch embeddings, transformer blocks, stochastic depth, CLS-token classification head |
| `ecg_vit/train.py` | Training entry point: `python ecg_vit/train.py <path_to_database_root>` |
| `ecg_vit/plot.py` | Utility to render a 12-lead ECG with the standard clinical grid |
| `ecg_vit/plot_attention.py` | Loads a trained ViT checkpoint, extracts last-layer self-attention, and overlays it on the ECG waveform: `python ecg_vit/plot_attention.py <ecg_file> <weights.h5> <lead_index>` |

---

## Dependencies

Install into a conda/virtual environment:

```bash
pip install numpy pandas scipy scikit-learn keras tensorflow \
            pywavelets xmltodict tqdm matplotlib seaborn scienceplots \
            wfdb shap
```

`scienceplots` is only required for the notebook (`ECG_include.ipynb`).  
`shap` is optional; `review_results.py` will skip SHAP analysis gracefully if it is not installed.  
`keras_cv` is required for `ecg_vit/vit.py` (provides `StochasticDepth`).

---

## Reproducing the Published Results

1. Place your ECG XMLs at `../ECGs/*.XML` and EHR spreadsheet at `Afib_data_v2.csv`.
2. Run `experiment.py` once with the save lines uncommented to generate `ready_EHR.csv`, `ready_ECG.npy`, `ready_Y.csv`.
3. Run the grid sweep (see `run.sh` in the original repo for the parameter ranges used).
4. Run `experiment_on_best2.py` for each row in `best_configs.csv` (rows 0–18).
5. Run `review_results.py --use_bmi_filter --res_dir res_best2 --out_dir res_review`.
6. Run `plot_curves.py` to produce the final figures.
7. Open `ECG_include.ipynb` and run all cells to reproduce the paper figures.
