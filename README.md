# PRISM

![PRISM logo](PRISM_logo.png)

```
    ╭─╮         ╭──╮      ╭╮        ╭────╮
  ──╯  ╰──╭─────╯  ╰──────╯╰────────╯    ╰──  λ →

  ██████╗ ██████╗ ██╗███████╗███╗   ███╗
  ██╔══██╗██╔══██╗██║██╔════╝████╗ ████║
  ██████╔╝██████╔╝██║███████╗██╔████╔██║
  ██╔═══╝ ██╔══██╗██║╚════██║██║╚██╔╝██║
  ██║     ██║  ██║██║███████║██║ ╚═╝ ██║
  ╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝

  Predictive Regression via Infrared Spectral Models
```

A Python machine-learning pipeline for building and comparing **Near-Infrared Spectroscopy (NIRS) calibration models** that predict agricultural traits from NIR spectra. Ten model types are evaluated across four spectral preprocessing methods, with unified best-model selection, SHAP/saliency explainability, and GPR uncertainty quantification.

![Python](https://img.shields.io/badge/Python-3.10%2B-306998?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-2d6a4f)

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Data Format](#data-format)
5. [Configuration](#configuration)
6. [Spectral Preprocessing](#spectral-preprocessing)
7. [ML Methods](#ml-methods)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Pipeline Architecture](#pipeline-architecture)
10. [Command-Line Interface](#command-line-interface)
11. [Explainability](#explainability)
12. [GPU Acceleration](#gpu-acceleration)
13. [Output Files](#output-files)
14. [Physics of NIR](#physics-of-near-infrared-spectroscopy)

---

## Overview

NIRS calibration models learn to predict chemical or biological traits from the absorbance spectrum of a sample. Once trained, a model can predict a trait from a new spectrum in under a second, replacing slow, expensive wet-chemistry or laboratory assays.

This pipeline:

- Reads any dataset in the **standard NIRS CSV format** (non-numeric headers = traits, numeric headers = wavelengths)
- Applies four **spectral preprocessing** methods: SNV, MSC, SG1, SG2
- Trains and cross-validates **10 model types** in a full grid search
- Performs **unified best-model selection** across all methods and preprocessings on the holdout R²
- Saves serialised `.pkl` model bundles ready for deployment or prediction on new samples
- Generates **wavelength importance** plots, SHAP maps, CNN saliency, and GPR uncertainty bands
- Supports **GPU acceleration** for CNN, XGBoost, and LightGBM
- Is **fully config-driven**: add a new dataset by editing one YAML file

---

## Example Data

The bundled `PRISM_config.yaml` uses the publicly available **sensAIfood** cereal NIR dataset:

> Pérez-Marín, D. et al. (2025). *sensAIfood — NIR spectra for cereal grains (barley, maize, wheat) with Moisture and Protein reference values.* University of Córdoba / sensAIfood consortium. Zenodo. [https://doi.org/10.5281/zenodo.16759587](https://doi.org/10.5281/zenodo.16759587)

| Dataset | Samples | Wavelengths | Range |
|---------|--------:|------------:|-------|
| Barley | 178 | 700 | 1100–2498 nm |
| Maize | 141 | 700 | 1100–2498 nm |
| Wheat | 149 | 700 | 1100–2498 nm |

Download the files from Zenodo and place them under `data/sensAIfood_UnivCordoba_v2/csv/` to reproduce the example results exactly.

---

## Installation

### Requirements

```
Python ≥ 3.10
PyTorch ≥ 2.0          (CUDA build recommended if GPU is available)
scikit-learn ≥ 1.3
xgboost ≥ 2.0
lightgbm ≥ 4.0
shap
pandas, numpy, scipy, matplotlib, pyyaml, joblib
```

### Install

```bash
git clone https://github.com/<your-org>/nirs-calibration.git
cd nirs-calibration
pip install -r requirements.txt

# GPU support: match the build to your CUDA driver version
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

---

## Quick Start

```bash
# 1. Place your data files in data/csv/<dataset>.nirs.csv  (see Data Format)

# 2. Edit PRISM_config.yaml to point to your files

# 3. Full pipeline run
python python/PRISM_pipeline.py

# 4. Fast exploratory run (PLS, LASSO, ElasticNet, SVR, LightGBM only)
python python/PRISM_pipeline.py --quick

# 5. Specific datasets and traits
python python/PRISM_pipeline.py --crops wheat --traits protein dm

# 6. Predict new samples from a saved model bundle
python python/predict.py \
    --model results_python/best_models/wheat_protein_pls_snv.pkl \
    --spectra new_spectra.csv
```

Results are written to `results_python/`.

---

## Data Format

The pipeline reads a **standardised CSV file per dataset**. Column roles are auto-detected by parsing each header as a number:

| Header type | Role |
|-------------|------|
| Non-numeric (e.g. `sample_id`, `protein`, `dm`) | Trait / metadata |
| Numeric (e.g. `400.0`, `402.0`, …, `2498.0`) | Spectral wavelength (nm) |

### Example layout

```
sample_id,dm,protein,ndf,...,400.0,402.0,...,2498.0
S001,88.4,12.1,45.3,...,0.312,0.318,...,0.891
S002,87.9,11.8,46.1,...,0.308,0.314,...,0.887
```

### Rules

- Column 1 should be a sample identifier (`sample_id` or similar).
- Any non-numeric column with **≥ 10 non-NaN values** is treated as a modellable trait.
- Known metadata names are auto-skipped: `sample_id`, `sample_number`, `position`, `id`, `part`.
- Trait columns may contain NaN; a trait is skipped if too few values are present.
- Spectral columns must have numeric headers (wavelength in nm). Non-uniform spacing and gaps are handled automatically.

> Different spectral resolutions are supported: 2 nm step (1050 channels) and 0.5 nm step (4200 channels) have both been tested.

---

## Configuration

A single YAML file (`PRISM_config.yaml`) at the project root drives all pipeline globals. It is loaded automatically at import time.

```yaml
# Example using the public sensAIfood dataset (Zenodo: 10.5281/zenodo.16759587)
crops:
  barley:
    file: data/sensAIfood_UnivCordoba_v2/csv/Barley_sensAIfood_UnivCordoba_v2.csv
  maize:
    file: data/sensAIfood_UnivCordoba_v2/csv/Maize_sensAIfood_UnivCordoba_v2.csv
  wheat:
    file: data/sensAIfood_UnivCordoba_v2/csv/Wheat_sensAIfood_UnivCordoba_v2.csv

trait_labels:
  Moisture: "Moisture %"
  Protein:  "Protein %"

# Restrict modelling to the two assay traits.
# (Year is numeric and would otherwise be auto-discovered.)
trait_include:
  barley: [Moisture, Protein]
  maize:  [Moisture, Protein]
  wheat:  [Moisture, Protein]

pipeline:
  seed:          42
  n_folds:       10
  test_size:     0.20
  full_methods:  [pls, lasso, enet, svr, rf, xgb, lgbm]
  quick_methods: [pls, lasso, enet, svr, lgbm]
  preprocs:      [snv, msc, sg1, sg2]
```

### Config sections

| Section | Purpose |
|---------|---------|
| `crops` | Maps dataset name → CSV path (relative to project root). Add any new dataset here. |
| `trait_labels` | Human-readable display labels for plots. Unrecognised traits use the raw column name. |
| `pipeline.seed` | Global random seed for reproducibility. |
| `pipeline.n_folds` | Cross-validation folds (default 10). |
| `pipeline.test_size` | Holdout fraction (default 0.20). |
| `pipeline.full_methods` | Methods used in the standard run. |
| `pipeline.quick_methods` | Methods used with `--quick`. |
| `pipeline.preprocs` | Preprocessing grid. |
| `trait_include` *(optional)* | Per-dataset trait whitelist. |

### Adding a new dataset

1. Place your CSV in `data/csv/` following the [Data Format](#data-format) schema
2. Add an entry under `crops:` in `PRISM_config.yaml` pointing to that file
3. Optionally add display labels under `trait_labels:` and a whitelist under `trait_include:`
4. Run the pipeline; the new dataset is picked up automatically

---

## Spectral Preprocessing

Raw NIR spectra carry physical noise (baseline offsets from particle-size variation, detector drift) on top of the chemical signal. Four standard preprocessing methods are supported and grid-searched over all model types:

### SNV (Standard Normal Variate)

Each spectrum is standardised independently to zero mean and unit standard deviation across its wavelengths:

$$x_\lambda^{\text{SNV}} = \frac{x_\lambda - \bar{x}}{\sigma_x}$$

Removes both additive and multiplicative scatter without requiring any training-set reference. **Best default choice**, especially for small datasets.

### MSC (Multiplicative Scatter Correction)

Each spectrum is regressed against the training-set mean spectrum $r$:

$$x_i \approx \hat{a}_i + \hat{b}_i \cdot r \quad\Rightarrow\quad x_i^{\text{MSC}} = \frac{x_i - \hat{a}_i}{\hat{b}_i}$$

The reference $r = \bar{x}_\text{train}$ is computed on training data only and stored in the model bundle and applied unchanged to new samples (no leakage).

### SG1 / SG2 (Savitzky-Golay Derivatives)

The SG filter differentiates and smooths simultaneously by fitting a local polynomial to a sliding window. Derivatives sharpen overlapping absorption peaks and remove polynomial baselines.

| Method | Derivative | Window | Poly degree | Trim |
|--------|-----------|--------|-------------|------|
| SG1    | 1st       | 11 pts | 3           | 5 wavelengths each end |
| SG2    | 2nd       | 11 pts | 3           | 5 wavelengths each end |

---

## ML Methods

Ten model types are evaluated. Seven run in the standard grid; three are advanced methods that also run across all preprocessings.

### Standard grid (7 methods)

| Method | Key hyperparameters | Notes |
|--------|-------------------|-------|
| **PLS** | Components 1–20 (CV-selected) | NIR standard; handles multicollinearity; interpretable |
| **LASSO** | α (log-spaced grid) | Sparse; zeroes uninformative wavelengths |
| **ElasticNet** | α, l1_ratio | LASSO + Ridge combination |
| **LinearSVR** | C, ε | Robust to outliers; large-margin generalisation |
| **Random Forest** | 500 trees, `max_features='sqrt'` | Ensemble average reduces variance |
| **XGBoost** | Gradient-boosted trees | GPU histogram split-finding |
| **LightGBM** | Leaf-wise boosting | Fast; efficient on high-dimensional spectra |

### Advanced methods (3)

**Stacking ensemble** uses five base learners (PLS, LASSO, LinearSVR, XGBoost, LightGBM) to produce out-of-fold predictions via cross-validation. A Ridge meta-learner is then trained on the OOF stack, eliminating data leakage:

$$\hat{y}_i = \text{Ridge}\!\left([\hat{y}^{\text{PLS}}_i,\;\hat{y}^{\text{LASSO}}_i,\;\hat{y}^{\text{SVR}}_i,\;\hat{y}^{\text{XGB}}_i,\;\hat{y}^{\text{LGBM}}_i]\right)$$

**GPR (Gaussian Process Regression)** returns a mean prediction $\mu(\mathbf{x})$ **and** a per-sample standard deviation $\sigma(\mathbf{x})$. PCA (20 components) reduces the spectral dimension before kernel computation. Kernel: RBF + WhiteKernel. Enables uncertainty-aware screening by flagging samples with low confidence.

**1D-CNN** is a five-layer 1D convolutional network treating the spectrum as a 1D signal. Trained with 5-fold CV and Adam optimiser. Gradient saliency maps are computed after training. Supports GPU acceleration.

---

## Evaluation Metrics

Three metrics are reported for every model:

| Metric | Formula | Notes |
|--------|---------|-------|
| $R^2$ | $1 - \dfrac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2}$ | Proportion of variance explained. Range $(-\infty, 1]$. |
| RMSE | $\sqrt{\dfrac{1}{n}\sum(y_i-\hat y_i)^2}$ | Error in trait units. Compare within one trait only. |
| RPD | $\sigma_y / \text{RMSE}$ | Scale-free; compare across traits with different units. |

Three $R^2$ values are reported per model: $R^2_\text{cal}$ (training), $R^2_\text{cv}$ (cross-validation), $R^2_\text{val}$ (holdout). **Use $R^2_\text{val}$ as the primary selection criterion**; it is computed on data the model has never seen.

### RPD guide

| RPD | $R^2$ (approx.) | Interpretation | Practical use |
|-----|----------------|----------------|--------------|
| < 1.5 | < 0.56 | Not reliable | Cannot be used for screening |
| 1.5–2.0 | 0.56–0.75 | Rough screening | Coarse variety ranking |
| 2.0–2.5 | 0.75–0.84 | Good screening | Reliable screening programmes |
| 2.5–3.0 | 0.84–0.89 | Quantitative | Quality grading |
| > 3.0 | > 0.89 | Excellent | Process control |

### Overfitting diagnostics

| Pattern | Diagnosis |
|---------|----------|
| $R^2_\text{cal} \approx R^2_\text{cv} \approx R^2_\text{val}$ | Well-calibrated, generalises well |
| $R^2_\text{cal} \gg R^2_\text{cv}$ | Overfitting (too many parameters) |
| $R^2_\text{cv} \gg R^2_\text{val}$ | Dataset shift or lucky split |
| $R^2_\text{cal} \approx R^2_\text{cv}$, both low | Underfitting (model too simple) |

---

## Pipeline Architecture

```
data/csv/*.nirs.csv
        │
        ▼
  load_nirs_data()
  (auto-detect trait vs spectral columns from header type)
        │
        ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  Preprocessing grid: SNV · MSC · SG1 · SG2                   │
  │                                                              │
  │  Standard grid (7 methods):                                  │
  │    PLS · LASSO · ElasticNet · LinearSVR                      │
  │    Random Forest · XGBoost · LightGBM                        │
  │                                                              │
  │  Advanced methods (all 4 preprocessings × 5-fold CV):        │
  │    Stacking · GPR · 1D-CNN                                   │
  └──────────────────────────────────────────────────────────────┘
        │
        ▼
  Unified best-selection
  (all 10 methods × 4 preprocessings, ranked by R²_val)
        │
        ▼
  Retrain winner → .pkl bundle
  (model + ref spectrum + preprocessor +
   wavelengths + y_test + y_pred + X_test_pp)
        │
        ▼
  Importance / saliency plots  →  results_python/
```

### Unified best-model selection

All 10 methods × 4 preprocessings compete per dataset × trait on $R^2_\text{val}$. The winner is saved as the canonical model:

```
Grid (7 × 4)  +  Stacking (4)  +  GPR (4)  +  CNN (4 × 5-fold)
                    ↓
        sort by R²_val, take first per (dataset, trait)
                    ↓
        best_model_configs.csv + retrained .pkl bundle
```

---

## Command-Line Interface

```bash
python python/PRISM_pipeline.py [OPTIONS]
```

| Flag | Effect |
|------|--------|
| `--config PATH` | Path to YAML config (default: `PRISM_config.yaml` in project root) |
| `--quick` | Fast run: PLS, LASSO, ElasticNet, SVR, LightGBM only |
| `--crops NAME ...` | Dataset subset, e.g. `--crops wheat sorghum` |
| `--traits NAME ...` | Trait subset, e.g. `--traits protein dm` |
| `--methods NAME ...` | Explicit method list, overrides `--quick`/full, e.g. `--methods pls svr lgbm` |
| `--preprocs NAME ...` | Preprocessing subset, e.g. `--preprocs snv sg1` |
| `--no-advanced` | Skip stacking, GPR, CNN |
| `--out DIR` | Override output directory (default: `results_python/`) |

---

## Explainability

Three complementary tools are provided after model selection:

### SHAP (SHapley Additive exPlanations)

Model-agnostic attribution based on cooperative game theory. Answers: *"how much did each wavelength contribute to this prediction?"*

$$\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!} \bigl[f(S \cup \{j\}) - f(S)\bigr]$$

**Implementation:** `KernelExplainer` with 30 background training samples and up to 20 test samples. Available for all standard methods and stacking. A consensus overlay normalises each model's SHAP profile to [0, 1] and overlays them; high agreement across models validates that important wavelength regions reflect genuine chemistry.

> SHAP describes wavelength importance for standard methods. GPR uses saliency instead (PCA internals would return component importance). CNN uses gradient saliency.

### CNN Gradient Saliency

The gradient of the output $\hat{y}$ with respect to each input wavelength:

$$\text{saliency}_\lambda = \mathbb{E}_{\mathbf{x} \in X_\text{test}}\!\left[\left|\frac{\partial \hat{y}}{\partial x_\lambda}\right|\right]$$

Wavelengths where a small change causes a large change in prediction are considered important. Fast, requires no background set, and can be recomputed from the saved `X_test_pp` in the `.pkl` bundle.

### GPR Prediction Uncertainty

GPR returns a per-sample standard deviation $\sigma(\mathbf{x})$ alongside the point prediction. Samples in sparse or unusual spectral regions receive higher $\sigma$, indicating lower model confidence.

| Coverage metric | Ideal | Formula |
|----------------|-------|---------|
| coverage_90 | 0.90 | $P(\|y - \mu\| \leq 1.645\sigma)$ |
| coverage_95 | 0.95 | $P(\|y - \mu\| \leq 1.960\sigma)$ |

A sample with $\sigma \gg \bar\sigma_\text{train}$ lies outside the calibration space; the model is extrapolating and the prediction should be treated with caution or verified by conventional measurement.

---

## GPU Acceleration

Only the Python pipeline benefits from GPU acceleration. Three of the ten model types use it:

| Method | GPU mechanism | Typical speedup |
|--------|-------------|----------------|
| 1D-CNN | PyTorch CUDA (full training + backprop) | 5–30× |
| XGBoost | GPU histogram split-finding | 3–10× |
| LightGBM | OpenCL GPU kernel | 3–8× |
| All others | CPU only (scikit-learn) | N/A |

GPU status is auto-detected via `torch.cuda.is_available()`. The pipeline falls back to CPU gracefully.

> **Driver note:** `cudaErrorNoKernelImageForDevice` means the installed PyTorch wheel does not include your GPU's compute capability. Check your driver's max CUDA version with `nvidia-smi` and reinstall the matching wheel (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu124`).

---

## Output Files

### Scripts

| Script | Purpose |
|--------|---------|
| `python/PRISM_pipeline.py` | Main pipeline: trains all methods and saves results |
| `python/predict.py` | Predict new samples from a saved `.pkl` bundle |
| `python/export_predictions.py` | Extract y_test / y_pred from all `.pkl` bundles → CSV |
| `python/visualize_models.py` | Per-model 3-panel spectrum / coefficient / contribution figures |
| `PRISM_config.yaml` | Pipeline configuration; edit to add datasets and tune parameters |

### Generated files

| File | Contents |
|------|---------|
| `results_python/all_model_results.csv` | All method × preproc × dataset × trait R²/RMSE/RPD |
| `results_python/best_model_configs.csv` | Best method + preprocessing per dataset × trait |
| `results_python/best_model_predictions.csv` | y_test and y_pred for each best model |
| `results_python/best_models/*.pkl` | Serialised bundle: model, ref spectrum, preprocessor, wavelengths, y_test, y_pred, X_test_pp |
| `results_python/model_transforms/*.png` | 3-panel figures: spectrum · coef/importance · contribution |
| `results_python/plots/raw_spectra.png` | Raw spectral overlay |
| `results_python/plots/preprocessing_comparison_*.png` | Before / after all preprocessing methods |
| `results_python/plots/trait_distributions.png` | Trait distribution plots |
| `results_python/plots/heatmap_*.png` | R²_val heatmaps (method × preprocessing) |
| `results_python/plots/importance_*.png` | Wavelength importance / saliency for best model |
| `results_python/plots/stacking_comparison.png` | Stacking vs best individual model |
| `results_python/plots/cnn_saliency.png` | CNN gradient saliency maps |
| `results_python/plots/gpr_uncertainty.png` | GPR prediction intervals with ±2σ bands |
| `results_python/plots/shap_*.png` | SHAP wavelength importance per model |
| `results_python/plots/shap_consensus_*.png` | Normalised SHAP overlay across models |

---

## Physics of Near-Infrared Spectroscopy

### Beer-Lambert Law

When light of intensity $I_0$ passes through a sample, the transmitted intensity $I$ obeys:

$$A(\lambda) = \log_{10}\!\left(\frac{I_0}{I}\right) = \varepsilon(\lambda)\, c\, \ell$$

where $\varepsilon(\lambda)$ is the molar absorptivity, $c$ is concentration, and $\ell$ is path length. NIR instruments report $A(\lambda)$ simultaneously across all wavelengths, forming a *chemical fingerprint*. This linearity is the theoretical foundation for all regression models in this pipeline.

### NIR Absorption Bands

The near-infrared region (700–2500 nm) is rich in overtone and combination bands of X–H molecular bonds:

| Bond | Wavelength region | Typical analytes |
|------|-----------------|-----------------|
| O–H (moisture) | ~1450 nm, ~1940 nm | Water content, dry matter |
| N–H (amine) | ~1500–1680 nm | Crude protein, amino acids |
| C–H (carbohydrates, lipids) | ~1720 nm, ~2300–2350 nm | Fibre, starch, digestibility |
| C=O (amide) | ~2050–2180 nm | Protein, organic acids |

Analytes without direct NIR absorption (e.g. fermentation products) are predicted **indirectly** via their correlation with chemical constituents that do absorb; the spectrum encodes substrate composition, not the product directly. These indirect predictions typically yield lower R² than traits with strong direct absorption signatures.

---

## Licence

MIT. See [LICENSE](LICENSE).

---

**Author:** Anestis Gkanogiannis · [ganoyan@gmail.com](mailto:ganoyan@gmail.com)
