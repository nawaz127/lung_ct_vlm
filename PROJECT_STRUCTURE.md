# Project Structure

This repository is organized as a sequential ten-sector research workflow. Data flows from audit to reproducible preparation, model experiments, controlled degradation, evaluation, explanation, and final reporting.

## Sector 01: Environment and Project Configuration

**Purpose:** validate Python, PyTorch, CUDA, paths, logging, seeds, and reproducibility settings.

**Important locations:** `configs/`, `src/utils/`, `scripts/check_environment.py`, `reports/audit/`, `tests/test_sector01/`.

**Inputs and outputs:** environment and local path settings in; an audit report and validated runtime configuration out. Depends on no research data and feeds every later sector.

## Sector 02: Dataset Audit and Data Management

**Purpose:** inspect the PNG dataset without modifying the original files. Discover images, verify classes and historical counts, inspect dimensions and integrity, hash exact duplicates, and generate metadata and visual audit reports.

**Important locations:** `src/data/audit/`, `src/data/metadata/`, `src/data/visualization/`, `data/raw/`, `data/metadata/`, `reports/dataset_audit/`.

**Inputs and outputs:** original `D:/lung data` or configured equivalent in; verified metadata, integrity findings, and audit reports out. Depends on Sector 01 and feeds Sector 03. Patient/examination IDs are unavailable and this limitation must remain explicit.

## Sector 03: Preprocessing and Data Splitting

**Purpose:** define RGB conversion, 224x224 resizing, normalization, train-only augmentation, leakage checks, and stratified image-level 70/15/15 splits.

**Important locations:** `src/data/preprocessing/`, `src/data/splitting/`, `src/data/datasets/`, `src/data/transforms/`, `data/processed/`, `data/splits/`, `reports/preprocessing/`.

**Inputs and outputs:** audited metadata in; split manifests and preprocessing metadata out. Depends on Sectors 01-02. Because grouping identifiers are unavailable, patient-level leakage cannot be ruled out.

## Sector 04: CNN Architecture Development

**Purpose:** provide consistent PyTorch interfaces for MobileNetV3-Large, EfficientNet-B0, and ResNet18, with three-class heads, transfer learning, parameter counts, feature extraction, and Grad-CAM-compatible layers.

**Important locations:** `src/models/`, `src/models/backbones/`, `src/models/classifiers/`, `src/models/heads/`, `src/models/initialization/`, `tests/test_sector04/`.

**Inputs and outputs:** model configuration and class schema in; initialized model definitions and summaries out. Depends on Sector 01 and the class contract from Sector 03.

## Sector 05: CNN Training Engine

**Purpose:** run six experiments: three architectures by two strategies, namely clean baseline and corruption-aware training. Include validation, loss, optimizer, scheduler, mixed precision, early stopping, checkpoints, TensorBoard, history, and GPU monitoring.

**Important locations:** `src/training/`, `scripts/train/`, `configs/experiments/`, `checkpoints/`, `outputs/training/`, `outputs/logs/`.

**Inputs and outputs:** split manifests, model definitions, and experiment YAML in; checkpoints, histories, and logs out. Depends on Sectors 01, 03, and 04. Test data is never used for selection.

## Sector 06: Image Quality Degradation

**Purpose:** generate deterministic Gaussian noise, Gaussian blur, and contrast reduction at three severity levels with fixed seeds, stable test indexing, metadata, and comparison visualizations.

**Important locations:** `src/corruptions/`, `configs/corruptions/`, `outputs/corruptions/`.

**Inputs and outputs:** clean indexed images and corruption configuration in; reproducible corruption metadata and visual artifacts out. Depends on Sectors 01 and 03 and supplies Sectors 05 and 08.

## Sector 07: Vision-Language Model Integration

**Purpose:** evaluate frozen BiomedCLIP as a supplementary zero-shot baseline using fixed prompt ensembles, cached text embeddings, image-text similarity, and recorded package/checkpoint/tokenizer provenance.

**Important locations:** `src/vlm/`, `configs/vlm/`, `checkpoints/vlm/`, `outputs/vlm/`.

**Inputs and outputs:** fixed prompts, clean/corrupted image inputs, and a recorded model checkpoint in; separate VLM predictions and logs out. Depends on Sectors 01, 03, and 06. No fine-tuning or test-set prompt optimization is permitted.

## Sector 08: Robustness and Classification Evaluation

**Purpose:** compute macro-F1 as the primary metric plus accuracy, balanced accuracy, macro precision/recall, per-class recall, confusion matrices, and statistically appropriate OvR ROC-AUC. Also compute clean-to-corrupted degradation and efficiency metrics.

**Important locations:** `src/evaluation/`, `outputs/evaluation/`, `outputs/predictions/`, `reports/evaluation/`.

**Inputs and outputs:** CNN/VLM predictions and condition metadata in; condition-wise metrics, degradation summaries, and efficiency measurements out. Depends on Sectors 05-07.

## Sector 09: Explainable AI

**Purpose:** implement Grad-CAM and Integrated Gradients plus SSIM, cosine similarity, and Spearman rank correlation for explanation stability.

**Important locations:** `src/xai/`, `outputs/xai/`, `outputs/explanations/`, `reports/xai/`.

**Inputs and outputs:** trained CNNs, indexed images, and predictions in; heatmaps, overlays, explanation arrays, and stability reports out. Depends on Sectors 04-06 and 08. Paired clean/corrupted explanations use the clean-image predicted class as the fixed attribution target while actual predictions are recorded independently. No clinical localization claim is valid without expert masks.

## Sector 10: Final Analysis and Reporting

**Purpose:** aggregate results, compare models and strategies, perform statistical analysis, compare robustness/XAI stability/efficiency, and produce publication-quality figures, tables, thesis material, and reproducibility documentation.

**Important locations:** `src/analysis/`, `reports/final/`, `reports/tables/`, `reports/figures/`, `reports/thesis/`, `notebooks/`, `docs/`.

**Inputs and outputs:** validated metrics, explanations, efficiency records, and provenance in; final comparisons and thesis-ready artifacts out. Depends on all preceding sectors.

## Execution order

1. Validate the environment and freeze configuration.
2. Audit the original dataset and verify counts.
3. Generate stratified image-level splits and preprocessing metadata.
4. Implement and test the three CNN families.
5. Run the six CNN training configurations.
6. Generate deterministic corruption conditions.
7. Run frozen BiomedCLIP separately.
8. Evaluate classification, robustness, and efficiency.
9. Generate explanations and stability metrics.
10. Aggregate, analyze, and report results.
