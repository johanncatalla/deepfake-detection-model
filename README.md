# DeepFake Detection (Image Classification in R + Keras)

This repository contains a full, **end‑to‑end pipeline for detecting deep‑fake images** built in R using the `keras` and `tensorflow` interfaces.  
Four convolutional‑neural‑network (CNN) architectures are trained, fine‑tuned, and evaluated, with utilities for class‑balancing, threshold optimisation, and (optional) ensembling.

> **TL;DR**  
> 1. Place your dataset under `data/` following the structure below.  
> 2. Open **RStudio**, set the working directory to the repo root, and install the listed packages.  
> 3. Run the scripts in the order shown in **🗺️ Workflow** – that’s it!

---

## 📂 Directory layout

```
.
├── data/
│   └── DeepFakeDetection/
│       ├── train/
│       │   ├── fake/
│       │   └── real/
│       ├── valid/
│       │   ├── fake/
│       │   └── real/
│       └── test/
│           ├── fake/
│           └── real/
└── scripts/
    ├── preprocessing.R
    ├── models.R
    ├── main.R
    ├── fine_tuning.R
    ├── class_weights.R
    ├── threshold_tuning.R
    ├── evaluation.R
    ├── final_evaluation.R
    └── ensemble.R        (placeholder)
```

*The `data/` folder in the zip already follows this hierarchy.*

---

## 🔧 Environment

| Component        | Version (tested) |
|------------------|------------------|
| R                | 4.3.2            |
| Python           | 3.11             |
| tensorflow (py)  | ≥ 2.15           |
| keras (R pkg)    | ≥ 2.13           |
| reticulate       | ≥ 1.34           |
| caret, pROC      | see `install.R`  |

> **Important** – `scripts/preprocessing.R` calls `reticulate::use_python()`; update the path to the Python binary of the virtual‑env/conda env where TensorFlow is installed.

Install R packages once:

```r
install.packages(c("keras","tensorflow","reticulate","caret","pROC"))
tensorflow::install_tensorflow()   # installs TF in the active python env
```

---

## 🗺️ Workflow

| Stage | Purpose | Script(s) |
|-------|---------|-----------|
| 1. Data pipeline | Define generators & basic augmentation | `preprocessing.R` |
| 2. Model definition | Build **four** CNNs + custom focal loss | `models.R` |
| 3. Baseline training | Train each model for *n = 20* epochs | `main.R` |
| 4. Fine‑tuning | Unfreeze top *k* layers for further training | `fine_tuning.R` |
| 5. Class balancing | Compute class weights & re‑train (optional) | `class_weights.R` |
| 6. Threshold search | Use ROC to pick best probability threshold | `threshold_tuning.R` |
| 7. Evaluation | Compute metrics, confusion matrices, ROC/AUC | `evaluation.R`, `final_evaluation.R` |
| 8. Ensembling ✨ | (future) Combine models via majority‑vote / averaging | `ensemble.R` |

Run each script sequentially in a fresh R session **or** source them from a driver script/notebook.

---

## 📝 Script‑by‑script details

| File | Role & Key Functions |
|------|----------------------|
| **`preprocessing.R`** | *Sets the stage.* Defines image generators for train/validation/test with resizing (`224×224`), rescaling, and light augmentation. Paths to `data/train`, `data/valid`, `data/test` are declared here, as well as global constants (`img_height`, `batch_size`, etc.). Generates three `keras_preprocessing$image_iterator` objects (`train_generator`, `validation_generator`, `test_generator`). |
| **`models.R`** | *Model zoo.* Implements:<br/>• **Model 1 – ResNet50** (ImageNet weights, top replaced)<br/>• **Model 2 – EfficientNet‑B0**<br/>• **Model 3 – Lightweight CNN + Squeeze‑and‑Excite** block<br/>• **Model 4 – EfficientNet‑B0 + Dual Attention** (channel & spatial).<br/>A custom `focal_loss(gamma, alpha)` is defined to address class‑imbalance, and each model is compiled with `optimizer_adam(lr = 1e‑4)`. |
| **`main.R`** | *Baseline training loop.* Trains the four networks for 20 epochs using early‑stopping (`val_loss`, patience = 5) and learning‑rate reduction. Histories are saved to `history1`…`history4`. |
| **`fine_tuning.R`** | *Second‑stage training.* For each transfer‑learning model (ResNet, EfficientNet), the final **50** layers are unfrozen and re‑trained at a smaller LR (`1e‑5`). Histories `history1_ft`… are appended. |
| **`class_weights.R`** | *Balancing act.* Tabulates image counts per class from `train_generator`, calculates inverse‑frequency weights, and re‑trains each model passing `class_weight = list("0"=…, "1"=…)`. |
| **`threshold_tuning.R`** | Uses `pROC::coords()` to pick the threshold on predicted probabilities that maximises the `Youden` index (closest to top‑left of the ROC curve). Returns per‑model metrics at the new threshold (Accuracy, Precision, Recall, F1, AUC). |
| **`evaluation.R`** | *Quick check.* Evaluates the **baseline** models on the test set with the default 0.5 threshold; writes confusion matrices and ROC objects. |
| **`final_evaluation.R`** | Re‑runs prediction after fine‑tuning **and** optimal thresholds; collates a tidy `summary_ft` data.frame for reporting. |
| **`ensemble.R`** | Placeholder for majority‑vote / probability‑average ensemble logic (not yet implemented). |
| **`install.R`** *(if present)* | Convenience script for installing required CRAN packages. |

---

## 📊 Results

After fine‑tuning & threshold optimisation you should see accuracies in the **60–65 %** range and observable gains in **Recall** (≈ 0.40–0.45) over the naïve 0.5 cut‑off.  
An example `summary_ft` output (yours will differ):

| Model | Thr. | Acc. | Prec. | Recall | F1 | AUC |
|-------|-----:|-----:|------:|-------:|---:|----:|
| ResNet50        | 0.547 | 0.60 | 0.61 | 0.41 | 0.49 | 0.52 |
| EffNetB0        | 0.454 | 0.59 | 0.64 | 0.29 | 0.39 | 0.52 |
| LightCNN + SE   | 0.473 | 0.62 | 0.71 | 0.31 | 0.43 | 0.55 |
| DualAttention   | —     | —    | —    | —    | —    | —    |

*(See your console for the exact table.)*

---

## 🤖 Inference on new images

1. Place images inside a folder, e.g. `predict/`.
2. Build a generator:

```r
new_gen <- flow_images_from_directory(
  "predict",
  test_datagen,            # re‑use the same preprocessing object
  target_size = c(224,224),
  batch_size  = 32,
  shuffle     = FALSE
)
pred <- model_3 %>% predict(new_gen)
```

3. Apply the tuned threshold from `threshold_tuning.R` to obtain class labels.

---

## 🚧 Road‑map

- [ ] Implement `scripts/ensemble.R` – soft‑voting & stacking  
- [ ] Export best model to **TensorFlow SavedModel** / **ONNX**  
- [ ] Add Grad‑CAM visualisations for explainability  
- [ ] CI workflow (GitHub Actions) for unit tests & linting

---

### ✨ Acknowledgements

Built with **R, Keras, TensorFlow, reticulate**, and caffeinated late nights.  
Contributions & PRs are welcome!

---
