# Enhancing Object Detection Robustness in Low-Light Conditions
## via Self-Calibrated Illumination (SCI) + YOLOv8

**Course:** COE-486 Computer Vision — American University of Sharjah  
**Authors:** Hessa Al Maktoum (G00097904) | Hend Al Owais (G00084770) | Jana Ibrahim (G00094967)

---

## Overview

This project reproduces and adapts the CVPR 2022 paper  
**"Toward Fast, Flexible, and Robust Low-Light Image Enhancement" (SCI)**  
and evaluates it as a preprocessing step for **YOLOv8-Large** object detection  
on the **Exclusively Dark (ExDark)** dataset.

**Pipeline:**
```
Low-light image → SCI/CLAHE Enhancement → YOLOv8-Large → Detections
```

---

## Project Structure

```
cv-project/
├── refined_project_workflow.ipynb   ← Main notebook (run this)
├── report_final.tex                 ← IEEE LaTeX report
├── sci_model.py                     ← SCI architecture in PyTorch
├── requirements.txt                 ← Python dependencies
├── models/                          ← Place model weights here
│   ├── yolov8l.pt                   ← Download separately (see below)
│   └── sci_weights.pth              ← Download separately (optional)
└── data/
    └── ExDark/
        └── SPIC/                    ← 3 sample low-light images
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch, OpenCV, Ultralytics YOLOv8, scikit-image, pandas, matplotlib

---

## Download Model Weights

### YOLOv8-Large (required)
```bash
mkdir -p models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt -O models/yolov8l.pt
```
Or on Windows (PowerShell):
```powershell
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt" -OutFile "models\yolov8l.pt"
```

### SCI Pretrained Weights (optional — see note below)
The SCI pretrained weights from the original authors can be placed at `models/sci_weights.pth`.  
If not available, the pipeline automatically falls back to **CLAHE** enhancement.

> **Why `sci_weights.pth` is not included in this repository:**  
> The original SCI authors (Ma et al., CVPR 2022) hosted their pretrained weights on a
> personal server that is **no longer accessible** at the time of this project (April–May 2026).
> We faithfully reproduced the full SCI architecture in PyTorch (see `sci_model.py` and the
> notebook's Section 3), but without the original pretrained weights the model produces
> random-initialisation outputs that are unsuitable for evaluation.  
>
> Rather than report misleading SCI results from an untrained model, **all experiments in the
> report use the CLAHE classical baseline** as the enhancement method. CLAHE serves as a
> deterministic, reproducible preprocessing step that isolates the effect of illumination
> enhancement on YOLOv8 detection—independent of any specific learned enhancer. The report
> explicitly discusses this choice in Section 6.4 ("CLAHE vs. SCI") and projects expected
> SCI gains based on the original paper's benchmarks.  
>
> If the pretrained weights become available in the future, simply place them at
> `models/sci_weights.pth` and re-run the notebook; the pipeline will automatically switch
> to SCI enhancement with no code changes required.

---

## Dataset Setup

This repo includes 3 SPIC sample images from ExDark in `data/ExDark/SPIC/`.

For full ExDark evaluation (7,363 images):
1. Download ExDark from: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset
2. Place images under `data/ExDark/`
3. The notebook will auto-detect and process them

---

## How to Run

### Step 1 — Open the notebook
```bash
jupyter notebook refined_project_workflow.ipynb
```

### Step 2 — Run all cells in order
The notebook is fully self-contained. Running all cells will:
1. Load the 3 ExDark sample images
2. Apply 5 data augmentation techniques (darkening, noise, blur, gamma, combined)
3. Enhance images using SCI (or CLAHE fallback)
4. Run YOLOv8-Large inference on raw vs. enhanced images
5. Compute metrics: confidence scores, detection counts, brightness, PSNR/SSIM
6. Save 7 comparison figures to `runs/figures/`
7. Print a full quantitative results table

### Step 3 — View results
After running, output figures are saved to `runs/figures/`:
- `01_raw_images.png` — raw low-light samples
- `02_augmentations.png` — augmentation variants
- `03_enhancement_quality.png` — raw vs enhanced with PSNR
- `04_detection_comparison.png` — side-by-side detection results
- `05_quantitative_bars.png` — confidence and brightness bar charts
- `06_augmented_bars.png` — results across augmentation types
- `07_confidence_hist.png` — confidence score distributions

---

## Testing (Inference Only)

This is an **inference-only** project (no model training required).  
To test on a custom image:

```python
from ultralytics import YOLO
import cv2

def clahe_enhance(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

model = YOLO('models/yolov8l.pt')
img = cv2.imread('your_dark_image.jpg')
enhanced = clahe_enhance(img)

# Baseline
results_raw = model.predict(img, conf=0.25)

# Enhanced
results_enh = model.predict(enhanced, conf=0.25)
```

---

## Key Results

| Image | Brightness Raw | Brightness Enhanced | Gain | Conf (Base) | Conf (Enhanced) |
|---|---|---|---|---|---|
| 2015_00003.png | 23.9 | 55.9 | +134% | 0.370 | 0.467 |
| 2015_02446.jpg | 18.4 | 32.1 | +74% | 0.544 | 0.408* |
| 2015_06400.jpg | 9.6 | 24.3 | +153% | 0.337 | 0.473 |

*Detection count increased from 6 → 12 (improved recall)

---

## References

1. Ma et al., *Toward Fast, Flexible, and Robust Low-Light Image Enhancement*, CVPR 2022
2. Loh & Chan, *Getting to Know Low-Light Images with the Exclusively Dark Dataset*, CVIU 2019
3. Jocher et al., *Ultralytics YOLO*, 2023
