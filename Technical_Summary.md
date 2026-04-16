# 🧬 Technical Deep-Dive: Lung Disease Classification

Detailed technical breakdown of the deep learning pipeline developed for chest X-ray analysis.

## Core Problem
Differentiating COVID-19, Normal, and Viral Pneumonia cases from grayscale radiographic imagery. 

## 🏗️ Architecture Design & Optimization

### 1. Model Comparisons (CNN vs. ResNet vs. EfficientNet)
| Parameter | CNN Baseline | ResNet50 | EfficientNet-B0 |
| :--- | :--- | :--- | :--- |
| **Strategy** | Scratch | Transfer Learning | Transfer Learning |
| **Depth** | 4 Blocks | 50 Layers | Scaled w/ MBConv |
| **Total Params** | 11.2M | 25.5M | 5.3M |
| **Accuracy** | 93.10% | 98.37% | **98.86%** |

### 2. Hyperparameter Optimization & Training Protocol
- **Optimizer**: Adam (Adaptive Moment Estimation) for stable convergence.
- **Learning Rate Strategy**: 
  - Phase 1 (Frozen): 0.001 (10 epochs)
  - Phase 2 (Fine-tuning): 0.0001 (15 epochs)
- **Batch Size**: 16 (optimized for RTX 3060 VRAM).
- **Loss Function**: `CrossEntropyLoss` with weights (where applicable) to handle sample imbalance.
- **Image Size**: 224x224 (pre-processed to match ImageNet input standards).

---

## 🛠️ Data Engineering & Augmentation

### Preprocessing Pipeline:
1.  **Resize**: Standardized all inputs to 224x224.
2.  **Normalization**: Applied ImageNet statistics (+0.485, +0.456, +0.406 mean; +0.229, +0.224, +0.225 std).
3.  **Stratified Sampling**: Ensured class distribution (70/15/15) was maintained across splits to prevent bias during validation.

### Augmentation Techniques (Training Only):
- **Rotation**: Randomly rotated by ±15° to simulate varying clinical imaging angles.
- **Affine Transforms**: Zoom (15%) and Shift (10%) to simulate different patient positions.
- **Horizontal Flipping**: Augments symmetry without losing diagnostic features.

---

## 🚀 Performance Insights & Explainability

### AUC & ROC Analysis:
The **EfficientNet-B0** model achieved near-perfect AUC scores (0.99+) across all three classes. 

### Confusion Matrix Observations:
- **COVID-19 vs. Viral Pneumonia**: Successfully resolved common overlaps using EfficientNet's compound scaling, which captures fine-grained textural features in the lung parenchyma better than standard CNNs.
- **High Sensitivity**: Achieved a recall of 98%+ for COVID-19 cases, a primary goal for early detection and isolation protocols.

---

## 📂 Installation & Requirements
- Python 3.9+ 
- PyTorch 2.5.1 / CUDA 12.1
- `timm` (PyTorch Image Models)
- `streamlit` (Inference Dashboard)

```bash
pip install -r requirements.txt
```
