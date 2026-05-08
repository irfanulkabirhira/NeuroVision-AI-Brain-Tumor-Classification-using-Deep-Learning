# 🧠 NeuroVision-AI: Brain Tumor Classification using Deep Learning

> **A comprehensive, multi-paradigm deep learning benchmark for automated MRI-based brain tumor classification — achieving 96.07% test accuracy with statistical validation.**

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-Research-green.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-Mendeley%20Data-red.svg)](https://data.mendeley.com/datasets/zwr4ntf94j/6)

---

## 📌 What Makes This Study Stand Out

Most published brain tumor classification papers evaluate **one or two models** on varying datasets, making cross-study comparisons unreliable. This project is different:

| Feature | This Study | Typical Prior Work |
|---|---|---|
| Number of architectures compared | **7** | 1–3 |
| Training paradigms covered | **3** (scratch, frozen, fine-tuned) | 1–2 |
| Cross-validation applied to **all** models | ✅ Yes | ❌ Rarely |
| Statistical significance testing (McNemar's) | ✅ Yes | ❌ Rarely |
| Hospital-verified, publicly released dataset | ✅ 11,153 images | Varies |
| Grad-CAM interpretability + t-SNE visualization | ✅ Yes | Partial |

We don't just report accuracy — we provide **evidence-based, statistically validated** conclusions about *which architecture to choose and why*, under controlled and fair experimental conditions.

---

## 🏆 Key Results at a Glance

| Architecture | Test Accuracy | CV Accuracy | Paradigm |
|---|---|---|---|
| **ResNet50 Fine-Tuned** | **96.07%** | 98.20% ± 0.48% | Fine-Tuned Pretrained |
| **DenseNet121 Fine-Tuned** | **95.91%** | 98.27% ± 0.19% | Fine-Tuned Pretrained |
| Custom CNN | 90.68% | 95.39% ± 0.71% | From Scratch |
| ResNet50 Frozen | 85.14% | 92.91% ± 0.52% | Frozen Pretrained |
| DenseNet121 Frozen | 84.87% | 91.63% ± 0.62% | Frozen Pretrained |
| ViT (From Scratch) | 77.17% | 78.49% ± 1.92% | From Scratch |
| Hybrid CNN-ViT | 76.79% | 78.40% ± 1.07% | From Scratch |

> **McNemar's test** confirmed that differences between paradigms are statistically significant (*p* < 0.001). Differences *within* paradigms (e.g., ResNet50 FT vs DenseNet121 FT) are **not** statistically significant (*p* = 0.822).

---

## 🧩 Dataset

The dataset was collected by the authors from **Epic Hospital** and **CSCR Hospital**, comprising **11,153 brain MRI images** across four diagnostic classes.

![Dataset Split](https://github.com/irfanulkabirhira/NeuroVision-AI-Brain-Tumor-Classification-using-Deep-Learning/blob/9bee5126befecdc68ecd4e2aedbf3135ba61b4b9/Datasest%20Splition.png)

| Class | Train | Test | Total |
|---|---|---|---|
| Glioma | 3,018 | 603 | 3,621 |
| Meningioma | 2,183 | 436 | 2,619 |
| No Tumor | 1,945 | 389 | 2,334 |
| Pituitary | 2,150 | 429 | 2,579 |
| **Total** | **9,296** | **1,857** | **11,153** |

📦 **Dataset DOI:** [10.17632/zwr4ntf94j.6](https://data.mendeley.com/datasets/zwr4ntf94j/6) — Fully public for reproducibility.

All images are de-identified. MRI sequences include T1-weighted, T2-weighted, and FLAIR.

---

## 🔬 Architectures Evaluated

### From Scratch
| Model | Parameters | Notes |
|---|---|---|
| **Custom CNN** | 11.2M | 3 conv blocks (32→64→128 filters), valid padding, ReLU, 50% dropout |
| **Vision Transformer (ViT)** | 0.7M | 16×16 patches, 8 transformer blocks, 4-head attention, 64-dim embedding |
| **Hybrid CNN-ViT** | 0.5M | Dual-branch: CNN features (128-dim) + ViT features (64-dim) concatenated |

### Pretrained on ImageNet (Frozen)
| Model | Total Params | Trainable Params |
|---|---|---|
| ResNet50 Frozen | 24.1M | 557,956 |
| DenseNet121 Frozen | 7.3M | 295,812 |

### Pretrained on ImageNet (Fine-Tuned)
| Model | Total Params | Trainable Params |
|---|---|---|
| ResNet50 Fine-Tuned | 24.1M | 9,489,284 (last 20 layers) |
| DenseNet121 Fine-Tuned | 7.3M | 937,220 (last 30 layers) |

---

## ⚙️ Experimental Setup

All seven models were trained under **identical, controlled conditions**:

- **Image size:** 224 × 224 pixels, normalized to [0, 1]
- **Augmentation (training only):** random horizontal/vertical flip, brightness ±15%, contrast 0.85–1.15, random 90° rotation
- **Optimizer:** Adam with categorical cross-entropy loss
- **Learning rates:** 1e-3 (CNN, ViT, frozen) | 1e-4 (Hybrid, fine-tuned)
- **Batch size:** 32 (most models) | 48 (Hybrid)
- **Max epochs:** 20 with EarlyStopping (patience=5) and ReduceLROnPlateau (factor=0.5, patience=3)
- **Random seed:** 42 (fixed across all runs)
- **Hardware:** NVIDIA Tesla T4 GPU (16 GB VRAM), Google Colab Pro

### Evaluation Strategy
1. **5-Fold Stratified Cross-Validation** on training set (9,296 images)
2. **Held-out test set evaluation** (1,857 images) using the best-fold model checkpoint
3. **McNemar's test** for pairwise statistical significance
4. **Grad-CAM** for model interpretability
5. **t-SNE** for feature space visualization
6. **AUC-ROC** (one-vs-rest) per class

---

## 🔄 Workflow

```
1. Dataset Loading & Path Indexing
         ↓
2. Preprocessing (resize → normalize → tf.data pipeline)
         ↓
3. Online Data Augmentation (training only)
         ↓
4. 5-Fold Stratified Cross-Validation
         ↓
5. Model Training (EarlyStopping + LR Scheduling)
         ↓
6. Held-Out Test Set Evaluation
         ↓
7. McNemar's Statistical Testing
         ↓
8. Figures: Confusion Matrix, ROC, t-SNE, Grad-CAM, Efficiency
```

---

## 📊 Detailed Findings

### Why Fine-Tuned Pretrained Models Win
Fine-tuning unlocks domain adaptation. ResNet50 *Frozen* achieved 85.14% but the same backbone *fine-tuned* achieved 96.07% — a **~11 percentage point jump** from simply unfreezing the last 20 layers. ImageNet features are informative but insufficient alone for medical imaging without adaptation.

### Why the Custom CNN Beats Frozen Pretrained Models
A purpose-built CNN with strong convolutional inductive biases (spatial locality, translation equivariance) trained entirely on domain-specific MRI data outperformed generic frozen ImageNet features. When fine-tuning is not available, **task-specific CNN > frozen transfer learning**.

### Why Transformers Underperform on This Dataset
Vision Transformers lack inductive biases and require orders of magnitude more data to learn spatial patterns from scratch. With ~9,300 training images — far below the 14M+ needed for competitive ViT training — both ViT (77.17%) and Hybrid CNN-ViT (76.79%) failed to match CNN-based approaches. The Hybrid's additional complexity actually *hurt* optimization due to competing gradient signals.

### The Glioma-Meningioma Challenge
Across **all seven models**, the most common misclassification was **glioma ↔ meningioma**. This is clinically expected — both can present with similar MRI signal intensities. Even DenseNet121 Fine-Tuned showed 91.87% glioma recall with most errors directed toward meningioma. This is not a model limitation; it reflects inherent ambiguity in standard MRI sequences.

### Efficiency Insight
DenseNet121 Fine-Tuned achieves **95.91% accuracy with only 7.3M parameters** vs ResNet50 Fine-Tuned's 24.1M for 96.07%. For resource-constrained clinical deployment, **DenseNet121 Fine-Tuned offers the best accuracy-to-efficiency ratio**.

---

## 📈 Comparison with Prior Work

| Study | Model | Accuracy | Dataset |
|---|---|---|---|
| Badza et al. (2020) | VGG16 | 96.56% | 3-class |
| Deepak & Ameer (2019) | GoogleNet + FT | 97.1% | 3-class |
| Díaz-Pernas et al. (2021) | Ensemble | 97.45% | 3-class |
| **This study** | **ResNet50 Fine-Tuned** | **96.07%** | **4-class, 11,153 images** |
| **This study** | **DenseNet121 Fine-Tuned** | **95.91%** | **4-class, 11,153 images** |

> Our results are **directly comparable or superior**, achieved on a harder 4-class problem with a larger dataset, with full cross-validation and statistical testing — which most prior studies lack.

---

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow 2.19.0** (Keras Functional API)
- **NumPy, Pandas** — data handling
- **Scikit-learn** — cross-validation, metrics, McNemar's test
- **Matplotlib, Seaborn** — visualization
- **OpenCV (cv2)** — image processing
- **Google Colab Pro** — NVIDIA T4 GPU training environment

---

## 💻 Getting Started

### Prerequisites
```bash
pip install tensorflow scikit-learn matplotlib seaborn opencv-python pandas numpy
```

### Dataset
Download from Mendeley Data:
```
DOI: 10.17632/zwr4ntf94j.6
URL: https://data.mendeley.com/datasets/zwr4ntf94j/6
```

Expected folder structure:
```
dataset/
├── Train/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Test/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

### Running the Code

**Account 1 — CNN, ViT, Hybrid CNN-ViT:**
Open `Account1_CNN_ViT_Hybrid.ipynb` in Google Colab (T4 GPU recommended). Update `DATA_ROOT_DRIVE` to your Drive path and run all cells. Results saved to `cv_account1.json`.

**Account 2 — ResNet50 & DenseNet121 (Frozen + Fine-Tuned):**
Open `Account2_Pretrained_Models.ipynb`. Same setup. Results saved to `cv_account2.json`.

**Account 3 — DenseNet121 Fine-Tuned (continuation):**
Open `Account3_DenseNet121_FT.ipynb`. Results saved to `cv_account3.json`.

**Final Merge + Figures:**
Run `Merge_and_Figures.ipynb` to combine all results, compute McNemar's test, and generate all figures.

---

## 📊 Output Figures

| Figure | Description |
|---|---|
| `fig1_confusion_matrices.png` | Normalized confusion matrices for all 7 models |
| `fig2_roc_curves.png` | One-vs-rest ROC curves with per-class AUC |
| `fig3_performance_bars.png` | CV accuracy & F1-score comparison |
| `fig4_computational_efficiency.png` | Training time per fold (T4 GPU) |
| `fig5_tsne.png` | t-SNE of CNN penultimate layer features |
| `fig6_gradcam_cnn.png` | Grad-CAM attention maps for CNN |
| `fig7_efficiency.png` | Accuracy vs. training time vs. parameter count |

---

## 📌 Limitations & Future Work

**Current Limitations:**
- Data from 2 hospitals only — external multi-center validation needed
- 2D slice-based classification (not volumetric 3D MRI)
- ViT/Hybrid models use modest sizes; larger pretrained medical transformers may differ
- 4-class scope — real-world diagnosis involves more pathologies

**Future Directions:**
- Swin Transformer and ConvNeXt with hierarchical locality biases
- Self-supervised pretraining on unlabeled MRI data
- 3D volumetric architectures
- Multimodal learning (MRI + genomic/clinical data)
- Prospective clinical validation studies

---

## 📜 Citation

If you use this dataset, code, or findings in your research, please cite:

```bibtex
@article{hira2024neurovision,
  title={Comparative Performance Evaluation of Convolutional Neural Networks, Vision Transformers, Pretrained, and Hybrid Architectures for Automated Brain Tumor Classification Using MRI Scans},
  author={Hira, MD Irfanul Kabir and Hasan, MD Mahmudul and Towsif, Abdullah Al and Faisal, Mohammad Imtiaz},
  journal={[Journal Name]},
  year={2024}
}
```

**Dataset DOI:** `10.17632/zwr4ntf94j.6`

---

## 🤝 Contributing

Contributions are welcome! To contribute:
1. Fork this repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📜 License

This project is released for **educational and research purposes**. The dataset is publicly available under the terms of the Mendeley Data repository.

---

## 👨‍💻 Authors

**Md Irfanul Kabir Hira** — Conceptualization, Methodology, Software, Validation, Formal Analysis, Investigation, Data Curation, Visualization, Writing  
📧 irfanhira11@niter.edu.bd | University of Dhaka, Bangladesh

**Md Mahmudul Hasan** — Methodology, Software, Validation, Investigation, Data Curation  
National Institute of Textile Engineering and Research, Bangladesh

**Abdullah Al Towsif** — Methodology, Software, Validation, Investigation  
National Institute of Textile Engineering and Research, Bangladesh

**Mohammad Imtiaz Faisal** — Writing, Review & Editing, Supervision  
Department of ECE, North South University, Bangladesh

---

⭐ **If this project helps your research, please give it a star!**
