# BoltLooseningDetection v2.2: Multi-Modal Attention Fusion Framework

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat-square&logo=pytorch)
![License](https://imgshields.io/badge/license-GPLv3-blue.svg?style=flat-square)
![Status](https://img.shields.io/badge/status-Active-success.svg?style=flat-square)
![Config](https://img.shields.io/badge/Config-222-brightgreen.svg?style=flat-square)

## 📖 Project Overview

**BoltLooseningDetection v2.2** is a high-performance deep learning framework designed for industrial bolt loosening detection. It employs an advanced **Multi-Modal "2-2-2" Architecture** that robustly classifies 16 distinct loosening states by fusing 1D vibration signals and their 2D time-frequency image representations.

### 🚀 Core Architecture ("222" Configuration)

The primary experimental setup utilizes the following high-performance components:

* **Image Encoder (Type 2):** **ResNet101** (Pretrained) adapted for 5-channel pseudo-image input.
* **Signal Encoder (Type 2):** **Hybrid Structure** (1D-CNN + Bi-LSTM + Transformer Encoder) for comprehensive feature extraction (local, temporal, and global).
* **Fusion Module (Type 2):** **Multi-Head Attention Fusion** for dynamic, weighted interaction between visual and signal features.

---

## 📂 Directory Structure

```text
BoltLooseningDetection/
├── config.json           # Core configuration file (set to "222" mode)
├── dataset.py            # Data loading, 5-channel image generation, and augmentation
├── generalization.py     # Evaluation script for generalization/unseen test cases
├── model.py              # Contains all model components: ResNet101, Hybrid Encoder, Attention Fusion
├── train.py              # Main training pipeline
├── checkpoints/          # Model weights storage
├── logs/                 # TensorBoard logs and Confusion Matrices
└── data/                 # Dataset directory (Expected location after download)
```

---

## ⚙️ Environmental Requirements

To ensure all dependencies are installed correctly, please run the following command:

```bash
pip install torch torchvision numpy pandas \
            scipy librosa opencv-python scikit-learn \
            matplotlib seaborn tqdm tensorboard kagglehub
```

> **Note:** `os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'` is automatically set in `train.py` for compatibility.

---

## 🔧 Configuration (`config.json`)

The default `config.json` is configured for the **"222"** high-performance mode:

```json
"modality": {
    "pseudo_image_mode": 1,   // 5-channel image mode
    "image_model": {
        "type": 2,            // 2 = ResNet101
        "in_channels": 5,
        "out_dim": 256,
        "pretrained": true
    },
    "signal_model": {
        "type": 2,            // 2 = Hybrid (CNN+LSTM+Transformer)
        "embed_dim": 256,
        "nhead": 8
    },
    "fusion": {
        "type": 2,            // 2 = Multi-Head Attention Fusion
        "num_heads": 4
    }
}
```

---

## 🚀 Usage

### 0. Download Dataset (Mandatory)

The dataset is hosted on Kaggle. Use `kagglehub` to download the data to your local machine. The script below will print the path to the downloaded files.

```python
import kagglehub
# Download the latest version of the dataset
path = kagglehub.dataset_download("oybekeraliev/vibration-dataset-for-bolt-loosening-detection")
print("Path to dataset files:", path)
```

> **Note:** Ensure the downloaded dataset structure matches the expectations of `dataset.py` (i.e., vibration files within the specified directory structure).

### 1. Training

Start the full training pipeline. The script automatically calculates signal statistics (mean/std) on the first run.

```bash
python train.py
```

* **Output:** The best model checkpoint is saved to `./checkpoints/best_model_*.pth`.
* **Logging:** Metrics (Loss/Accuracy) are logged to TensorBoard; Confusion Matrix is saved to `./logs`.

### 2. Generalization Test

Evaluate the trained model on specific unseen cases or the full dataset for detailed metrics.

```bash
python generalization.py
```

* **Output:** Detailed classification report (Precision, Recall, F1-Score) and per-case accuracy.

---

## 📊 Visualization

You can monitor the training progress and visualize results using TensorBoard:

```bash
tensorboard --logdir=./logs
```

The project also automatically generates **Confusion Matrix Heatmaps (PNG)** in the `./logs` folder after testing.

---

## 📝 License

This project is open-sourced under the **GNU General Public License v3.0 (GPLv3)**.
