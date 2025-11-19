# **Cobra-Net**: Multi-Modal Attention Fusion Framework for Bolt Loosening Detection

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/license-GPLv3-blue.svg?style=flat-square)
![Status](https://img.shields.io/badge/status-Active-success.svg?style=flat-square)
![Config](https://img.shields.io/badge/Config-222-brightgreen.svg?style=flat-square)

## 📖 Project Overview

**Cobra-Net** is a high-performance deep learning framework specifically engineered for automated industrial bolt loosening detection. It employs an advanced **Multi-Modal "2-2-2" Architecture** that robustly classifies 16 distinct loosening states by dynamically fusing 1D raw vibration signals and their generated 2D time-frequency image representations. This architecture is optimized to enhance feature discriminability, particularly in marginal classification cases.

### 🚀 Core Architecture ("222" Configuration)

The primary experimental setup leverages a robust combination of state-of-the-art models for maximum performance:

* **Image Encoder (Type 2 - Visual Feature Extraction):** **ResNet101** (Pretrained on ImageNet) adapted for processing 5-channel pseudo-image inputs (including Spectrograms and Phase/SVD features).
* **Signal Encoder (Type 2 - Sequential Feature Extraction):** **Hybrid Structure** combining 1D-CNN, Bi-LSTM, and a Transformer Encoder to capture local convolutions, temporal dependencies, and long-range global context from the vibration data.
* **Fusion Module (Type 2 - Inter-Modal Fusion):** **Multi-Head Attention Fusion** which dynamically calculates the weighted interaction (attention scores) between the visual and sequential feature embeddings before final classification.

---

## 📂 Directory Structure

```text
BoltLooseningDetection/
├── config.json           # Core configuration file (set to "222" mode)
├── dataset.py            # Data loading, 5-channel image generation, and augmentation utilities
├── generalization.py     # Evaluation script for generalization/unseen test cases
├── model.py              # Contains all core components: ResNet101, Hybrid Encoder, and Attention Fusion
├── train.py              # Main training pipeline script
├── checkpoints/          # Storage directory for trained model weights and checkpoints
├── logs/                 # Directory for TensorBoard logs and intermediate Confusion Matrices
└── data/                 # Expected location for the downloaded dataset files
```

---

## ⚙️ Environmental Requirements

To ensure all necessary dependencies are installed correctly, please execute the following command:

```bash
pip install torch torchvision numpy pandas \
            scipy librosa opencv-python scikit-learn \
            matplotlib seaborn tqdm tensorboard kagglehub
```

> **Note:** `os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'` is automatically set within the `train.py` script to ensure environment compatibility.

---

## 🔧 Configuration (`config.json`)

The default `config.json` is configured for the highest-performance **"222"** operational mode:

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

This project relies on a **publicly available dataset** hosted on Kaggle, originally developed by Oybek Eraliev. You can find the dataset directly here: [Vibration Dataset for Bolt Loosening Detection](https://www.kaggle.com/datasets/oybekeraliev/vibration-dataset-for-bolt-loosening-detection?resource=download).

Utilize the `kagglehub` library to seamlessly download the data to your local machine. The script below will automatically retrieve the files and print the path to the downloaded directory.

```python
import kagglehub
# Download the latest version of the dataset
path = kagglehub.dataset_download("oybekeraliev/vibration-dataset-for-bolt-loosening-detection")
print("Path to dataset files:", path)
```

> **Important:** Verify that the structure of the downloaded dataset matches the directory expectations of `dataset.py` (i.e., vibration files must be located within the specified structure).

### 1. Training

Initiate the full training pipeline. The script automatically computes necessary signal statistics (mean/std) upon the first run.

```bash
python train.py
```

* **Output:** The best model checkpoint based on validation performance is saved to `./checkpoints/best_model_*.pth`.
* **Logging:** Training metrics (Loss/Accuracy) are logged to TensorBoard; the final Confusion Matrix is saved to `./logs`.

### 2. Generalization Test

Evaluate the fully trained model on specific unseen cases or the complete dataset for detailed performance metrics.

```bash
python generalization.py
```

* **Output:** Comprehensive classification report (Precision, Recall, F1-Score) and per-case accuracy are printed to the console.

---

## 📊 Visualization

You can actively monitor the training progress, loss curves, and view intermediate results using TensorBoard:

```bash
tensorboard --logdir=./logs
```

Additionally, the project automatically generates high-resolution **Confusion Matrix Heatmaps (PNG)** in the `./logs` folder immediately after testing is complete.

---

## 📝 License

This project is open-sourced under the terms of the **GNU General Public License v3.0 (GPLv3)**.
