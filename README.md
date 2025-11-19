# BoltLooseningDetection v2.2: Multi-Modal Attention Fusion Framework

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)
![Status](https://img.shields.io/badge/status-Active-success.svg?style=flat-square)
![Config](https://img.shields.io/badge/Config-222-brightgreen.svg?style=flat-square)

## 📖 Project Overview

**BoltLooseningDetection v2.2** 是一个高性能的深度学习框架，专为工业螺栓松动检测设计。它采用先进的 **多模态 "2-2-2" 架构**，结合 1D 振动信号和 2D 时频图像，实现了 16 种不同松动状态的鲁棒分类。

## 📂 Directory Structure

```text
BoltLooseningDetection/
├── config.json           # 核心配置 (设置为 "222" 模式)
├── dataset.py            # 数据加载、5通道图像生成、增强
├── generalization.py     # 泛化性测试脚本
├── model.py              # ResNet101, Hybrid Signal Encoder, Attention Fusion
├── train.py              # 主训练流程
├── checkpoints/          # 模型权重存储
├── logs/                 # TensorBoard 日志和混淆矩阵
└── data/                 # 数据集目录 (Case1 - Case16)
```

---

## ⚙️ Environmental Requirements

要确保所有依赖项都正确安装，请运行以下命令：

```bash
pip install torch torchvision numpy pandas \\
            scipy librosa opencv-python scikit-learn \\
            matplotlib seaborn tqdm tensorboard
```

> **Note:** `os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'` 已在 `train.py` 中自动设置以确保兼容性。

---

## 🔧 Configuration (`config.json`)

默认 `config.json` 设置为 **"222"** 高性能模式：

```json
"modality": {
    "pseudo_image_mode": 1,   // 5-channel mode
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

### 1. Training

启动完整的训练流程。脚本会在首次运行时自动计算信号统计数据 (mean/std)。

```bash
python train.py
```

* **Output:** 最佳模型保存到 `./checkpoints/best_model_*.pth`。
* **Logging:** 指标 (损失/精度) 记录到 TensorBoard；混淆矩阵保存到 `./logs`。

### 2. Generalization Test

在特定的未见案例或完整数据集上评估训练好的模型，以获取详细指标。

```bash
python generalization.py
```

* **Output:** 详细的分类报告 (精确度、召回率、F1) 和分案例精度。

---

## 📊 Visualization

您可以使用 TensorBoard 监控训练进度并查看混淆矩阵：

```bash
tensorboard --logdir=./logs
```

项目还会在测试后自动在 `./logs` 文件夹中生成 **混淆矩阵热图 (PNG)**。

---

## 📝 License

本项目在 MIT 许可证下开源。
