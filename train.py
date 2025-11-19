import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

# 确保兼容性
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, \
    classification_report

import logging
from dataset import create_dataloaders, calculate_signal_stats  # <--- 导入 calculate_signal_stats
from model import build_model
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LabelSmoothingLoss(nn.Module):
    """
    带标签平滑的交叉熵损失函数。
    """

    def __init__(self, smoothing=0.1, num_classes=16):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.num_classes = num_classes

    def forward(self, outputs, labels):
        """
        计算带标签平滑的损失。
        """
        log_probs = nn.functional.log_softmax(outputs, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, labels.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class EarlyStopping:
    """
    早停机制，用于在验证损失不再改善时停止训练，防止过拟合。
    """

    def __init__(self, patience=10, verbose=False, delta=0, path='best_model_8.9.pth'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        """
        根据当前验证损失决定是否早停并保存模型。
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        保存模型权重到指定路径。
        """
        if self.verbose:
            logging.info(f'Validation loss decreased ({-self.best_score:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)


def train_epoch(model, dataloaders, criterion, optimizer, device, writer, epoch, config):
    """
    执行一个完整的训练 epoch，实现多尺度训练。
    """
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    loaders_iterators = {seq_len: iter(loader) for seq_len, loader in dataloaders.items()}
    total_batches = sum(len(loader) for loader in dataloaders.values())
    signal_model_type = config['modality']['signal_model']['type']

    tqdm_loader = tqdm(range(total_batches), desc=f"[Train Epoch {epoch + 1}]", leave=False)

    for i in tqdm_loader:
        # 在不同序列长度的数据加载器之间交替
        seq_len = list(loaders_iterators.keys())[i % len(loaders_iterators)]

        try:
            image_data, signal_data, labels = next(loaders_iterators[seq_len])
        except StopIteration:
            # 如果迭代器耗尽，重新创建
            loaders_iterators[seq_len] = iter(dataloaders[seq_len])
            image_data, signal_data, labels = next(loaders_iterators[seq_len])

        # 处理 dataset.py 中可能返回的 None 样本
        if image_data is None:
            continue

        # 将数据移动到指定设备
        image_data, signal_data, labels = image_data.to(device), signal_data.to(device), labels.to(device)

        # 统一信号数据维度
        # 在 dataset.py 中，我们返回的 sig_tensor 形状是 (1, seq_len)
        # 在 DataLoader 堆叠后，信号数据形状是 (B, 1, seq_len)

        # 针对 Hybrid (Type 2, CNN) 模型的输入形状 (B, 1, S)
        if signal_model_type == 2:
            # 信号数据在 dataloader 堆叠后形状是 (B, 1, S) 或 (B, 1, 1, S)
            if signal_data.dim() == 4:
                signal_data = signal_data.squeeze(1).squeeze(1)  # 得到 (B, S)
            elif signal_data.dim() == 3 and signal_data.size(1) == 1:
                signal_data = signal_data.squeeze(1)  # 得到 (B, S)

            # 最终需要 (B, C, S)，即 (B, 1, S)
            signal_data = signal_data.unsqueeze(1)

            assert signal_data.dim() == 3 and signal_data.size(1) == 1, \
                f"train_epoch: signal_data has incorrect final shape for CNN. Got {signal_data.shape}"

        # 针对 RNN/LSTM (Type 0/1) 模型的输入形状 (B, S, C)
        elif signal_model_type in [0, 1]:
            # 信号数据在 dataloader 堆叠后形状是 (B, 1, S) 或 (B, 1, 1, S)
            if signal_data.dim() == 4:
                signal_data = signal_data.squeeze(1).squeeze(1)  # 得到 (B, S)
            elif signal_data.dim() == 3 and signal_data.size(1) == 1:
                signal_data = signal_data.squeeze(1)  # 得到 (B, S)

            # 最终需要 (B, S, C)，即 (B, S, 1)
            signal_data = signal_data.unsqueeze(-1)

            assert signal_data.dim() == 3 and signal_data.size(-1) == 1, \
                f"train_epoch: signal_data has incorrect final shape for RNN/LSTM. Got {signal_data.shape}"

        optimizer.zero_grad()
        outputs = model(image_data, signal_data)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        tqdm_loader.set_postfix(loss=loss.item(), seq_len=seq_len)

    avg_loss = running_loss / total_batches
    avg_acc = accuracy_score(all_labels, all_preds) if all_labels else 0
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Accuracy/train', avg_acc, epoch)
    return avg_loss, avg_acc


def validate_epoch(model, dataloaders, criterion, device, writer, epoch, config):
    """
    执行一个完整的验证 epoch。
    """
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    total_batches = sum(len(loader) for loader in dataloaders.values())
    signal_model_type = config['modality']['signal_model']['type']

    tqdm_loader = tqdm(range(total_batches), desc=f"[Val Epoch {epoch + 1}]", leave=False)

    with torch.no_grad():
        loaders_iterators = {seq_len: iter(loader) for seq_len, loader in dataloaders.items()}
        for i in tqdm_loader:
            seq_len = list(loaders_iterators.keys())[i % len(loaders_iterators)]
            try:
                image_data, signal_data, labels = next(loaders_iterators[seq_len])
            except StopIteration:
                loaders_iterators[seq_len] = iter(dataloaders[seq_len])
                image_data, signal_data, labels = next(loaders_iterators[seq_len])

            if image_data is None:
                continue

            image_data, signal_data, labels = image_data.to(device), signal_data.to(device), labels.to(device)

            # 统一信号数据维度 (与 train_epoch 保持一致)
            if signal_model_type == 2:  # CNN
                if signal_data.dim() == 4:
                    signal_data = signal_data.squeeze(1).squeeze(1)
                elif signal_data.dim() == 3 and signal_data.size(1) == 1:
                    signal_data = signal_data.squeeze(1)
                signal_data = signal_data.unsqueeze(1)

                assert signal_data.dim() == 3 and signal_data.size(1) == 1, \
                    f"validate_epoch: signal_data has incorrect final shape for CNN. Got {signal_data.shape}"

            elif signal_model_type in [0, 1]:  # RNN/LSTM
                if signal_data.dim() == 4:
                    signal_data = signal_data.squeeze(1).squeeze(1)
                elif signal_data.dim() == 3 and signal_data.size(1) == 1:
                    signal_data = signal_data.squeeze(1)
                signal_data = signal_data.unsqueeze(-1)

                assert signal_data.dim() == 3 and signal_data.size(-1) == 1, \
                    f"validate_epoch: signal_data has incorrect final shape for RNN/LSTM. Got {signal_data.shape}"

            outputs = model(image_data, signal_data)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            tqdm_loader.set_postfix(loss=loss.item(), seq_len=seq_len)

    avg_loss = running_loss / total_batches
    avg_acc = accuracy_score(all_labels, all_preds) if all_labels else 0
    writer.add_scalar('Loss/val', avg_loss, epoch)
    writer.add_scalar('Accuracy/val', avg_acc, epoch)
    return avg_loss, avg_acc


def test_model(model, dataloaders, device, config):
    """
    在测试集上评估模型的最终性能。
    """
    model.eval()
    all_preds, all_labels = [], []
    signal_model_type = config['modality']['signal_model']['type']

    for seq_len, dataloader in dataloaders.items():
        tqdm_loader = tqdm(dataloader, desc=f"[Test - {seq_len}]", leave=False)
        with torch.no_grad():
            for image_data, signal_data, labels in tqdm_loader:
                if image_data is None:
                    continue

                image_data, signal_data, labels = image_data.to(device), signal_data.to(device), labels.to(device)

                # 统一信号数据维度 (与 train_epoch 保持一致)
                if signal_model_type == 2:  # CNN
                    if signal_data.dim() == 4:
                        signal_data = signal_data.squeeze(1).squeeze(1)
                    elif signal_data.dim() == 3 and signal_data.size(1) == 1:
                        signal_data = signal_data.squeeze(1)
                    signal_data = signal_data.unsqueeze(1)

                elif signal_model_type in [0, 1]:  # RNN/LSTM
                    if signal_data.dim() == 4:
                        signal_data = signal_data.squeeze(1).squeeze(1)
                    elif signal_data.dim() == 3 and signal_data.size(1) == 1:
                        signal_data = signal_data.squeeze(1)
                    signal_data = signal_data.unsqueeze(-1)

                outputs = model(image_data, signal_data)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    # 生成详细的分类报告
    report = classification_report(all_labels, all_preds,
                                   target_names=[f'Case {i + 1}' for i in range(config['data']['num_classes'])])

    return acc, f1, precision, recall, cm, report


def plot_confusion_matrix(cm, labels, title='Confusion Matrix'):
    """
    绘制并返回混淆矩阵图。
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt.gcf()


def main():
    """
    主函数，项目的入口。
    """
    config_file = 'config.json'
    with open(config_file, 'r') as f:
        config = json.load(f)

    # --- 关键修正：检查并记录信号归一化统计量 ---
    if 'signal_mean' not in config['data'] or 'signal_std' not in config['data']:
        logging.info("Signal mean/std not found in config. Calculating and recording them now...")
        try:
            # 调用修正后的 dataset.py 中的函数
            signal_mean, signal_std = calculate_signal_stats(config)

            # 更新 config 字典
            config['data']['signal_mean'] = float(signal_mean)
            config['data']['signal_std'] = float(signal_std)

            # 写回 config.json
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

            logging.info("---------------------------------------------------------------------------------")
            logging.info(f"SUCCESS: Calculated and recorded 'signal_mean': {signal_mean:.6f}")
            logging.info(f"SUCCESS: Calculated and recorded 'signal_std': {signal_std:.6f}")
            logging.info("Config file has been updated. Proceeding with DataLoader creation.")
            logging.info("---------------------------------------------------------------------------------")

        except Exception as e:
            logging.critical(f"FATAL ERROR: Failed to calculate signal statistics. Please check data files. Error: {e}")
            return
    else:
        logging.info(
            f"Using existing signal stats: Mean={config['data']['signal_mean']:.6f}, Std={config['data']['signal_std']:.6f}")

    # --- 结束关键修正 ---

    log_dir = config['train']['log_dir']
    model_dir = config['train']['model_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    if config['train']['visualization_tool'] == 'tensorboard':
        writer = SummaryWriter(log_dir)
    elif config['train']['visualization_tool'] == 'wandb':
        raise NotImplementedError("WandB support not yet implemented.")
    else:
        writer = None

    device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    try:
        # create_dataloaders 现在将使用 config 中固化的 mean/std
        train_loaders, val_loaders, test_loaders = create_dataloaders(config)
    except Exception as e:
        logging.error(f"Failed to create dataloaders: {e}")
        return

    model = build_model(config).to(device)

    if config['train']['loss_type'] == 'weighted_loss':
        class_weights = torch.tensor(config['train']['class_weights'], dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logging.info(f"Using Weighted CrossEntropyLoss with weights: {class_weights}")
    elif config['train']['loss_type'] == 'label_smoothing':
        # 检查 smoothing 参数是否存在，如果不存在则使用默认值 0.1
        smoothing_param = config['train'].get('smoothing', 0.1)
        criterion = LabelSmoothingLoss(smoothing=smoothing_param,
                                       num_classes=config['data']['num_classes'])
        logging.info(f"Using LabelSmoothingLoss with smoothing: {smoothing_param}")
    else:
        criterion = nn.CrossEntropyLoss()
        logging.info("Using standard CrossEntropyLoss.")

    optimizer = optim.AdamW(model.parameters(), lr=config['train']['learning_rate'], weight_decay=1e-4)

    scheduler = None
    if config['train']['scheduler']['use_scheduler']:
        scheduler_type = config['train']['scheduler']['type']
        if scheduler_type == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config['train']['scheduler']['factor'],
                patience=config['train']['scheduler']['patience'],
            )
        else:
            logging.warning(f"Scheduler type '{scheduler_type}' not implemented. Training without a scheduler.")

    early_stopping = EarlyStopping(patience=config['train']['early_stopping_patience'], verbose=True,
                                   path=os.path.join(model_dir, config['train']['best_model_name']))  # 使用 config 中的名称

    for epoch in range(config['train']['epochs']):
        train_loss, train_acc = train_epoch(model, train_loaders, criterion, optimizer, device, writer, epoch, config)
        val_loss, val_acc = validate_epoch(model, val_loaders, criterion, device, writer, epoch, config)

        if scheduler:
            scheduler.step(val_loss)

        logging.info(
            f"Epoch {epoch + 1}/{config['train']['epochs']} [Train]: Loss={train_loss:.4f}, Acc={train_acc:.4f} | [Val]: Loss={val_loss:.4f}, Acc={val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered. Training finished.")
            break

    writer.close()

    if os.path.exists(early_stopping.path):
        model.load_state_dict(torch.load(early_stopping.path))
        test_acc, test_f1, test_precision, test_recall, test_cm, test_report = test_model(model, test_loaders, device,
                                                                                          config)

        logging.info(f"Final Test Accuracy: {test_acc:.4f}")
        logging.info(f"Final Test F1-Score (Weighted): {test_f1:.4f}")
        logging.info(f"Final Test Precision (Weighted): {test_precision:.4f}")
        logging.info(f"Final Test Recall (Weighted): {test_recall:.4f}")
        logging.info(f"Classification Report:\n{test_report}")

        cm_fig = plot_confusion_matrix(test_cm, labels=[f'Case {i + 1}' for i in range(config['data']['num_classes'])])
        cm_fig.savefig(
            os.path.join(log_dir, f'confusion_matrix_{config["train"]["best_model_name"].replace(".pth", ".png")}'))
    else:
        logging.warning("Best model checkpoint not found. Skipping final evaluation.")


if __name__ == '__main__':
    main()