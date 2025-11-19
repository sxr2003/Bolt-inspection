import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 导入必要的模块 (确保这些文件存在于您的项目目录中)
from dataset import GeneralizationDataset, generalization_collate_fn 
from model import build_model
from torch.utils.data import DataLoader

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ----------------------------- #
#       绘图函数 (从 train.py 复制)
# ----------------------------- #
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


# ----------------------------- #
#       数据加载与测试函数
# ----------------------------- #

def create_generalization_dataloader(config, seq_len, case_ids_to_load):
    """
    创建泛化数据集和 DataLoader。
    """
    signal_mean = config['data'].get('signal_mean', 0.0)
    signal_std = config['data'].get('signal_std', 1.0)
    
    # 临时配置，用于指定当前的序列长度
    temp_config = config.copy()
    temp_config['data']['seq_len'] = seq_len

    dataset = GeneralizationDataset(
        config=temp_config, 
        signal_mean=signal_mean, 
        signal_std=signal_std,
        generalization_case_ids=case_ids_to_load
    )
    
    # 如果数据集为空，返回 None
    if len(dataset) == 0:
        return None

    dataloader = DataLoader(
        dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=False,
        num_workers=config['data']['num_workers'], 
        collate_fn=generalization_collate_fn, 
        pin_memory=True
    )
    return dataloader

def generalization_test(model, dataloaders, device, config):
    """
    在泛化数据集上评估模型的性能，并按 Case ID 汇总结果。
    """
    model.eval()
    all_preds, all_labels = [], []
    all_case_ids = []
    signal_model_type = config['modality']['signal_model']['type']
    
    num_classes = config['data']['num_classes']
    
    # 遍历所有序列长度的 DataLoader
    for seq_len, dataloader in dataloaders.items():
        if dataloader is None:
            continue
            
        logging.info(f"Starting generalization test for sequence length: {seq_len}")
        tqdm_loader = tqdm(dataloader, desc=f"[Generalization - {seq_len}]", leave=False)
        with torch.no_grad():
            for image_data, signal_data, labels, case_ids in tqdm_loader:
                if image_data is None:
                    continue

                image_data, signal_data, labels = image_data.to(device), signal_data.to(device), labels.to(device)

                # 统一信号数据维度 (与 train.py 保持一致)
                if signal_model_type == 2:  # CNN (B, C, S) -> (B, 1, S)
                    if signal_data.dim() == 4:
                        signal_data = signal_data.squeeze(1).squeeze(1)
                    elif signal_data.dim() == 3 and signal_data.size(1) == 1:
                        signal_data = signal_data.squeeze(1)
                    signal_data = signal_data.unsqueeze(1)
                elif signal_model_type in [0, 1]:  # RNN/LSTM (B, S, C) -> (B, S, 1)
                    if signal_data.dim() == 4:
                        signal_data = signal_data.squeeze(1).squeeze(1)
                    elif signal_data.dim() == 3 and signal_data.size(1) == 1:
                        signal_data = signal_data.squeeze(1)
                    signal_data = signal_data.unsqueeze(-1)

                outputs = model(image_data, signal_data)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_case_ids.extend(case_ids)
                
    if not all_labels:
        logging.warning("No data samples were processed during generalization test. Cannot compute metrics.")
        return 0.0, 0.0, pd.DataFrame(), "No Data"

    # 1. 总体指标
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # 2. 按 Case ID 汇总结果
    results = pd.DataFrame({
        'Case_ID': all_case_ids,
        'True_Label': all_labels,
        'Predicted_Label': all_preds
    })
    
    # 修正: 使用 as_index=False 避免 ValueError
    case_metrics = results.groupby('Case_ID', as_index=False).apply(
        lambda x: pd.Series({
            'Accuracy': accuracy_score(x['True_Label'], x['Predicted_Label']),
            'F1_Score': f1_score(x['True_Label'], x['Predicted_Label'], average='weighted', zero_division=0)
        })
    )

    # 3. 详细分类报告 (修正: 明确指定 labels)
    target_names = [f'Case {i + 1}' for i in range(num_classes)]
    all_possible_labels = list(range(num_classes))
    
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=target_names,
        labels=all_possible_labels, # 明确指定所有 16 个类别标签 (0-15)
        zero_division=0
    )

    return acc, f1, case_metrics, report


# ----------------------------- #
#       主函数
# ----------------------------- #

def main():
    """
    泛化实验主函数。
    """
    config_file = 'config.json'
    with open(config_file, 'r') as f:
        config = json.load(f)

    device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')
    log_dir = config['train']['log_dir']
    model_dir = config['train']['model_dir']
    num_classes = config['data']['num_classes']
    
    # -----------------------------------------------------
    #         泛化实验配置（核心修改点）
    # -----------------------------------------------------
    
    # 修正: 定义用于泛化测试的 Case ID 为所有类别
    GENERALIZATION_CASES = [f"Case{i}" for i in range(1, num_classes + 1)]
    
    # 定义测试使用的序列长度
    SEQ_LENS_TO_TEST = config['data']['seq_lens']

    # 检查最佳模型路径
    best_model_path = os.path.join(model_dir, config['train']['best_model_name'])
    if not os.path.exists(best_model_path):
        logging.critical(f"FATAL ERROR: Best model not found at {best_model_path}. Please run train.py first.")
        return

    # -----------------------------------------------------
    #         模型加载与数据准备
    # -----------------------------------------------------
    
    model = build_model(config).to(device)
    # 忽略 FutureWarning:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    logging.info(f"Loaded best model from {best_model_path} for generalization testing.")

    # 创建泛化 DataLoader 字典
    generalization_loaders = {}
    for seq_len in SEQ_LENS_TO_TEST:
        loader = create_generalization_dataloader(config, seq_len, GENERALIZATION_CASES)
        if loader is not None:
            generalization_loaders[seq_len] = loader

    if not generalization_loaders:
        logging.error("All generalization data loaders are empty. Cannot proceed with testing.")
        return

    # -----------------------------------------------------
    #         运行泛化测试
    # -----------------------------------------------------
    
    overall_acc, overall_f1, case_metrics, report = generalization_test(
        model, generalization_loaders, device, config
    )

    # -----------------------------------------------------
    #         结果输出
    # -----------------------------------------------------
    logging.info("=" * 80)
    logging.info(f"| Generalization Experiment Results (Cases: {', '.join(GENERALIZATION_CASES)}) |")
    logging.info("=" * 80)
    
    if report != "No Data":
        logging.info(f"Overall Accuracy: {overall_acc:.4f}")
        logging.info(f"Overall F1-Score (Weighted): {overall_f1:.4f}")
        
        logging.info("\nMetrics per Case ID:")
        print(case_metrics.to_markdown(index=False)) # 打印表格格式的结果
        
        logging.info("\nDetailed Classification Report:")
        print(report)

        # 保存按 Case ID 汇总的结果到 CSV 文件
        results_path = os.path.join(log_dir, f'generalization_results_all_cases.csv')
        case_metrics.to_csv(results_path, index=False)
        logging.info(f"Per-case metrics saved to {results_path}")
        
        # 绘制并保存泛化测试的混淆矩阵
        
        # 重新收集所有预测和真实标签以绘制 CM
        all_preds = []
        all_labels = []
        for seq_len, dataloader in generalization_loaders.items():
            if dataloader is None: continue
            for image_data, signal_data, labels, _ in dataloader:
                 if image_data is None: continue
                 # 重新执行 train.py 中 signal_data 的维度处理逻辑
                 signal_data = signal_data.to(device)
                 if config['modality']['signal_model']['type'] == 2:
                    if signal_data.dim() == 4: signal_data = signal_data.squeeze(1).squeeze(1)
                    elif signal_data.dim() == 3 and signal_data.size(1) == 1: signal_data = signal_data.squeeze(1)
                    signal_data = signal_data.unsqueeze(1)
                 elif config['modality']['signal_model']['type'] in [0, 1]:
                     if signal_data.dim() == 4: signal_data = signal_data.squeeze(1).squeeze(1)
                     elif signal_data.dim() == 3 and signal_data.size(1) == 1: signal_data = signal_data.squeeze(1)
                     signal_data = signal_data.unsqueeze(-1)
                 
                 # 预测
                 outputs = model(image_data.to(device), signal_data)
                 _, predicted = torch.max(outputs.data, 1)
                 all_preds.extend(predicted.cpu().numpy())
                 all_labels.extend(labels.cpu().numpy())
             
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
        cm_fig = plot_confusion_matrix(cm, labels=[f'Case {i + 1}' for i in range(num_classes)], title=f'Generalization CM (All {num_classes} Cases)')
        
        save_filename = 'generalization_confusion_matrix_all_cases.png'
        save_path = os.path.join(log_dir, save_filename)
        cm_fig.savefig(save_path)
        
        logging.info(f"Confusion matrix saved to {save_path}")
    else:
        logging.error("Generalization test failed due to no data being processed.")


if __name__ == '__main__':
    main()