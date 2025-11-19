import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import random
import logging
from tqdm import tqdm
from scipy.signal import stft
import librosa
import warnings

# 阻止 librosa 相关的 UserWarning
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

# 配置日志记录，方便追踪错误
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ----------------------------- #
#       Utility Functions
# ----------------------------- #

def pad_signal(signal, target_length):
    """
    填充或截断信号到目标长度。
    """
    if len(signal) >= target_length:
        return signal[:target_length]
    padding = target_length - len(signal)
    return np.pad(signal, (0, padding), 'constant')


def compute_stft(signal, nperseg, noverlap):
    """
    计算信号的短时傅立叶变换(STFT)。
    """
    # 确保 nperseg 不大于信号长度，避免 STFT 崩溃
    if nperseg > len(signal):
        nperseg = len(signal)
    # 避免 nperseg 为 0
    if nperseg == 0:
        return np.array([[]])
        
    f, t, Zxx = stft(signal, nperseg=nperseg, noverlap=noverlap)
    return Zxx


def normalize_img(x):
    """
    将numpy数组归一化到[0, 1]区间。
    """
    x = x - np.min(x)
    max_val = np.max(x)
    if max_val > 1e-8:
        x = x / max_val
    return x.astype(np.float32)


def low_rank_approx(mat, rank=5):
    """
    使用SVD对矩阵进行低秩近似，并修复广播问题。
    """
    try:
        # 避免奇异情况
        if mat.size == 0 or min(mat.shape) == 0:
            return np.zeros_like(mat)

        # 确保 rank 不超过矩阵的有效秩
        K = min(mat.shape)
        if K == 0:
             return np.zeros_like(mat)
        rank = min(rank, K)
        
        # 1. 执行 SVD
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        
        # 2. 截断奇异值
        S[rank:] = 0
        
        # 3. 构造对角矩阵 Sigma
        return (U * S[None, :]) @ Vh 
        
    except np.linalg.LinAlgError as e:
        logging.warning(f"SVD failed for low rank approximation: {e}. Returning zero matrix.")
        return np.zeros_like(mat)
    except Exception as e:
        # 捕获可能的广播错误
        return np.zeros_like(mat)

def augment_signal(signal_segment, sr, config, case_label=None):
    """
    【核心增强】对信号片段应用时域和频域增强，根据 case_label 对 Case 11/13 启用激进增强。

    参数:
        signal_segment (np.ndarray): 信号片段。
        sr (int): 采样率。
        config (dict): 项目配置字典。
        case_label (int): 样本的类别标签 (0-15)。Case 11 对应标签 10，Case 13 对应标签 12。
    
    返回:
        np.ndarray: 增强后的信号片段。
    """
    
    # ---------------------------------------------------- #
    # >>>>> 关键修改：定义针对 Case 11 (Label 10) 和 Case 13 (Label 12) 的激进参数 <<<<<
    # Case 11 的 Label 是 10, Case 13 的 Label 是 12
    IS_CRITICAL_CASE = (case_label == 10) or (case_label == 12) 

    # 从 config 中获取默认参数
    default_params = config.get('augment', {}).get('signal', {}).get('params', {})
    signal_len = len(signal_segment)
    
    # 激进增强参数 (基于默认参数，但更广、更激进)
    # 基于配置文件中的默认值进行了放大：
    CRITICAL_PARAMS = {
        'time_stretch_range': [0.95, 1.05],    # 默认 [0.9, 1.1]
        'freq_shift_range': 1,               # 默认 3
        'noise_amp': 0.001,                  # 默认 0.005 (放大 3 倍)
        'amplitude_scale_range': [0.95, 1.05], # 默认 [0.9, 1.1]
        'dc_offset_amp': 0.01,                # 默认 0.1 (放大 2 倍)
        'random_cutout_p': 0.2,              # 默认 0.6 (提高概率)
        'cutout_len_range': [0.001, 0.01],     # 默认 [0.01, 0.1] (放大范围)
        'p_augment': 0.2                    # 整体增强操作的发生概率
    }
    
    # 辅助函数，用于获取参数：如果为关键工况则取激进参数，否则取默认参数
    def get_param(key, default_key=None):
        if IS_CRITICAL_CASE and key in CRITICAL_PARAMS:
            return CRITICAL_PARAMS[key]
        return default_params.get(default_key if default_key else key, CRITICAL_PARAMS.get(key))


    # ---------------------------------------------------- #
    # 整体增强操作的发生概率控制 (Case 11/13 几乎必发生增强)
    # 非关键 Case 仍有小概率跳过增强 (基于默认的 max(1-p) 进行估计，此处简化为 0.95)
    if not IS_CRITICAL_CASE and random.random() > 0.98: 
        return signal_segment
    
    # 1. 随机幅度缩放
    amp_range = get_param('amplitude_scale_range')
    if random.random() < default_params.get('amplitude_scale_p', 0.8):
        scale_factor = random.uniform(*amp_range)
        signal_segment *= scale_factor

    # 2. 随机直流偏移 (DC Offset)
    offset_amp = get_param('dc_offset_amp')
    if random.random() < default_params.get('dc_offset_p', 0.8):
        offset = random.uniform(-offset_amp, offset_amp)
        signal_segment += offset
        
    # 3. 加性噪声
    noise_amp = get_param('noise_amp')
    if random.random() < default_params.get('noise_p', 0.5):
        noise = np.random.randn(len(signal_segment)) * noise_amp
        signal_segment += noise

    # 4. 随机切除 (Random Cutout)
    cutout_p = get_param('random_cutout_p')
    cutout_len_range = get_param('cutout_len_range')
    if random.random() < cutout_p:
        cutout_len_ratio = random.uniform(*cutout_len_range) 
        cutout_len = int(signal_len * cutout_len_ratio)
        if cutout_len > 0 and signal_len > cutout_len:
            start_idx = random.randint(0, signal_len - cutout_len)
            signal_segment[start_idx:start_idx + cutout_len] = 0 # 设为零

    # 5. 时域拉伸 (Time Stretch)
    stretch_range = get_param('time_stretch_range')
    if random.random() < default_params.get('time_stretch_p', 0.5):
        stretch_factor = random.uniform(*stretch_range)
        stretched_signal = librosa.effects.time_stretch(
            signal_segment.astype(np.float32),
            rate=stretch_factor,
        )
        # 确保信号长度不变
        signal_segment = librosa.util.fix_length(stretched_signal, size=signal_len)

    # 6. 频率偏移 (Pitch Shift)
    shift_range = get_param('freq_shift_range')
    if random.random() < default_params.get('pitch_shift_p', 0.5):
        n_steps = random.uniform(-shift_range, shift_range)
        signal_segment = librosa.effects.pitch_shift(
            signal_segment,
            sr=sr,
            n_steps=n_steps,
        )

    return signal_segment

def augment_image(image, config):
    """
    对伪图像应用数据增强。
    """
    params = config.get('augment', {}).get('image', {})
    h, w, c = image.shape

    # 1. 随机色彩抖动（亮度&对比度）
    if random.random() < params.get('color_jitter_p', 0.5):
        brightness = random.uniform(1 - params.get('brightness', 0.2), 1 + params.get('brightness', 0.2))
        contrast = random.uniform(1 - params.get('contrast', 0.2), 1 + params.get('contrast', 0.2))
        image = np.clip(image * contrast + brightness, 0, 1)
        
    # 2. 随机水平翻转
    if random.random() < params.get('h_flip_p', 0.5):
        image = np.flip(image, axis=1).copy()

    # 3. 随机旋转
    if random.random() < params.get('rotate_p', 0.3):
        angle = random.uniform(*params.get('rotate_range', [-10, 10]))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        
        # 对所有通道应用相同的变换
        transformed_image = np.zeros_like(image)
        for i in range(c):
            transformed_image[:, :, i] = cv2.warpAffine(image[:, :, i], M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        image = transformed_image


    # 4. 随机擦除 (Random Erasing)
    if random.random() < params.get('random_erasing_p', 0.5):
        area_range = params.get('erasing_area_range', [0.02, 0.1])
        ratio_range = params.get('erasing_ratio_range', [0.3, 3.33])

        area = random.uniform(*area_range) * h * w
        ratio = random.uniform(*ratio_range)

        erasing_h = int(np.sqrt(area / ratio))
        erasing_w = int(np.sqrt(area * ratio))

        if erasing_h < h and erasing_w < w:
            x1 = random.randint(0, h - erasing_h)
            y1 = random.randint(0, w - erasing_w)
            image[x1:x1 + erasing_h, y1:y1 + erasing_w, :] = 0 # 擦除区域置为零

    return image


def generate_pseudo_image_from_signal(signal, config):
    """
    根据信号生成多通道伪图像。
    """
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    image_mode = config['modality'].get('pseudo_image_mode', 1)
    image_size = tuple(config['data'].get('image_size', [224, 224]))
    hop_length = config['data'].get('hop_length', 128)
    sr = config['data'].get('sr', 16000)

    try:
        # 动态计算 nperseg，确保不大于信号长度
        nperseg = int(2 ** np.ceil(np.log2(hop_length * 4)))
        if nperseg > len(signal):
            nperseg = len(signal)
        noverlap = hop_length
        
        if nperseg <= 0 or nperseg > len(signal):
            raise ValueError("Invalid nperseg size.")

        f, t, Zxx = stft(signal, fs=sr, nperseg=nperseg, noverlap=noverlap)
        S_db = 20 * np.log10(np.abs(Zxx) + 1e-6)
    except Exception as e:
        logging.error(f"STFT computation failed for signal of length {len(signal)}: {e}. Returning zero image.")
        num_channels = 5 if image_mode == 1 else 1
        return np.zeros(image_size + (num_channels,), dtype=np.float32)

    if Zxx.shape[0] == 0 or Zxx.shape[1] == 0:
        num_channels = 5 if image_mode == 1 else 1
        return np.zeros(image_size + (num_channels,), dtype=np.float32)

    channels_to_stack = []
    # 通道1：对数幅度谱
    ch1 = normalize_img(S_db)
    channels_to_stack.append(ch1)

    if image_mode == 1:
        phase = np.angle(Zxx)
        # 通道2：相位梯度（时间方向）
        ch2 = np.gradient(phase, axis=1)
        channels_to_stack.append(normalize_img(ch2))
        # 通道3：相位梯度（频率方向）
        ch3 = -np.gradient(phase, axis=0)
        channels_to_stack.append(normalize_img(ch3))
        # 通道4：幅度谱的低秩近似
        ch4 = low_rank_approx(np.abs(Zxx))
        channels_to_stack.append(normalize_img(ch4))
        # 通道5：功率谱
        ch5 = np.abs(Zxx) ** 2
        channels_to_stack.append(normalize_img(ch5))

    # 将所有通道调整到指定尺寸
    # 注意：cv2.resize 接受 (width, height)
    img_list = [cv2.resize(ch, image_size[::-1], interpolation=cv2.INTER_LINEAR) for ch in channels_to_stack]
    # 将通道堆叠起来，形成最终的伪图像
    img_np = np.stack(img_list, axis=-1)

    return img_np.astype(np.float32)


def calculate_signal_stats(config):
    """
    计算所有信号的均值和标准差用于归一化，并一次性加载所有信号。
    """
    data_dir = config['data']['data_dir']
    case_ids = config['data']['case_ids']
    all_signals_flat = []
    loaded_signals_info = []

    for cid in tqdm(case_ids, desc="Loading all signals and calculating stats"):
        # 假设文件路径结构为: data_dir/CaseX/CaseX_800.csv
        filepath = os.path.join(data_dir, cid, f"{cid}_800.csv")
        try:
            # 假设信号文件是单列数据
            signal = np.loadtxt(filepath, delimiter=',')
            # 确保 signal 是 1D 数组
            if signal.ndim > 1:
                signal = signal.flatten()
                
            all_signals_flat.append(signal)
            # 假设 CaseID 决定标签 (Case1 -> 0, Case2 -> 1, ...)
            label = int(cid.replace("Case", "")) - 1
            loaded_signals_info.append({
                'case_id': cid,
                'signal': signal, # 信号数据被存储在内存中
                'label': label
            })
        except FileNotFoundError:
            logging.warning(f"Signal file not found at {filepath}. Skipping.")
        except Exception as e:
            logging.error(f"Error loading signal file {filepath}: {e}. Skipping.")


    if not all_signals_flat:
        raise ValueError("No signal data found to calculate stats. Check your data_dir and case_ids.")

    all_signals_flat = np.concatenate(all_signals_flat, axis=0)
    return np.mean(all_signals_flat), np.std(all_signals_flat), loaded_signals_info


class BoltLooseningDataset(Dataset):
    """
    螺栓松动检测数据集类。
    数据从内存中加载，并按时间比例划分 train/val/test。
    """

    def __init__(self, config, split='train', signal_mean=0.0, signal_std=1.0, loaded_signals_info=None):
        self.config = config
        self.split = split
        self.signal_mean = signal_mean
        self.signal_std = signal_std
        
        if loaded_signals_info is None:
            raise ValueError("loaded_signals_info must be provided for one-time loading.")
        self.loaded_signals_info = loaded_signals_info
        
        # 根据配置获取序列长度
        if 'seq_len' in config['data']:
            self.seq_len = config['data']['seq_len']
        else:
            # 使用 seq_lens 列表中的第一个值作为默认值
            self.seq_len = config['data']['seq_lens'][0]
            
        # 计算滑动窗口步长
        self.step_ratio = config['data'].get('step_ratio', 0.5)
        self.step = int(self.seq_len * self.step_ratio)
        self.augment_flag = split == 'train'

        # 准备所有样本的索引
        self.samples = self._prepare_samples()

        if not self.samples:
            raise RuntimeError(f"No valid data samples found for split '{split}' with seq_len {self.seq_len}.")

        logging.info(f"Initialized {split} dataset with {len(self.samples)} samples for seq_len {self.seq_len}.")

    def _prepare_samples(self):
        """
        从加载的原始数据中准备样本索引。
        """
        samples = []
        split_ratio = self.config['data'].get('split_ratio', [0.7, 0.15, 0.15])

        for sig_idx_in_list, info in enumerate(self.loaded_signals_info):
            signal_full = info['signal']
            label = info['label']

            if len(signal_full) < self.seq_len or self.step <= 0:
                continue

            # 计算总共的样本分段数
            num_segments = (len(signal_full) - self.seq_len) // self.step + 1
            
            # 使用起始索引
            start_indices = np.arange(num_segments) * self.step

            # 根据split_ratio划分训练、验证和测试集 (按时间序列的段进行划分)
            train_end = int(num_segments * split_ratio[0])
            val_end = train_end + int(num_segments * split_ratio[1])

            if self.split == 'train':
                split_start_indices = start_indices[:train_end]
            elif self.split == 'val':
                split_start_indices = start_indices[train_end:val_end]
            else:
                split_start_indices = start_indices[val_end:]

            # 为每个分段创建一个样本条目
            for seg_start in split_start_indices:
                samples.append({
                    'sig_idx_in_list': sig_idx_in_list,
                    'seg_start': seg_start,
                    'label': label,
                })
        return samples

    def __len__(self):
        """返回数据集中的样本总数。"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取指定索引的样本。从内存中的信号数组中切片。
        """
        sample = self.samples[idx]

        try:
            # 从内存中获取原始信号切片
            signal_full = self.loaded_signals_info[sample['sig_idx_in_list']]['signal']
            seg_start = sample['seg_start']
            label = sample['label'] # 获取标签
            
            # 使用 .copy() 确保后续增强操作不会影响原始数据
            sig_segment = signal_full[seg_start: seg_start + self.seq_len].copy() 
            sig_segment = pad_signal(sig_segment, self.seq_len)

            # --- 信号增强 (仅训练集) ---
            if self.augment_flag and self.config.get('augment', {}).get('signal', {}).get('use_augment', False):
                # >>>>> 关键修改：调用 augment_signal 时传入 label <<<<<
                sig_segment = augment_signal(sig_segment, self.config['data'].get('sr', 16000), self.config, case_label=label)

            # 信号归一化
            sig_segment_norm = (sig_segment - self.signal_mean) / (self.signal_std + 1e-8)

            # 调整信号张量格式为 (channels, sequence_length) -> (1, seq_len)
            sig_tensor = torch.tensor(sig_segment_norm, dtype=torch.float32).unsqueeze(0)

            # --- 图像生成与增强 (仅训练集) ---
            # 使用归一化后的信号生成图像
            img_np = generate_pseudo_image_from_signal(sig_segment_norm, self.config)

            if self.augment_flag and self.config.get('augment', {}).get('image', {}).get('use_augment', False):
                img_np = augment_image(img_np, self.config)

            # 将图像转换为 (C, H, W) 格式的张量
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()

            return img_tensor, sig_tensor, label

        except Exception as e:
            # 捕获异常
            sig_id = self.loaded_signals_info[sample['sig_idx_in_list']]['case_id']
            logging.error(f"Failed to process sample {idx} (sig_id: {sig_id}, seg_start: {sample['seg_start']}): {e}. Returning None.")
            return None, None, None


def pad_collate_fn(batch):
    """
    用于过滤无效样本并合并有效样本到批次中的`collate_fn`。
    """
    # 过滤掉 `__getitem__` 中返回的 None 样本
    batch = [item for item in batch if item is not None and item[0] is not None]
    if not batch:
        return None, None, None

    img_data_list = [item[0] for item in batch]
    sig_data_list = [item[1] for item in batch]
    labels_list = [item[2] for item in batch]

    # 将样本列表堆叠成批次张量
    img_data_batch = torch.stack(img_data_list, dim=0)
    sig_data_batch = torch.stack(sig_data_list, dim=0)
    labels_batch = torch.tensor(labels_list, dtype=torch.long)

    return img_data_batch, sig_data_batch, labels_batch


def create_dataloaders(config):
    """
    为每个指定的信号长度创建数据加载器。
    """
    if not os.path.exists(config['data']['data_dir']):
        raise FileNotFoundError(f"Data directory not found at {config['data']['data_dir']}.")

    seq_lens = config['data'].get('seq_lens')
    if not seq_lens or not isinstance(seq_lens, list):
        raise ValueError("Config must have 'data.seq_lens' as a list of integers.")

    logging.info("Loading all signals and calculating statistics...")
    try:
        # 核心改变：一次性加载所有数据并计算统计量
        signal_mean, signal_std, loaded_signals_info = calculate_signal_stats(config)
    except (ValueError, FileNotFoundError) as e:
        logging.error(f"Failed to load all signals or calculate signal statistics: {e}")
        return {}, {}, {}

    if not loaded_signals_info:
        raise RuntimeError("No signals were successfully loaded.")

    train_loaders = {}
    val_loaders = {}
    test_loaders = {}

    for seq_len in seq_lens:
        temp_config = config.copy()
        temp_config['data'] = config['data'].copy()
        temp_config['data']['seq_len'] = seq_len
        temp_config['data']['split_ratio'] = config['data'].get('split_ratio', [0.7, 0.15, 0.15])

        try:
            # 创建数据集实例，传入一次性加载的数据
            train_dataset = BoltLooseningDataset(temp_config, 'train', signal_mean, signal_std, loaded_signals_info)
            val_dataset = BoltLooseningDataset(temp_config, 'val', signal_mean, signal_std, loaded_signals_info)
            test_dataset = BoltLooseningDataset(temp_config, 'test', signal_mean, signal_std, loaded_signals_info)

            # 创建 DataLoader 实例
            train_loaders[seq_len] = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True,
                                                num_workers=config['data']['num_workers'], collate_fn=pad_collate_fn,
                                                pin_memory=True)
            val_loaders[seq_len] = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False,
                                                num_workers=config['data']['num_workers'], collate_fn=pad_collate_fn,
                                                pin_memory=True)
            test_loaders[seq_len] = DataLoader(test_dataset, batch_size=config['data']['batch_size'], shuffle=False,
                                                num_workers=config['data']['num_workers'], collate_fn=pad_collate_fn,
                                                pin_memory=True)
        except RuntimeError as e:
            logging.error(f"Failed to create dataloader for seq_len {seq_len}: {e}. Skipping this scale.")
            continue
        except Exception as e:
            logging.error(
                f"An unexpected error occurred while creating dataloader for seq_len {seq_len}: {e}. Skipping this scale.")
            continue

    if not train_loaders:
        raise RuntimeError("No training dataloaders were created. Please check your data and config.")

    logging.info("DataLoaders created successfully for all specified scales.")
    return train_loaders, val_loaders, test_loaders
# ----------------------------- #
#       Generalization Dataset
# ----------------------------- #

class GeneralizationDataset(Dataset):
    """
    螺栓松动检测泛化数据集类。
    专门用于加载 config 中 generalization_dir 路径下的数据，进行 Case 泛化测试。
    """

    def __init__(self, config, signal_mean=0.0, signal_std=1.0, generalization_case_ids=None):
        self.config = config
        self.signal_mean = signal_mean
        self.signal_std = signal_std
        self.case_ids_to_load = generalization_case_ids if generalization_case_ids is not None else config['data']['case_ids']
        
        # 泛化数据集不需要数据增强
        self.augment_flag = False 
        
        if 'seq_len' in config['data']:
            self.seq_len = config['data']['seq_len']
        else:
            self.seq_len = config['data']['seq_lens'][0]
        
        self.step_ratio = config['data'].get('step_ratio', 0.5)
        self.step = int(self.seq_len * self.step_ratio)

        # 核心改变：加载泛化目录下的信号数据
        self.loaded_signals_info = self._load_generalization_signals()
        
        # 准备所有样本的索引
        self.samples = self._prepare_samples()

        if not self.samples:
            logging.warning(f"No valid data samples found for generalization cases {self.case_ids_to_load} with seq_len {self.seq_len}.")

        logging.info(f"Initialized Generalization dataset with {len(self.samples)} samples from cases {self.case_ids_to_load}.")


    def _load_generalization_signals(self):
        """
        加载指定 Case ID 的信号数据，从 generalization_dir 路径。
        """
        data_dir = self.config['data']['generalization_dir'] # 使用泛化目录
        loaded_signals_info = []

        logging.info(f"Loading generalization signals from {data_dir} for cases: {self.case_ids_to_load}...")
        
        for cid in tqdm(self.case_ids_to_load, desc="Loading generalization signals"):
            # 假设文件路径结构为: generalization_dir/CaseX/CaseX_800.csv
            filepath = os.path.join(data_dir, cid, f"{cid}_800.csv")
            
            try:
                # 假设信号文件格式与训练数据相同
                signal = np.loadtxt(filepath, delimiter=',')
                if signal.ndim > 1:
                    signal = signal.flatten()
                    
                label = int(cid.replace("Case", "")) - 1
                
                loaded_signals_info.append({
                    'case_id': cid,
                    'signal': signal, 
                    'label': label
                })
            except FileNotFoundError:
                logging.warning(f"Generalization signal file not found at {filepath}. Skipping.")
            except Exception as e:
                logging.error(f"Error loading generalization file {filepath}: {e}. Skipping.")
        
        return loaded_signals_info


    def _prepare_samples(self):
        """
        从加载的原始数据中准备样本索引（不划分 train/val/test）。
        """
        samples = []
        for sig_idx_in_list, info in enumerate(self.loaded_signals_info):
            signal_full = info['signal']
            label = info['label']

            if len(signal_full) < self.seq_len or self.step <= 0:
                continue

            num_segments = (len(signal_full) - self.seq_len) // self.step + 1
            start_indices = np.arange(num_segments) * self.step
            
            # 泛化集使用所有分段
            for seg_start in start_indices:
                samples.append({
                    'sig_idx_in_list': sig_idx_in_list,
                    'seg_start': seg_start,
                    'label': label,
                    'case_id': info['case_id'] # 添加 case_id 便于按工况分析
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        从内存中的信号数组中切片并生成图像/信号张量。
        """
        sample = self.samples[idx]
        
        try:
            # 数据获取逻辑与 BoltLooseningDataset 相同
            signal_full = self.loaded_signals_info[sample['sig_idx_in_list']]['signal']
            seg_start = sample['seg_start']
            
            sig_segment = signal_full[seg_start: seg_start + self.seq_len].copy() 
            sig_segment = pad_signal(sig_segment, self.seq_len)

            # 泛化测试无需增强

            # 信号归一化 (使用全局统计量)
            sig_segment_norm = (sig_segment - self.signal_mean) / (self.signal_std + 1e-8)

            # 调整信号张量格式为 (1, seq_len)
            sig_tensor = torch.tensor(sig_segment_norm, dtype=torch.float32).unsqueeze(0)

            # 图像生成
            img_np = generate_pseudo_image_from_signal(sig_segment_norm, self.config)

            # 泛化测试无需图像增强

            # 将图像转换为 (C, H, W) 格式的张量
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()

            return img_tensor, sig_tensor, sample['label'], sample['case_id'] # 返回 case_id
            
        except Exception as e:
            logging.error(f"Failed to process generalization sample {idx} (case_id: {sample['case_id']}, seg_start: {sample['seg_start']}): {e}. Returning None.")
            return None, None, None, None


def generalization_collate_fn(batch):
    """
    用于 GeneralizationDataset 的 collate_fn。
    """
    batch = [item for item in batch if item is not None and item[0] is not None]
    if not batch:
        return None, None, None, None

    img_data_list = [item[0] for item in batch]
    sig_data_list = [item[1] for item in batch]
    labels_list = [item[2] for item in batch]
    case_ids_list = [item[3] for item in batch]

    img_data_batch = torch.stack(img_data_list, dim=0)
    sig_data_batch = torch.stack(sig_data_list, dim=0)
    labels_batch = torch.tensor(labels_list, dtype=torch.long)

    return img_data_batch, sig_data_batch, labels_batch, case_ids_list