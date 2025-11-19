import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101, ResNet50_Weights, ResNet101_Weights
from collections import OrderedDict
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ----------------------------- #
#       Utility Components
# ----------------------------- #
class PositionalEncoding(nn.Module):
    """
    为Transformer模型提供位置信息编码。

    参数:
        d_model (int): 输入特征的维度。
        dropout (float): Dropout率。
        max_len (int): 序列的最大长度。

    作用:
        - 引入序列中token的顺序信息。
        - 使用正弦和余弦函数生成位置编码，并加到输入嵌入中。
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 创建一个形状为 (max_len, d_model) 的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        # 计算位置（position）和div_term，用于正弦/余弦函数的参数
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        # 将正弦和余弦函数应用到位置编码矩阵中
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # 将位置编码注册为模型的缓冲区，它不会被视为模型参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播，将位置编码添加到输入张量中。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)。

        返回:
            torch.Tensor: 带有位置编码的输出张量。
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ----------------------------- #
#       Image Encoders
# ----------------------------- #
class CustomCNNImageEncoder(nn.Module):
    """
    一个简单的自定义卷积神经网络（CNN）图像编码器。

    参数:
        in_channels (int): 输入图像的通道数。
        out_dim (int): 输出特征向量的维度。

    作用:
        - 使用一系列卷积层和池化层提取图像特征。
        - 最后通过一个全连接层将特征展平并映射到指定维度。
    """

    def __init__(self, in_channels, out_dim):
        super(CustomCNNImageEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 使用自适应平均池化将特征图大小变为 1x1
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        """
        前向传播。

        参数:
            x (torch.Tensor): 输入图像张量，形状为 (batch_size, in_channels, H, W)。

        返回:
            torch.Tensor: 图像特征向量，形状为 (batch_size, out_dim)。
        """
        x = self.conv_layers(x).squeeze()
        if x.dim() == 1:  # 处理 batch_size 为 1 的情况
            x = x.unsqueeze(0)
        x = self.fc(x)
        return x


def get_image_encoder(config):
    """
    根据配置创建并返回图像编码器。

    参数:
        config (dict): 项目配置字典，包含模型类型、维度等信息。

    返回:
        torch.nn.Module: 实例化后的图像编码器。
    """
    model_cfg = config['modality']['image_model']
    encoder_type = model_cfg['type']
    out_dim = model_cfg['out_dim']
    # 根据伪图像模式确定输入通道数
    in_channels = 1 if config['modality']['pseudo_image_mode'] == 0 else 5
    pretrained = model_cfg['pretrained']

    if in_channels != model_cfg['in_channels']:
        logging.warning(
            f"Config mismatch: `image_model.in_channels` is {model_cfg['in_channels']}, but `pseudo_image_mode` suggests {in_channels}. Overriding `in_channels` for model creation.")

    if encoder_type == 0:  # CustomCNN
        return CustomCNNImageEncoder(in_channels, out_dim)
    elif encoder_type == 1:  # ResNet50
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        # 修改 ResNet 的第一层卷积，以适应多通道输入
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 修改 ResNet 的全连接层，以匹配所需的输出维度
        model.fc = nn.Linear(model.fc.in_features, out_dim)
        return model
    elif encoder_type == 2:  # ResNet101
        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet101(weights=weights)
        # 修改 ResNet 的第一层卷积
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 修改 ResNet 的全连接层
        model.fc = nn.Linear(model.fc.in_features, out_dim)
        return model
    else:
        raise ValueError(f"Invalid image encoder type: {encoder_type}. Expected 0, 1, or 2.")


# ----------------------------- #
#       Signal Encoders
# ----------------------------- #
class SignalRNNEncoder(nn.Module):
    """
    使用RNN作为时序信号编码器。

    参数:
        in_channels (int): 输入特征维度（通常为 1）。
        out_dim (int): 输出特征向量的维度。

    作用:
        - 接收时序信号，通过RNN网络处理。
        - 返回RNN最后一个隐藏状态作为信号的特征表示。
    """

    def __init__(self, in_channels, out_dim):
        super(SignalRNNEncoder, self).__init__()
        self.hidden_size = out_dim
        # batch_first=True 使得输入形状为 (batch, seq_len, features)
        self.rnn = nn.RNN(in_channels, self.hidden_size, batch_first=True)

        assert in_channels == 1, "SignalRNNEncoder expects input_size to be 1."

    def forward(self, x):
        """
        前向传播。

        参数:
            x (torch.Tensor): 输入信号张量，形状为 (batch_size, seq_len, 1)。

        返回:
            torch.Tensor: 信号特征向量，形状为 (batch_size, out_dim)。
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() == 4:
            x = x.squeeze(1).squeeze(-1)
            x = x.unsqueeze(-1)
        assert x.dim() == 3, f"SignalRNNEncoder input must be 3D, but got {x.dim()}D tensor."
        assert x.size(-1) == 1, f"SignalRNNEncoder input's last dimension must be 1, but got {x.size(-1)}."
        # RNN返回所有时间步的输出和最后一个隐藏状态
        _, hn = self.rnn(x)
        return hn.squeeze(0)  # 提取并返回最后一个隐藏状态作为特征


class SignalLSTMEncoder(nn.Module):
    """
    使用LSTM作为时序信号编码器。

    参数:
        in_channels (int): 输入特征维度（通常为 1）。
        out_dim (int): 输出特征向量的维度。

    作用:
        - 接收时序信号，通过LSTM网络处理。
        - 返回LSTM最后一个隐藏状态作为信号的特征表示。
    """

    def __init__(self, in_channels, out_dim):
        super(SignalLSTMEncoder, self).__init__()
        self.hidden_size = out_dim
        self.lstm = nn.LSTM(in_channels, self.hidden_size, batch_first=True)

    def forward(self, x):
        """
        前向传播。

        参数:
            x (torch.Tensor): 输入信号张量，形状为 (batch_size, seq_len, 1)。

        返回:
            torch.Tensor: 信号特征向量，形状为 (batch_size, out_dim)。
        """
        # LSTM返回所有时间步的输出，最后一个隐藏状态和最后一个细胞状态
        _, (hn, cn) = self.lstm(x)
        return hn.squeeze(0)  # 提取并返回最后一个隐藏状态作为特征


class SignalHybridEncoder(nn.Module):
    """
    一个混合信号编码器，结合了CNN、LSTM和Transformer。

    参数:
        in_channels (int): 输入特征维度（通常为 1）。
        out_dim (int): 输出特征向量的维度。
        embed_dim (int): LSTM和Transformer的嵌入维度。
        nhead (int): Transformer的多头注意力头数。
        num_layers (int): Transformer编码器层的数量。
        dropout (float): Dropout率。

    作用:
        - CNN (`Conv1d`) 提取信号的局部时域特征。
        - LSTM 捕捉局部特征中的时序依赖。
        - Transformer编码器利用自注意力机制捕捉全局特征和长距离依赖。
        - 最终通过一个全连接层输出固定维度的特征向量。
    """

    def __init__(self, in_channels, out_dim, embed_dim, nhead, num_layers, dropout):
        super(SignalHybridEncoder, self).__init__()
        # 1D 卷积层用于提取信号的局部特征
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # LSTM用于捕捉卷积后序列的时序信息
        self.lstm = nn.LSTM(64, embed_dim, batch_first=True, bidirectional=True)
        # 位置编码器
        self.pos_encoder = PositionalEncoding(embed_dim * 2, dropout)
        # Transformer编码器层，利用自注意力机制
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim * 2,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=num_layers
        )
        self.fc = nn.Linear(embed_dim * 2, out_dim)

    def forward(self, x):
        """
        前向传播。

        参数:
            x (torch.Tensor): 输入信号张量，形状为 (batch_size, 1, seq_len)。

        返回:
            torch.Tensor: 信号特征向量，形状为 (batch_size, out_dim)。
        """
        # CNN处理，输入形状 (B, C, S)，例如 (16, 1, 512)
        x = self.cnn(x)  # 输出形状 (B, 64, S/2)
        # 调整维度以匹配LSTM输入 (B, S, C)
        x = x.permute(0, 2, 1)  # 输出形状 (16, 256, 64)
        # LSTM处理
        x, _ = self.lstm(x)  # 输出形状 (B, S/2, embed_dim*2)
        # Transformer处理
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # 对序列维度进行平均池化，得到固定长度的特征向量
        x = x.mean(dim=1)
        # 全连接层映射到最终输出维度
        x = self.fc(x)
        return x


def get_signal_encoder(config):
    """
    根据配置创建并返回信号编码器。

    参数:
        config (dict): 项目配置字典。

    返回:
        torch.nn.Module: 实例化后的信号编码器。
    """
    model_cfg = config['modality']['signal_model']
    encoder_type = model_cfg['type']
    in_channels = model_cfg['in_channels']
    out_dim = config['modality']['image_model']['out_dim']

    if encoder_type == 0:
        return SignalRNNEncoder(in_channels, out_dim)
    elif encoder_type == 1:
        return SignalLSTMEncoder(in_channels, out_dim)
    elif encoder_type == 2:
        embed_dim = model_cfg.get('embed_dim', 256)
        nhead = model_cfg.get('nhead', 8)
        num_layers = model_cfg.get('num_layers', 2)
        dropout = model_cfg.get('dropout', 0.1)
        return SignalHybridEncoder(in_channels, out_dim, embed_dim, nhead, num_layers, dropout)
    else:
        raise ValueError(f"Invalid signal encoder type: {encoder_type}. Expected 0, 1, or 2.")


# ----------------------------- #
#       Fusion Modules
# ----------------------------- #
class SimpleConcatenation(nn.Module):
    """
    简单的特征拼接融合模块。

    参数:
        input_dim (int): 单个模态的输入特征维度。

    作用:
        - 将两个特征向量简单拼接起来，然后通过一个全连接层降维。
    """

    def __init__(self, input_dim):
        super(SimpleConcatenation, self).__init__()
        self.fc = nn.Linear(input_dim * 2, input_dim)

    def forward(self, image_features, signal_features):
        """
        前向传播。

        参数:
            image_features (torch.Tensor): 图像特征向量，形状为 (batch_size, input_dim)。
            signal_features (torch.Tensor): 信号特征向量，形状为 (batch_size, input_dim)。

        返回:
            torch.Tensor: 融合后的特征向量，形状为 (batch_size, input_dim)。
        """
        # 沿维度1拼接两个特征
        combined = torch.cat([image_features, signal_features], dim=1)
        return self.fc(combined)


class AttentionFusion(nn.Module):
    """
    基于加权注意力的特征融合模块。

    参数:
        input_dim (int): 输入特征的维度。

    作用:
        - 学习一个权重，表示每个模态对最终预测的贡献度。
        - 使用Softmax对权重进行归一化，然后进行加权求和。
    """

    def __init__(self, input_dim):
        super(AttentionFusion, self).__init__()
        self.W_i = nn.Linear(input_dim, input_dim)
        self.W_s = nn.Linear(input_dim, input_dim)
        self.v = nn.Linear(input_dim, 1)

    def forward(self, image_features, signal_features):
        """
        前向传播。

        参数:
            image_features (torch.Tensor): 图像特征向量，形状为 (batch_size, input_dim)。
            signal_features (torch.Tensor): 信号特征向量，形状为 (batch_size, input_dim)。

        返回:
            torch.Tensor: 融合后的特征向量，形状为 (batch_size, input_dim)。
        """
        # 计算注意力权重
        alpha_i = self.v(torch.tanh(self.W_i(image_features)))
        alpha_s = self.v(torch.tanh(self.W_s(signal_features)))
        # 对权重进行归一化
        weights = torch.softmax(torch.cat([alpha_i, alpha_s], dim=1), dim=1)
        # 加权求和得到融合特征
        fused_features = weights[:, 0].unsqueeze(1) * image_features + weights[:, 1].unsqueeze(1) * signal_features
        return fused_features


class MultiHeadAttentionFusion(nn.Module):
    """
    基于多头自注意力的特征融合模块。

    参数:
        input_dim (int): 输入特征的维度。
        num_heads (int): 多头注意力的头数。
        dropout (float): Dropout率。

    作用:
        - 允许模型在不同的表示子空间中共同关注来自不同模态的信息。
        - 相比简单注意力，能够更复杂、更精细地捕捉模态间的关系。
    """

    def __init__(self, input_dim, num_heads=4, dropout=0.1):
        super(MultiHeadAttentionFusion, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(input_dim * 2, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_features, signal_features):
        """
        前向传播。

        参数:
            image_features (torch.Tensor): 图像特征向量，形状为 (batch_size, input_dim)。
            signal_features (torch.Tensor): 信号特征向量，形状为 (batch_size, input_dim)。

        返回:
            torch.Tensor: 融合后的特征向量，形状为 (batch_size, input_dim)。
        """
        # 将图像特征作为query
        query = image_features.unsqueeze(1)
        # 将图像和信号特征堆叠作为key和value
        key = torch.stack([image_features, signal_features], dim=1)
        value = key
        # 执行多头注意力计算
        attn_output, _ = self.multihead_attn(
            query=query, key=key, value=value
        )
        # 添加残差连接，进行层归一化和Dropout
        fused_features = attn_output.squeeze(1) + image_features
        fused_features = self.layer_norm(fused_features)
        fused_features = self.dropout(fused_features)

        return fused_features


def get_fusion_module(config):
    """
    根据配置创建并返回特征融合模块。

    参数:
        config (dict): 项目配置字典。

    返回:
        torch.nn.Module: 实例化后的融合模块。
    """
    fusion_cfg = config['modality']['fusion']
    fusion_type = fusion_cfg['type']
    input_dim = config['modality']['image_model']['out_dim']

    if fusion_type == 0:
        return SimpleConcatenation(input_dim)
    elif fusion_type == 1:
        return AttentionFusion(input_dim)
    elif fusion_type == 2:
        num_heads = fusion_cfg.get('num_heads', 4)
        dropout = fusion_cfg.get('dropout', 0.1)
        return MultiHeadAttentionFusion(input_dim, num_heads, dropout)
    else:
        raise ValueError(f"Invalid fusion type: {fusion_type}. Expected 0, 1, or 2.")


# ----------------------------- #
#       Main Model
# ----------------------------- #
class MultiModalNet(nn.Module):
    """
    多模态深度学习主网络。

    参数:
        config (dict): 项目配置字典。

    作用:
        - 封装图像编码器、信号编码器、融合模块和分类器。
        - 实现了端到端的多模态特征提取、融合和分类流程。
    """

    def __init__(self, config):
        super(MultiModalNet, self).__init__()
        # 根据配置构建各个模块
        self.image_encoder = get_image_encoder(config)
        self.signal_encoder = get_signal_encoder(config)
        self.fusion = get_fusion_module(config)

        fusion_out_dim = config['modality']['image_model']['out_dim']
        self.classifier = nn.Linear(fusion_out_dim, config['data']['num_classes'])

    def forward(self, image_data, signal_data):
        """
        前向传播。

        参数:
            image_data (torch.Tensor): 图像输入数据。
            signal_data (torch.Tensor): 信号输入数据。

        返回:
            torch.Tensor: 模型的最终预测输出。
        """
        # 通过各自的编码器提取特征
        image_features = self.image_encoder(image_data)
        signal_features = self.signal_encoder(signal_data)
        # 确保特征维度一致
        if image_features.shape[1] != signal_features.shape[1]:
            raise ValueError(
                f"Dimension mismatch between image and signal features. Got {image_features.shape[1]} and {signal_features.shape[1]}.")
        # 通过融合模块合并特征
        fused_features = self.fusion(image_features, signal_features)
        # 通过分类器得到最终预测结果
        output = self.classifier(fused_features)

        return output


def build_model(config):
    """
    一个辅助函数，用于根据配置实例化并返回 MultiModalNet。

    参数:
        config (dict): 项目配置字典。

    返回:
        MultiModalNet: 实例化后的主模型。
    """
    return MultiModalNet(config)