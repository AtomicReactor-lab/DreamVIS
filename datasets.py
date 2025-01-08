import os
import torch
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
from data_processor.vis_query_dataset import VisQueryDataset

def build_vis_query_dataset(is_train, args):
    """构建视觉查询数据集"""
    data_dir = os.path.dirname(args.data_path)  # 使用目录而不是单个文件
    
    # 获取所有样本的总数
    dataset = VisQueryDataset(data_dir, indices=None, window_size=args.window_size)
    total_samples = len(dataset)
    
    # 划分训练集和验证集
    if is_train:
        indices = range(0, int(total_samples * 0.8))
    else:
        indices = range(int(total_samples * 0.8), total_samples)
    
    # 创建数据集
    dataset = VisQueryDataset(
        data_dir,
        indices=indices,
        window_size=args.window_size
    )
    
    return dataset

class DataAugmentationForVisQuery(object):
    """视觉查询任务的数据增强
    
    这里可以添加特定的数据增强方法，比如:
    1. EEG信号的随机噪声
    2. 时间窗口的随机偏移
    3. 通道mask等
    """
    def __init__(self, args=None):
        self.args = args
    
    def __call__(self, x):
        # 这里可以实现具体的数据增强逻辑
        return x

def build_transform(is_train, args):
    """构建数据转换
    
    Args:
        is_train: 是否为训练模式
        args: 参数配置
    
    Returns:
        transform: 数据转换函数
    """
    if is_train:
        # 训练时使用数据增强
        transform = DataAugmentationForVisQuery(args)
    else:
        # 验证时不使用数据增强
        transform = None
    
    return transform 