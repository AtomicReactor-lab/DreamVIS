import os
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class VisQueryDataset(Dataset):
    def __init__(self, data_dir, indices=None, window_size=200):
        """
        Args:
            data_dir: 包含所有预处理后的npz文件的目录
            indices: 用于训练/验证集划分的索引
            window_size: EEG信号窗口大小
        """
        self.data_dir = data_dir
        self.window_size = window_size
        
        # 获取所有预处理后的数据文件
        labeled_dir = os.path.join(data_dir, 'labeled')
        self.data_files = sorted(Path(labeled_dir).glob("*_preprocessed_labeled.npz"))
        print(f"Found {len(self.data_files)} data files")
        
        # 计算总样本数并构建文件索引映射
        self.file_index_map = []  # (file_idx, trial_idx)
        self.total_samples = 0
        
        for file_idx, file_path in enumerate(self.data_files):
            try:
                with np.load(file_path, allow_pickle=True) as data:
                    trials = data['trials']
                    for trial_idx, trial in enumerate(trials):
                        if isinstance(trial, dict) and 'eeg_data' in trial and 'labels' in trial:
                            if 'vis_query' in trial['labels']:  # 确保有vis_query标签
                                self.file_index_map.append((file_idx, trial_idx))
                                self.total_samples += 1
                    
                    print(f"Added {self.total_samples - len(self.file_index_map) + 1} samples from {file_path.name}")
            except Exception as e:
                print(f"Warning: Error processing {file_path.name}: {e}")
                continue
        
        print(f"\nTotal samples found: {self.total_samples}")
        if self.total_samples == 0:
            raise ValueError("No valid samples found in the dataset!")
        
        # 使用指定的索引进行划分
        if indices is not None:
            self.file_index_map = [self.file_index_map[i] for i in indices]
            self.total_samples = len(indices)
        
        print(f"Dataset initialized with {self.total_samples} samples")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        file_idx, trial_idx = self.file_index_map[idx]
        file_path = self.data_files[file_idx]
        
        try:
            with np.load(file_path, allow_pickle=True) as data:
                trial = data['trials'][trial_idx]
                
                # 获取EEG数据（合并table、query和vis）
                eeg_data = np.concatenate([
                    trial['eeg_data']['table'],
                    trial['eeg_data']['query'],
                    trial['eeg_data']['vis']
                ], axis=1)  # 沿时间维度拼接
                
                # 获取vis_query作为标签
                vis_query = trial['labels']['vis_query']
                
                # 转换为tensor
                eeg = torch.FloatTensor(eeg_data)
                label = torch.LongTensor([1])  # 临时使用1作为标签，后续需要根据vis_query生成真实标签
                
                return {
                    'eeg': eeg,
                    'label': label,
                    'vis_query': vis_query  # 保存原始vis_query以供后续处理
                }
        except Exception as e:
            print(f"Error loading sample {idx} from {file_path}: {e}")
            return self.__getitem__((idx + 1) % self.total_samples) 