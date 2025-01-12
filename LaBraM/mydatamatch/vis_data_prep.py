import numpy as np
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm

class VisQueryDataset(Dataset):
    def __init__(self, data_dir, phase='train'):
        self.data_dir = data_dir
        self.files = []
        self.query_to_class = {}
        self.trials = []
        
        # 获取所有文件
        files = [f for f in os.listdir(data_dir) if f.endswith('_preprocessed_labeled.npz')]
        
        print(f"\nLoading {phase} dataset:")
        for file in tqdm(files, desc=f"Loading {phase} data"):
            data = np.load(os.path.join(data_dir, file), allow_pickle=True)
            trials = data['trials']
            for trial in trials:
                eeg = np.concatenate([
                    trial['eeg_data']['table'],
                    trial['eeg_data']['query']
                ], axis=1)
                vis_query = trial['labels']['vis_query']
                class_id = self._convert_query_to_class(vis_query)
                self.trials.append({
                    'eeg': eeg,
                    'class_id': class_id
                })
    
    def __len__(self):
        """返回数据集中样本的总数"""
        return len(self.trials)
    
    def _convert_query_to_class(self, query):
        if query not in self.query_to_class:
            self.query_to_class[query] = len(self.query_to_class)
        return self.query_to_class[query]
    
    def __getitem__(self, idx):
        trial = self.trials[idx]
        eeg = trial['eeg']  # 应该是[N, T]形状
        # 确保输出是[N, T]形状，让rearrange在训练时处理窗口划分
        return torch.FloatTensor(eeg), trial['class_id']
