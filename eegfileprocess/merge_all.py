import numpy as np
import os
from pathlib import Path

def load_npz_safely(file_path):
    """安全加载npz文件"""
    try:
        # 使用allow_pickle=True来加载数据
        data = np.load(file_path, allow_pickle=True)
        # 将数据转换为标准numpy数组
        eeg_data = data['eeg_data'].astype(np.float32)
        labels = data['labels'].astype(np.int64)
        info = data.get('info', None)
        if info is not None:
            info = np.array(info, dtype=object)
        return eeg_data, labels, info
    except Exception as e:
        print(f"处理文件 {os.path.basename(file_path)} 时出错: {str(e)}")
        return None, None, None

def merge_data_files(data_dir):
    """合并所有预处理后的数据文件"""
    print("开始合并数据文件...")
    
    # 获取所有预处理后的数据文件
    data_files = sorted(Path(data_dir).glob("test*_preprocessed_labeled.npz"))
    
    all_eeg_data = []
    all_labels = []
    all_info = []
    total_trials = 0
    
    # 处理每个文件
    for file_path in data_files:
        eeg_data, labels, info = load_npz_safely(file_path)
        if eeg_data is not None:
            all_eeg_data.append(eeg_data)
            all_labels.append(labels)
            if info is not None:
                all_info.append(info)
            total_trials += len(eeg_data)
            print(f"成功处理文件 {file_path.name}, 包含 {len(eeg_data)} 个试次")
    
    # 合并数据
    if all_eeg_data:
        merged_eeg = np.concatenate(all_eeg_data, axis=0)
        merged_labels = np.concatenate(all_labels, axis=0)
        merged_info = np.concatenate(all_info, axis=0) if all_info else None
        
        # 保存合并后的数据
        output_path = os.path.join(data_dir, 'all_labeled_data.npz')
        np.savez(output_path,
                 eeg_data=merged_eeg,
                 labels=merged_labels,
                 info=merged_info)
        
        print("\n合并完成！")
        print(f"总试次数: {total_trials}")
        print(f"数据已保存至: {output_path}")
        print(f"EEG数据形状: {merged_eeg.shape}")
        print(f"标签形状: {merged_labels.shape}")
    else:
        print("\n警告：没有成功处理任何数据文件！")

if __name__ == "__main__":
    data_dir = "/home/shiyao/EEG/eegprocessed"
    merge_data_files(data_dir) 