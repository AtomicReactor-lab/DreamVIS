import os
import numpy as np
from glob import glob

def load_and_merge_data(data_dir="/home/shiyao/EEG/eegprocessed/labeled"):
    """从labeled文件夹加载并合并所有数据"""
    print(f"Loading data from {data_dir}")
    
    # 获取所有.npy文件
    eeg_files = sorted(glob(os.path.join(data_dir, "*_eeg.npy")))
    label_files = sorted(glob(os.path.join(data_dir, "*_label.npy")))
    
    print(f"Found {len(eeg_files)} EEG files and {len(label_files)} label files")
    
    # 初始化列表存储数据
    all_eeg_data = []
    all_labels = []
    
    # 逐个加载并合并数据
    for eeg_file, label_file in zip(eeg_files, label_files):
        try:
            print(f"Processing {os.path.basename(eeg_file)}")
            eeg_data = np.load(eeg_file)
            label_data = np.load(label_file)
            
            all_eeg_data.append(eeg_data)
            all_labels.append(label_data)
            
            print(f"EEG shape: {eeg_data.shape}, Label shape: {label_data.shape}")
            
        except Exception as e:
            print(f"Error loading {eeg_file}: {str(e)}")
            continue
    
    # 合并所有数据
    eeg_data = np.concatenate(all_eeg_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    print("\nFinal data shapes:")
    print(f"EEG data: {eeg_data.shape}")
    print(f"Labels: {labels.shape}")
    
    # 保存合并后的数据
    output_path = "/home/shiyao/EEG/eegprocessed/all_labeled_data_merged.npz"
    print(f"\nSaving merged data to {output_path}")
    np.savez(output_path, 
             eeg_data=eeg_data.astype(np.float32),  # 确保数据类型
             labels=labels.astype(np.int64))
    
    # 验证保存的数据
    print("\nVerifying saved data...")
    loaded = np.load(output_path)
    print(f"Loaded data shapes:")
    print(f"EEG data: {loaded['eeg_data'].shape}")
    print(f"Labels: {loaded['labels'].shape}")
    
    return output_path

if __name__ == "__main__":
    output_path = load_and_merge_data()
    print(f"\nData processing completed. New data file: {output_path}") 