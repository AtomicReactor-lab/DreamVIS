import mne
import numpy as np
from pathlib import Path
import os

def preprocess_eeg(file_path, output_dir):
    """
    EEG数据预处理函数
    参数:
        file_path: 输入的.vhdr文件路径
        output_dir: 输出目录
    """
    print(f"\n开始处理文件: {Path(file_path).name}")
    
    # 读取数据
    raw = mne.io.read_raw_brainvision(file_path, preload=True)
    print(f"\n原始数据信息:")
    print(f"采样率: {raw.info['sfreq']} Hz")
    print(f"通道数: {len(raw.ch_names)}")
    print(f"数据时长: {raw.times[-1]:.1f} 秒")
    
    # 1. 重采样到 200 Hz
    print("\n重采样到 200 Hz...")
    raw.resample(200, npad='auto', n_jobs=4)
    
    # 2. 带通滤波 (0.1-75 Hz)
    print("\n应用带通滤波 (0.1-75 Hz)...")
    raw.filter(l_freq=0.1, h_freq=75.0, 
              method='fir', 
              phase='zero-double',
              fir_window='hamming',
              n_jobs=4)
    
    # 3. 陷波滤波 (50 Hz)
    print("\n应用陷波滤波 (50 Hz)...")
    raw.notch_filter(freqs=50, 
                    method='fir', 
                    phase='zero-double',
                    fir_window='hamming',
                    n_jobs=4)
    
    # 4. 单位转换为 μV
    print("\n转换单位为 μV...")
    if raw.info['chs'][0]['unit'] == 107:  # 1e-7 V = 0.1 μV
        raw.apply_function(lambda x: x * 1e6)
        ch_types = {ch: 'eeg' for ch in raw.ch_names}
        raw.set_channel_types(ch_types)
        for ch in raw.info['chs']:
            ch['unit'] = 106  # 106 represents μV
    
    # 保存为EEGLAB格式
    file_name = Path(file_path).stem
    output_path = os.path.join(output_dir, file_name + '_preprocessed.set')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换为EEGLAB格式并保存
    raw.export(output_path, fmt='eeglab', overwrite=True)
    print(f"\n数据已保存至: {output_path}")
    
    return raw

def batch_process(input_dir, output_dir):
    """
    批量处理指定目录下的所有.vhdr文件
    """
    # 获取所有.vhdr文件
    vhdr_files = list(Path(input_dir).glob('*.vhdr'))
    total_files = len(vhdr_files)
    
    print(f"\n找到 {total_files} 个.vhdr文件")
    
    # 处理每个文件
    for i, vhdr_file in enumerate(vhdr_files, 1):
        print(f"\n处理进度: {i}/{total_files}")
        try:
            preprocess_eeg(str(vhdr_file), output_dir)
            print(f"成功处理: {vhdr_file.name}")
        except KeyboardInterrupt:
            print("\n用户中断处理。已完成文件数: ", i-1)
            break
        except Exception as e:
            print(f"处理文件 {vhdr_file.name} 时出错: {str(e)}")
            continue

if __name__ == "__main__":
    # 设置输入和输出目录
    input_dir = "/home/shiyao/EEG/eegsignal/241009"
    output_dir = "/home/shiyao/EEG/eegprocessed/241009"
    
    # 批量处理
    batch_process(input_dir, output_dir)