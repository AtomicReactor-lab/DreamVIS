import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def segment_by_trial_phase(raw):
    """
    将数据按照试次和阶段进行分段
    每个试次包含三个阶段：table, query, vis
    """
    print("\n开始分段处理...")
    
    # 初始化trials列表
    trials = []
    
    # 获取所有事件
    events, event_id = mne.events_from_annotations(raw)
    print(f"找到事件ID: {event_id}")
    
    # 重新映射事件ID，忽略 'New Segment'
    event_map = {
        'Stimulus/S  1': 1,
        'Stimulus/S  2': 2,
        'Stimulus/S  3': 3
    }
    
    # 创建新的事件数组
    new_events = []
    for event in events:
        event_code = event[2]
        event_name = list(event_id.keys())[list(event_id.values()).index(event_code)]
        if event_name in event_map:
            new_event = event.copy()
            new_event[2] = event_map[event_name]
            new_events.append(new_event)
    
    new_events = np.array(new_events)
    
    # 获取各类型事件
    s1_events = new_events[new_events[:, 2] == 1]  # S1事件
    s2_events = new_events[new_events[:, 2] == 2]  # S2事件
    s3_events = new_events[new_events[:, 2] == 3]  # S3事件
    
    print(f"找到事件数量: S1={len(s1_events)}, S2={len(s2_events)}, S3={len(s3_events)}")
    
    # 确保三种事件数量相同
    n_trials = min(len(s1_events), len(s2_events), len(s3_events))
    
    if n_trials == 0:
        print("警告: 未找到完整的试次")
        return trials
    
    # 处理第一个试次（包含开始到第一个S1的table阶段）
    first_trial = {
        'table': raw.get_data(start=0, stop=s1_events[0][0]),
        'query': raw.get_data(start=s1_events[0][0], stop=s2_events[0][0]),
        'vis': raw.get_data(start=s2_events[0][0], stop=s3_events[0][0]),
        'times': {
            'table': (0, s1_events[0][0]),
            'query': (s1_events[0][0], s2_events[0][0]),
            'vis': (s2_events[0][0], s3_events[0][0])
        }
    }
    trials.append(first_trial)
    
    # 处理后续试次
    for i in range(1, n_trials):
        s1_time = s1_events[i][0]
        s2_time = s2_events[i][0]
        s3_time = s3_events[i][0]
        
        # table阶段从上一个试次的s3到当前试次的s1
        table_start = s3_events[i-1][0]
        
        trial = {
            'table': raw.get_data(start=table_start, stop=s1_time),
            'query': raw.get_data(start=s1_time, stop=s2_time),
            'vis': raw.get_data(start=s2_time, stop=s3_time),
            'times': {
                'table': (table_start, s1_time),
                'query': (s1_time, s2_time),
                'vis': (s2_time, s3_time)
            }
        }
        trials.append(trial)
    
    print(f"成功分段: 共{len(trials)}个试次")
    return trials

def normalize_trial_lengths(trials, target_seconds=15, sampling_rate=200):
    """
    将所有试次的各个阶段标准化到指定长度
    """
    target_length = int(target_seconds * sampling_rate)
    normalized_trials = []
    
    print(f"\n开始标准化试次长度...")
    print(f"目标长度: {target_seconds}秒 ({target_length}个采样点)")
    
    for i, trial in enumerate(trials):
        normalized_trial = {}
        for phase in ['table', 'query', 'vis']:
            current_length = trial[phase].shape[1]
            if current_length == 0:
                print(f"警告: 试次{i+1}的{phase}阶段长度为0，跳过此试次")
                continue
                
            # 使用scipy的resample函数进行重采样
            normalized_trial[phase] = mne.filter.resample(
                trial[phase],
                up=target_length,
                down=current_length,
                npad='auto',
                n_jobs=4
            )
        
        # 只有当所有阶段都成功处理时才添加到结果中
        if len(normalized_trial) == 3:  # 确保所有三个阶段都存在
            normalized_trial['times'] = {
                'table': (0, target_length),
                'query': (0, target_length),
                'vis': (0, target_length)
            }
            normalized_trial['original_times'] = trial['times']
            normalized_trials.append(normalized_trial)
    
    print(f"标准化完成: {len(normalized_trials)}个试次")
    return normalized_trials

def save_trials(trials, output_path):
    """
    保存处理后的试次数据
    """
    # 创建输出目录
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 将trials数据转换为可保存的格式
    data_dict = {
        'trials': trials,
        'info': {
            'n_trials': len(trials),
            'phases': ['table', 'query', 'vis'],
            'sampling_rate': 200
        }
    }
    
    # 保存为.npz格式
    np.savez(output_path + '_trials.npz', **data_dict)
    print(f"\n数据已保存至: {output_path}_trials.npz")
    
    # 打印保存的数据结构
    print("\n保存的数据结构:")
    print("data_dict/")
    print("  ├─trials/")
    print(f"  │  └─{len(trials)}个试次")
    print("  └─info/")
    print(f"     ├─n_trials: {len(trials)}")
    print("     ├─phases: ['table', 'query', 'vis']")
    print("     └─sampling_rate: 200")

def print_data_structure(trials):
    """
    打印数据结构树
    """
    print("\n数据结构树:")
    print("trials/")
    for i, trial in enumerate(trials):
        print(f"  ├─trial_{i+1}/")
        for phase in ['table', 'query', 'vis']:
            shape = trial[phase].shape
            print(f"  │  ├─{phase}: shape={shape}, dtype={trial[phase].dtype}")
        print("  │  ├─times/")
        for phase in ['table', 'query', 'vis']:
            start, stop = trial['times'][phase]
            print(f"  │  │  ├─{phase}: ({start}, {stop})")
        if 'original_times' in trial:
            print("  │  └─original_times/")
            for phase in ['table', 'query', 'vis']:
                start, stop = trial['original_times'][phase]
                print(f"  │     ├─{phase}: ({start}, {stop})")
    print("\n")

def read_and_segment_eeg(file_path):
    """
    读取并分段EEG数据
    """
    # 读取EEGLAB格式的数据
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    print(f"\n读取数据: {Path(file_path).name}")
    print(f"采样率: {raw.info['sfreq']} Hz")
    print(f"通道数: {len(raw.ch_names)}")
    print(f"数据时长: {raw.times[-1]:.1f} 秒")
    
    # 按试次分段
    trials = segment_by_trial_phase(raw)
    
    # 打印原始分段数据结构
    print("\n原始分段数据结构:")
    print_data_structure(trials)
    
    # 标准化长度
    normalized_trials = normalize_trial_lengths(trials)
    
    # 打印标准化后的数据结构
    print("\n标准化后的数据结构:")
    print_data_structure(normalized_trials)
    
    return normalized_trials

def batch_process(input_dir, output_dir):
    """
    批量处理指定目录下的所有.set文件
    """
    # 获取所有.set文件
    set_files = list(Path(input_dir).glob('*_preprocessed.set'))
    total_files = len(set_files)
    
    print(f"\n找到 {total_files} 个.set文件")
    
    # 处理每个文件
    for i, set_file in enumerate(set_files, 1):
        print(f"\n处理进度: {i}/{total_files}")
        print(f"当前文件: {set_file.name}")
        
        try:
            # 处理单个文件
            trials = read_and_segment_eeg(str(set_file))
            
            # 构建输出路径
            output_path = Path(output_dir) / set_file.stem
            
            # 保存处理后的数据
            save_trials(trials, str(output_path))
            
            print(f"成功处理: {set_file.name}")
            
        except KeyboardInterrupt:
            print("\n用户中断处理。已完成文件数: ", i-1)
            break
        except Exception as e:
            print(f"处理文件 {set_file.name} 时出错: {str(e)}")
            continue
    
    print(f"\n批量处理完成。成功处理 {i} 个文件。")

if __name__ == "__main__":
    # 设置输入和输出目录
    input_dir = "/home/shiyao/EEG/eegprocessed/241009"
    output_dir = "/home/shiyao/EEG/eegprocessed/segmented"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 批量处理
    batch_process(input_dir, output_dir)
