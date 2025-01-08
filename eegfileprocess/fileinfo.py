import mne
import numpy as np
import matplotlib.pyplot as plt

def print_data_structure(raw):
    """
    打印MNE Raw对象的数据结构树状图
    参数:
        raw: MNE Raw对象
    """
    print("\n📊 MNE Raw对象结构:")
    print("├── info")
    print("│   ├── sfreq:", raw.info['sfreq'], "Hz")
    print("│   ├── nchan:", raw.info['nchan'])
    print("│   ├── ch_names:", raw.ch_names[:3], "...")
    print("│   └── meas_date:", raw.info['meas_date'])
    
    data = raw.get_data()
    print("├── data (numpy array)")
    print("│   ├── shape:", data.shape)
    print("│   ├── dtype:", data.dtype)
    print("│   └── memory size:", f"{data.nbytes / (1024*1024):.2f} MB")
    
    print("├── times")
    print("│   ├── length:", len(raw.times))
    print("│   ├── duration:", f"{raw.times[-1]:.2f} seconds")
    print("│   └── time points:", raw.times[:3], "...")
    
    events, event_id = mne.events_from_annotations(raw)
    print("└── annotations")
    print("    ├── event count:", len(events))
    print("    └── event types:", event_id)

def read_brainvision_eeg(eeg_file_path):
    """
    读取BrainVision格式的EEG文件
    参数:
        eeg_file_path: .eeg文件的路径（或对应的.vhdr文件路径）
    返回:
        raw: MNE Raw对象，包含EEG数据
    """
    # 读取数据
    raw = mne.io.read_raw_brainvision(eeg_file_path, preload=True)
    
    # 打印数据结构树状图
    print_data_structure(raw)

    # 打印基本信息
    print("\nEEG文件基本信息:")
    print(f"采样率: {raw.info['sfreq']} Hz")
    print(f"通道数: {len(raw.ch_names)}")
    print(f"通道名称: {raw.ch_names}")
    print(f"数据时长: {raw.times[-1]:.2f} 秒")
    
    # 获取数据数组并展示更多信息
    data = raw.get_data()
    print(f"\n数据形状: {data.shape}")  # (通道数, 采样点数)
    print("\n数据结构详情:")
    print(f"数据类型: {data.dtype}")
    print(f"每个通道的前5个采样点: ")
    print(data[:, :5])  # 显示所有通道的前5个采样点
    
    # 显示统计信息
    print("\n数据统计信息:")
    print(f"最大值: {np.max(data):.2f}")
    print(f"最小值: {np.min(data):.2f}")
    print(f"平均值: {np.mean(data):.2f}")
    print(f"标准差: {np.std(data):.2f}")
    
    # 显示事件信息
    events, event_ids = mne.events_from_annotations(raw)
    print("\n事件信息:")
    print(f"事件总数: {len(events)}")
    print("\n事件ID对应关系:")
    print(event_ids)
    print("\n前5个事件:")
    print("格式: [采样点位置, 0, 事件类型]")
    if len(events) > 0:
        print(events[:5])
    else:
        print("没有找到事件标记")
    
    return raw

def plot_eeg_data(raw, duration=10, n_channels=5):
    """
    绘制EEG数据的前几个通道
    参数:
        raw: MNE Raw对象
        duration: 要显示的时间长度（秒）
        n_channels: 要显示的通道数
    """
    # 绘制前n个通道的数据
    raw.plot(duration=duration, n_channels=n_channels)
    plt.show()

if __name__ == "__main__":
    # 修改为使用.vhdr文件
    eeg_file = "/home/shiyao/EEG/eegsignal/241009/test001.vhdr"  # 使用.vhdr文件
    
    try:
        # 读取数据
        raw = read_brainvision_eeg(eeg_file)
        
        # 绘制数据
        plot_eeg_data(raw)
        
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
