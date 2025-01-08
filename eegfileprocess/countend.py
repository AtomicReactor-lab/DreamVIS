from pathlib import Path
import re
from datetime import datetime

def get_last_marker(vmrk_path):
    """读取.vmrk文件中的最后一个marker编号"""
    with open(vmrk_path, 'r') as f:
        lines = f.readlines()
        for line in reversed(lines):
            if 'Mk' in line and 'Stimulus,S' in line:
                match = re.search(r'Mk(\d+)=', line)
                if match:
                    return int(match.group(1))
    return None

def get_vis_starts():
    """从mapping.txt读取每个test文件的VIS Start信息"""
    mapping = {}
    with open("/home/shiyao/EEG/eegsignal/241009/mapping.txt.txt", 'r') as f:
        lines = f.readlines()
        
    current_file = None
    for line in lines:
        if line.startswith("Filename:"):
            current_file = line.split(":")[1].strip()
        elif line.startswith("VIS Start:"):
            vis_start = line.split(":")[1].strip()
            if '-' in vis_start:
                vis_start = vis_start.split('-')[0].strip()
            try:
                mapping[current_file] = int(vis_start)
            except ValueError:
                continue
    return mapping

def get_last_marker_info(vmrk_path):
    """读取.vmrk文件中的最后一个marker的编号和S值"""
    with open(vmrk_path, 'r') as f:
        lines = f.readlines()
        for line in reversed(lines):
            if 'Mk' in line and 'Stimulus,S' in line:
                mk_match = re.search(r'Mk(\d+)=', line)
                s_match = re.search(r'Stimulus,S\s+(\d+)', line)
                if mk_match and s_match:
                    return int(mk_match.group(1)), int(s_match.group(1))
    return None, None

def calculate_ends():
    """计算每个test文件的end值"""
    # 创建输出文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"/home/shiyao/EEG/eegprocessed/vis_ranges_{timestamp}.txt")
    
    vmrk_dir = Path("/home/shiyao/EEG/eegsignal/241009")
    vis_starts = get_vis_starts()
    test_files = sorted(list(vis_starts.keys()))
    
    with open(output_file, 'w') as f:
        f.write("每个test文件的VIS范围：\n")
        f.write("-" * 50 + "\n")
        
        for i, current_file in enumerate(test_files):
            s = vis_starts[current_file]
            
            vmrk_path = vmrk_dir / f"{current_file}.vmrk"
            if not vmrk_path.exists():
                continue
                
            mk, s_value = get_last_marker_info(vmrk_path)
            if mk is None:
                continue
            
            # 检查最后一个marker是否为S  3（只输出到终端）
            if s_value != 3:
                print(f"警告: {current_file} 的最后一个marker不是S  3 (实际值: S  {s_value})")
            
            # 根据mk计算end值，减1得到实际的最后VIS编号
            end = s + (mk - 1) // 3 - 1
            
            # 检查与下一个文件的连续性（只输出到终端）
            if i < len(test_files) - 1:
                next_file = test_files[i + 1]
                next_start = vis_starts[next_file]
                if end + 1 != next_start:
                    print(f"警告: {current_file} 与 {next_file} 不连续 ({end + 1} != {next_start})")
            
            # 文件中只输出基本信息
            f.write(f"文件: {current_file}\n")
            f.write(f"  Start: {s}\n")
            f.write(f"  End: {end}\n")
            f.write(f"  Last Marker: {mk}\n")
            f.write(f"  包含VIS数量: {end - s + 1}\n")
            f.write("-" * 50 + "\n")
    
    print(f"计算完成，结果已保存至: {output_file}")

if __name__ == "__main__":
    calculate_ends()
