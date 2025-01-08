from pathlib import Path
import re
import sys
from datetime import datetime

def get_last_marker(vmrk_path):
    """读取.vmrk文件中的最后一个marker编号"""
    with open(vmrk_path, 'r') as f:
        lines = f.readlines()
        for line in reversed(lines):
            if 'Mk' in line and 'Stimulus,S' in line:
                # 格式类似: Mk49=Stimulus,S  3,572563,1,0
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

def check_markers():
    """检查marker数量是否符合公式 3*(s1-s)+1=mk"""
    # 创建输出文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"/home/shiyao/EEG/eegprocessed/241009/marker_check_{timestamp}.txt")
    
    vmrk_dir = Path("/home/shiyao/EEG/eegsignal/241009")
    vis_starts = get_vis_starts()
    test_files = sorted(list(vis_starts.keys()))
    
    with open(output_file, 'w') as f:
        f.write("检查不一致的文件：\n")
        f.write("-" * 50 + "\n")
        
        for i in range(len(test_files) - 1):
            current_file = test_files[i]
            next_file = test_files[i + 1]
            
            s = vis_starts[current_file]
            s1 = vis_starts[next_file]
            
            vmrk_path = vmrk_dir / f"{current_file}.vmrk"
            if not vmrk_path.exists():
                continue
                
            mk = get_last_marker(vmrk_path)
            if mk is None:
                continue
            
            # 根据mk反推s1'
            s1_calc = s + (mk - 1) // 3
            
            expected_mk = 3 * (s1 - s) + 1
            if mk != expected_mk:
                f.write(f"文件: {current_file}\n")
                f.write(f"  VIS Start (s): {s}\n")
                f.write(f"  Next VIS Start (s1): {s1}\n")
                f.write(f"  根据mk计算的VIS Start (s1'): {s1_calc}\n")
                f.write(f"  Last Marker (mk): {mk}\n")
                f.write(f"  Expected Marker: {expected_mk}\n")
                f.write(f"  差异: {mk - expected_mk}\n")
                f.write("-" * 50 + "\n")
    
    print(f"检查完成，结果已保存至: {output_file}")

if __name__ == "__main__":
    check_markers()
