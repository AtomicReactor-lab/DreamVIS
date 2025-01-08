import numpy as np
from pathlib import Path
import sys

def load_vis_ranges(ranges_file):
    """从vis_ranges文件加载每个test文件的VIS范围"""
    ranges = {}
    current_file = None
    
    with open(ranges_file, 'r') as f:
        for line in f:
            if line.startswith("文件: "):
                current_file = line.split(": ")[1].strip()
            elif line.strip().startswith("Start: "):
                start = int(line.split(": ")[1].strip())
                ranges[current_file] = {'start': start}
            elif line.strip().startswith("End: "):
                end = int(line.split(": ")[1].strip())
                ranges[current_file]['end'] = end
            elif line.strip().startswith("Last Marker: "):
                mk = int(line.split(": ")[1].strip())
                ranges[current_file]['last_marker'] = mk
            elif line.strip().startswith("包含VIS数量: "):
                count = int(line.split(": ")[1].strip())
                ranges[current_file]['count'] = count
    return ranges

def print_tree(obj, indent=0, max_array=3):
    """递归打印对象的结构树"""
    prefix = "│   " * indent
    
    if isinstance(obj, dict):
        for i, (key, value) in enumerate(obj.items()):
            is_last = i == len(obj) - 1
            branch = "└── " if is_last else "├── "
            
            if isinstance(value, np.ndarray):
                shape = value.shape
                print(f"{prefix}{branch}{key}: np.array{shape}")
                if len(value.shape) == 1 and len(value) <= max_array:
                    child_prefix = "    " * (indent + 1) if is_last else "│   " * (indent + 1)
                    print(f"{child_prefix}值: {value}")
            else:
                print(f"{prefix}{branch}{key}")
                next_prefix = "    " if is_last else "│   "
                print_tree(value, indent + 1, max_array)
                
    elif isinstance(obj, list):
        print(f"{prefix}List[{len(obj)}]")
        if len(obj) > 0:
            print(f"{prefix}├── [0]")
            print_tree(obj[0], indent + 1, max_array)
            if len(obj) > 1:
                print(f"{prefix}└── ... ({len(obj)-1} more items)")
                
    elif isinstance(obj, np.ndarray):
        print(f"{prefix}np.array{obj.shape}")
        if len(obj.shape) == 1 and len(obj) <= max_array:
            print(f"{prefix}└── 值: {obj}")
            
    else:
        print(f"{prefix}{obj}")

def check_labeled_data(test_number):
    """检查指定test文件的标签数据"""
    # 构建文件名
    test_name = f"test{test_number:03d}"
    file_path = Path(f"/home/shiyao/EEG/eegprocessed/labeled/{test_name}_preprocessed_labeled.npz")
    
    # 加载vis_ranges信息
    ranges_file = Path("/home/shiyao/EEG/eegprocessed/vis_ranges_20250106_065622.txt")
    vis_ranges = load_vis_ranges(ranges_file)
    expected_range = vis_ranges.get(test_name, {})
    
    if not file_path.exists():
        print(f"错误: 找不到文件 {file_path}")
        return
    
    # 加载数据
    try:
        data = np.load(file_path, allow_pickle=True)
        
        print(f"\n检查文件: {file_path}")
        print("=" * 80)
        
        # 打印文件结构
        print("\n文件结构")
        print("=" * 20)
        for key in data.files:
            print(f"├── {key}")
        
        # 打印info信息
        print("\nInfo信息")
        print("=" * 20)
        print_tree(data['info'].item())
        
        # 打印trials信息
        print("\nTrials信息")
        print("=" * 20)
        trials = data['trials']
        print(f"总试次数: {len(trials)}")
        
        # 打印最后一个trial的详细信息
        print("\n最后一个Trial的详细信息")
        print("=" * 20)
        print_tree(trials[-1])
        
        # 打印VIS ID信息和比较
        print("\nVIS ID信息与预期比较")
        print("=" * 20)
        vis_ids = [trial['labels']['vis_id'] for trial in trials]
        actual_range = {
            'start': min(vis_ids),
            'end': max(vis_ids),
            'count': len(vis_ids)
        }
        
        print("实际数据:")
        print(f"├── 范围: {actual_range['start']} - {actual_range['end']}")
        print(f"└── 总数: {actual_range['count']}")
        
        print("\nvis_ranges中的预期值:")
        print(f"├── 范围: {expected_range['start']} - {expected_range['end']}")
        print(f"├── 总数: {expected_range['count']}")
        print(f"└── Last Marker: {expected_range['last_marker']}")
        
        # 检查差异
        if actual_range['start'] != expected_range['start']:
            print(f"\n[!] Start值不匹配: 实际={actual_range['start']}, 预期={expected_range['start']}")
        if actual_range['end'] != expected_range['end']:
            print(f"\n[!] End值不匹配: 实际={actual_range['end']}, 预期={expected_range['end']}")
        if actual_range['count'] != expected_range['count']:
            print(f"\n[!] 试次数量不匹配: 实际={actual_range['count']}, 预期={expected_range['count']}")
        
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python check.py <test_number>")
        print("例如: python check.py 1")
        sys.exit(1)
    
    try:
        test_number = int(sys.argv[1])
        check_labeled_data(test_number)
    except ValueError:
        print("错误: 请输入有效的test编号")
        sys.exit(1)