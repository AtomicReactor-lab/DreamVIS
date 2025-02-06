import numpy as np
import h5py
import json
from pathlib import Path
from bs4 import BeautifulSoup

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
    return ranges

def extract_vql_from_html(html_file):
    """从HTML文件中提取最后一个VQL注释"""
    vql = None
    with open(html_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '<!-- <h4>VIS Query: </h4><p>' in line:
                vql = line.strip()
    return vql

def extract_table_from_html(html_file):
    """从container div中提取完整的表格HTML内容"""
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
        soup = BeautifulSoup(content, 'html.parser')
        container = soup.find('div', {'class': 'container'})
        if container:
            table = container.find('table')
            if table:
                return str(table)
    return None

def process_test_with_labels(test_file, vis_ranges):
    """处理单个test文件并添加标签"""
    # 1. 加载EEG数据
    npz_path = Path(f"/home/shiyao/EEG/eegprocessed/segmented/{test_file}_preprocessed_trials.npz")
    eeg_data = np.load(npz_path, allow_pickle=True)
    trials = eeg_data['trials']
    
    # 2. 获取该文件的VIS范围
    vis_range = vis_ranges[test_file]
    start_vis = vis_range['start']
    end_vis = vis_range['end']
    
    # 3. 设置HTML文件路径
    html_base_dir = Path("/home/shiyao/EEG/eegmaterial/FINAL_EDITION_cleaned_id_JOIN_rename_EEGNV")
    
    # 4. 创建输出目录
    output_path = Path(f"/home/shiyao/EEG/eegprocessed/labeled/{test_file}_preprocessed_labeled.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 5. 创建新的数据结构来存储带标签的数据
    labeled_trials = []
    
    # 6. 处理每个VIS ID
    for vis_id in range(start_vis, end_vis + 1):
        table_path = html_base_dir / f"VIS_{vis_id}_table.html"
        query_path = html_base_dir / f"VIS_{vis_id}_nlquery.html"
        
        if table_path.exists() and query_path.exists():
            trial_idx = vis_id - start_vis
            if trial_idx >= len(trials):
                continue
                
            trial_data = trials[trial_idx]
            
            # 读取table内容
            table_content = extract_table_from_html(table_path)
            
            # 读取query内容和自然语言查询
            with open(query_path, 'r', encoding='utf-8') as f:
                content = f.read()
                soup = BeautifulSoup(content, 'html.parser')
                container = soup.find('div', {'class': 'container'})
                nl_query = container.text.strip() if container else None
            
            # 提取VQL
            vql = extract_vql_from_html(query_path)
            
            # 创建带标签的试次数据
            labeled_trial = {
                'eeg_data': trial_data,
                'labels': {
                    'vis_id': vis_id,
                    'table': table_content,
                    'nl_query': nl_query,
                    'vis_query': vql
                }
            }
            labeled_trials.append(labeled_trial)
    
    # 7. 保存带标签的数据
    np.savez(output_path, 
             trials=labeled_trials,
             info={
                 'n_trials': len(labeled_trials),
                 'sampling_rate': 200,
                 'phases': ['table', 'query', 'vis'],
                 'vis_range': {'start': start_vis, 'end': end_vis}
             })
    
    print(f"完成 {test_file}: 处理了 {len(labeled_trials)} 个试次 (VIS范围: {start_vis}-{end_vis})")
    return labeled_trials

def batch_process_all_tests():
    """批量处理所有test文件"""
    ranges_file = Path("/home/shiyao/EEG/eegprocessed/vis_ranges_20250106_065622.txt")
    vis_ranges = load_vis_ranges(ranges_file)
    
    print("开始处理所有文件...")
    
    for test_file in sorted(vis_ranges.keys()):
        try:
            process_test_with_labels(test_file, vis_ranges)
        except Exception as e:
            print(f"处理 {test_file} 时出错: {str(e)}")
            continue
    
    print("\n所有文件处理完成！")

if __name__ == "__main__":
    batch_process_all_tests()
