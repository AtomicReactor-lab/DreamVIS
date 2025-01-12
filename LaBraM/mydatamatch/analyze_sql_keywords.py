import numpy as np
from pathlib import Path
from collections import defaultdict
import json
import re

def extract_sql_keywords(query: str) -> set:
    """提取SQL查询中的关键字"""
    # 移除图表类型
    sql = ' '.join(query.split()[1:])
    
    # 定义SQL关键字模式
    keywords = set()
    
    # 常见SQL关键字
    common_keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'ASC', 'DESC', 'AND', 'OR', 'IN', 'LIKE']
    for keyword in common_keywords:
        if keyword in sql:
            keywords.add(keyword)
    
    # 聚合函数
    aggregations = ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']
    for agg in aggregations:
        if f"{agg}(" in sql:
            keywords.add(agg)
    
    return keywords

def extract_chart_type(query: str) -> str:
    """提取并验证图表类型"""
    # 已知的有效图表类型
    valid_chart_types = {'BAR', 'LINE', 'PIE', 'SCATTER'}
    
    # 获取第一个词并转换为大写
    chart_type = query.split()[0].upper()
    
    # 验证是否为有效图表类型
    if chart_type in valid_chart_types:
        return chart_type
    else:
        print(f"警告: 发现未知图表类型 '{chart_type}' in query: {query}")
        return chart_type

def analyze_keywords():
    """分析所有查询中的SQL关键字"""
    # 加载数据
    data_path = Path("/home/shiyao/EEG/eegprocessed/all_labeled_data.npz")
    data = np.load(data_path, allow_pickle=True)
    labels = data['labels']
    
    # 统计
    all_keywords = set()
    keyword_counts = defaultdict(int)
    keyword_combinations = defaultdict(int)
    chart_types = defaultdict(int)  # 改用计数器而不是集合
    
    print("正在分析SQL关键字...")
    
    for label in labels:
        query = label['vis_query']
        
        # 记录图表类型
        chart_type = extract_chart_type(query)
        chart_types[chart_type] += 1
        
        # 提取并统计关键字
        query_keywords = extract_sql_keywords(query)
        all_keywords.update(query_keywords)
        
        # 统计每个关键字的出现次数
        for keyword in query_keywords:
            keyword_counts[keyword] += 1
        
        # 统计关键字组合
        keyword_combinations[' '.join(sorted(query_keywords))] += 1
    
    # 打印结果
    print("\n=== SQL关键字分析 ===")
    
    print("\n1. 图表类型统计:")
    print("-" * 40)
    for chart, count in sorted(chart_types.items(), key=lambda x: (-x[1], x[0])):
        percentage = (count / len(labels)) * 100
        print(f"- {chart}: {count}次 ({percentage:.1f}%)")
    
    print("\n2. SQL关键字列表:")
    print("-" * 40)
    for keyword in sorted(all_keywords):
        count = keyword_counts[keyword]
        percentage = (count / len(labels)) * 100
        print(f"- {keyword}: {count}次 ({percentage:.1f}%)")
    
    print("\n3. 常见关键字组合 (前10个):")
    print("-" * 40)
    sorted_combinations = sorted(keyword_combinations.items(), key=lambda x: (-x[1], x[0]))
    for i, (combination, count) in enumerate(sorted_combinations[:10], 1):
        percentage = (count / len(labels)) * 100
        print(f"\n组合{i} ({count}次, {percentage:.1f}%):")
        print(f"  {combination}")
    
    # 保存结果
    result = {
        'chart_types': list(chart_types),
        'keywords': {k: v for k, v in keyword_counts.items()},
        'combinations': {k: v for k, v in keyword_combinations.items()},
        'total_queries': len(labels)
    }
    
    output_path = Path("/home/shiyao/EEG/eegprocessed/sql_keywords_analysis.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n分析结果已保存至: {output_path}")

if __name__ == "__main__":
    analyze_keywords() 