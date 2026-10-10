"""
计算并比较多个算法在给定recall下的QPS提升
柱状图展示QPS对比和相对于baseline的提升百分比
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict


FONT_SIZE = 21
BARTOP_FONT_SIZE = 12

STYLE_MAPPING = {
    "Shard HNSW": {
        "color": "#ff7f0e",
        "linestyle": "-",
        "marker": "s",
        "hatch": "-"  
    },
    "TITAN": {
        "color": "#2ca02c",
        "linestyle": "-",
        "marker": "^",
        "hatch": "//"  # 斜线
    },
    "TITAN-Overlap-2": {
        "color": "#ff7f0e",
        "linestyle": "--",
        "marker": "D",
        "hatch": "+"  # 加号
    },
}

recall_target = 0.95
bias = 0.01
DEFAULT_CONFIG = {
    "deep100m_overlap2": {
        "csv_path": "../results/deep100m_multi_shard_overlap_2_dstb.csv",
        "display_name": "TITAN-Overlap-2"
    },
    "deep100m_graph_partition": {
        "csv_path": "../results/deep100m_multi_shard_graph_dstb.csv",
        "display_name": "TITAN"
    },
    "sift100m_overlap2": {
        "csv_path": "../results/sift100m_multi_shard_overlap_2_dstb.csv",
        "display_name": "TITAN-Overlap-2"
    },
    "sift100m_graph_partition": {
        "csv_path": "../results/sift100m_multi_shard_graph_dstb.csv",
        "display_name": "TITAN"
    },
    "space100m_overlap2": {
        "csv_path": "../results/space100m_multi_shard_overlap_2_dstb.csv",
        "display_name": "TITAN-Overlap-2"
    },
    "space100m_graph_partition": {
        "csv_path": "../results/space100m_multi_shard_graph_dstb.csv",
        "display_name": "TITAN"
    },
    "wiki20m_overlap2": {
        "csv_path": "../results/wiki20m_1mbkt_overlap.csv",
        "display_name": "TITAN-Overlap-2"
    },
    "wiki20m_graph_partition": {
        "csv_path": "../results/wiki20m_1mbkt_graph.csv",
        "display_name": "TITAN"
    },
}


def get_qps_at_recall(csv_path, recall_target, bias):
    """
    从CSV文件中获取指定recall范围内的平均QPS
    
    Args:
        csv_path: CSV文件路径
        recall_target: 目标recall值
        bias: 允许的误差范围
    
    Returns:
        平均QPS值，如果没有找到则返回None
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_path}")
        return None
    
    # 检查必要的列
    if 'recall' not in df.columns or 'qps' not in df.columns:
        print(f"错误: {csv_path} 缺少必要的列 (recall 和/或 qps)")
        return None
    
    # 计算recall范围
    recall_min = recall_target - bias
    recall_max = recall_target + bias
    
    # 筛选在recall范围内的数据
    filtered_df = df[(df['recall'] >= recall_min) & (df['recall'] <= recall_max)]
    
    if len(filtered_df) == 0:
        print(f"警告: {csv_path} 在 recall 范围 [{recall_min:.4f}, {recall_max:.4f}] 内没有数据")
        return None
    
    # 返回最大QPS
    max_qps = filtered_df['qps'].max()
    return max_qps

def plot_qps_comparison(recall_target, bias, config, output_path=None):
    """
    绘制QPS对比柱状图（分组柱状图）
    
    Args:
        recall_target: 目标recall值
        bias: 允许的误差范围
        config: 配置字典
        output_path: 输出图片路径，如果为None则显示图片
    """
    # 按数据集前缀分组
    grouped_data = defaultdict(dict)
    
    for key, cfg in config.items():
        qps = get_qps_at_recall(cfg["csv_path"], recall_target, bias)
        if qps is not None:
            # 提取数据集前缀（如 deep10m, sift10m）
            # 匹配 deep10m, sift10m, deep100m 等格式
            match = re.match(r'([a-z]+\d+[mk]?)', key)
            dataset_prefix = match.group(1) if match else key.split('_')[0]
            method_name = cfg["display_name"]
            grouped_data[dataset_prefix][method_name] = qps
    
    if len(grouped_data) == 0:
        print("错误: 没有有效的数据可以绘制")
        return
    
    # 获取所有数据集和方法，按照config中的顺序
    # 先收集所有数据集，按照在config中首次出现的顺序
    datasets_ordered = []
    datasets_seen = set()
    for key in config.keys():
        match = re.match(r'([a-z]+\d+[mk]?)', key)
        dataset_prefix = match.group(1) if match else key.split('_')[0]
        if dataset_prefix in grouped_data and dataset_prefix not in datasets_seen:
            datasets_ordered.append(dataset_prefix)
            datasets_seen.add(dataset_prefix)
    
    datasets = datasets_ordered
    
    # 按照config中的顺序保持方法顺序
    all_methods = []
    seen = set()
    for cfg in config.values():
        method_name = cfg["display_name"]
        if method_name not in seen:
            all_methods.append(method_name)
            seen.add(method_name)
    
    
    # 设置柱状图参数
    num_methods = len(all_methods)
    # 使用固定的总宽度，让同一组的柱子紧挨着
    total_group_width = 0.6  # 每组的总宽度
    bar_width = total_group_width / num_methods  # 每个柱子的宽度
    x_pos = np.arange(len(datasets))
    
    fig, ax = plt.subplots(figsize=(13, 6))
    
    # 收集所有 bars 和 labels 用于图例
    all_bars = []
    all_labels = []
    
    # 为每个方法绘制柱状图
    for i, method in enumerate(all_methods):
        values = []
        for dataset in datasets:
            qps = grouped_data[dataset].get(method, 0)
            # 转换为kQPS（除以1000）
            values.append(qps / 1000.0)
        
        # 计算offset，让同一组的柱子紧挨着（无间隔）
        # 第一个方法从 -total_group_width/2 开始，每个方法紧挨着排列
        offset = -total_group_width / 2 + i * bar_width
        style = STYLE_MAPPING[method]
        bars = ax.bar(x_pos + offset, values, bar_width, 
                     label=method, 
                     color="white",           # 白色填充
                     edgecolor="black",       # 黑色边框
                     linewidth=0.9,
                     hatch=style["hatch"])    # 使用不同的填充图案区分
        
        # 收集第一个 bar 和 label（用于图例）
        if len(bars) > 0:
            all_bars.append(bars[0])
            all_labels.append(method)
        
        # 在柱状图上标注数值（显示k单位）
        for bar, val in zip(bars, values):
            if val > 0:  # 只标注有效数据
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}',
                       ha='center', va='bottom', fontsize=BARTOP_FONT_SIZE)
    
    # 设置图表属性
    ax.set_ylabel('Throughput (QPS)', fontsize=FONT_SIZE)
    ax.set_xlabel('Dataset', fontsize=FONT_SIZE)
    # ax.set_title(f'QPS Comparison at Recall {recall_target:.2f} (±{bias:.2f})', 
    #             fontsize=FONT_SIZE, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, fontsize=FONT_SIZE, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=FONT_SIZE)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # 保存或显示图片
    if output_path:
        plt.savefig(output_path, dpi=500, bbox_inches='tight')
        print(f"\n图表已保存到: {output_path}")
        
        # 创建单独的图例文件
        fig_legend = plt.figure(figsize=(5, 0.4))
        fig_legend.legend(all_bars, all_labels, loc="center", ncol=len(all_methods), fontsize=FONT_SIZE)
        fig_legend.tight_layout()
        plt.axis("off")
        legend_file = output_path.replace('.pdf', '_legend.pdf')
        fig_legend.savefig(legend_file, bbox_inches="tight")
        print(f"图例已保存到: {legend_file}")
    else:
        plt.show()
    
    # 打印统计信息
    print("\n" + "="*70)
    print(f"Recall Target: {recall_target:.2f} (±{bias:.2f})")
    print(f"Recall Range: [{recall_target - bias:.4f}, {recall_target + bias:.4f}]")
    print("="*70)
    for dataset in datasets:
        print(f"\n{dataset}:")
        for method in all_methods:
            qps = grouped_data[dataset].get(method, None)
            if qps:
                print(f"  {method:15s}: {qps:10.2f} QPS")
    print("="*70)
    


def main():
    config = DEFAULT_CONFIG
    output_path = f"../results/paper_figs/multi_shard_qps_comparison_{recall_target:.2f}.pdf"
    plot_qps_comparison(recall_target, bias, config, output_path=output_path)


if __name__ == "__main__":
    main()
