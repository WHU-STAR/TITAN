
"""
Python script to process space100m dataset
将space100m的二进制格式数据转换为fvecs和ivecs格式
"""

from usearch.io import load_matrix
from pathlib import Path
import torch
import numpy as np
import os
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_writer import write_fvecs, write_ivecs


base_path = "/home/adrian/code/vcluster-system/data/ann-spacev-100m/base.100M.i8bin"
query_path = "/home/adrian/code/vcluster-system/data/ann-spacev-100m/query.30K.i8bin"
groundtruth_path = "/home/adrian/code/vcluster-system/data/ann-spacev-100m/groundtruth.30K.i32bin"
ids_path = "/home/adrian/code/vcluster-system/data/ann-spacev-100m/ids.100M.i32bin"
distances_path = "/home/adrian/code/vcluster-system/data/ann-spacev-100m/groundtruth.30K.f32bin"

def recaclulate_groundtruth(ids: np.ndarray, groundtruth: np.ndarray):
    """
    使用GPU加速重新计算groundtruth，将原来的ID值替换为对应的下标索引
    
    Args:
        ids: 原始ID数组，包含每个向量对应的ID
        groundtruth: 原始groundtruth，包含查询结果对应的ID值
    
    Returns:
        recaclulated_groundtruth: 更新后的groundtruth，ID值被替换为对应的下标索引
    """
    
    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 将numpy数组转换为torch张量并移到GPU，确保数据类型一致
    ids_tensor = torch.from_numpy(ids.flatten()).long().to(device)
    gt_tensor = torch.from_numpy(groundtruth).long().to(device)
    
    print(f"ids张量形状: {ids_tensor.shape}, groundtruth张量形状: {gt_tensor.shape}")
    
    # 将groundtruth展平以便批量处理
    gt_flat = gt_tensor.flatten()
    result_flat = torch.zeros_like(gt_flat, dtype=torch.long, device=device)
    
    # 使用torch的searchsorted进行批量查找
    sorted_ids, sorted_indices = torch.sort(ids_tensor)
    search_positions = torch.searchsorted(sorted_ids, gt_flat)
    
    # 检查是否找到匹配项
    valid_mask = (search_positions < len(sorted_ids)) & (sorted_ids[search_positions] == gt_flat)
    
    # 设置结果，确保数据类型匹配
    result_flat[valid_mask] = sorted_indices[search_positions[valid_mask]]
    result_flat[~valid_mask] = gt_flat[~valid_mask]  # 未找到的保持原值

    # 如果有没找到的
    if torch.sum(~valid_mask) > 0:
        print(f"警告: 有 {torch.sum(~valid_mask)} 个ID在ids数组中找不到")
    
    # 将结果reshape回原来的形状
    result_tensor = result_flat.reshape(groundtruth.shape)
    
    # 将结果转回numpy数组
    recaclulated_groundtruth = result_tensor.cpu().numpy()
    
    # 显示前几个结果作为验证
    print("前几个重新计算的结果:")
    for i in range(min(3, len(groundtruth))):
        for j in range(min(5, len(groundtruth[i]))):
            print(f"  查询{i}, 邻居{j}: {groundtruth[i][j]} -> {recaclulated_groundtruth[i][j]}")
    
    return recaclulated_groundtruth

def process_space100m_dataset(output_dir: str):
    """
    处理space100m数据集，转换为fvecs和ivecs格式
    
    Args:
        output_dir: 输出目录
    """
    print("=== 开始处理SPACE100M数据集 ===")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 读取二进制文件
    print(f"读取二进制文件...")
    base_view = load_matrix(base_path, dtype=np.int8, view=True)
    ids = load_matrix(ids_path, dtype=np.int32)
    queries = load_matrix(query_path, dtype=np.int8)
    groundtruth = load_matrix(groundtruth_path, dtype=np.int32)
    distances = load_matrix(distances_path, dtype=np.float32)
    
    print(f"\n数据统计:")
    print(f"  Base数据: {base_view.shape[0]:,} 个向量, 维度: {base_view.shape[1]}")
    print(f"  Query数据: {queries.shape[0]:,} 个向量, 维度: {queries.shape[1]}")
    print(f"  匹配数据: {groundtruth.shape[0]:,} 个查询, 每个查询 {groundtruth.shape[1]} 个邻居")
    print(f"  距离数据: {distances.shape[0]:,} 个查询, 每个查询 {distances.shape[1]} 个距离")
    print(f"  ID数据: {ids.shape[0]:,} 个ID")
    
    # 将int8数据转换为float32 (fvecs格式需要float32)
    print(f"\n转换数据类型...")
    base_data = base_view.astype(np.float32)
    queries_data = queries.astype(np.float32)
    
    # 重新计算groundtruth
    print(f"\n重新计算groundtruth...")
    recaclulated_groundtruth = recaclulate_groundtruth(ids, groundtruth)
    
    # 保存base数据
    base_filename = output_path / "space100m_base.fvecs"
    write_fvecs(str(base_filename), base_data)
    
    # 保存queries数据
    queries_filename = output_path / "space100m_queries.fvecs"
    write_fvecs(str(queries_filename), queries_data)
    
    # 保存ground truth数据 (matches -> ivecs)
    gt_filename = output_path / "space100m_groundtruth.ivecs"
    write_ivecs(str(gt_filename), recaclulated_groundtruth)
    
    # # 保存距离数据 (distances -> fvecs)
    # distances_filename = output_path / "space100m_distances.fvecs"
    # write_fvecs(str(distances_filename), distances)
    
    # 保存ID数据 (ids -> ivecs，但需要reshape)
    # if len(ids.shape) == 1:
    #     # 如果ids是一维的，reshape为二维
    #     ids_reshaped = ids.reshape(-1, 1)
    # else:
    #     ids_reshaped = ids
    
    # ids_filename = output_path / "space100m_ids.ivecs"
    # write_ivecs(str(ids_filename), ids_reshaped)
    
    print(f"\n=== 处理完成 ===")
    print(f"输出文件:")
    print(f"  Base: {base_filename}")
    print(f"  Queries: {queries_filename}")
    print(f"  Ground Truth: {gt_filename}")
    # print(f"  Distances: {distances_filename}")
    # print(f"  IDs: {ids_filename}")
    
    file_lst = [base_filename, queries_filename, gt_filename]

    # 显示文件大小
    for filename in file_lst:
        if filename.exists():
            size_mb = filename.stat().st_size / (1024 * 1024)
            print(f"  {filename.name}: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="处理SPACE100M数据集，转换为fvecs和ivecs格式")
    parser.add_argument("--output", type=str,
                       default="/home/adrian/code/vcluster-system/data/ann-spacev-100m",
                       help="输出目录路径")
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    input_files = [base_path, query_path, groundtruth_path, ids_path, distances_path]
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"错误: 输入文件不存在: {file_path}")
            return
    
    # 处理数据集
    process_space100m_dataset(args.output)

if __name__ == "__main__":
    main()