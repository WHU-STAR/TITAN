"""
Python script to process sift100m dataset
将sift100m的hdf5格式数据转换为与deep文件夹相同的fvecs和ivecs格式
"""

import h5py
import numpy as np
import struct
import sys
import os
from pathlib import Path
import argparse


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_writer import write_fvecs, write_ivecs


def process_sift100m_dataset(hdf5_path: str, output_dir: str):
    """
    处理sift100m数据集，转换为fvecs和ivecs格式
    
    Args:
        hdf5_path: hdf5文件路径
        output_dir: 输出目录
    """
    print("=== 开始处理SIFT100M数据集 ===")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 读取hdf5文件
    print(f"读取HDF5文件: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        print("HDF5文件结构:")
        for key in f.keys():
            print(f"  {key}: {f[key].shape}, dtype={f[key].dtype}")
        
        # 读取数据
        train_data = f['train'][:]  # (100000000, 128) - 基础向量
        test_data = f['test'][:]    # (1000, 128) - 查询向量
        neighbors = f['neighbors'][:]  # (1000, 1000) - 每个查询的最近邻索引
        distances = f['distances'][:]  # (1000, 1000) - 对应的距离
    
    print(f"\n数据统计:")
    print(f"  训练数据 (base): {train_data.shape[0]:,} 个向量, 维度: {train_data.shape[1]}")
    print(f"  测试数据 (queries): {test_data.shape[0]:,} 个向量, 维度: {test_data.shape[1]}")
    print(f"  邻居数据: {neighbors.shape[0]:,} 个查询, 每个查询 {neighbors.shape[1]} 个邻居")
    
    # 保存base数据 (train -> fvecs)
    base_filename = output_path / "sift100m_base.fvecs"
    write_fvecs(str(base_filename), train_data)
    
    # 保存queries数据 (test -> fvecs)
    queries_filename = output_path / "sift100m_queries.fvecs"
    write_fvecs(str(queries_filename), test_data)
    
    # 保存ground truth数据 (neighbors -> ivecs)
    gt_filename = output_path / "sift100m_groundtruth.ivecs"
    write_ivecs(str(gt_filename), neighbors)
    
    print(f"\n=== 处理完成 ===")
    print(f"输出文件:")
    print(f"  Base: {base_filename}")
    print(f"  Queries: {queries_filename}")
    print(f"  Ground Truth: {gt_filename}")
    
    # 显示文件大小
    for filename in [base_filename, queries_filename, gt_filename]:
        if filename.exists():
            size_mb = filename.stat().st_size / (1024 * 1024)
            print(f"  {filename.name}: {size_mb:.1f} MB")

            
def main():
    parser = argparse.ArgumentParser(description="处理SIFT100M数据集，转换为fvecs和ivecs格式")
    parser.add_argument("--input", type=str, 
                       default="/home/adrian/code/vcluster-system/data/sift100m/sift100m-128-euclidean.hdf5",
                       help="输入HDF5文件路径")
    parser.add_argument("--output", type=str,
                       default="/home/adrian/code/vcluster-system/data/sift100m",
                       help="输出目录路径")
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    # 处理数据集
    process_sift100m_dataset(args.input, args.output)



if __name__ == "__main__":
    main()