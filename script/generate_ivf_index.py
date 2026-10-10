"""
Python script to generate IVF index for vector datasets
支持的数据集: sift, siftsmall, deep1m, deep10m, deep100m, gist1m, wiki20m
"""

import argparse
import numpy as np
import os
import sys
import time
import struct
from typing import Tuple
import tqdm
import faiss

# 添加当前目录到路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import get_dataset

# IVF索引文件格式常量
IVF_MAGIC = 0x49564F46  # "IVOF"
IVF_VERSION = 2  # 版本2支持PQ


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="生成 IVF 索引",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据集参数
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="deep10m",
        help="数据集名称"
    )
    
    # IVF 索引参数
    parser.add_argument(
        "--nlist", 
        type=int, 
        default=100000,
        help="IVF 聚类中心数量 (nlist)"
    )
    
    parser.add_argument(
        "--nprobe", 
        type=int, 
        default=1000,
        help="搜索时探测的聚类中心数量 (nprobe)"
    )
    
    parser.add_argument(
        "--metric", 
        type=str, 
        default="L2",
        choices=["L2", "IP"],
        help="距离度量类型 (L2 或 IP)"
    )
    
    # 输出参数
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="../data/hakes_index",
        help="输出索引文件路径"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="启用详细输出"
    )
    
    # 训练参数
    parser.add_argument(
        "--train-samples", 
        type=int, 
        default=None,
        help="用于训练聚类的样本数量 (默认使用全部数据)"
    )
    
    parser.add_argument(
        "--random-seed", 
        type=int, 
        default=42,
        help="随机种子"
    )
    
    # 输出格式参数
    parser.add_argument(
        "--format", 
        type=str, 
        default="binary",
        choices=["faiss", "binary"],
        help="输出格式: faiss (原始faiss格式) 或 binary (与Go代码兼容的二进制格式)"
    )
    
    # PQ量化参数
    parser.add_argument(
        "--pq-bits",
        type=int,
        default=None,
        # choices=[None, 8, 16,25, 32, 48,50, 64,96,128],
        help="PQ量化子向量数量 (m)，None表示使用Flat索引。可选值: 8, 16, 32, 64。注意：维度必须能被m整除"
    )
    
    return parser.parse_args()


def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载数据集"""
    print(f"正在加载数据集: {dataset_name}")
    
    try:
        base_data, query_data, ground_truth = get_dataset(dataset_name)
        print(f"数据集加载完成:")
        print(f"  - 基础数据: {base_data.shape}")
        print(f"  - 查询数据: {query_data.shape}")
        print(f"  - 真实标签: {ground_truth.shape}")
        return base_data, query_data, ground_truth
    except Exception as e:
        print(f"加载数据集失败: {e}")
        sys.exit(1)


def build_ivf_index(
    data: np.ndarray, 
    nlist: int, 
    metric: str = "L2",
    train_samples: int = None,
    random_seed: int = 42,
    verbose: bool = False,
    pq_bits: int = None
):
    """构建 IVF 索引（支持 Flat 和 PQ）
    
    Args:
        pq_bits: PQ量化子向量数量 (m)，None表示使用Flat索引
    """
    
    dimension = data.shape[1]
    
    print(f"开始构建 IVF 索引...")
    print(f"  - 数据维度: {data.shape}")
    print(f"  - 聚类中心数: {nlist}")
    print(f"  - 距离度量: {metric}")
    if pq_bits is not None:
        print(f"  - PQ量化: m={pq_bits}, nbits=8 (256码字)")
    else:
        print(f"  - 索引类型: Flat (无压缩)")
    
    # 设置随机种子
    np.random.seed(random_seed)
    faiss.omp_set_num_threads(128) # 设置线程数
    
    # 选择距离度量
    if metric == "L2":
        metric_type = faiss.METRIC_L2
    elif metric == "IP":
        metric_type = faiss.METRIC_INNER_PRODUCT
    else:
        raise ValueError(f"不支持的距离度量: {metric}")
    
    # 创建量化器
    quantizer = faiss.IndexFlat(dimension, metric_type)
    
    # 创建 IVF 索引（Flat 或 PQ）
    if pq_bits is not None:
        # 验证维度能否被m整除
        if dimension % pq_bits != 0:
            raise ValueError(f"向量维度 {dimension} 必须能被 PQ子向量数 {pq_bits} 整除")
        # nbits=8 表示每个子向量使用8bit，对应256个码字（2^8=256）
        nbits = 8
        index = faiss.IndexIVFPQ(quantizer, dimension, nlist, pq_bits, nbits, metric_type)
        # 设置为非残差模式，编码原始向量（简化实现）
        index.by_residual = False
    else:
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, metric_type)
    
    # 准备训练数据
    if train_samples is None or train_samples >= len(data):
        train_data = data
    else:
        # 随机采样训练数据
        indices = np.random.choice(len(data), train_samples, replace=False)
        train_data = data[indices]
    
    print(f"  - 训练数据大小: {train_data.shape}")
    
    # 训练索引
    print("正在训练 IVF 索引...")
    start_time = time.time()
    index.train(train_data.astype(np.float32))
    training_time = time.time() - start_time
    print(f"训练完成，耗时: {training_time:.2f} 秒")
    
    # 添加数据到索引
    print("正在添加数据到索引...")
    start_time = time.time()
    index.add(data.astype(np.float32))
    build_time = time.time() - start_time
    print(f"索引构建完成，耗时: {build_time:.2f} 秒")
    
    if verbose:
        print(f"索引统计信息:")
        print(f"  - 总向量数: {index.ntotal}")
        print(f"  - 聚类中心数: {index.nlist}")
        print(f"  - 是否已训练: {index.is_trained}")
        if pq_bits is not None:
            print(f"  - PQ码本大小: {index.pq.M} 个子向量, {index.pq.ksub} 个码字/子向量")
            print(f"  - PQ编码大小: {index.pq.code_size} 字节/向量")
    
    return index


def save_index_binary(index, data: np.ndarray, output_dir: str, dataset: str, nprobe: int, metric: str):
    """保存索引到二进制文件，格式与Go代码兼容（支持Flat和PQ）
    
    Args:
        index: faiss.IndexIVFFlat 或 faiss.IndexIVFPQ
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 检测索引类型
    is_pq = isinstance(index, faiss.IndexIVFPQ)
    pq_m = index.pq.M if is_pq else 0
    pq_ksub = index.pq.ksub if is_pq else 0
    
    # 批量分配所有向量到聚类
    print("正在批量分配向量到聚类...")
    data_f32 = data.astype(np.float32)
    distances, cluster_ids = index.quantizer.search(data_f32, 1)
    cluster_ids = cluster_ids.squeeze()  # shape: (N,)

    # 使用计数分桶（O(N)）代替全排序，将相同聚类的样本放到连续区间
    nlist = index.nlist
    counts = np.bincount(cluster_ids, minlength=nlist).astype(np.int64)  # shape: (nlist,)
    offsets = np.zeros(nlist + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)

    # 构建一个全局的排列索引 perm，使得同一聚类的样本连续
    perm = np.empty(len(cluster_ids), dtype=np.uint32)
    write_ptr = offsets[:-1].copy()
    for j, cid in enumerate(cluster_ids):
        pos = write_ptr[cid]
        perm[pos] = j
        write_ptr[cid] += 1

    output_path = os.path.join(output_dir, f"{dataset}_{nlist}_ivf_index.bin")
    if is_pq:
        output_path = os.path.join(output_dir, f"{dataset}_{nlist}_ivfpq_{pq_m}_index.bin")
    
    print(f"正在保存索引到二进制文件: {output_path}")
    print(f"  - 索引类型: {'PQ' if is_pq else 'Flat'}")

    with open(output_path, 'wb') as f:
        # 写入IVFHeader
        header_part1 = struct.pack('<IIIII',
                                  IVF_MAGIC,     # Magic (uint32)
                                  IVF_VERSION,   # Version (uint32)
                                  index.d,       # Dim (int32)
                                  index.nlist,   # NList (int32)
                                  nprobe)        # NProbe (int32)

        metric_bytes = metric.encode('utf-8')[:16]
        metric_padded = metric_bytes + b'\x00' * (16 - len(metric_bytes))
        
        # Header part2: IsTrained (1 byte) + PQ标志 (1 byte) + PQ参数 (2 bytes) + padding (4 bytes)
        is_trained = 1 if index.is_trained else 0
        pq_flag = 1 if is_pq else 0
        header_part2 = struct.pack('<BBHH',
                                  is_trained,    # IsTrained (uint8)
                                  pq_flag,       # PQ标志 (uint8)
                                  pq_m,          # PQ子向量数 (uint16)
                                  pq_ksub)       # PQ码本大小 (uint16)

        f.write(header_part1)
        f.write(metric_padded)
        f.write(header_part2)

        # 写入聚类中心
        print(f"正在写入 {index.nlist} 个聚类中心...")
        for i in range(index.nlist):
            centroid = index.quantizer.reconstruct(i).astype(np.float32)
            f.write(centroid.tobytes())

        # 写入PQ码本（如果使用PQ）
        if is_pq:
            print(f"正在写入 PQ 码本 (M={pq_m}, ksub={pq_ksub})...")
            # PQ码本结构: M个子量化器，每个有ksub个码字，每个码字维度为dsub = dim/M
            dsub = index.d // pq_m
            # 使用 faiss.vector_to_array 获取码本
            centroids = faiss.vector_to_array(index.pq.centroids)
            codebooks = np.asarray(centroids, dtype=np.float32).reshape(pq_m, pq_ksub, dsub)
            # 按顺序写入码本
            for m in range(pq_m):
                for k in range(pq_ksub):
                    code = codebooks[m, k, :].astype(np.float32)
                    f.write(code.tobytes())

        # 写入倒排列表
        print("正在写入倒排列表...")
        for i in range(nlist):
            start = offsets[i]
            end = offsets[i + 1]
            vector_count = int(end - start)
            f.write(struct.pack('<I', vector_count))
            if vector_count > 0:
                point_ids = perm[start:end]
                f.write(point_ids.tobytes())
                
                if is_pq:
                    # PQ模式：写入编码后的向量（code_size字节/向量）
                    # 批量获取原始向量并编码
                    vectors = data_f32[point_ids]
                    # compute_codes 接受向量数组，返回编码数组
                    codes = index.pq.compute_codes(vectors)
                    f.write(codes.tobytes())
                else:
                    # Flat模式：写入原始向量
                    vectors = data_f32[point_ids]
                    f.write(vectors.tobytes())

    print(f"二进制索引已保存到: {output_path}")
    print(f"索引统计:")
    print(f"  - 总向量数: {len(data)}")
    print(f"  - 聚类数: {index.nlist}")
    if is_pq:
        print(f"  - PQ参数: M={pq_m}, ksub={pq_ksub}, code_size={index.pq.code_size}")
    # 使用 counts 统计非空聚类数量
    try:
        non_empty = int((counts > 0).sum())
    except Exception:
        non_empty = 0
    print(f"  - 非空聚类数: {non_empty}")


def save_index(index: faiss.IndexIVFFlat, output_path: str):
    """保存索引到文件（原始faiss格式）"""
    print(f"正在保存索引到: {output_path}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存索引
    faiss.write_index(index, output_path)
    print(f"索引已保存到: {output_path}")


def test_index(index, query_data: np.ndarray, ground_truth: np.ndarray, nprobe: int, k: int = 10):
    """测试索引性能"""
    print(f"正在测试索引性能...")
    print(f"  - nprobe: {nprobe}")
    print(f"  - top-k: {k}")
    
    # 设置 nprobe
    index.nprobe = nprobe
    
    # 执行搜索
    start_time = time.time()
    distances, indices = index.search(query_data.astype(np.float32), k)
    search_time = time.time() - start_time
    
    print(f"搜索完成，耗时: {search_time:.2f} 秒")
    print(f"平均每次查询耗时: {search_time / len(query_data) * 1000:.2f} ms")
    
    # 计算召回率 (如果有真实标签)
    if ground_truth is not None and ground_truth.shape[1] >= k:
        recall = 0
        for i in range(len(query_data)):
            retrieved = set(indices[i])
            relevant = set(ground_truth[i][:k])
            if len(relevant) > 0:
                recall += len(retrieved.intersection(relevant)) / len(relevant)
        
        recall /= len(query_data)
        print(f"Top-{k} 召回率: {recall:.4f}")


def main():
    """主函数"""
    args = parse_arguments()
    
    print("=" * 60)
    print("IVF 索引生成器")
    print("=" * 60)
    
    # 加载数据集
    base_data, query_data, ground_truth = load_dataset(args.dataset)
    
    # 构建 IVF 索引
    index = build_ivf_index(
        data=base_data,
        nlist=args.nlist,
        metric=args.metric,
        train_samples=args.train_samples,
        random_seed=args.random_seed,
        verbose=args.verbose,
        pq_bits=args.pq_bits
    )
    
    # 测试索引性能
    test_index(index, query_data, ground_truth, args.nprobe)
    
    # 保存索引
    if args.format == "binary":
        save_index_binary(index, base_data, args.output_dir,args.dataset, args.nprobe, args.metric)
    else:
        save_index(index, args.output_dir)
    
    print("=" * 60)
    print("IVF 索引生成完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
