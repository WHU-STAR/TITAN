import hnswlib
import argparse
import numpy as np
import os
import time
from data_loader import get_dataset
from utils import Timer


def main():
    parser = argparse.ArgumentParser(description='HNSW索引测试工具')
    parser.add_argument('--dataset', type=str, required=True,
                       help='数据集名称 (如: sift, siftsmall, deep1m, space100m等)')
    parser.add_argument('--index_path', type=str, 
                       help='HNSW索引保存路径')
    parser.add_argument('--k', type=int, default=10,
                       help='返回的最近邻数量 (默认: 10)')
    parser.add_argument('--M', type=int, default=16,
                       help='HNSW参数M，每个节点的最大连接数 (默认: 16)')
    parser.add_argument('--ef_construction', type=int, default=200,
                       help='HNSW构建时的ef参数 (默认: 200)')
    parser.add_argument('--ef_search', type=int, default=50,
                       help='HNSW搜索时的ef参数 (默认: 50)')
    parser.add_argument('--max_elements', type=int, default=None,
                       help='索引中最大元素数量，如果未指定则使用数据集大小')
    parser.add_argument('--rebuild', action='store_true',
                       help='强制重建索引，即使索引文件已存在')
    
    args = parser.parse_args()
    
    print(f"=== HNSW索引测试 ===")
    print(f"数据集: {args.dataset}")
    if args.index_path is None:
        args.index_path = f"{args.dataset}_single_hnsw.bin"
    print(f"索引路径: {args.index_path}")
    print(f"k: {args.k}")
    print(f"M: {args.M}")
    print(f"ef_construction: {args.ef_construction}")
    print(f"ef_search: {args.ef_search}")
    
    # 加载数据集
    print(f"\n正在加载数据集 '{args.dataset}'...")
    timer = Timer()
    timer.tick()
    
    try:
        base_data, query_data, groundtruth_data = get_dataset(args.dataset)
        timer.tuck("数据加载")
        
        print(f"Base数据形状: {base_data.shape}")
        print(f"Query数据形状: {query_data.shape}")
        print(f"Ground truth数据形状: {groundtruth_data.shape}")
        
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return
    
    # 获取数据维度和大小
    num_elements, dim = base_data.shape
    if args.max_elements is None:
        args.max_elements = num_elements
    
    print(f"数据维度: {dim}")
    print(f"Base数据数量: {num_elements:,}")
    print(f"Query数据数量: {query_data.shape[0]:,}")
    
    # 构建或加载HNSW索引
    index = build_or_load_hnsw_index(
        base_data, args.index_path, args.M, args.ef_construction, 
        args.max_elements, args.rebuild
    )
    
    # 执行搜索
    print(f"\n=== 开始搜索 ===")
    print(f"ef_search: {args.ef_search}")
    
    # 设置搜索参数
    index.set_ef(args.ef_search)
    
    # 执行批量搜索
    timer.tick()
    labels, distances = index.knn_query(query_data, k=args.k)
    search_time = timer.tuck("搜索")
    
    print(f"搜索完成，平均每个查询时间: {search_time / query_data.shape[0] * 1000:.3f}ms")
    
    # 计算recall
    recall = calculate_recall(labels, groundtruth_data, args.k)
    print(f"\nRecall@{args.k}: {recall:.4f}")
    
    print(f"\n=== 测试完成 ===")


def build_or_load_hnsw_index(base_data, index_path, M, ef_construction, max_elements, rebuild=False):
    """
    构建或加载HNSW索引
    """
    timer = Timer()
    
    # 检查索引文件是否存在
    if index_path is not None and os.path.exists(index_path) and not rebuild:
        print(f"\n正在加载现有索引: {index_path}")
        timer.tick()
        
        # 创建索引对象并加载
        index = hnswlib.Index(space='l2', dim=base_data.shape[1])
        index.load_index(index_path, max_elements=max_elements)
        
        timer.tuck("索引加载")
        print(f"索引加载完成，包含 {index.get_current_count()} 个元素")
        
    else:
        print(f"\n正在构建HNSW索引...")
        print(f"参数: M={M}, ef_construction={ef_construction}")
        
        timer.tick()
        
        # 创建新索引
        index = hnswlib.Index(space='l2', dim=base_data.shape[1])
        index.init_index(max_elements=max_elements, M=M, ef_construction=ef_construction)
        
        # 批量添加数据
        batch_size = 10000  # 批量大小
        num_batches = (len(base_data) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(base_data))
            
            batch_data = base_data[start_idx:end_idx]
            batch_ids = np.arange(start_idx, end_idx)
            
            index.add_items(batch_data, batch_ids)
            
            if (i + 1) % 10 == 0 or i == num_batches - 1:
                print(f"已添加批次 {i + 1}/{num_batches}, 元素数量: {end_idx}")
        
        build_time = timer.tuck("索引构建")
        print(f"索引构建完成，总元素数: {index.get_current_count():,}")
        
        # 保存索引
        print(f"正在保存索引到: {index_path}")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        index.save_index(index_path)
        print(f"索引保存完成")
    
    return index


def calculate_recall(predictions, ground_truth, k):
    """
    计算recall@k
    
    Args:
        predictions: 搜索结果，形状为 (n_queries, k)
        ground_truth: 真实最近邻，形状为 (n_queries, n_neighbors)
        k: 计算recall的k值
    
    Returns:
        recall@k的值
    """
    if predictions.shape[0] != ground_truth.shape[0]:
        raise ValueError("预测结果和ground truth的查询数量不匹配")
    
    total_recall = 0.0
    n_queries = predictions.shape[0]
    
    for i in range(n_queries):
        # 获取前k个预测结果
        pred_k = set(predictions[i, :k])
        
        # 获取前k个真实结果
        true_k = set(ground_truth[i, :k])
        
        # 计算交集
        intersection = pred_k.intersection(true_k)
        
        # 计算该查询的recall
        recall_i = len(intersection) / k
        total_recall += recall_i
    
    return total_recall / n_queries


if __name__ == "__main__":
    main()