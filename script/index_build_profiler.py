"""
Profile index construction time for Shard HNSW, Pyramid, and TITAN.

Each method records per-phase timings. For distributed deployment, wall-clock
build time uses max(worker_hnsw_times) for parallel worker index builds.
"""

from __future__ import annotations

import gc
import math
import os
from typing import Any, Literal

import hnswlib
import kahip
import networkx as nx
import numpy as np
from tqdm import tqdm

from bucket_cluster_algorithm import graph_partition
from data_processor import (
    EF,
    M,
    THREAD_NUM,
    np_find_nearest_point_idx_and_furthest_distance,
)
from hierarchical_kmeans import hierarchical_KMeans
from node import Bucket, Node

DEFAULT_M = M
DEFAULT_EF = EF
DEFAULT_THREADS = THREAD_NUM


def _build_hnsw(
    data: np.ndarray,
    ids: np.ndarray,
    dim: int,
    M: int = DEFAULT_M,
    ef_construction: int = DEFAULT_EF,
    num_threads: int = DEFAULT_THREADS,
) -> hnswlib.Index:
    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(
        max_elements=max(len(data), 1),
        ef_construction=ef_construction,
        M=M,
    )
    if len(data) > 0:
        index.add_items(data, ids=ids, num_threads=num_threads)
    return index


def _parallel_wall_time(worker_times: list[float], overhead: float) -> float:
    if not worker_times:
        return overhead
    return overhead + max(worker_times)


def _summarize_profile(
    method: str,
    dataset_schema: str,
    num_vectors: int,
    num_workers: int,
    phases: dict[str, Any],
    worker_hnsw_times: list[float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overhead = sum(v for k, v in phases.items() if k != "worker_hnsw_total")
    result = {
        "method": method,
        "dataset_schema": dataset_schema,
        "num_vectors": num_vectors,
        "num_workers": num_workers,
        "phases": phases,
        "worker_hnsw_times": worker_hnsw_times,
        "worker_hnsw_max": max(worker_hnsw_times) if worker_hnsw_times else 0.0,
        "worker_hnsw_sum": sum(worker_hnsw_times),
        "overhead_sequential": overhead,
        "wall_time_sequential": overhead + sum(worker_hnsw_times),
        "wall_time_parallel": _parallel_wall_time(worker_hnsw_times, overhead),
    }
    if extra:
        result.update(extra)
    return result


# --- CSV schema (shared by measure / estimate / plot) ---

PHASE_TO_CSV = {
    "hierarchical_kmeans": "phase_hierarchical_kmeans",
    "meta_sample_hnsw": "phase_meta_sample_hnsw",
    "kahip_partition": "phase_kahip_partition",
    "graph_partition": "phase_graph_partition",
    "assign_points": "phase_assign_points",
    "bucket_prep": "phase_bucket_prep",
    "nearest_points": "phase_nearest_points",
    "meta_hnsw": "phase_meta_hnsw",
    "data_split": "phase_data_split",
    "worker_hnsw_max": "phase_worker_hnsw_max",
}

METHOD_LABELS = {
    "shard": "Shard HNSW",
    "pyramid": "Pyramid",
    "titan": "TITAN",
}

SCALE_DATASETS = {
    "1m": ["deep1m", "sift1m"],
    "10m": ["deep10m", "sift10m"],
}

CSV_BASE_FIELDS = [
    "target_dataset",
    "num_vectors",
    "method",
    "method_label",
    "data_source",
    "source_dataset",
    "wall_time_parallel_sec",
    "wall_time_sequential_sec",
]


def profile_json_path(measure_dir: str, schema: str, method: str) -> str:
    return os.path.join(measure_dir, f"{schema}_{method}.json")


def load_profile_json(path: str) -> dict[str, Any]:
    import json

    with open(path) as f:
        return json.load(f)


def profile_to_csv_row(profile: dict[str, Any]) -> dict[str, Any]:
    """Convert a measured profile JSON to a standard CSV row."""
    phases = profile.get("phases", {})
    row = {
        "target_dataset": profile["dataset_schema"],
        "num_vectors": profile["num_vectors"],
        "method": profile["method"],
        "method_label": METHOD_LABELS.get(profile["method"], profile["method"]),
        "data_source": "measured",
        "source_dataset": profile["dataset_schema"],
        "wall_time_parallel_sec": round(profile["wall_time_parallel"], 2),
        "wall_time_sequential_sec": round(profile["wall_time_sequential"], 2),
    }
    for phase_key, csv_col in PHASE_TO_CSV.items():
        if phase_key == "worker_hnsw_max":
            continue
        val = phases.get(phase_key)
        if val is not None and float(val) > 0:
            row[csv_col] = round(float(val), 2)
    worker_max = profile.get("worker_hnsw_max", 0.0)
    if worker_max > 0:
        row["phase_worker_hnsw_max"] = round(float(worker_max), 2)
    return row


def estimate_to_csv_row(est: dict[str, Any]) -> dict[str, Any]:
    """Convert extrapolate_to_target() output to a standard CSV row."""
    row = {
        "target_dataset": est["target_schema"],
        "num_vectors": est["num_vectors"],
        "method": est["method"],
        "method_label": METHOD_LABELS.get(est["method"], est["method"]),
        "data_source": "estimated",
        "source_dataset": est["source_schema"],
        "wall_time_parallel_sec": round(est["wall_time_parallel_estimated"], 2),
        "wall_time_sequential_sec": round(est["wall_time_sequential_estimated"], 2),
    }
    for phase_key, val in est.get("estimated_phases", {}).items():
        csv_col = PHASE_TO_CSV.get(phase_key, f"phase_{phase_key}")
        if csv_col == "phase_worker_hnsw_max" or val > 0:
            row[csv_col] = round(float(val), 2)
    return row


def write_csv_rows(rows: list[dict[str, Any]], path: str) -> None:
    import csv

    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    phase_fields = sorted({k for row in rows for k in row if k.startswith("phase_")})
    fieldnames = CSV_BASE_FIELDS + phase_fields
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def profile_shard(
    data: np.ndarray,
    dataset_schema: str,
    num_workers: int = 16,
    M: int = DEFAULT_M,
    ef_construction: int = DEFAULT_EF,
    num_threads: int = DEFAULT_THREADS,
) -> dict[str, Any]:
    import time

    n = len(data)
    labels = np.arange(n)
    data_list = np.array_split(data, num_workers)
    labels_list = np.array_split(labels, num_workers)

    t0 = time.perf_counter()
    data_split = time.perf_counter() - t0

    worker_times: list[float] = []
    worker_sizes: list[int] = []
    for i in range(num_workers):
        node = Node.from_buckets(
            [Bucket.from_data(data_list[i], labels_list[i])],
            np.array([i]),
            np.array([i]),
        )
        t_start = time.perf_counter()
        _build_hnsw(
            node.data,
            node.absolute_label_indices,
            node.data.shape[1],
            M=M,
            ef_construction=ef_construction,
            num_threads=num_threads,
        )
        worker_times.append(time.perf_counter() - t_start)
        worker_sizes.append(len(node.data))
        gc.collect()

    phases = {
        "data_split": data_split,
        "worker_hnsw_total": sum(worker_times),
    }
    return _summarize_profile(
        "shard",
        dataset_schema,
        n,
        num_workers,
        phases,
        worker_times,
        {"worker_vector_counts": worker_sizes},
    )


def profile_pyramid(
    data: np.ndarray,
    dataset_schema: str,
    sample_num: int = 1_000_000,
    num_workers: int = 16,
    imblance_factor: float = 0.03,
    M: int = DEFAULT_M,
    ef_construction: int = DEFAULT_EF,
    num_threads: int = DEFAULT_THREADS,
    assign_probe_size: int = 100_000,
) -> dict[str, Any]:
    import time

    n = len(data)
    dim = data.shape[1]
    if sample_num > n:
        sample_num = n

    np.random.seed(42)
    t_start = time.perf_counter()
    sampled_indices = np.random.choice(n, size=sample_num, replace=False)
    sampled_points = data[sampled_indices]
    sample_ids = np.arange(sample_num, dtype=np.uint32)
    meta_hnsw = _build_hnsw(
        sampled_points,
        sample_ids,
        dim,
        M=M,
        ef_construction=ef_construction,
        num_threads=num_threads,
    )
    meta_sample_hnsw = time.perf_counter() - t_start

    t_start = time.perf_counter()
    k_neighbors = min(10, sample_num - 1)
    neighbors, _ = meta_hnsw.knn_query(
        sampled_points, k=k_neighbors, num_threads=num_threads
    )
    G = nx.Graph()
    G.add_nodes_from(range(sample_num))
    for i in range(sample_num):
        for neighbor_idx in neighbors[i]:
            if neighbor_idx != i:
                G.add_edge(i, int(neighbor_idx))
    kahip_graph = kahip.kahip_graph()
    kahip_graph.set_num_nodes(sample_num)
    for u, v in G.edges():
        kahip_graph.add_undirected_edge(u, v, 1)
        kahip_graph.set_weight(u, 1)
        kahip_graph.set_weight(v, 1)
    vwgt, xadj, adjcwgt, adjncy = kahip_graph.get_csr_arrays()
    _, blocks = kahip.kaffpa(
        vwgt, xadj, adjcwgt, adjncy, num_workers, imblance_factor, 0, 0, 0
    )
    sample_partition_labels = np.array(blocks, dtype=np.int32)
    kahip_partition = time.perf_counter() - t_start

    t_start = time.perf_counter()
    point_partition_labels = np.zeros(n, dtype=np.int32)
    point_partition_labels[sampled_indices] = sample_partition_labels
    non_sampled_mask = np.ones(n, dtype=bool)
    non_sampled_mask[sampled_indices] = False
    non_sampled_indices = np.where(non_sampled_mask)[0]
    num_unsampled = len(non_sampled_indices)
    if num_unsampled > 0:
        non_sampled_data = data[non_sampled_indices]
        batch_size = 10_000
        for i in range(0, len(non_sampled_data), batch_size):
            batch = non_sampled_data[i : i + batch_size]
            nearest_samples, _ = meta_hnsw.knn_query(
                batch, k=1, num_threads=num_threads
            )
            batch_indices = non_sampled_indices[i : i + batch_size]
            point_partition_labels[batch_indices] = sample_partition_labels[
                nearest_samples.flatten()
            ]
    assign_points = time.perf_counter() - t_start

    assign_per_point = None
    if num_unsampled == 0 and assign_probe_size > 0:
        probe_n = min(assign_probe_size, n)
        probe_data = data[:probe_n]
        t_probe = time.perf_counter()
        meta_hnsw.knn_query(probe_data, k=1, num_threads=num_threads)
        assign_per_point = (time.perf_counter() - t_probe) / probe_n
    elif num_unsampled > 0:
        assign_per_point = assign_points / num_unsampled

    worker_times: list[float] = []
    worker_sizes: list[int] = []
    for node_id in range(num_workers):
        node_mask = point_partition_labels == node_id
        node_data = data[node_mask]
        node_labels = np.where(node_mask)[0]
        if len(node_data) == 0:
            worker_times.append(0.0)
            worker_sizes.append(0)
            continue
        node = Node.from_buckets(
            [Bucket.from_data(node_data, node_labels)],
            np.array([node_id]),
            np.array([node_id]),
        )
        t_w = time.perf_counter()
        _build_hnsw(
            node.data,
            node.absolute_label_indices,
            dim,
            M=M,
            ef_construction=ef_construction,
            num_threads=num_threads,
        )
        worker_times.append(time.perf_counter() - t_w)
        worker_sizes.append(len(node.data))
        gc.collect()

    phases = {
        "meta_sample_hnsw": meta_sample_hnsw,
        "kahip_partition": kahip_partition,
        "assign_points": assign_points,
        "worker_hnsw_total": sum(worker_times),
    }
    return _summarize_profile(
        "pyramid",
        dataset_schema,
        n,
        num_workers,
        phases,
        worker_times,
        {
            "sample_num": sample_num,
            "num_unsampled_assigned": num_unsampled,
            "assign_per_point_sec": assign_per_point,
            "worker_vector_counts": worker_sizes,
        },
    )


def profile_titan(
    data: np.ndarray,
    dataset_schema: str,
    bucket_num: list[int],
    num_workers: int = 16,
    num_overlap: int = 2,
    bucket_cluster_method: Literal["graph_partition"] = "graph_partition",
    M: int = DEFAULT_M,
    ef_construction: int = DEFAULT_EF,
    num_threads: int = DEFAULT_THREADS,
    cache_dir: str | None = "../data/cluster_cache",
) -> dict[str, Any]:
    import time

    n = len(data)
    dim = data.shape[1]

    t_start = time.perf_counter()
    bucket_labels, bucket_centroids = hierarchical_KMeans(
        data,
        bucket_num,
        verbose=False,
        dataset_name=dataset_schema if cache_dir is not None else None,
        cache_dir=cache_dir,
    )
    hierarchical_kmeans = time.perf_counter() - t_start
    num_buckets = bucket_centroids.shape[0]

    t_start = time.perf_counter()
    if bucket_cluster_method == "graph_partition":
        bucket_node_assignments = graph_partition(
            bucket_centroids, num_workers, num_max_assignments=num_overlap
        )
    else:
        raise ValueError(f"Unsupported bucket_cluster_method: {bucket_cluster_method}")
    graph_partition_time = time.perf_counter() - t_start

    t_start = time.perf_counter()
    sort_indices = np.argsort(bucket_labels)
    sorted_bucket_labels = bucket_labels[sort_indices]
    unique_buckets, bucket_starts = np.unique(sorted_bucket_labels, return_index=True)
    bucket_ends = np.append(bucket_starts[1:], len(sorted_bucket_labels))
    bucket_data_map: dict[int, np.ndarray] = {}
    for i, bucket_idx in enumerate(unique_buckets):
        start = bucket_starts[i]
        end = bucket_ends[i]
        bucket_data_map[int(bucket_idx)] = sort_indices[start:end]
    bucket_prep = time.perf_counter() - t_start

    worker_times: list[float] = []
    worker_sizes: list[int] = []
    for i in range(num_workers):
        node_bucket_indices = [
            bucket_idx
            for bucket_idx in range(num_buckets)
            if i in bucket_node_assignments[bucket_idx]
        ]
        node_buckets = []
        for bucket_idx in node_bucket_indices:
            if bucket_idx not in bucket_data_map:
                continue
            bucket_indices = bucket_data_map[bucket_idx]
            node_buckets.append(Bucket.from_data(data[bucket_indices], bucket_indices))
        if not node_buckets:
            worker_times.append(0.0)
            worker_sizes.append(0)
            continue
        node_bucket_indices_arr = np.array(node_bucket_indices)
        node = Node.from_buckets(
            node_buckets,
            node_bucket_indices_arr,
            bucket_centroids[node_bucket_indices_arr],
        )
        t_w = time.perf_counter()
        _build_hnsw(
            node.data,
            node.absolute_label_indices,
            dim,
            M=M,
            ef_construction=ef_construction,
            num_threads=num_threads,
        )
        worker_times.append(time.perf_counter() - t_w)
        worker_sizes.append(len(node.data))
        gc.collect()

    t_start = time.perf_counter()
    nearest_point_labels = np.zeros(num_buckets, dtype=np.int32)
    bucket_radiuses = np.zeros(num_buckets, dtype=np.float32)
    for b in range(int(bucket_labels.max()) + 1):
        if b not in bucket_data_map:
            nearest_point_labels[b] = -1
            continue
        current_bucket_indices = bucket_data_map[b]
        bucket_data = data[current_bucket_indices]
        if len(bucket_data) == 0:
            continue
        nearest_point_index, furthest_distance = (
            np_find_nearest_point_idx_and_furthest_distance(
                bucket_data,
                bucket_centroids[b],
                indices=current_bucket_indices,
                space="l2",
            )
        )
        nearest_point_labels[b] = nearest_point_index
        bucket_radiuses[b] = furthest_distance
    nearest_points = time.perf_counter() - t_start

    t_start = time.perf_counter()
    _build_hnsw(
        bucket_centroids,
        np.arange(num_buckets, dtype=np.int32),
        dim,
        M=M,
        ef_construction=ef_construction,
        num_threads=num_threads,
    )
    meta_hnsw = time.perf_counter() - t_start

    phases = {
        "hierarchical_kmeans": hierarchical_kmeans,
        "graph_partition": graph_partition_time,
        "bucket_prep": bucket_prep,
        "nearest_points": nearest_points,
        "meta_hnsw": meta_hnsw,
        "worker_hnsw_total": sum(worker_times),
    }
    return _summarize_profile(
        "titan",
        dataset_schema,
        n,
        num_workers,
        phases,
        worker_times,
        {
            "num_buckets": num_buckets,
            "num_overlap": num_overlap,
            "worker_vector_counts": worker_sizes,
        },
    )


def hnsw_time_scale(t: float, n_src: int, n_tgt: int) -> float:
    """Scale HNSW build time assuming O(n log n) complexity."""
    if t <= 0 or n_src <= 0 or n_tgt <= 0:
        return 0.0
    if n_src == n_tgt:
        return t
    ratio = n_tgt / n_src
    log_ratio = math.log2(n_tgt) / math.log2(n_src)
    return t * ratio * log_ratio


def linear_time_scale(t: float, ratio: float) -> float:
    return t * ratio if t > 0 else 0.0


def bucket_count(schema: str) -> int:
    """Return total bucket count from parameters.yaml bucket_num lists."""
    bucket_nums = {
        "deep1m": [100, 1000],
        "deep10m": [1000, 1000],
        "deep100m_1mbkt": [1000, 1000],
        "sift1m": [100, 1000],
        "sift10m": [1000, 1000],
        "sift100m_1mbkt": [1000, 1000],
    }
    if schema not in bucket_nums:
        raise KeyError(f"Unknown schema for bucket count: {schema}")
    total = 1
    for n in bucket_nums[schema]:
        total *= n
    return total


def vector_count(schema: str) -> int:
    counts = {
        "deep1m": 1_000_000,
        "deep10m": 10_000_000,
        "deep100m_1mbkt": 100_000_000,
        "sift1m": 1_000_000,
        "sift10m": 10_000_000,
        "sift100m_1mbkt": 100_000_000,
    }
    if schema not in counts:
        raise KeyError(f"Unknown schema for vector count: {schema}")
    return counts[schema]


def extrapolate_to_target(
    profile: dict[str, Any],
    source_schema: str,
    target_schema: str,
    num_workers: int = 16,
    pyramid_sample_num: int = 1_000_000,
) -> dict[str, Any]:
    """Extrapolate measured profile from source_schema to target_schema."""
    src_n = vector_count(source_schema)
    tgt_n = vector_count(target_schema)
    n_ratio = tgt_n / src_n
    src_buckets = bucket_count(source_schema)
    tgt_buckets = bucket_count(target_schema)
    bucket_ratio = tgt_buckets / src_buckets
    per_worker_src = src_n / num_workers
    per_worker_tgt = tgt_n / num_workers

    method = profile["method"]
    phases = profile["phases"]
    worker_max = profile["worker_hnsw_max"]
    estimated_phases: dict[str, float] = {}

    if method == "shard":
        estimated_phases["data_split"] = phases.get("data_split", 0.0)
        worker_est = hnsw_time_scale(worker_max, int(per_worker_src), int(per_worker_tgt))
        estimated_phases["worker_hnsw_max"] = worker_est
        overhead = estimated_phases["data_split"]
        wall_parallel = overhead + worker_est

    elif method == "pyramid":
        sample_num = min(pyramid_sample_num, tgt_n)
        meta_est = hnsw_time_scale(
            phases.get("meta_sample_hnsw", 0.0),
            min(profile.get("sample_num", sample_num), src_n),
            sample_num,
        )
        kahip_est = linear_time_scale(
            phases.get("kahip_partition", 0.0),
            sample_num / min(profile.get("sample_num", src_n), src_n),
        )
        measured_assign = phases.get("assign_points", 0.0)
        num_unsampled_src = profile.get("num_unsampled_assigned", 0)
        assign_per_point = profile.get("assign_per_point_sec")
        tgt_unsampled = max(tgt_n - sample_num, 0)
        if num_unsampled_src > 0:
            assign_est = linear_time_scale(measured_assign, tgt_unsampled / num_unsampled_src)
        elif assign_per_point is not None:
            assign_est = assign_per_point * tgt_unsampled
        else:
            assign_est = linear_time_scale(measured_assign, n_ratio)
        worker_est = hnsw_time_scale(worker_max, int(per_worker_src), int(per_worker_tgt))
        estimated_phases.update(
            {
                "meta_sample_hnsw": meta_est,
                "kahip_partition": kahip_est,
                "assign_points": assign_est,
                "worker_hnsw_max": worker_est,
            }
        )
        overhead = meta_est + kahip_est + assign_est
        wall_parallel = overhead + worker_est

    elif method == "titan":
        kmeans_est = linear_time_scale(
            phases.get("hierarchical_kmeans", 0.0), n_ratio * bucket_ratio
        )
        partition_est = hnsw_time_scale(
            phases.get("graph_partition", 0.0), src_buckets, tgt_buckets
        )
        bucket_prep_est = linear_time_scale(phases.get("bucket_prep", 0.0), n_ratio)
        nearest_est = linear_time_scale(
            phases.get("nearest_points", 0.0), bucket_ratio
        )
        meta_est = hnsw_time_scale(
            phases.get("meta_hnsw", 0.0), src_buckets, tgt_buckets
        )
        worker_est = hnsw_time_scale(worker_max, int(per_worker_src), int(per_worker_tgt))
        estimated_phases.update(
            {
                "hierarchical_kmeans": kmeans_est,
                "graph_partition": partition_est,
                "bucket_prep": bucket_prep_est,
                "nearest_points": nearest_est,
                "meta_hnsw": meta_est,
                "worker_hnsw_max": worker_est,
            }
        )
        overhead = (
            kmeans_est + partition_est + bucket_prep_est + nearest_est + meta_est
        )
        wall_parallel = overhead + worker_est
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "method": method,
        "source_schema": source_schema,
        "target_schema": target_schema,
        "num_vectors": tgt_n,
        "num_workers": num_workers,
        "estimated_phases": estimated_phases,
        "wall_time_parallel_estimated": wall_parallel,
        "wall_time_sequential_estimated": overhead
        + linear_time_scale(profile["worker_hnsw_sum"], n_ratio * (math.log2(per_worker_tgt) / math.log2(per_worker_src) if per_worker_src > 1 else 1)),
        "scaling_notes": {
            "vector_ratio": n_ratio,
            "bucket_ratio": bucket_ratio,
            "hnsw_model": "O(n log n)",
            "parallel_model": "max(worker_hnsw) + sequential_overhead",
        },
    }
