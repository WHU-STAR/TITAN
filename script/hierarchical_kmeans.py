import os
import numpy as np
from typing import Union
import h5py
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm
from utils import Timer
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

# Buckets larger than this that also span the entire dataset (i.e. the level-0
# bucket at 1B scale) are clustered by streaming over the memmap instead of
# gathering all of X into RAM (which would OOM). 20M keeps every real subset
# bucket on the fast eager path; only the full-dataset bucket is streamed.
# Override via env var HK_STREAM_THRESHOLD (e.g. for forcing it on small data
# during testing).
STREAM_THRESHOLD = int(os.environ.get("HK_STREAM_THRESHOLD", 20_000_000))


def _stream_kmeans_full(X, cluster_count, fit_sample=5_000_000, pred_chunk=200_000, seed=0):
    """Cluster a huge full-dataset bucket without materializing all of X.

    Fits MiniBatchKMeans on a random subsample (only that much is paged in),
    then assigns every row by predicting over the memmap-backed X in contiguous
    chunks. Returns (labels[int32, len N], centers[n_clusters, dim]).

    For 1B x 128 float32 this peaks at ~8 GiB (4 GiB label array + ~2.5 GiB
    fit sample + predict temp) instead of a 512 GiB full gather.
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    fit_n = min(n, fit_sample)
    # Sample with replacement: cheap (no N-sized permutation); duplicates are
    # negligible at fit_n/n and harmless for centroid seeding.
    sample_idx = rng.integers(0, n, size=fit_n)
    sample = X[sample_idx]  # pages ~fit_n*dim*4 bytes (2.5 GiB for 5M x 128)
    km = MiniBatchKMeans(
        n_clusters=cluster_count,
        n_init="auto",
        random_state=seed,
        batch_size=min(1024, fit_n),
    )
    km.fit(sample)
    centers = km.cluster_centers_

    labels = np.empty(n, dtype=np.int32)
    for s in tqdm(range(0, n, pred_chunk), desc="stream-predict", leave=False):
        labels[s : s + pred_chunk] = km.predict(X[s : s + pred_chunk])
    return labels, centers


def hierarchical_KMeans(
    X: np.ndarray,
    center_count_ls: list,
    reused_labels: Union[np.ndarray, None] = None,
    verbose: bool = False,
    cache_dir: str = "../data/cluster_cache",
    dataset_name: str = None,
):
    """
    Hierarchical KMeans with HDF5-based persistent caching.
    - cache_dir: directory to store cache files
    - dataset_name: used for cache file naming
    """
    
    def generate_cache_file_name(dataset_name, center_count_ls):
        return os.path.join(
            cache_dir,
            f"{dataset_name}_hier_kmeans_{'-'.join(map(str, center_count_ls))}.h5"
        )

    do_reuse_cache = (dataset_name is not None and cache_dir is not None and reused_labels is None)
    reused_center_count_ls = []
    if do_reuse_cache:
        os.makedirs(cache_dir, exist_ok=True)
        center_count_ls_len = len(center_count_ls)
        cache_reuseable = False
        # find the nearest common ancestor
        for i in range(len(center_count_ls)):
            upperbound = center_count_ls_len - i
            cache_file = generate_cache_file_name(dataset_name, center_count_ls[:upperbound])
            if os.path.exists(cache_file):
                reused_center_count_ls = center_count_ls[:upperbound]
                cache_reuseable = True
                break
    
        if cache_reuseable:
            print(f"Reused hierarchical KMeans result from cache: {cache_file}")
            with h5py.File(cache_file, "r") as f:
                reused_labels = np.array(f["bucket_labels"][:])
                centers = np.array(f["centers"][:])
        else:
            do_reuse_cache = False

    print("using cache" if do_reuse_cache else "not using cache")
    timer = Timer()
    timer.tick()
    if not do_reuse_cache:
        centers = np.zeros(X.shape[1])
    if reused_labels is None:
        bucket_labels = np.zeros(X.shape[0]).astype(int)
    else:
        bucket_labels = reused_labels.copy()
    for level, i in enumerate(center_count_ls):
        if level < len(reused_center_count_ls):
            # no need to compute this level
            continue
        if verbose:
            print(f"Calculating for center count of {i}")

        new_labels = np.zeros(X.shape[0], dtype=np.int64)
        new_centers_list = []
        # Sort once so each bucket's members are contiguous, then iterate by
        # slice. This replaces a per-bucket `bucket_labels == j` full-N mask
        # scan (O(B*N) per level) and the in-loop `np.concatenate` of centers
        # (O(n^2)). Together these dominated the 1B build (~3.8 days on level-1)
        # and the output is byte-identical to the previous version.
        sort_idx = np.argsort(bucket_labels, kind="stable")
        sorted_labels = bucket_labels[sort_idx]
        unique_buckets, starts = np.unique(sorted_labels, return_index=True)
        ends = np.append(starts[1:], sorted_labels.shape[0])

        label_offset = 0
        for bi in tqdm(range(unique_buckets.shape[0])):
            member_idx = sort_idx[starts[bi] : ends[bi]]
            applied_vectors_count = int(ends[bi] - starts[bi])
            cluster_count = min(i, applied_vectors_count)

            if (
                applied_vectors_count > STREAM_THRESHOLD
                and applied_vectors_count == X.shape[0]
            ):
                # Full-dataset bucket (level 0 at 1B): stream over the memmap,
                # never gather all of X. cluster_count << n here, so every
                # cluster is populated -> keep all centers (no remap needed).
                _labels, _centers = _stream_kmeans_full(X, cluster_count)
                unique_labels = np.arange(cluster_count)
                # _labels is indexed by original row; member_idx is all rows.
                new_labels[member_idx] = _labels[member_idx] + label_offset
            else:
                bucket_data = X[member_idx]
                if cluster_count > 5000:
                    kmeans = MiniBatchKMeans(
                        n_clusters=cluster_count, n_init="auto", random_state=0
                    )
                else:
                    kmeans = KMeans(n_clusters=cluster_count, n_init="auto", random_state=0)
                _labels = kmeans.fit_predict(bucket_data)
                _centers = kmeans.cluster_centers_
                unique_labels = np.unique(_labels)          # 这里依赖于一个特性：unique_labels是升序的
                if unique_labels.shape[0] < cluster_count:
                    print("Warning: KMeans produced fewer clusters than requested.")
                    # 实际聚类数量小于目标数量，此时中心点仍然是目标数量，需要进一步处理
                    _centers = _centers[unique_labels]      # 只保留实际存在的中心点，下标改变
                    _labels = np.searchsorted(unique_labels, _labels)  # 将标签重新映射到新的下标
                new_labels[member_idx] = _labels + label_offset

            new_centers_list.append(_centers)
            label_offset += unique_labels.shape[0]

        bucket_labels = new_labels
        centers = (
            np.vstack(new_centers_list) if new_centers_list else np.empty((0, X.shape[1]))
        )
        # Save this level's result to cache
        if dataset_name is not None and cache_dir is not None:
            level_cache_file = generate_cache_file_name(
                dataset_name, center_count_ls[: level + 1]
            )
            with h5py.File(level_cache_file, "w") as f:
                f.create_dataset("bucket_labels", data=bucket_labels)
                f.create_dataset("centers", data=centers)
            print(f"Saved level {level} KMeans result to cache: {level_cache_file}")

    print(
        f"Reused labels: {reused_labels.max() + 1 if reused_labels is not None else 'No'}",
    )
    timer.tuck("Calculate residual norm")
    final_ulabels = np.unique(bucket_labels)
    assert final_ulabels.shape[0] == bucket_labels.max() + 1, "Labels are not continuous"
    assert bucket_labels.max() + 1 == centers.shape[0], "Labels and centers size mismatch"
    return bucket_labels, centers

def _process_bucket_kmeans_parallel(args):
    """Helper function for parallel processing of bucket KMeans"""
    bucket_id, bucket_data, cluster_count, random_state = args
    
    if cluster_count <= 0 or len(bucket_data) == 0:
        return bucket_id, np.array([]), np.zeros(len(bucket_data), dtype=int)
    
    # Use MiniBatchKMeans for large clusters to improve performance
    if cluster_count > 5000:
        kmeans = MiniBatchKMeans(
            n_clusters=cluster_count, 
            n_init="auto", 
            random_state=random_state,
            batch_size=min(1000, len(bucket_data))
        )
    else:
        kmeans = KMeans(
            n_clusters=cluster_count, 
            n_init="auto", 
            random_state=random_state
        )
    
    try:
        labels = kmeans.fit_predict(bucket_data)
        centers = kmeans.cluster_centers_
        
        # Handle case where actual clusters < target clusters
        if labels.max() + 1 < cluster_count:
            cluster_count = labels.max() + 1
            centers = centers[:cluster_count]
            
        return bucket_id, centers, labels
    except Exception as e:
        print(f"Error processing bucket {bucket_id}: {e}")
        return bucket_id, np.array([]), np.zeros(len(bucket_data), dtype=int)


def hierarchical_KMeans_opt(
    X: np.ndarray,
    center_count_ls: list,
    reused_labels: Union[np.ndarray, None] = None,
    verbose: bool = False,
    cache_dir: str = "../data/cluster_cache",
    dataset_name: str = None,
    n_jobs: int = None,
    batch_size: int = 1000,
):
    """
    Optimized Hierarchical KMeans with parallel processing and batch processing.
    
    Key optimizations:
    1. Parallel processing of buckets using ThreadPoolExecutor
    2. Batch processing to manage memory usage
    3. Improved memory efficiency with reduced array operations
    4. Smart caching strategy
    
    Args:
        X: Input data array
        center_count_ls: List of center counts for each level
        reused_labels: Previously computed labels to reuse
        verbose: Whether to print progress information
        cache_dir: Directory to store cache files
        dataset_name: Used for cache file naming
        n_jobs: Number of parallel jobs (default: cpu_count())
        batch_size: Batch size for processing buckets
    """
    
    def generate_cache_file_name(dataset_name, center_count_ls):
        return os.path.join(
            cache_dir,
            f"{dataset_name}_hier_kmeans_{'-'.join(map(str, center_count_ls))}.h5"
        )

    # Set default number of jobs
    if n_jobs is None:
        n_jobs = min(cpu_count(), 8)  # Limit to 8 to avoid memory issues
    
    do_reuse_cache = (dataset_name is not None and cache_dir is not None and reused_labels is None)
    reused_center_count_ls = []
    if do_reuse_cache:
        os.makedirs(cache_dir, exist_ok=True)
        center_count_ls_len = len(center_count_ls)
        cache_reuseable = False
        # find the nearest common ancestor
        for i in range(len(center_count_ls)):
            upperbound = center_count_ls_len - i
            cache_file = generate_cache_file_name(dataset_name, center_count_ls[:upperbound])
            if os.path.exists(cache_file):
                reused_center_count_ls = center_count_ls[:upperbound]
                cache_reuseable = True
                break
    
        if cache_reuseable:
            print(f"Reused hierarchical KMeans result from cache: {cache_file}")
            with h5py.File(cache_file, "r") as f:
                reused_labels = np.array(f["bucket_labels"][:])
                centers = np.array(f["centers"][:])
        else:
            do_reuse_cache = False

    print("using cache" if do_reuse_cache else "not using cache")
    timer = Timer()
    timer.tick()
    
    if not do_reuse_cache:
        centers = np.zeros(X.shape[1])
    if reused_labels is None:
        bucket_labels = np.zeros(X.shape[0]).astype(int)
    else:
        bucket_labels = reused_labels.copy()
    
    for level, i in enumerate(center_count_ls):
        if level < len(reused_center_count_ls):
            # no need to compute this level
            continue
            
        if verbose:
            print(f"Calculating for center count of {i}")
        
        # Get unique bucket labels and their counts
        unique_buckets, bucket_counts = np.unique(bucket_labels, return_counts=True)
        num_buckets = len(unique_buckets)
        
        if verbose:
            print(f"Processing {num_buckets} buckets with {n_jobs} parallel jobs")
        
        # Process buckets in parallel with batch processing
        new_labels = np.zeros(X.shape[0]).astype(int)
        new_centers = []
        label_offset = 0
        
        # Process in batches to manage memory
        for batch_start in range(0, num_buckets, batch_size):
            batch_end = min(batch_start + batch_size, num_buckets)
            batch_buckets = unique_buckets[batch_start:batch_end]
            
            # Prepare arguments for parallel processing
            args_list = []
            for bucket_id in batch_buckets:
                mask = bucket_labels == bucket_id
                bucket_data = X[mask]
                cluster_count = min(i, len(bucket_data))
                args_list.append((bucket_id, bucket_data, cluster_count, 0))
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                results = list(tqdm(
                    executor.map(_process_bucket_kmeans_parallel, args_list),
                    total=len(args_list),
                    desc=f"Processing buckets {batch_start}-{batch_end}"
                ))
            
            # Process results
            for bucket_id, centers_result, labels_result in results:
                if len(centers_result) > 0:
                    # Find the original indices for this bucket
                    bucket_mask = bucket_labels == bucket_id
                    bucket_indices = np.where(bucket_mask)[0]
                    
                    # Update labels with offset
                    new_labels[bucket_indices] = labels_result + label_offset
                    
                    # Store centers
                    new_centers.append(centers_result)
                    
                    # Update label offset
                    label_offset += len(centers_result)
        
        # Convert centers list to array efficiently
        if new_centers:
            centers = np.vstack(new_centers)
        else:
            centers = np.empty((0, X.shape[1]))
        
        bucket_labels = new_labels
        
        # Save this level's result to cache
        if dataset_name is not None and cache_dir is not None:
            level_cache_file = generate_cache_file_name(
                dataset_name, center_count_ls[: level + 1]
            )
            with h5py.File(level_cache_file, "w") as f:
                f.create_dataset("bucket_labels", data=bucket_labels)
                f.create_dataset("centers", data=centers)
            print(f"Saved level {level} KMeans result to cache: {level_cache_file}")

    print(
        f"Reused labels: {reused_labels.max() + 1 if reused_labels is not None else 'No'}",
    )
    timer.tuck("Calculate residual norm")
    assert bucket_labels.max() + 1 == centers.shape[0], "Labels and centers size mismatch"
    return bucket_labels, centers
