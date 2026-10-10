import struct
import threading
from queue import Queue
from typing import Optional

import numpy as np
from node import Node


def save_node_data(node: Node, save_path: str):
    """Save node data to a binary file that can be read by Go.

    File format:
    - Header:
        - Dimensions (uint32): number of dimensions in data
        - Data count (uint32): number of data points
        - Bucket indices count (uint32)
        - Absolute label indices count (uint32)
    - Data arrays:
        - Bucket indices (int32 array)
        - Absolute label indices (int32 array)
        - Data size per bucket (int32 array)
        - Bucket data (float32 array)
    """

    # Prepare header values
    dimensions = node.data.shape[1]
    data_count = len(node.data)
    bucket_indices_count = len(node.bucket_indices)
    absolute_indices_count = len(node.absolute_label_indices)
    data_size_per_bucket_count = len(node.data_size_per_bucket)

    print("Saving node data:")
    print(f"- Dimensions: {dimensions}")
    print(f"- Data count: {data_count}")
    print(
        f"- Array sizes: bucket_indices={bucket_indices_count}, absolute_indices={absolute_indices_count}, data_size_per_bucket={data_size_per_bucket_count}"
    )

    with open(save_path, "wb") as f:
        # Write header using little-endian format
        f.write(
            struct.pack(
                "<IIIII",  # Little-endian, 4 uint32
                dimensions,
                data_count,
                bucket_indices_count,
                absolute_indices_count,
                data_size_per_bucket_count,
            )
        )

        # Convert arrays to correct type and ensure C-contiguous memory layout
        bucket_indices_int32 = node.bucket_indices.astype(
            np.int32, order="C", copy=False
        )
        absolute_label_indices_int32 = node.absolute_label_indices.astype(
            np.int32, order="C", copy=False
        )
        data_size_per_bucket_int32 = node.data_size_per_bucket.astype(
            np.int32, order="C", copy=False
        )

        # Convert bucket data to float32
        bucket_data_float32_flat = np.concatenate(
            [
                node.bucket(i).astype(np.float32, order="C", copy=False)
                for i in range(node.num_buckets)
            ]
        )

        # Write data arrays in little-endian format
        f.write(bucket_indices_int32.tobytes())
        f.write(absolute_label_indices_int32.tobytes())
        f.write(data_size_per_bucket_int32.tobytes())
        f.write(bucket_data_float32_flat.tobytes())

        # Verify file size
        file_size = f.tell()
        expected_size = (
            5 * 4  # Header (5 uint32s)
            + bucket_indices_count * 4  # Bucket indices (int32)
            + absolute_indices_count * 4  # Absolute label indices (int32)
            + data_size_per_bucket_count * 4  # Data size per bucket (int32)
            + bucket_data_float32_flat.size * 4  # Bucket data (float32)
        )

        if file_size != expected_size:
            raise ValueError(
                f"File size mismatch: got {file_size}, expected {expected_size}"
            )

    print(f"Successfully saved node data to {save_path}")


def save_node_data_streamed(
    save_path: str,
    dim: int,
    bucket_ids: np.ndarray,
    bucket_indices_list: list,
    data: np.ndarray,
    on_bucket_data=None,
    chunk_rows: int = None,
):
    """Memory-lean variant of save_node_data.

    Writes the exact same file format but never materialises the whole node in
    RAM: header and index arrays are written from cheap per-bucket counts, then
    each bucket's data is gathered from ``data`` (typically a memmap) in chunks
    of ``chunk_rows`` rows, handed to ``on_bucket_data(chunk, ids)`` (used to
    feed hnswlib incrementally), written to the file, and released.

    Args:
        bucket_ids: bucket id of each entry in bucket_indices_list (ascending).
        bucket_indices_list: per-bucket arrays of global vector indices.
        data: the full base array (memmap ok); must be indexable by those.
        on_bucket_data: optional callback(chunk_data, chunk_ids) per chunk,
            invoked on the single gather so HNSW build and file write share it.
        chunk_rows: rows per gather chunk; None = one chunk per bucket.
    """
    bucket_ids = np.asarray(bucket_ids)
    sizes = np.array([len(ix) for ix in bucket_indices_list], dtype=np.int32)
    num_buckets = len(bucket_indices_list)
    data_count = int(sizes.sum())

    print("Saving node data (streamed):")
    print(f"- Dimensions: {dim}")
    print(f"- Data count: {data_count}")
    print(f"- Buckets: {num_buckets}")

    with open(save_path, "wb") as f:
        f.write(
            struct.pack(
                "<IIIII",
                dim,
                data_count,
                num_buckets,       # bucket_indices count
                data_count,        # absolute label indices count
                num_buckets,       # data_size_per_bucket count
            )
        )
        f.write(bucket_ids.astype(np.int32, order="C", copy=False).tobytes())
        for ix in bucket_indices_list:
            f.write(np.asarray(ix).astype(np.int32, order="C", copy=False).tobytes())
        f.write(sizes.tobytes())

        # Merge consecutive small buckets into ~TARGET_UNIT_ROWS work units:
        # one gather + one hnswlib add_items + one write per unit. With
        # ~1000-row buckets (sift1b: 101M vectors / 96k buckets per node),
        # per-bucket units meant ~100k tiny add_items calls -- each paying
        # hnswlib's thread-pool spin-up -- while gather and write stayed
        # single-threaded and serialized (process idled at ~9% CPU).
        # Units never split a bucket and rows keep bucket order, so the
        # output file stays byte-identical.
        TARGET_UNIT_ROWS = chunk_rows if chunk_rows is not None else 2_000_000
        units = []  # (start_bucket, end_bucket)
        unit_start = 0
        unit_rows = 0
        for b in range(num_buckets):
            n = int(sizes[b])
            if unit_rows > 0 and unit_rows + n > TARGET_UNIT_ROWS:
                units.append((unit_start, b))
                unit_start = b
                unit_rows = 0
            unit_rows += n
        if unit_start < num_buckets:
            units.append((unit_start, num_buckets))

        # Prefetch pipeline: gather unit N+1 on a worker thread while unit N
        # feeds hnswlib (CPU) and is written to disk, overlapping random
        # memmap reads with index construction.
        q = Queue(maxsize=1)

        def _gather(unit):
            b0, b1 = unit
            parts = [np.asarray(bucket_indices_list[b]) for b in range(b0, b1)]
            idx = np.concatenate(parts) if len(parts) > 1 else parts[0]
            idx = np.ascontiguousarray(idx, dtype=np.int64)
            chunk = np.ascontiguousarray(data[idx], dtype=np.float32)
            return idx, chunk

        def _producer():
            for unit in units:
                q.put(_gather(unit))
            q.put(None)

        th = threading.Thread(target=_producer, daemon=True)
        th.start()

        done_rows = 0
        while True:
            item = q.get()
            if item is None:
                break
            idx, chunk = item
            if on_bucket_data is not None:
                on_bucket_data(chunk, idx)
            f.write(chunk.tobytes())
            done_rows += chunk.shape[0]
            if done_rows // 10_000_000 != (done_rows - chunk.shape[0]) // 10_000_000:
                print(f"  streamed {done_rows:,}/{data_count:,} rows", flush=True)
            del idx, chunk

        file_size = f.tell()
        expected_size = (
            5 * 4
            + num_buckets * 4          # bucket indices
            + data_count * 4           # absolute label indices
            + num_buckets * 4          # data size per bucket
            + data_count * dim * 4     # bucket data
        )
        if file_size != expected_size:
            raise ValueError(
                f"File size mismatch: got {file_size}, expected {expected_size}"
            )

    print(f"Successfully saved node data to {save_path}")


def save_meta_index_data(
    bucket_centroids: np.ndarray,
    bucket_node_labels: np.ndarray,
    nearest_point_labels: np.ndarray,
    bucket_radiuses: np.ndarray,
    save_path: str,
    bucket_sizes: Optional[np.ndarray] = None,
    total_num_vectors: Optional[int] = None,
):
    """
    Save meta index data to a binary file that can be read by Go.
    File format:
    - Header:
        - Dimensions (uint32): number of dimensions in data
        - Data count (uint32): number of buckets
    - Data arrays:
        - Bucket node labels (uint32 array)
        - Bucket centroids (float32 array)
    """

    print("Num Buckets:", bucket_centroids.shape[0])
    print("Dim:", bucket_centroids.shape[1])

    with open(save_path, "wb") as f:
        # Write dimensions
        f.write(
            struct.pack(
                "III",
                bucket_centroids.shape[0],  # num_buckets
                bucket_centroids.shape[1],  # dim
                bucket_node_labels.shape[
                    1
                ],  # max num_nodes for adaptive kmeans and fixed num_nodes for kmeans
            )
        )

        # Write bucket node labels as uint32
        f.write(bucket_node_labels.astype(np.uint32).tobytes())

        # Write bucket centroids as float32
        f.write(bucket_centroids.astype(np.float32).tobytes())
        f.write(nearest_point_labels.astype(np.int32).tobytes())
        f.write(bucket_radiuses.astype(np.float32).tobytes())

        if bucket_sizes is not None:
            sizes = bucket_sizes.astype(np.uint32, order="C", copy=False)
            if sizes.shape[0] != bucket_centroids.shape[0]:
                raise ValueError(
                    f"bucket_sizes length {sizes.shape[0]} != num_buckets {bucket_centroids.shape[0]}"
                )
            f.write(sizes.tobytes())
            if total_num_vectors is None:
                total_num_vectors = int(sizes.sum())
            f.write(struct.pack("<I", int(total_num_vectors)))

def write_fvecs(filename: str, data: np.ndarray):
    """
    将numpy数组写入fvecs格式文件
    
    Args:
        filename: 输出文件名
        data: numpy数组，形状为(n_vectors, dimension)
    """
    print(f"正在写入fvecs文件: {filename}")
    print(f"数据形状: {data.shape}")
    
    with open(filename, 'wb') as f:
        for vector in data:
            # 写入维度信息 (4字节)
            dim = len(vector)
            f.write(struct.pack('i', dim))
            # 写入向量数据 (4*dim字节)
            f.write(struct.pack('f' * dim, *vector))
    
    print(f"成功写入 {len(data)} 个向量到 {filename}")

    
def write_ivecs(filename: str, data: np.ndarray):
    """
    将numpy数组写入ivecs格式文件
    
    Args:
        filename: 输出文件名
        data: numpy数组，形状为(n_vectors, dimension)
    """
    print(f"正在写入ivecs文件: {filename}")
    print(f"数据形状: {data.shape}")
    
    with open(filename, 'wb') as f:
        for vector in data:
            # 写入维度信息 (4字节)
            dim = len(vector)
            f.write(struct.pack('i', dim))
            # 写入向量数据 (4*dim字节)
            f.write(struct.pack('i' * dim, *vector))
    
    print(f"成功写入 {len(data)} 个向量到 {filename}")