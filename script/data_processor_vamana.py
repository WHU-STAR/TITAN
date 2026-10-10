import os
import yaml
from typing import Literal, Union
from node import Node, Bucket
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm
from data_loader import get_dataset, get_data_paths
from utils import Timer
from pyfzf.pyfzf import FzfPrompt
import torch
import pickle
import struct
import vamanalib


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

NUM_OVERLAP = 3

space_hnsw_to_torch = {
    "l2": "euclidean",
    "cosine": "cosine",
}


def find_nearest_point(
    data: np.ndarray, target: np.ndarray, space: str = "l2"
) -> np.ndarray:
    """
    Find the nearest point in data to the target point
    """
    if space == "l2":
        distances = np.sum((data - target) ** 2, axis=1)
    elif space == "cosine":
        data_norm = np.linalg.norm(data, axis=1)
        target_norm = np.linalg.norm(target)
        distances = 1 - np.dot(data, target) / (data_norm * target_norm)
    else:
        raise NotImplementedError(f"Distance metric {space} not implemented")

    nearest_idx = np.argmin(distances)
    return data[nearest_idx]


def hierarchical_KMeans(
    X: np.ndarray,
    center_count_ls: list,
    reused_labels: Union[np.ndarray, None] = None,
    verbose: bool = False,
):
    """
    Hierarchical KMeans
    examples:
    labels1, _ = hierarchical_KMeans(data_with_10m_elements, [100, 100], None, true)        # produces a KMeans result containing 100 x 100 = 10000 centers
    final_labels = hierarchical_KMeans(data_with_10m_elements, [100], labels1, False)        # produces a KMeans result reusing previous labels, resulting in about 10000 * 100 = 1m centers(some buckets have less than 100 elements will automatically have fewer centers)
    """
    timer = Timer()
    timer.tick()
    centers = np.zeros(X.shape[1])
    if reused_labels is None:
        labels = np.zeros(X.shape[0]).astype(int)
    else:
        labels = reused_labels.copy()
    for i in center_count_ls:
        label_offset = 0
        new_centers = np.empty((0, X.shape[1]))
        new_labels = np.zeros(X.shape[0]).astype(int)
        if verbose:
            print(f"Calculating for center count of {i}")
        for j in tqdm(range(int(labels.max() + 1))):
            current_applied_vectors_mask = labels == j
            current_applied_vectors = X[current_applied_vectors_mask]
            applied_vectors_count = current_applied_vectors.shape[0]
            cluster_count = min(i, applied_vectors_count)
            if cluster_count > 5000:
                kmeans = MiniBatchKMeans(
                    n_clusters=cluster_count, n_init="auto", random_state=0
                )
            else:
                kmeans = KMeans(n_clusters=cluster_count, n_init="auto", random_state=0)
            _labels = kmeans.fit_predict(current_applied_vectors)
            _centers = kmeans.cluster_centers_

            # Replace each centroid with its nearest real point
            _real_centers = np.zeros_like(_centers)
            for k in range(cluster_count):
                cluster_data = current_applied_vectors[_labels == k]
                if len(cluster_data) > 0:  # ensure the cluster is not empty
                    _real_centers[k] = find_nearest_point(
                        cluster_data, _centers[k], "l2"
                    )
                else:
                    _real_centers[k] = _centers[
                        k
                    ]  # fallback to computed centroid if cluster is empty

            new_labels[current_applied_vectors_mask] = _labels + label_offset
            new_centers = np.concatenate((new_centers, _real_centers))
            label_offset += _labels.max() + 1
        labels = new_labels
        centers = np.array(new_centers)
    print(
        f"Reused labels: {reused_labels.max() + 1 if reused_labels is not None else 'No'}",
    )
    timer.tuck("Calculate residual norm")
    return labels, centers


def create_persistent_kmeans_data(data, num_clusters, path):
    """
    Create a persistent kmean object
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    data_dict = {
        "base": data,
        "labels": labels,
        "centroids": centroids,
    }
    with open(path, "wb") as f:
        pickle.dump(data_dict, f)
    return data_dict


def load_persistent_kmeans_data(path):
    with open(path, "rb") as f:
        saved_dict = pickle.load(f)
    return saved_dict


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


def save_meta_index_data(
    bucket_centroids: np.ndarray,
    bucket_node_labels: np.ndarray,
    save_path: str,
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
                "II",
                bucket_centroids.shape[0],  # num_buckets
                bucket_centroids.shape[1],  # dim
            )
        )

        # Write bucket node labels as uint32
        f.write(bucket_node_labels.astype(np.uint32).tobytes())

        # Write bucket centroids as float32
        f.write(bucket_centroids.astype(np.float32).tobytes())


def generate_path_script(destdir: str, dataset_name: str):
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    data_paths = get_data_paths(dataset_name)
    # create a new file under destdir called get_data_path.sh
    query_path = data_paths["query"]
    groundtruth_path = data_paths["groundtruth"]
    file_name = os.path.join(destdir, "get_data_path.sh")
    with open(file_name, "w") as f:
        f.write(f'echo "{query_path}"\n')
        f.write(f'echo "{groundtruth_path}"\n')

    os.system(f"chmod +x {file_name}")


def pre_partition_buckets(
    dataset_name: str,
    method_name: str,
    data: np.ndarray,
    bucket_labels: np.ndarray,
    bucket_centroids: np.ndarray,
    num_nodes: int,
    bucket_cluster_method: Literal["kmeans", "spectral"] = "kmeans",
    alpha: float = 1.2,
    R: int = 128,
    L: int = 100,
    ef_search: int = 400,
):
    num_buckets = bucket_centroids.shape[0]
    t_data = torch.from_numpy(data).float().to(DEVICE)
    t_bucket_labels = torch.from_numpy(bucket_labels).to(DEVICE)
    t_labels = torch.arange(len(data)).to(DEVICE)

    buckets_node_labels = np.zeros(num_buckets, dtype=np.int32)
    if bucket_cluster_method == "kmeans":
        bucket_kmeans = KMeans(n_clusters=num_nodes)
        # 每一个bucket质心进行二次聚类，得到的node label
        buckets_node_labels = bucket_kmeans.fit_predict(bucket_centroids)
    else:
        # TODO: implement spectral clustering
        raise ValueError(f"Unsupported bucket cluster method: {bucket_cluster_method}")

    dirpath = f"data/{dataset_name}_worker_data_vamana_{method_name}"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    # 为每个节点创建数据
    for i in range(num_nodes):
        # 当前节点对应的bucket indices
        current_node_bucket_indices = np.where(buckets_node_labels == i)[0]
        current_node_bucket_indices_list = current_node_bucket_indices.tolist()
        node_buckets = []
        for _bucket_index in tqdm(current_node_bucket_indices_list):
            t_current_bucket_data = t_data[t_bucket_labels == _bucket_index]
            t_current_bucket_indices = t_labels[t_bucket_labels == _bucket_index]
            bucket = Bucket.from_data(
                t_current_bucket_data.cpu().numpy(),
                t_current_bucket_indices.cpu().numpy(),
            )
            node_buckets.append(bucket)
        _current_node_bucket_centroids = bucket_centroids[
            current_node_bucket_indices_list
        ]

        # Create Node object with Vamana parameters
        node = Node.from_buckets(
            node_buckets,
            current_node_bucket_indices,
            _current_node_bucket_centroids,
        )
        vamana_index = vamanalib.Vamana(
            dim=node.data.shape[1],
            max_points=len(node.data),
            alpha=alpha,
            R=R,
            L=L,
            ef_search=ef_search,
        )

        for idx, point in enumerate(node.data):
            label = node.absolute_label_indices[idx]
            vamana_index.add_point(point, label)
        print("point added Successfully")
        vamana_index.build_index()
        vamana_index.save_index(f"{dirpath}/worker-{i}_vamana.bin")

        save_node_data(node, f"{dirpath}/worker-{i}.bin")

    meta_vamana = vamanalib.Vamana(
        dim=bucket_centroids.shape[1],
        max_points=bucket_centroids.shape[0],
        alpha=alpha,
        R=R,
        L=L,
        ef_search=ef_search,
    )
    print("len of bucket_centroids", len(bucket_centroids))
    for i, point in enumerate(bucket_centroids):
        meta_vamana.add_point(point, i)
    # build meta index
    meta_vamana.build_index()
    meta_vamana.save_index(f"{dirpath}/meta_vamana.bin")
    # Save bucket node labels separately for building metaIndex
    save_meta_index_data(
        bucket_centroids,
        buckets_node_labels,
        f"{dirpath}/meta_index_data.bin",
    )

    generate_path_script(dirpath, dataset_name)


def naive_partition_buckets(
    dataset_name: str,
    method_name: str,
    data: np.ndarray,
    num_nodes: int,
    alpha: float = 1.2,
    R: int = 128,
    L: int = 100,
    ef_search: int = 200,
):
    # 读取数据，并平均分配到每个节点
    def split_data(data, num_nodes):
        return np.array_split(data, num_nodes)

    labels = np.arange(len(data))
    dataList = split_data(data, num_nodes)
    print("Data split completed")
    labelsList = split_data(labels, num_nodes)
    dirpath = f"data/{dataset_name}_worker_data_vamana_{method_name}"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    for i in range(num_nodes):
        node = Node.from_buckets(
            [Bucket.from_data(dataList[i], labelsList[i])],
            np.array([i]),
            np.array([i]),
        )
        vamana_index = vamanalib.Vamana(
            dim=node.data.shape[1],
            max_points=len(node.data),
            alpha=alpha,
            R=R,
            L=L,
            ef_search=ef_search,
        )
        for idx, point in enumerate(node.data):
            label = node.absolute_label_indices[idx]
            vamana_index.add_point(point, label)
        print("point added Successfully")
        vamana_index.build_index()
        vamana_index.save_index(f"{dirpath}/worker-{i}_vamana.bin")
        save_node_data(node, f"{dirpath}/worker-{i}.bin")
    generate_path_script(dirpath, dataset_name)


def generate_help_message(param_config, partition_config):
    help_message = "Usage: python data_processor.py <dataset_name> <partition_method>\n"
    help_message += "Valid dataset names are:\n"
    for key in param_config:
        help_message += f"- {key}\n"
    help_message += "Valid partition methods are:\n"
    for key in partition_config:
        help_message += f"- {key}\n"
    return help_message


def print_help(param_config, partition_config):
    print(generate_help_message(param_config, partition_config))


if __name__ == "__main__":
    with open("../config/parameters.yaml", "r") as f:
        param_config = yaml.safe_load(f)
    with open("../config/partitionConfigVamana.yaml", "r") as f:
        partition_config = yaml.safe_load(f)
    print(partition_config)

    param_config_keys = list(param_config.keys())
    partition_config_keys = list(partition_config.keys())
    fzf = FzfPrompt()
    print("Choose dataset:")
    chosen_dataset = fzf.prompt(param_config_keys)[0]
    print(f"Chosen dataset: {chosen_dataset}")
    print("Choose partition method:")
    chosen_method = fzf.prompt(partition_config_keys)[0]
    print(f"Chosen method: {chosen_method}")
    if chosen_dataset is None or chosen_method is None:
        print("No dataset or method chosen.")
        exit(1)

    # if (
    #     args.d is None
    #     or args.m is None
    #     or not args.d in param_config
    #     or not args.m in partition_config
    # ):
    #     print_help(param_config, partition_config)
    #     exit(1)

    dataset_config = param_config[chosen_dataset]
    dataset_name = dataset_config["dataset"]
    bucket_num = dataset_config["bucket_num"]
    partition_params = partition_config[chosen_method]

    if not isinstance(bucket_num, list):
        bucket_num = [bucket_num]
    base, query, ground_truth = get_dataset(dataset_name)
    if partition_params["method"] == "similarity":
        labels, centroids = hierarchical_KMeans(base, bucket_num, verbose=True)
        pre_partition_buckets(
            dataset_name,
            chosen_method,
            base,
            labels,
            centroids,
            partition_params["partition_num"],
            alpha=partition_params["alpha"],
            R=partition_params["R"],
            L=partition_params["L"],
            ef_search=400,
        )

    elif partition_params["method"] == "naive":
        naive_partition_buckets(
            dataset_name,
            chosen_method,
            base,
            partition_params["partition_num"],
            alpha=partition_params["alpha"],
            R=partition_params["R"],
            L=partition_params["L"],
            ef_search=400,
        )
