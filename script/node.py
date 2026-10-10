import numpy as np
from typing import List, Dict


class Bucket:
    """
    does NOT store:
        - centroids
    """

    absolute_labels: np.ndarray  # stores absolute global labels
    data: np.ndarray

    def __init__(
        self,
        data: np.ndarray,
        absolute_labels: np.ndarray,
    ):
        self.data = data
        self.absolute_labels = absolute_labels

    def __repr__(self):
        return f"Bucket(data={self.data.shape}, labels={self.absolute_labels.shape})"

    @classmethod
    def from_data(
        cls,
        data: np.ndarray,
        label: np.ndarray,
    ):
        return cls(data, label)


class Node:
    buckets: List[Bucket]  # list of buckets
    centroids: np.ndarray  # duplicate of part of centroids in Cluster
    bucket_dict: Dict[int, Bucket]  # NOT USED
    absolute_label_indices: np.ndarray  # stores absolute global labels from all buckets

    def __init__(
        self,
        data: np.ndarray,
        data_size_per_bucket: np.ndarray,
        bucket_indices: np.ndarray,
        absolute_label_indices: np.ndarray,
        centroids: np.ndarray,
    ):
        self.data = data
        self.data_size_per_bucket = data_size_per_bucket
        self.bucket_indices = bucket_indices
        self.absolute_label_indices = absolute_label_indices
        self.centroids = centroids

    def __repr__(self):
        return f"Node(buckets={self.num_buckets})"

    @classmethod
    def from_buckets(
        cls,
        buckets: List[Bucket],
        bucket_indices: np.ndarray,
        centroids: np.ndarray,
    ):
        data = np.concatenate([bucket.data for bucket in buckets])
        absolute_label_indices = np.concatenate(
            [bucket.absolute_labels for bucket in buckets]
        )
        data_size_per_bucket = np.array([bucket.data.shape[0] for bucket in buckets])
        if min(data_size_per_bucket) == 0:
            raise ValueError("Empty bucket detected!")
        print("Number of buckets:", len(buckets))
        print("Data size per bucket:", data_size_per_bucket)
        return cls(
            data,
            data_size_per_bucket,
            bucket_indices,
            absolute_label_indices,
            centroids,
        )

    @property
    def num_buckets(self):
        """
        Get the number of buckets.
        """
        return len(self.data_size_per_bucket)

    def bucket(self, index: int):
        """
        Get the data of the i-th bucket.
        """
        return self.data[
            self.data_size_per_bucket[:index].sum() : self.data_size_per_bucket[
                : index + 1
            ].sum()
        ]
