import os
from typing import Literal
import numpy as np

# Above this size, vector files are loaded lazily via np.memmap (zero-copy,
# read-only) instead of np.fromfile, which would materialise the whole file in
# RAM and then .copy() it again. memmap pages data in on demand, so the OS only
# ever keeps the touched pages resident. This is what lets multi-hundred-GB
# datasets (e.g. deep1B, ~384 GB of vectors) be processed at all.
_MEMMAP_THRESHOLD_BYTES = 2 * 1024**3  # 2 GiB


def _should_memmap(fname):
    """True when the file is large enough that an eager load would be wasteful."""
    try:
        return os.path.getsize(fname) > _MEMMAP_THRESHOLD_BYTES
    except OSError:
        return False

data_name_aliases = {
    "sift1m": "sift",
    "sift10k": "siftsmall",
}

data_paths_dict = {
    "sift": {
        "base": "../data/sift/sift_base.fvecs",
        "query": "../data/sift/sift_query.fvecs",
        "groundtruth": "../data/sift/sift_groundtruth.ivecs",
    },
    "siftsmall": {
        "base": "../data/siftsmall/siftsmall_base.fvecs",
        "query": "../data/siftsmall/siftsmall_query.fvecs",
        "groundtruth": "../data/siftsmall/siftsmall_groundtruth.ivecs",
    },
    "deep1m": {
        "base": "../data/deep/deep1M_base.fvecs",
        "query": "../data/deep/deep1B_queries.fvecs",
        "groundtruth": "../data/deep/deep1M_groundtruth.ivecs",
    },
    "deep10m": {
        "base": "../data/deep/deep10M_base.fvecs",
        "query": "../data/deep/deep1B_queries.fvecs",
        "groundtruth": "../data/deep/deep10M_groundtruth.ivecs",
    },
    "deep100m": {
        "base": "../data/deep/deep100M_base.fvecs",
        "query": "../data/deep/deep1B_queries.fvecs",
        "groundtruth": "../data/deep/deep100M_groundtruth.ivecs",
    },
    "gist1m": {
        "base": "../data/gist/gist_base.fvecs",
        "query": "../data/gist/gist_query.fvecs",
        "groundtruth": "../data/gist/gist_groundtruth.ivecs",
    },
    "sift10m": {
        "base": "../data/sift10m/sift10m_base.fvecs",
        "query": "../data/sift10m/sift10m_queries.fvecs",
        "groundtruth": "../data/sift10m/sift100m_groundtruth.ivecs",
    },
    "sift100m": {
        "base": "../data/sift100m/sift100m_base.fvecs",
        "query": "../data/sift100m/sift100m_queries.fvecs",
        "groundtruth": "../data/sift100m/sift100m_groundtruth.ivecs",
    },
    "space100m": {
        "base": "../data/ann-spacev-100m/space100m_base.fvecs",
        "query": "../data/ann-spacev-100m/space100m_queries.fvecs",
        "groundtruth": "../data/ann-spacev-100m/space100m_groundtruth.ivecs",
    },
    "wiki100m": {
        "base": "../data/ann-wiki-100m/wiki100m_base.fvecs",
        "query": "../data/ann-wiki-100m/wiki100m_queries.fvecs",
        "groundtruth": "../data/ann-wiki-100m/wiki100m_groundtruth.ivecs",
    },
    "wiki50m": {
        "base": "../data/ann-wiki-50m/wiki50m_base.fvecs",
        "query": "../data/ann-wiki-50m/wiki50m_queries.fvecs",
        "groundtruth": "../data/ann-wiki-50m/wiki50m_groundtruth.ivecs",
    },
    "wiki20m": {
        "base": "../data/ann-wiki-20m/wiki20m_base.fvecs",
        "query": "../data/ann-wiki-20m/wiki20m_queries.fvecs",
        "groundtruth": "../data/ann-wiki-20m/wiki20m_groundtruth.ivecs",
    },
    "laion-i2i-20m": {
        "base": "../data/laion-i2i-20m/laion_i2i_base.fvecs",
        "query": "../data/laion-i2i-20m/laion_i2i_query.fvecs",
        "groundtruth": "../data/laion-i2i-20m/laion_i2i_groundtruth.ivecs",
    },
    "laion-t2i-20m": {
        "base": "../data/laion-t2i-20m/laion_t2i_base.fvecs",
        "query": "../data/laion-t2i-20m/laion_t2i_query.fvecs",
        "groundtruth": "../data/laion-t2i-20m/laion_t2i_groundtruth.ivecs",
    },
    "laion-i2i-10m": {
        "base": "../data/laion-i2i-10m/laion_i2i_base.fvecs",
        "query": "../data/laion-i2i-10m/laion_i2i_query.fvecs",
        "groundtruth": "../data/laion-i2i-10m/laion_i2i_groundtruth.ivecs",
    },
    "laion-t2i-10m": {
        "base": "../data/laion-t2i-10m/laion_t2i_base.fvecs",
        "query": "../data/laion-t2i-10m/laion_t2i_query.fvecs",
        "groundtruth": "../data/laion-t2i-10m/laion_t2i_groundtruth.ivecs",
    },
    "sift1b": {
        "base": "../data/sift1b/sift1b_base.fbin",
        "query": "../data/sift1b/bigann_query.bvecs",
        "groundtruth": "../data/sift1b/bigann_groundtruth.ivecs",
    },
}


def get_data_paths(dataset: str) -> dict:
    if dataset in data_name_aliases:
        dataset = data_name_aliases[dataset]
    if dataset not in data_paths_dict:
        raise ValueError(f"Invalid dataset: {dataset}")
    return data_paths_dict[dataset]


def get_dataset(
    dataset: Literal["sift", "siftsmall", "deep1m", "deep10m", "sift1m", "deep100m", "gist1m", "sift100m", "space100m", "wiki100m", "wiki50m", "wiki20m", "laion-i2i-20m","laion-t2i-20m","laion-i2i-10m","laion-t2i-10m"],
) -> tuple:
    data: str = dataset
    if dataset in data_name_aliases:
        data = data_name_aliases[dataset]
    if data not in data_paths_dict:
        raise ValueError(f"Invalid dataset: {dataset}")
    data_paths = data_paths_dict[data]

    base = load_data(data_paths["base"])
    query = _load_opt(data_paths["query"])
    ground_truth = _load_opt(data_paths["groundtruth"], "ivecs")
    return base, query, ground_truth


def _load_opt(file_name: str, file_type: str = "fvecs"):
    """Load a data file, or return None if it does not exist (e.g. missing
    query/groundtruth for a build-only dataset like sift1b)."""
    if not os.path.exists(file_name):
        print(f"[data_loader] optional file not found, skipping: {file_name}")
        return None
    return load_data(file_name, file_type)


def load_data(file_name: str, file_type: str = "fvecs") -> np.ndarray:
    if file_type == "fvecs":
        # Auto-route .fbin (memmap-friendly BIN format) to read_fbin.
        if file_name.endswith(".fbin"):
            return read_fbin(file_name)
        return read_fvecs(file_name)
    if file_type == "ivecs":
        return read_ivecs(file_name)
    if file_type == "fbin":
        return read_fbin(file_name)
    raise ValueError("Invalid file type")


def read_ivecs(fname):
    if not _should_memmap(fname):
        # Small file: eager load + copy keeps the original (writable,
        # contiguous) behaviour for unit-test-scale data.
        a = np.fromfile(fname, dtype="int32")
        d = a[0]
        return a.reshape(-1, d + 1)[:, 1:].copy()

    # Large file: memmap lazily and return a strided, zero-copy view. Reads page
    # the underlying bytes in on demand; nothing is duplicated in RAM.
    d = np.fromfile(fname, dtype=np.int32, count=1)[0]
    rec_len = d + 1
    raw = np.memmap(fname, dtype=np.int32, mode="r")
    n = raw.size // rec_len
    return raw[: n * rec_len].reshape(n, rec_len)[:, 1:]


def read_fvecs(fname):
    return read_ivecs(fname).view("float32")


def read_bvecs(fname):
    if not _should_memmap(fname):
        a = np.fromfile(fname, dtype="uint8")
        d = a.view("int32")[0]
        return a.reshape(-1, d + 4)[:, 4:].copy()

    # memmap as raw bytes; each record = 4-byte dim header + d uint8 values.
    d = np.fromfile(fname, dtype=np.int32, count=1)[0]
    rec_len = d + 4
    raw = np.memmap(fname, dtype=np.uint8, mode="r")
    n = raw.size // rec_len
    return raw[: n * rec_len].reshape(n, rec_len)[:, 4:]


def read_fbin(fname):
    shape = np.fromfile(fname, dtype=np.uint32, count=2)
    if float(shape[0]) * shape[1] * 4 > 2_000_000_000:
        data = np.memmap(fname, dtype=np.float32, offset=8, mode="r").reshape(shape)
    else:
        data = np.fromfile(fname, dtype=np.float32, offset=8).reshape(shape)
    return data
