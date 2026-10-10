# Search engines

`c_api` exposes the C++ indexes. `go_api` wraps those libraries for the worker process.

- `hnsw` is the HNSW index.
- `ivf` scans selected buckets.
- `vamana` is the Vamana index.

## Build

From the repository root:

```bash
make engine
```

The shared libraries are written to `internal/engine/build`.
