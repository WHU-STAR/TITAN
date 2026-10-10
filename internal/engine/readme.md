# Search Wrapper

This is a search wrapper for distribute vector search engine, we use `cgo` to combine go and c++ code, `c_api` contains the c functions, which can be called by `go_api`.

`example` is the test for search link, do not contains implement of any algorithms.
`hnsw` is the implement of the real search wrapper in `hnsw` algorithm.
`vamana` is the implement of the `vamana` index.

## Build

```bash
1. mkdir build
2. cd build
3. cmake ..
4. make
```

## Test

```bash
# To test the search link, by doing this, you can check whether you know how to use the search wrapper.
go run example.go

# To test the hnsw algorithm
go run hnsw.go

# To test the vamana algorithm
go run vamana.go
```

## Usage

If you want to add a new algorithm or revise the existing algorithm, you need to add the corresponding c++ code in `c_api` and the corresponding go code in `go_api`.
