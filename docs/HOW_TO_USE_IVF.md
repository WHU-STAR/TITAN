# HOW TO USE IVF

## 1. Generate IVF index

```bash
cd ./script && python generate_ivf_index.py --dataset deep10m --nlist 10000 --nprobe 10 --format binary --output ../data/workerdata/meta_ivf_index.bin
```

## 2. Test IVF index

```bash
go run cmd/test/debug-ivf-labels/test_siftsmall.go
```