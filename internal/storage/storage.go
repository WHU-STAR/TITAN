package storage

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/WHU-STAR/TITAN/internal/definitions"
	"github.com/WHU-STAR/TITAN/pkg/logger"
	"go.uber.org/zap"
)

type NodeData struct {
	Num                       uint32
	Dim                       uint32
	BucketIndicesCount        uint32
	AbsoluteLabelIndicesCount uint32
	BucketIndices             []uint32
	AbsoluteLabelIndices      []int32
	DataSizePerBucket         []int32
	Buckets                   []definitions.Bucket
}

type MetaIndexData struct {
	NumBuckets         uint32
	Dim                uint32
	BucketNodeLabels   [][]uint32
	Centroids          [][]float32
	NearestPointLabels []uint32
	BucketRadiuses     []float32
	BucketSizes        []uint32 // |B_b| per bucket; nil for legacy meta index files
	TotalNumVectors    uint32   // N in |B_b|/N; 0 for legacy files
}

type QueryData struct {
	Dim     uint32
	Num     uint32
	Queries [][]float32
}

type GroundTruth struct {
	Num         uint32
	GroundTruth [][]uint32
}

func ReadFVECS(path string) ([][]float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var vectors [][]float32
	for {
		// 读取维度（4字节）
		dimBuf := make([]byte, 4)
		_, err := io.ReadFull(file, dimBuf)
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, err
		}
		dim := int(binary.LittleEndian.Uint32(dimBuf))

		// 读取向量数据（4*dim字节）
		dataBuf := make([]byte, 4*dim)
		_, err = io.ReadFull(file, dataBuf)
		if err != nil {
			return nil, err
		}

		// 转换为float32切片
		vector := make([]float32, dim)
		for i := 0; i < dim; i++ {
			start := i * 4
			end := start + 4
			bits := binary.LittleEndian.Uint32(dataBuf[start:end])
			vector[i] = math.Float32frombits(bits)
		}

		vectors = append(vectors, vector)
	}
	return vectors, nil
}

func ReadIVECS(path string) ([][]uint32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var vectors [][]uint32
	for {
		// 读取维度（4字节）
		dimBuf := make([]byte, 4)
		_, err := io.ReadFull(file, dimBuf)
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, err
		}
		dim := int(binary.LittleEndian.Uint32(dimBuf))

		// 读取向量数据（4*dim字节）
		dataBuf := make([]byte, 4*dim)
		_, err = io.ReadFull(file, dataBuf)
		if err != nil {
			return nil, err
		}

		// 转换为int32切片
		vector := make([]uint32, dim)
		for i := 0; i < dim; i++ {
			start := i * 4
			end := start + 4
			vector[i] = uint32(binary.LittleEndian.Uint32(dataBuf[start:end]))
		}

		vectors = append(vectors, vector)
	}
	return vectors, nil
}

func LoadGroundTruth(path string) (*GroundTruth, error) {
	// 嵌套一层，获取元数据
	groundTruth, err := ReadIVECS(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read ground truth: %v", err)
	}
	return &GroundTruth{
		Num:         uint32(len(groundTruth)),
		GroundTruth: groundTruth,
	}, nil
}

func LoadQuerySet(path string) (*QueryData, error) {
	// 嵌套一层，获取元数据
	queries, err := ReadFVECS(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read queries: %v", err)
	}
	return &QueryData{
		Dim:     uint32(len(queries[0])),
		Num:     uint32(len(queries)),
		Queries: queries,
	}, nil
}

func LoadMetaIndexData(path string, searchMethod string) (*MetaIndexData, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	// Read header (num_buckets dim as uint32)
	var numBuckets, dim uint32
	if err := binary.Read(file, binary.LittleEndian, &numBuckets); err != nil {
		return nil, fmt.Errorf("failed to read num_buckets: %v", err)
	}

	if err := binary.Read(file, binary.LittleEndian, &dim); err != nil {
		return nil, fmt.Errorf("failed to read dim for meta Index: %v", err)
	}

	var numNodes uint32
	if err := binary.Read(file, binary.LittleEndian, &numNodes); err != nil {
		return nil, fmt.Errorf("failed to read num_nodes for meta Index: %v", err)
	}

	// Read bucket Node labels
	bucketNodeLabels := make([][]uint32, numBuckets)
	fmt.Printf("Loading meta index data with dimensions: %d, data count: %d\n", dim, numBuckets)

	for i := range bucketNodeLabels {
		bucketNodeLabels[i] = make([]uint32, numNodes)
		if err := binary.Read(file, binary.LittleEndian, bucketNodeLabels[i]); err != nil {
			return nil, fmt.Errorf("failed to read bucketNodeLabels: %v", err)
		}
	}

	// Read bucket centroids
	centroids := make([][]float32, numBuckets)
	fmt.Printf("Loading centroids with dimensions: %d, data count: %d\n", dim, numBuckets)
	for i := range centroids {
		centroids[i] = make([]float32, dim)
		if err := binary.Read(file, binary.LittleEndian, centroids[i]); err != nil {
			return nil, fmt.Errorf("failed to read centroids: %v", err)
		}
	}

	// Read nearest point labels
	nearestPointLabels_ := make([]int32, numBuckets)
	if err := binary.Read(file, binary.LittleEndian, nearestPointLabels_); err != nil {
		return nil, fmt.Errorf("failed to read nearest_point_labels: %v", err)
	}

	nearestPointLabels := make([]uint32, numBuckets)
	for i, label := range nearestPointLabels_ {
		if label < 0 {
			return nil, fmt.Errorf("invalid nearest point label: %d at index %d", label, i)
		}
		nearestPointLabels[i] = uint32(label)
	}

	bucketRadiuses := make([]float32, numBuckets)
	if searchMethod == "rf" || searchMethod == "hf" || searchMethod == "hf-v2" {
		if err := binary.Read(file, binary.LittleEndian, bucketRadiuses); err != nil {
			return nil, fmt.Errorf("failed to read bucket radiuses: %v", err)
		}
	}

	var bucketSizes []uint32
	var totalNumVectors uint32
	if searchMethod == "rf" || searchMethod == "hf" || searchMethod == "hf-v2" {
		curPos, err := file.Seek(0, io.SeekCurrent)
		if err != nil {
			return nil, fmt.Errorf("failed to seek meta index file: %v", err)
		}
		fi, err := file.Stat()
		if err == nil && fi.Size() >= curPos+int64(numBuckets)*4+4 {
			sizes := make([]uint32, numBuckets)
			if err := binary.Read(file, binary.LittleEndian, sizes); err == nil {
				if err := binary.Read(file, binary.LittleEndian, &totalNumVectors); err == nil {
					bucketSizes = sizes
				}
			}
		}
	}

	return &MetaIndexData{
		NumBuckets:         numBuckets,
		BucketNodeLabels:   bucketNodeLabels,
		Centroids:          centroids,
		Dim:                dim,
		NearestPointLabels: nearestPointLabels,
		BucketRadiuses:     bucketRadiuses,
		BucketSizes:        bucketSizes,
		TotalNumVectors:    totalNumVectors,
	}, nil
}

func LoadNodeData(path string) (*NodeData, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	// Read header values
	var (
		dim                       uint32
		num                       uint32
		bucketIndicesCount        uint32
		absoluteLabelIndicesCount uint32
		dataSizePerBucketCount    uint32
	)

	if err := binary.Read(file, binary.LittleEndian, &dim); err != nil {
		return nil, fmt.Errorf("failed to read dimensions: %v", err)
	}
	if err := binary.Read(file, binary.LittleEndian, &num); err != nil {
		return nil, fmt.Errorf("failed to read data count: %v", err)
	}
	if err := binary.Read(file, binary.LittleEndian, &bucketIndicesCount); err != nil {
		return nil, fmt.Errorf("failed to read bucket_indices_count: %v", err)
	}
	if err := binary.Read(file, binary.LittleEndian, &absoluteLabelIndicesCount); err != nil {
		return nil, fmt.Errorf("failed to read absolute_label_indices_count: %v", err)
	}
	if err := binary.Read(file, binary.LittleEndian, &dataSizePerBucketCount); err != nil {
		return nil, fmt.Errorf("failed to read data_size_per_bucket_count: %v", err)
	}

	fmt.Printf("Loading node data with dimensions: %d, data count: %d\n", dim, num)
	fmt.Printf("Array sizes: bucket_indices=%d, absolute_label_indices=%d\n",
		bucketIndicesCount, absoluteLabelIndicesCount)

	// Read bucket indices as int32
	bucketIndices := make([]uint32, bucketIndicesCount)
	if err := binary.Read(file, binary.LittleEndian, bucketIndices); err != nil {
		return nil, fmt.Errorf("failed to read bucket indices: %v", err)
	}

	// Read absolute label indices as int32
	absoluteLabelIndices := make([]int32, absoluteLabelIndicesCount)
	if err := binary.Read(file, binary.LittleEndian, absoluteLabelIndices); err != nil {
		return nil, fmt.Errorf("failed to read absolute label indices: %v", err)
	}

	// Read data size per bucket as int32
	dataSizePerBucket := make([]int32, dataSizePerBucketCount)
	if err := binary.Read(file, binary.LittleEndian, dataSizePerBucket); err != nil {
		return nil, fmt.Errorf("failed to read data size per bucket: %v", err)
	}

	// Calculate total size of bucket data
	totalDataSize := int(0)
	for _, size := range dataSizePerBucket {
		totalDataSize += int(size) * int(dim) // 每个点都是 dim 维的
	}

	// Read bucket data as float32
	bucketData := make([]float32, totalDataSize)
	if err := binary.Read(file, binary.LittleEndian, bucketData); err != nil {
		return nil, fmt.Errorf("failed to read bucket data: %v", err)
	}

	// Convert bucket data to buckets
	buckets := make([]definitions.Bucket, bucketIndicesCount)
	offset := 0
	pointOffset := 0
	logger.Debug("len buckets", zap.Int("len", len(buckets)))
	logger.Debug("len dataSizePerBucket", zap.Int("len", len(dataSizePerBucket)))

	for i := range buckets {
		numPoints := int(dataSizePerBucket[i]) // 这是点的数量
		buckets[i] = definitions.Bucket{
			BucketId: bucketIndices[i],
			Count:    int32(numPoints),
			Points:   make([]definitions.Point, numPoints),
		}

		// Convert flat data to points
		for j := 0; j < numPoints; j++ {
			pointStart := offset + j*int(dim)
			pointEnd := pointStart + int(dim)
			buckets[i].Points[j] = definitions.Point{
				PointId: uint32(absoluteLabelIndices[pointOffset+j]), // 使用累积的 pointOffset
				Data:    bucketData[pointStart:pointEnd],
			}
		}
		offset += numPoints * int(dim) // 每个 bucket 的数据偏移量需要乘以维度
		pointOffset += numPoints       // 累积点的偏移量
	}

	// Verify we've read the entire file
	if _, err := file.Read(make([]byte, 1)); err != io.EOF {
		return nil, fmt.Errorf("file contains extra data")
	}

	return &NodeData{
		Num:                       num,
		Dim:                       dim,
		BucketIndicesCount:        bucketIndicesCount,
		AbsoluteLabelIndicesCount: absoluteLabelIndicesCount,
		BucketIndices:             bucketIndices,
		AbsoluteLabelIndices:      absoluteLabelIndices,
		DataSizePerBucket:         dataSizePerBucket,
		Buckets:                   buckets,
	}, nil
}
