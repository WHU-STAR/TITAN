package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/WHU-STAR/TITAN/internal/definitions"
	"github.com/WHU-STAR/TITAN/internal/engine/go_api/ivf"
)

func buildIndex(dim int, bucketIdList []uint32) *ivf.Index {
	return ivf.NewIndex(dim, bucketIdList)
}

func generateRandVectors(numVectors, dim int) [][]float32 {
	vectors := make([][]float32, numVectors)
	for i := 0; i < numVectors; i++ {
		vectors[i] = make([]float32, dim)
		var sum float32
		for j := 0; j < dim; j++ {
			vectors[i][j] = rand.Float32()
			sum += vectors[i][j] * vectors[i][j]
		}
		// 归一化向量
		norm := float32(math.Sqrt(float64(sum)))
		for j := 0; j < dim; j++ {
			vectors[i][j] /= norm
		}
	}
	return vectors
}

// 计算两个向量之间的 L2 距离
func computeL2Distance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

// 使用暴力搜索找到最近邻
func bruteForceKNN(query []float32, vectors [][]float32, k int) ([]uint32, []float32) {
	type distanceInfo struct {
		id       uint32
		distance float32
	}

	// 计算所有距离
	distances := make([]distanceInfo, len(vectors))
	for i := range vectors {
		distances[i] = distanceInfo{
			id:       uint32(i),
			distance: computeL2Distance(query, vectors[i]),
		}
	}

	// 按距离排序
	sort.Slice(distances, func(i, j int) bool {
		return distances[i].distance < distances[j].distance
	})

	// 返回前 k 个结果
	ids := make([]uint32, k)
	dists := make([]float32, k)
	for i := 0; i < k; i++ {
		ids[i] = distances[i].id
		dists[i] = distances[i].distance
	}
	return ids, dists
}

// 计算召回率
func computeRecall(groundTruth, result []uint32) float32 {
	if len(groundTruth) == 0 || len(result) == 0 {
		return 0
	}

	// 将 ground truth 转换为 map 以便快速查找
	truthMap := make(map[uint32]bool)
	for _, id := range groundTruth {
		truthMap[id] = true
	}

	// 计算结果中有多少在 ground truth 中
	var hits float32
	for _, id := range result {
		if truthMap[id] {
			hits++
		}
	}

	return hits / float32(len(groundTruth))
}

func main() {
	const (
		dim             = 128  // 向量维度
		numBuckets      = 100  // bucket 数量
		pointsPerBucket = 1000 // 每个 bucket 的点数量
		totalPoints     = numBuckets * pointsPerBucket
		topK            = 10 // 检索 top-k
	)

	// 初始化 bucket ID 列表
	bucketIdList := make([]uint32, numBuckets)
	for i := range bucketIdList {
		bucketIdList[i] = uint32(i + 1)
	}

	// 创建索引
	index := buildIndex(dim, bucketIdList)

	// 生成随机向量
	vectors := generateRandVectors(totalPoints, dim)
	fmt.Printf("Generated %d vectors with dimension %d\n", len(vectors), dim)
	fmt.Printf("Number of buckets: %d\n", len(bucketIdList))

	// 构建 buckets
	buckets := make([]definitions.Bucket, numBuckets)
	for i := range buckets {
		buckets[i].BucketId = uint32(i + 1)
		buckets[i].Count = int32(pointsPerBucket)
		buckets[i].Points = make([]definitions.Point, pointsPerBucket)
		for j := 0; j < pointsPerBucket; j++ {
			globalPointId := uint32(i*pointsPerBucket + j)
			buckets[i].Points[j] = definitions.Point{
				PointId: globalPointId,
				Data:    vectors[globalPointId],
			}
		}
	}

	// 构建索引
	fmt.Println("Building index...")
	err := index.AddItems(buckets)
	if err != nil {
		fmt.Printf("Failed to build index: %v\n", err)
		return
	}
	fmt.Println("Index built successfully")

	// 进行多次查询测试
	numQueries := 100
	var totalRecall float32

	fmt.Printf("\nRunning %d queries...\n", numQueries)
	for i := 0; i < numQueries; i++ {
		queryVector := generateRandVectors(1, dim)[0]

		// 使用 IVF 索引搜索
		result, distances, err := index.SearchKNN(queryVector, bucketIdList, topK)
		if err != nil {
			fmt.Printf("Search failed: %v\n", err)
			continue
		}

		// 计算 ground truth
		groundTruthIds, groundTruthDists := bruteForceKNN(queryVector, vectors, topK)

		// 计算召回率
		recall := computeRecall(groundTruthIds, result)
		totalRecall += recall

		if i == 0 { // 只打印第一次查询的详细结果
			fmt.Printf("\nQuery %d results:\n", i+1)
			fmt.Println("IVF results:")
			for j := 0; j < min(5, len(result)); j++ {
				fmt.Printf("  Label: %d, Distance: %.6f\n", result[j], distances[j])
			}
			fmt.Println("Ground truth:")
			for j := 0; j < min(5, len(groundTruthIds)); j++ {
				fmt.Printf("  Label: %d, Distance: %.6f\n", groundTruthIds[j], groundTruthDists[j])
			}
			fmt.Printf("Recall: %.4f\n", recall)
		}
	}

	avgRecall := totalRecall / float32(numQueries)
	fmt.Printf("\nAverage recall over %d queries: %.4f\n", numQueries, avgRecall)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
