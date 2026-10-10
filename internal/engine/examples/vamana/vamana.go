package main

import (
	"fmt"
	"math/rand"
	"sort"

	"github.com/WHU-STAR/TITAN/internal/engine/go_api/vamana"
)

// set parameters
const (
	dimension  = 128  // vector dimension
	numPoints  = 1000 // number of points
	numQueries = 100  // number of queries
	R          = 56   // maximum degree (R in paper)
	L          = 100  // length of candidate list (L in paper)
	alpha      = 1.2  // robust prune parameter
	k          = 10   // top-k
	efSearch   = 200  // search-time candidate list length (L in paper)
)

func generateRandomVector(dimension uint32) []float32 {
	vec := make([]float32, dimension)
	for i := range vec {
		vec[i] = rand.Float32()*2 - 1 // [-1, 1]
	}
	return vec
}

func computeDistance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(sum)
}

func TestVamanaBasic() {
	// create index
	index, err := vamana.NewVamanaIndex(dimension, numPoints, alpha, R, L, efSearch)
	if err != nil {
		fmt.Println("Failed to create index:", err)
		return
	}
	defer index.Free()

	points := make([][]float32, numPoints)
	for i := uint32(0); i < numPoints; i++ {
		points[i] = generateRandomVector(dimension)
		err := index.AddPoint(points[i], i)
		if err != nil {
			fmt.Println("Failed to add point:", err)
			return
		}
	}

	err = index.BuildIndex()
	if err != nil {
		fmt.Println("Failed to build index:", err)
		return
	}
	fmt.Println("Index built success")

	queries := make([][]float32, numQueries)
	for i := uint32(0); i < numQueries; i++ {
		queries[i] = generateRandomVector(dimension)
	}

	// calculate average recall
	totalRecall := 0.0
	for i := uint32(0); i < numQueries; i++ {
		query := queries[i]

		// calculate ground truth
		var exactDistances []struct {
			id       uint32
			distance float32
		}
		for j := uint32(0); j < numPoints; j++ {
			dist := computeDistance(query, points[j])
			exactDistances = append(exactDistances, struct {
				id       uint32
				distance float32
			}{j, dist})
		}

		// Sort ground truth
		sort.Slice(exactDistances, func(i, j int) bool {
			return exactDistances[i].distance < exactDistances[j].distance
		})

		// Search
		ids, distances, err := index.Search(query, k)
		if err != nil {
			fmt.Printf("Search failed for query %d: %v\n", i, err)
			continue
		}

		// verify result size
		if uint32(len(ids)) != k || uint32(len(distances)) != k {
			fmt.Printf("Unexpected result size for query %d\n", i)
			continue
		}

		// calculate recall
		correct := 0
		groundTruthSet := make(map[uint32]struct{})
		for j := 0; j < int(k); j++ {
			groundTruthSet[exactDistances[j].id] = struct{}{}
		}
		for _, id := range ids {
			if _, exists := groundTruthSet[id]; exists {
				correct++
			}
		}

		recall := float64(correct) / float64(k)
		totalRecall += recall
		fmt.Printf("Query %d recall: %.2f\n", i, recall)
	}

	// calculate and verify average recall
	averageRecall := totalRecall / float64(numQueries)
	if averageRecall < 0.8 {
		fmt.Println("Average recall too low:", averageRecall)
		return
	}
	fmt.Printf("Average recall success: %.2f\n", averageRecall)

	// save index
	index.SaveIndex("index.bin")
}

func TestLoadIndex() {
	index, err := vamana.LoadIndex("index.bin")
	if err != nil {
		fmt.Println("Failed to load index:", err)
		return
	}
	defer index.Free()
	index.PrintParams()
	fmt.Println("loaded index success!")
	// // test search
	k := 10
	query := generateRandomVector(128) // 使用与创建索引时相同的维度
	ids, distances, err := index.Search(query, uint32(k))
	if err != nil {
		fmt.Printf("Search failed: %v\n", err)
		return
	}
	fmt.Printf("Search success: ids=%v, distances=%v\n", ids, distances)
}

func TestSearchWithStartPoint() {
	// 加载索引
	index, err := vamana.LoadIndex("../../data/sift10k_worker_data_similarity_16/worker-0_vamana.bin")
	if err != nil {
		fmt.Println("Failed to load index:", err)
		return
	}
	defer index.Free()

	// 测试参数
	k := 10
	query := generateRandomVector(128)

	// 先用普通搜索找到一些点
	ids, distances, err := index.Search(query, uint32(k))
	if err != nil {
		fmt.Printf("Normal search failed: %v\n", err)
		return
	}
	fmt.Printf("Normal search results: ids=%v, distances=%v\n", ids, distances)

	// 使用找到的第一个点作为起始点进行搜索
	if len(ids) > 0 {
		// 获取第一个结果点的向量作为起始点
		startPoint := ids[0]
		idsWithStart, distancesWithStart, err := index.SearchWithStartPoint(query, startPoint, uint32(k))
		if err != nil {
			fmt.Printf("Search with start point failed: %v\n", err)
			return
		}
		fmt.Printf("Search with start point results: ids=%v, distances=%v\n", idsWithStart, distancesWithStart)

		// 比较两次搜索的结果
		fmt.Println("Comparing results:")
		fmt.Printf("Normal search found %d results\n", len(ids))
		fmt.Printf("Search with start point found %d results\n", len(idsWithStart))
	}
}

func TestMetaIndex() {
	index, err := vamana.LoadIndex("../../data/sift10k_worker_data_similarity_16/meta_vamana.bin")
	if err != nil {
		fmt.Println("Failed to load index:", err)
		return
	}
	defer index.Free()
	// test search
	k := 10
	query := generateRandomVector(128) // 使用与创建索引时相同的维度
	index.PrintParams()
	ids, distances, err := index.Search(query, uint32(k))
	if err != nil {
		fmt.Printf("Search failed: %v\n", err)
		return
	}
	fmt.Printf("Search success: ids=%v, distances=%v\n", ids, distances)
}

func main() {
	TestMetaIndex()
	TestVamanaBasic()
	TestLoadIndex()
	TestSearchWithStartPoint()
}
