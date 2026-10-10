package main

import (
	"fmt"
	"math/rand"
	"runtime"
	"time"

	"github.com/WHU-STAR/TITAN/internal/engine/go_api/hnsw"
)

func init() {
	rand.New(rand.NewSource(1))
}

func toMegaBytes(bytes uint64) float64 {
	return float64(bytes) / 1024 / 1024
}

func traceMemStats() {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	var result = make([]float64, 8)
	// 堆中分配的对象总数
	result[0] = float64(ms.HeapObjects)
	// 堆中分配的对象字节数
	result[1] = toMegaBytes(ms.HeapAlloc)
	// 已经分配给对象的累计字节数
	result[2] = toMegaBytes(ms.TotalAlloc)
	result[3] = toMegaBytes(ms.HeapSys)
	// ！！ 闲置的内存
	result[4] = toMegaBytes(ms.HeapIdle)
	result[5] = toMegaBytes(ms.HeapReleased)
	// 空闲但还未还给操作系统的内存
	result[6] = toMegaBytes(ms.HeapIdle - ms.HeapReleased)
	result[7] = toMegaBytes(ms.Alloc)

	fmt.Printf("%d\t", time.Now().Unix())
	for _, v := range result {
		fmt.Printf("%.2f\t", v)
	}
	fmt.Printf("\n")
	time.Sleep(2 * time.Second)
}

func exampleBatchBuild() *hnsw.Index {
	dim := 128
	maxElements := 100000
	M := 16
	efConstruction := 100
	numVectors := 10000
	numGoroutines := 10

	// create new index
	index := hnsw.NewIndex(uint32(dim), uint32(maxElements), uint32(M), uint32(efConstruction), "l2")

	// generate new vectors
	vectors := generateRandVectors(numVectors, dim)
	ids := make([]uint32, numVectors)
	for i := 0; i < numVectors; i++ {
		ids[i] = uint32(i)
	}
	// add items to index
	startTime := time.Now()
	err := index.AddItems(vectors, ids, numGoroutines)
	if err != nil {
		res := fmt.Errorf("failed to add items to index: %v", err)
		fmt.Println(res)
		return nil
	}
	buildTime := time.Since(startTime)
	fmt.Printf("Index built in %v\n", buildTime)
	return index
}

func exampleSingleSearch(index *hnsw.Index) {
	// set search parameters
	ef := 100
	k := 10
	index.SetEf(ef)

	// search
	fmt.Println("Starting single search...")
	query := generateRandVectors(10, 128)
	startTime := time.Now()
	for i := 0; i < 10; i++ {
		label, distance, err := index.SearchKNN(query[i], k)
		if err != nil {
			errmes := fmt.Errorf("search failed: %v", err)
			fmt.Println(errmes)
			return
		}
		fmt.Printf("query %d results:\n", i)
		for j := 0; j < k; j++ {
			fmt.Printf("label: %d, distance: %f\n", label[j], distance[j])
		}
	}
	searchTime := time.Since(startTime)
	fmt.Printf("Search completed in %v\n", searchTime)
}

func exampleBatchSearch(index *hnsw.Index) {
	// set search parameters
	ef := 100
	k := 10
	numGoroutines := 10000
	index.SetEf(ef)

	// generate queries
	queries := generateRandVectors(10000, 128)

	// search
	startTime := time.Now()
	fmt.Println("Starting batch search...")
	_, _, err := index.BatchSearchKnn(queries, k, numGoroutines)
	if err != nil {
		errmes := fmt.Errorf("search failed: %v", err)
		fmt.Println(errmes)
		return
	}
	searchTime := time.Since(startTime)
	fmt.Printf("Search completed in %v\n", searchTime)

	// print results
	// for i := 0; i < 10; i++ {
	// 	fmt.Printf("query %d results:\n", i)
	// 	for j := 0; j < k; j++ {
	// 		fmt.Printf("label: %d, distance: %f\n", labels[i][j], distances[i][j])
	// 	}
	// }
}

func exampleLoadIndex(path string) *hnsw.Index {
	index, err := hnsw.LoadIndex(path, 128, "l2")
	if err != nil {
		errmes := fmt.Errorf("failed to load index: %v", err)
		fmt.Println(errmes)
	}
	fmt.Println("index loaded")
	return index
}

func main() {
	// Test batch build
	buildedindex := exampleBatchBuild()

	// Test save index
	fmt.Println("Save Index")
	buildedindex.SaveIndex("index.bin")

	// Test load index
	index := exampleLoadIndex("index.bin")

	exampleSingleSearch(index)

	fmt.Println("Avg Dist Computations:", index.GetAvgDistComputations())
	fmt.Println("Avg Hops:", index.GetAvgHops())
	fmt.Println("Query Count:", index.GetQueryCount())

	// var vec1, vec2 [][]float32
	// for i := 0; i < 100; i++ {
	// 	vec1 = append(vec1, buildedindex.GetVectorByLabel(uint32(i), 128))
	// 	vec2 = append(vec2, index.GetVectorByLabel(uint32(i), 128))
	// }

	// fmt.Println("Vectors are equal? :", reflect.DeepEqual(vec1, vec2))

	// // Test single search
	// exampleSingleSearch(index)

	// // Test batch search
	// exampleBatchSearch(index)

	// fmt.Println("Max elements:", index.GetMaxElements())
	// fmt.Println("Current elements:", index.GetCurrentElementCount())
	// fmt.Println("Deleted Count:", index.GetDeletedCount())

	// // Test unload index
	// fmt.Println("Before Unload")
	// traceMemStats()
	// index.Unload()
	// fmt.Println("After Unload")
	// traceMemStats()

}

func generateRandVectors(numVectors, dim int) [][]float32 {
	vectors := make([][]float32, numVectors)
	for i := 0; i < numVectors; i++ {
		vectors[i] = make([]float32, dim)
		for j := 0; j < dim; j++ {
			vectors[i][j] = rand.Float32()
		}
	}
	return vectors
}
