package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/WHU-STAR/TITAN/internal/engine/go_api/bf"
	"github.com/WHU-STAR/TITAN/internal/engine/go_api/hnsw"
)

func init() {
	rand.New(rand.NewSource(1))
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

func exampleBatchBuild() (*bf.BruteForceIndex, *hnsw.Index) {
	dim := 128
	maxElements := 100000
	M := 16
	numVectors := 10000
	numGoroutines := 10

	// create new bfindex
	bfindex := bf.NewIndex(uint32(dim), uint32(maxElements), uint32(M), "l2")
	hnswindex := hnsw.NewIndex(uint32(dim), uint32(maxElements), uint32(M), uint32(100), "l2")

	// generate new vectors
	vectors := generateRandVectors(numVectors, dim)
	ids := make([]uint32, numVectors)
	for i := 0; i < numVectors; i++ {
		ids[i] = uint32(i)
	}
	// add items to index
	startTime := time.Now()
	err := bfindex.AddItems(vectors, ids, numGoroutines)
	if err != nil {
		res := fmt.Errorf("failed to add items to index: %v", err)
		fmt.Println(res)
		return nil, nil
	}
	err = hnswindex.AddItems(vectors, ids, numGoroutines)
	if err != nil {
		res := fmt.Errorf("failed to add items to hnsw index: %v", err)
		fmt.Println(res)
		return nil, nil
	}

	buildTime := time.Since(startTime)
	fmt.Printf("Index built in %v\n", buildTime)
	return bfindex, hnswindex
}

func exampleSingleSearch(bfindex *bf.BruteForceIndex, hnswindex *hnsw.Index) {
	fmt.Println("Starting single search")
	startime := time.Now()
	query := generateRandVectors(10, 128)
	for i := 0; i < 10; i++ {
		label, distance, err := bfindex.SearchKNN(query[i], 10)
		if err != nil {
			errmes := fmt.Errorf("search failed: %v", err)
			fmt.Println(errmes)
			return
		}
		fmt.Printf("query %d results:\n", i)
		for j := 0; j < 10; j++ {
			fmt.Printf("label: %d, distance: %f\n", label[j], distance[j])
		}

		label, distance, err = hnswindex.SearchKNN(query[i], 10)
		if err != nil {
			errmes := fmt.Errorf("search failed: %v", err)
			fmt.Println(errmes)
			return
		}
		fmt.Printf("hnsw query %d results:\n", i)
		for j := 0; j < 10; j++ {
			fmt.Printf("label: %d, distance: %f\n", label[j], distance[j])
		}
	}

	searchTime := time.Since(startime)
	fmt.Printf("Search completed in %v\n", searchTime)

}

func exampleLoadIndex(path string, dim int) *bf.BruteForceIndex {
	index, err := bf.LoadIndex(path, dim, "l2")
	if err != nil {
		errmes := fmt.Errorf("failed to load index: %v", err)
		fmt.Println(errmes)
	}
	fmt.Println("index loaded")
	return index
}

func main() {

	bfIndex, hnswindex := exampleBatchBuild()
	fmt.Println("Index built successfully")
	bfIndex.SaveIndex("bf_index.bin")
	hnswindex.SaveIndex("hnsw_index.bin")
	loadedbfIndex := exampleLoadIndex("bf_index.bin", 128)
	exampleSingleSearch(loadedbfIndex, hnswindex)

}
