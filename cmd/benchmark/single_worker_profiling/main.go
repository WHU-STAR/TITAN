package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/WHU-STAR/TITAN/internal/engine/go_api/hnsw"
	"github.com/WHU-STAR/TITAN/internal/storage"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
)

type Config struct {
	WorkerDataDir      string     `yaml:"worker_data_dir"`
	DistributionJSON   string     `yaml:"distribution_json"`
	QueryPath          string     `yaml:"query_path"`
	GroundTruthPath    string     `yaml:"ground_truth_path"`
	OutputPrefix       string     `yaml:"output_prefix"`
	Dim                int        `yaml:"dim"`
	Metric             string     `yaml:"metric"`
	TopK               int        `yaml:"topk"`
	BucketRanges       [][]int    `yaml:"bucket_ranges"`
	MaxQueriesPerRange int        `yaml:"max_queries_per_range"`
	WarmupQueries      int        `yaml:"warmup_queries"`
	RepeatCount        int        `yaml:"repeat_count"`
	HNSW               HNSWConfig `yaml:"hnsw"`
	IVF                IVFConfig  `yaml:"ivf"`
}

type HNSWConfig struct {
	EfSearch []int `yaml:"ef_search"`
}

type IVFConfig struct {
	NProbe []int `yaml:"n_probe"`
}

type QueryDistribution struct {
	QueryID           int            `json:"query_id"`
	WorkersAndBuckets map[string]int `json:"workers_and_buckets"`
}

type QueryInfo struct {
	QueryID       int
	WorkerID      int
	BucketCount   int
	BucketIndices []uint32
}

type BenchmarkResult struct {
	QueryID      int
	WorkerID     int
	BucketCount  int
	Method       string
	Param        int
	TopK         int
	LocalGTCount int
	QPS          float64
	Recall       float64
	AvgLatencyMs float64
}

var (
	configPath  string
	mode        string
	workerID    int
	outputDir   string
	allWorkers  bool
	parallelism int
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "single_worker_profiling",
		Short: "Profile single worker QPS for HNSW and IVF methods",
		RunE:  runProfiling,
	}

	rootCmd.Flags().StringVarP(&configPath, "config", "c", "script/min_nodes_analysis/single_worker_profiling_config.yaml", "Path to config file")
	rootCmd.Flags().StringVarP(&mode, "mode", "m", "default", "Configuration mode to use")
	rootCmd.Flags().IntVarP(&workerID, "worker", "w", -1, "Worker ID to test (-1 for auto-detect all workers)")
	rootCmd.Flags().BoolVarP(&allWorkers, "all", "a", false, "Process all workers automatically")
	rootCmd.Flags().IntVarP(&parallelism, "parallel", "p", 1, "Number of workers to process in parallel")
	rootCmd.Flags().StringVarP(&outputDir, "output", "o", "script/min_nodes_analysis/results", "Output directory for CSV files")

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func runProfiling(cmd *cobra.Command, args []string) error {
	fmt.Printf("=== Single Worker QPS Profiling ===\n")
	fmt.Printf("Config: %s, Mode: %s\n\n", configPath, mode)

	config, err := loadConfig(configPath, mode)
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	workerIDs, err := detectWorkerIDs(config, workerID, allWorkers)
	if err != nil {
		return fmt.Errorf("failed to detect workers: %w", err)
	}
	fmt.Printf("Workers to process: %v\n\n", workerIDs)

	queryData, err := loadQueries(config)
	if err != nil {
		return fmt.Errorf("failed to load queries: %w", err)
	}
	fmt.Printf("Loaded %d queries\n", queryData.Num)

	groundTruth, err := storage.LoadGroundTruth(config.GroundTruthPath)
	if err != nil {
		return fmt.Errorf("failed to load ground truth: %w", err)
	}
	fmt.Printf("Loaded ground truth with %d entries\n\n", groundTruth.Num)

	if parallelism > 1 && len(workerIDs) > 1 {
		return runParallel(config, queryData, groundTruth, workerIDs)
	}

	return runSequential(config, queryData, groundTruth, workerIDs)
}

func detectWorkerIDs(config *Config, specifiedWorkerID int, processAll bool) ([]int, error) {
	if !processAll && specifiedWorkerID >= 0 {
		return []int{specifiedWorkerID}, nil
	}

	var workerIDs []int
	entries, err := os.ReadDir(config.WorkerDataDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read worker data dir: %w", err)
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if strings.HasPrefix(name, "worker-") && strings.HasSuffix(name, ".bin") {
			idStr := strings.TrimSuffix(strings.TrimPrefix(name, "worker-"), ".bin")
			id, err := strconv.Atoi(idStr)
			if err == nil {
				workerIDs = append(workerIDs, id)
			}
		}
	}

	if len(workerIDs) == 0 {
		return nil, fmt.Errorf("no worker files found in %s", config.WorkerDataDir)
	}

	sort.Ints(workerIDs)
	return workerIDs, nil
}

func runSequential(config *Config, queryData *storage.QueryData, groundTruth *storage.GroundTruth, workerIDs []int) error {
	var allResults []BenchmarkResult
	successCount := 0

	for _, wid := range workerIDs {
		fmt.Printf("\n========== Processing Worker %d ==========\n", wid)
		results, err := processWorker(config, queryData, groundTruth, wid)
		if err != nil {
			fmt.Printf("Warning: Worker %d failed: %v\n", wid, err)
			continue
		}
		allResults = append(allResults, results...)
		successCount++
	}

	if len(allResults) > 0 {
		combinedPath := filepath.Join(outputDir, fmt.Sprintf("%s_all_workers.csv", config.OutputPrefix))
		if err := saveResults(allResults, combinedPath); err != nil {
			fmt.Printf("Warning: Failed to save combined results: %v\n", err)
		} else {
			fmt.Printf("\nCombined results saved to: %s\n", combinedPath)
		}
	}

	fmt.Printf("\n=== Profiling Complete ===\n")
	fmt.Printf("Successfully processed %d/%d workers\n", successCount, len(workerIDs))
	fmt.Printf("Total results: %d\n", len(allResults))

	return nil
}

func runParallel(config *Config, queryData *storage.QueryData, groundTruth *storage.GroundTruth, workerIDs []int) error {
	type workerResult struct {
		workerID int
		results  []BenchmarkResult
		err      error
	}

	sem := make(chan struct{}, parallelism)
	resultsChan := make(chan workerResult, len(workerIDs))

	for _, wid := range workerIDs {
		sem <- struct{}{}
		go func(workerID int) {
			defer func() { <-sem }()
			results, err := processWorker(config, queryData, groundTruth, workerID)
			resultsChan <- workerResult{workerID: workerID, results: results, err: err}
		}(wid)
	}

	var allResults []BenchmarkResult
	successCount := 0
	for range workerIDs {
		wr := <-resultsChan
		if wr.err != nil {
			fmt.Printf("Warning: Worker %d failed: %v\n", wr.workerID, wr.err)
			continue
		}
		fmt.Printf("Worker %d completed: %d results\n", wr.workerID, len(wr.results))
		allResults = append(allResults, wr.results...)
		successCount++
	}

	if len(allResults) > 0 {
		combinedPath := filepath.Join(outputDir, fmt.Sprintf("%s_all_workers.csv", config.OutputPrefix))
		if err := saveResults(allResults, combinedPath); err != nil {
			fmt.Printf("Warning: Failed to save combined results: %v\n", err)
		} else {
			fmt.Printf("\nCombined results saved to: %s\n", combinedPath)
		}
	}

	fmt.Printf("\n=== Profiling Complete ===\n")
	fmt.Printf("Successfully processed %d/%d workers (parallelism: %d)\n", successCount, len(workerIDs), parallelism)
	fmt.Printf("Total results: %d\n", len(allResults))

	return nil
}

func processWorker(config *Config, queryData *storage.QueryData, groundTruth *storage.GroundTruth, wid int) ([]BenchmarkResult, error) {
	queryInfos, err := loadQueryDistribution(config, wid)
	if err != nil {
		return nil, fmt.Errorf("failed to load query distribution: %w", err)
	}

	if len(queryInfos) == 0 {
		fmt.Printf("No queries found for worker %d, skipping\n", wid)
		return nil, nil
	}
	fmt.Printf("Found %d queries for worker %d\n", len(queryInfos), wid)

	workerBinPath := filepath.Join(config.WorkerDataDir, fmt.Sprintf("worker-%d.bin", wid))
	nodeData, err := storage.LoadNodeData(workerBinPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load worker data: %w", err)
	}
	fmt.Printf("Worker %d: %d buckets, %d points\n", wid, nodeData.BucketIndicesCount, nodeData.AbsoluteLabelIndicesCount)

	var results []BenchmarkResult

	hnswResults, err := benchmarkHNSWForWorker(config, queryData.Queries, groundTruth.GroundTruth, queryInfos, nodeData, wid)
	if err != nil {
		fmt.Printf("Warning: HNSW benchmark failed for worker %d: %v\n", wid, err)
	} else {
		results = append(results, hnswResults...)
	}

	ivfResults, err := benchmarkIVFForWorker(config, queryData.Queries, groundTruth.GroundTruth, queryInfos, nodeData, wid)
	if err != nil {
		fmt.Printf("Warning: IVF benchmark failed for worker %d: %v\n", wid, err)
	} else {
		results = append(results, ivfResults...)
	}

	outputPath := filepath.Join(outputDir, fmt.Sprintf("%s_worker%d.csv", config.OutputPrefix, wid))
	if err := saveResults(results, outputPath); err != nil {
		return nil, fmt.Errorf("failed to save results: %w", err)
	}
	fmt.Printf("Worker %d results saved to: %s\n", wid, outputPath)

	return results, nil
}

func loadConfig(path, mode string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var allConfigs map[string]Config
	if err := yaml.Unmarshal(data, &allConfigs); err != nil {
		return nil, err
	}

	config, ok := allConfigs[mode]
	if !ok {
		return nil, fmt.Errorf("mode '%s' not found in config", mode)
	}

	if config.TopK == 0 {
		config.TopK = 20
	}
	if config.MaxQueriesPerRange == 0 {
		config.MaxQueriesPerRange = 50
	}
	if config.WarmupQueries == 0 {
		config.WarmupQueries = 10
	}
	if config.RepeatCount == 0 {
		config.RepeatCount = 5
	}

	return &config, nil
}

func loadQueries(config *Config) (*storage.QueryData, error) {
	return storage.LoadQuerySet(config.QueryPath)
}

func loadQueryDistribution(config *Config, targetWorkerID int) ([]QueryInfo, error) {
	jsonPath := config.DistributionJSON
	if !filepath.IsAbs(jsonPath) {
		scriptDir := filepath.Dir(configPath)
		jsonPath = filepath.Join(scriptDir, jsonPath)
	}

	data, err := os.ReadFile(jsonPath)
	if err != nil {
		return nil, err
	}

	var distributions []QueryDistribution
	if err := json.Unmarshal(data, &distributions); err != nil {
		return nil, err
	}

	var queryInfos []QueryInfo
	workerIDStr := strconv.Itoa(targetWorkerID)

	for _, dist := range distributions {
		bucketCount, ok := dist.WorkersAndBuckets[workerIDStr]
		if !ok || bucketCount == 0 {
			continue
		}

		inRange := false
		for _, r := range config.BucketRanges {
			if len(r) >= 2 && bucketCount >= r[0] && bucketCount <= r[1] {
				inRange = true
				break
			}
		}
		if !inRange && len(config.BucketRanges) > 0 {
			continue
		}

		queryInfos = append(queryInfos, QueryInfo{
			QueryID:     dist.QueryID,
			WorkerID:    targetWorkerID,
			BucketCount: bucketCount,
		})
	}

	sort.Slice(queryInfos, func(i, j int) bool {
		return queryInfos[i].BucketCount < queryInfos[j].BucketCount
	})

	if config.MaxQueriesPerRange > 0 && len(config.BucketRanges) > 0 {
		var filtered []QueryInfo
		for _, r := range config.BucketRanges {
			if len(r) < 2 {
				continue
			}
			minB, maxB := r[0], r[1]
			count := 0
			for _, qi := range queryInfos {
				if qi.BucketCount >= minB && qi.BucketCount <= maxB {
					filtered = append(filtered, qi)
					count++
					if count >= config.MaxQueriesPerRange {
						break
					}
				}
			}
		}
		queryInfos = filtered
	}

	return queryInfos, nil
}

func benchmarkHNSWForWorker(config *Config, queries [][]float32, groundTruth [][]uint32, queryInfos []QueryInfo, nodeData *storage.NodeData, wid int) ([]BenchmarkResult, error) {
	fmt.Printf("[Worker %d] Benchmarking HNSW...\n", wid)

	indexPath := filepath.Join(config.WorkerDataDir, fmt.Sprintf("worker-%d.hnsw", wid))
	if _, err := os.Stat(indexPath); os.IsNotExist(err) {
		indexPath = filepath.Join(config.WorkerDataDir, fmt.Sprintf("worker-%d_hnsw.bin", wid))
	}

	spaceType := strings.ToLower(config.Metric)
	if spaceType == "ip" || spaceType == "innerproduct" {
		spaceType = "ip"
	} else {
		spaceType = "l2"
	}

	hnswIndex, err := hnsw.LoadIndex(indexPath, config.Dim, spaceType)
	if err != nil {
		return nil, fmt.Errorf("failed to load HNSW index: %w", err)
	}
	defer hnswIndex.Unload()

	workerPointSet := buildWorkerPointSet(nodeData)
	fmt.Printf("[Worker %d] Built point set with %d points\n", wid, len(workerPointSet))

	var results []BenchmarkResult

	for _, efSearch := range config.HNSW.EfSearch {
		hnswIndex.SetEf(efSearch)

		if config.WarmupQueries > 0 {
			for i := 0; i < config.WarmupQueries && i < len(queryInfos); i++ {
				qi := queryInfos[i]
				if qi.QueryID < len(queries) {
					_, _, _ = hnswIndex.SearchKNN(queries[qi.QueryID], config.TopK)
				}
			}
		}

		for _, qi := range queryInfos {
			if qi.QueryID >= len(queries) || qi.QueryID >= len(groundTruth) {
				continue
			}

			query := queries[qi.QueryID]
			globalGT := groundTruth[qi.QueryID]
			localGT := filterLocalGroundTruth(globalGT, workerPointSet, config.TopK)

			if len(localGT) == 0 {
				continue
			}

			var totalLatency time.Duration
			var lastLabels []uint32

			for r := 0; r < config.RepeatCount; r++ {
				start := time.Now()
				labels, _, err := hnswIndex.SearchKNN(query, config.TopK)
				elapsed := time.Since(start)

				if err != nil {
					continue
				}

				totalLatency += elapsed
				lastLabels = labels
			}

			avgLatency := totalLatency / time.Duration(config.RepeatCount)
			qps := float64(time.Second) / float64(avgLatency)
			recall := calculateRecall(localGT, lastLabels)

			results = append(results, BenchmarkResult{
				QueryID:      qi.QueryID,
				WorkerID:     qi.WorkerID,
				BucketCount:  qi.BucketCount,
				Method:       "HNSW",
				Param:        efSearch,
				TopK:         config.TopK,
				LocalGTCount: len(localGT),
				QPS:          qps,
				Recall:       recall,
				AvgLatencyMs: float64(avgLatency.Microseconds()) / 1000.0,
			})
		}
	}

	return results, nil
}

func benchmarkIVFForWorker(config *Config, queries [][]float32, groundTruth [][]uint32, queryInfos []QueryInfo, nodeData *storage.NodeData, wid int) ([]BenchmarkResult, error) {
	fmt.Printf("[Worker %d] Benchmarking IVF (Pure Go) (%d buckets, %d points)...\n",
		wid, nodeData.BucketIndicesCount, nodeData.AbsoluteLabelIndicesCount)

	bucketMap := buildBucketDataMap(nodeData)
	fmt.Printf("[Worker %d] Built bucket data map with %d buckets\n", wid, len(bucketMap))

	workerPointSet := buildWorkerPointSet(nodeData)
	pointToBucket := buildPointToBucketMap(nodeData)

	var results []BenchmarkResult

	nProbeValues := config.IVF.NProbe
	if len(nProbeValues) == 0 {
		nProbeValues = []int{0}
	}

	for _, nProbe := range nProbeValues {
		useAllGTBuckets := (nProbe == 0)

		if config.WarmupQueries > 0 {
			for i := 0; i < config.WarmupQueries && i < len(queryInfos); i++ {
				qi := queryInfos[i]
				if qi.QueryID < len(queries) && qi.QueryID < len(groundTruth) {
					globalGT := groundTruth[qi.QueryID]
					localGT := filterLocalGroundTruth(globalGT, workerPointSet, config.TopK)
					gtBuckets := findBucketsForGT(localGT, pointToBucket)
					bucketsToSearch := selectIVFBuckets(gtBuckets, nProbe, useAllGTBuckets)
					if len(bucketsToSearch) > 0 {
						_ = searchBucketsKNN(queries[qi.QueryID], bucketsToSearch, bucketMap, config.TopK)
					}
				}
			}
		}

		for _, qi := range queryInfos {
			if qi.QueryID >= len(queries) || qi.QueryID >= len(groundTruth) {
				continue
			}

			query := queries[qi.QueryID]
			globalGT := groundTruth[qi.QueryID]
			localGT := filterLocalGroundTruth(globalGT, workerPointSet, config.TopK)

			if len(localGT) == 0 {
				fmt.Printf("[Worker %d] Query %d: No local GT, skipping IVF benchmark\n", wid, qi.QueryID)
				continue
			}

			gtBuckets := findBucketsForGT(localGT, pointToBucket)
			if len(gtBuckets) == 0 {
				fmt.Printf("[Worker %d] Query %d: No GT buckets found, skipping IVF benchmark\n", wid, qi.QueryID)
				continue
			}

			bucketsToSearch := selectIVFBuckets(gtBuckets, nProbe, useAllGTBuckets)
			actualBucketCount := len(bucketsToSearch)

			// 计算搜索范围内的GT（在bucketsToSearch中的GT）
			searchedGT := filterGTInBuckets(localGT, pointToBucket, bucketsToSearch)

			var totalLatency time.Duration
			var lastLabels []uint32

			for r := 0; r < config.RepeatCount; r++ {
				start := time.Now()
				lastLabels = searchBucketsKNN(query, bucketsToSearch, bucketMap, config.TopK)
				elapsed := time.Since(start)
				totalLatency += elapsed
			}

			avgLatency := totalLatency / time.Duration(config.RepeatCount)
			qps := float64(time.Second) / float64(avgLatency)
			// 计算搜索范围内的recall（在搜索的bucket中找到了多少GT）
			recall := calculateRecall(searchedGT, lastLabels)

			paramValue := nProbe
			if useAllGTBuckets {
				paramValue = actualBucketCount
			}

			results = append(results, BenchmarkResult{
				QueryID:      qi.QueryID,
				WorkerID:     qi.WorkerID,
				BucketCount:  qi.BucketCount, // 与HNSW一致，使用GT所在的bucket总数
				Method:       "IVF",
				Param:        paramValue,
				TopK:         config.TopK,
				LocalGTCount: len(searchedGT), // 搜索范围内的GT数
				QPS:          qps,
				Recall:       recall,
				AvgLatencyMs: float64(avgLatency.Microseconds()) / 1000.0,
			})
		}
	}

	return results, nil
}

type PointData struct {
	PointID uint32
	Data    []float32
}

type BucketData struct {
	BucketID uint32
	Points   []PointData
}

func buildBucketDataMap(nodeData *storage.NodeData) map[uint32]*BucketData {
	bucketMap := make(map[uint32]*BucketData, len(nodeData.Buckets))
	for _, bucket := range nodeData.Buckets {
		bd := &BucketData{
			BucketID: bucket.BucketId,
			Points:   make([]PointData, len(bucket.Points)),
		}
		for i, p := range bucket.Points {
			bd.Points[i] = PointData{
				PointID: p.PointId,
				Data:    p.Data,
			}
		}
		bucketMap[bucket.BucketId] = bd
	}
	return bucketMap
}

type distanceResult struct {
	pointID  uint32
	distance float32
}

func searchBucketsKNN(query []float32, bucketIDs []uint32, bucketMap map[uint32]*BucketData, topK int) []uint32 {
	var candidates []distanceResult

	for _, bucketID := range bucketIDs {
		bucket, ok := bucketMap[bucketID]
		if !ok {
			continue
		}
		for _, point := range bucket.Points {
			dist := computeL2Distance(query, point.Data)
			candidates = append(candidates, distanceResult{
				pointID:  point.PointID,
				distance: dist,
			})
		}
	}

	if len(candidates) == 0 {
		return nil
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].distance < candidates[j].distance
	})

	k := topK
	if k > len(candidates) {
		k = len(candidates)
	}

	results := make([]uint32, k)
	for i := 0; i < k; i++ {
		results[i] = candidates[i].pointID
	}
	return results
}

func computeL2Distance(a, b []float32) float32 {
	if len(a) != len(b) {
		return math.MaxFloat32
	}

	var sum float32
	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

func selectIVFBuckets(gtBuckets []uint32, nProbe int, useAll bool) []uint32 {
	if useAll || nProbe <= 0 || nProbe >= len(gtBuckets) {
		return gtBuckets
	}
	return gtBuckets[:nProbe]
}

// filterGTInBuckets 返回在指定bucket中的GT点
func filterGTInBuckets(localGT []uint32, pointToBucket map[uint32]uint32, buckets []uint32) []uint32 {
	bucketSet := make(map[uint32]bool, len(buckets))
	for _, b := range buckets {
		bucketSet[b] = true
	}

	var filtered []uint32
	for _, gtID := range localGT {
		if bucketID, ok := pointToBucket[gtID]; ok {
			if bucketSet[bucketID] {
				filtered = append(filtered, gtID)
			}
		}
	}
	return filtered
}

func selectBuckets(allBuckets []uint32, nProbe int) []uint32 {
	if nProbe >= len(allBuckets) {
		return allBuckets
	}
	if nProbe <= 0 {
		nProbe = 1
	}
	return allBuckets[:nProbe]
}

func buildWorkerPointSet(nodeData *storage.NodeData) map[uint32]bool {
	pointSet := make(map[uint32]bool, len(nodeData.AbsoluteLabelIndices))
	for _, label := range nodeData.AbsoluteLabelIndices {
		if label >= 0 {
			pointSet[uint32(label)] = true
		}
	}
	return pointSet
}

func buildPointToBucketMap(nodeData *storage.NodeData) map[uint32]uint32 {
	pointToBucket := make(map[uint32]uint32)

	pointOffset := 0
	for i, bucketID := range nodeData.BucketIndices {
		numPoints := int(nodeData.DataSizePerBucket[i])
		for j := 0; j < numPoints; j++ {
			idx := pointOffset + j
			if idx < len(nodeData.AbsoluteLabelIndices) {
				pointID := nodeData.AbsoluteLabelIndices[idx]
				if pointID >= 0 {
					pointToBucket[uint32(pointID)] = bucketID
				}
			}
		}
		pointOffset += numPoints
	}

	return pointToBucket
}

func findBucketsForGT(localGT []uint32, pointToBucket map[uint32]uint32) []uint32 {
	bucketSet := make(map[uint32]bool)
	for _, gtID := range localGT {
		if bucketID, ok := pointToBucket[gtID]; ok {
			bucketSet[bucketID] = true
		}
	}

	buckets := make([]uint32, 0, len(bucketSet))
	for bucketID := range bucketSet {
		buckets = append(buckets, bucketID)
	}
	return buckets
}

func filterLocalGroundTruth(globalGT []uint32, workerPointSet map[uint32]bool, topK int) []uint32 {
	var localGT []uint32
	for _, gtID := range globalGT {
		if workerPointSet[gtID] {
			localGT = append(localGT, gtID)
		}
		if len(localGT) >= topK {
			break
		}
	}
	return localGT
}

func calculateRecall(localGT, results []uint32) float64 {
	if len(localGT) == 0 {
		return 0.0
	}
	if len(results) == 0 {
		return 0.0
	}

	gtSet := make(map[uint32]bool, len(localGT))
	for _, gtID := range localGT {
		gtSet[gtID] = true
	}

	correct := 0
	for _, r := range results {
		if gtSet[r] {
			correct++
		}
	}

	return float64(correct) / float64(len(localGT))
}

func saveResults(results []BenchmarkResult, outputPath string) error {
	if err := os.MkdirAll(filepath.Dir(outputPath), 0755); err != nil {
		return err
	}

	file, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	header := []string{"query_id", "worker_id", "bucket_count", "method", "param", "topk", "local_gt_count", "qps", "recall", "avg_latency_ms"}
	if err := writer.Write(header); err != nil {
		return err
	}

	for _, r := range results {
		row := []string{
			strconv.Itoa(r.QueryID),
			strconv.Itoa(r.WorkerID),
			strconv.Itoa(r.BucketCount),
			r.Method,
			strconv.Itoa(r.Param),
			strconv.Itoa(r.TopK),
			strconv.Itoa(r.LocalGTCount),
			fmt.Sprintf("%.2f", r.QPS),
			fmt.Sprintf("%.4f", r.Recall),
			fmt.Sprintf("%.4f", r.AvgLatencyMs),
		}
		if err := writer.Write(row); err != nil {
			return err
		}
	}

	return nil
}
