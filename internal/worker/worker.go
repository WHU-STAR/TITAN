package worker

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net"
	"runtime"
	"runtime/debug"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/WHU-STAR/TITAN/config"
	"github.com/WHU-STAR/TITAN/internal/definitions"
	"github.com/WHU-STAR/TITAN/internal/engine/go_api/ivf"
	"github.com/WHU-STAR/TITAN/internal/rpc"
	"github.com/WHU-STAR/TITAN/internal/storage"
	"github.com/WHU-STAR/TITAN/pkg/logger"
	pb "github.com/WHU-STAR/TITAN/protocol/pb"
	"go.uber.org/zap"
	"google.golang.org/grpc"
)

const (
	MAX_FAILED_HEARTBEATS      = 3               // 最大失败心跳次数
	HB_INTERVAL                = time.Second * 3 // 心跳间歇
	ONGOING_INTERNAL_WEIGHT    = 2
	ONGOING_META_WEIGHT        = 3
	ONGOING_AGGREGATION_WEIGHT = 4
	PENDING_AGGREGATION_WEIGHT = 5
	PENDING_TASK_WEIGHT        = 7
)

var (
	ErrNodeDataNotFound      = errors.New("node data file not found")
	ErrMetaIndexNotSupported = errors.New("meta index is not supported")
	ErrUnknownTaskType       = errors.New("unknown task type")
	ErrLoadDataFailed        = errors.New("failed to load data")
	ErrAddItemsFailed        = errors.New("failed to add items")
	ErrFileNotFound          = errors.New("file not found")
)

type ProfileTimer struct {
	TotalTime     time.Duration
	PostQueryTime time.Duration
	TimerLock     sync.Mutex
}

type IndexWrapper struct {
	Index    Index
	IvfIndex *ivf.Index
}

// Interface for query operations
type QueryOperations interface {
	InternalQuery(ctx context.Context, query *definitions.Query) ([]uint32, []float32, error)
	MetaQuery(query []float32, topk int64) (map[uint32][]uint32, error)
}

// Interface for task processing
type TaskProcessor interface {
	ProcessTask(ctx context.Context, task *definitions.Task) error
	ProcessNodeQuery(ctx context.Context, task *definitions.Task) error
	ProcessMetaQuery(ctx context.Context, task *definitions.Task) error
}

type Worker struct {
	// Identify and Network config
	ID              string          // 格式：worker-0
	Address         string          // 地址
	Mode            string          // 模式：singleNode, distributed
	strategy        string          // 策略：naive, similarity
	metaK           int             // 元索引的k值
	ivfThreshold    int             // 本地索引的阈值
	metaIndexType   string          // 元索引的类型
	nodeIndexType   string          // 节点索引的类型
	searchMethod    string          // 搜索类型：default, rf(radius filtering), hf(heuristic filtering)
	pruningStrategy PruningStrategy // bucket 剪枝策略
	pruningLambda   float64         // 剪枝算法的λ参数，控制剪枝强度/召回保证

	// Indices
	NodeIndex                *IndexWrapper
	MetaIndex                Index
	bucketNodeLabels         [][]uint32
	nearestPointsToCentroids []uint32  // 每个bucket对应的最近点到中心点的索引
	bucketRadiuses           []float32 // 每个bucket对应的半径
	bucketSizes              []uint32  // 每个bucket的向量数量 |B_b|
	totalNumVectors          uint32    // 数据集总向量数 N
	absoluteNodeIndex        []int32   // 用于元索引的绝对索引
	BucketIndicesCount       uint32
	// bucketCentroids    [][]float32

	// task management
	taskChannel chan *definitions.Task
	stopCh      chan struct{}

	// Counters (atmoic)
	ongoingInternalQueryCount int32 // !!!!!需要使用atomic访问!!!!!
	ongoingMetaQueryCount     int32 // !!!!!需要使用atomic访问!!!!!
	pendingAggregationCount   int32 // !!!!!需要使用atomic访问!!!!!
	ongoingAggregationCount   int32 // !!!!!需要使用atomic访问!!!!!
	totalInternalQueries      int32
	totalMetaQueries          int32

	// Global workers
	GlobalWorkers map[string]string // global workers and their addresses

	/* ****************** TEMPORARY VARIABLES**************************** */
	// totalInternalQueryTime     time.Duration
	// totalInternalQueryTimeLock sync.Mutex
	// totalMetaQueryTime         time.Duration
	// totalMetaQueryTimeLock     sync.Mutex
	// Performance tracking
	InternalQueryTimer ProfileTimer
	MetaQueryTimer     ProfileTimer
	AggregationTimer   ProfileTimer

	TotalBucketCount int64 // 所有被统计的桶的总数
	// Global statistics
	TotalFilteredBucketCount int64 // 被过滤掉的桶的总数
}

func FormatID(i int) string {
	return fmt.Sprintf("worker-%d", i)
}

func NewWorker(id string, address string, mode string, strategy string) *Worker {
	metaK := config.Config.MetaK
	configinfo := config.Config.GetInfo()
	logger.Info("config: ", zap.String("config", configinfo))
	// logger.Debug("config: ", zap.String("config", configinfo))
	return &Worker{
		ID:                 id,
		Address:            address,
		Mode:               mode,
		metaK:              metaK[0],
		ivfThreshold:       config.Config.Thresh[0],
		metaIndexType:      config.Config.MetaIndexType,
		nodeIndexType:      config.Config.NodeIndexType,
		searchMethod:       config.Config.SearchMethod,
		pruningStrategy:    ParsePruningStrategy(config.Config.PruningStrategy),
		strategy:           strategy,
		pruningLambda:      0.8,
		taskChannel:        make(chan *definitions.Task, 10000),
		stopCh:             make(chan struct{}),
		GlobalWorkers:      make(map[string]string),
		MetaQueryTimer:     ProfileTimer{time.Duration(0), time.Duration(0), sync.Mutex{}},
		InternalQueryTimer: ProfileTimer{time.Duration(0), time.Duration(0), sync.Mutex{}},
		NodeIndex:          &IndexWrapper{Index: &HNSWIndex{}, IvfIndex: &ivf.Index{}},
	}
}

// InitMetaIndex 初始化元索引
func (w *Worker) InitMetaIndex() (bool, error) {
	metaIndexDataPath := "../data/workerdata/meta_index_data.bin"
	var metaIndexPath string
	indexType := w.metaIndexType
	if indexType == "bf" || indexType == "hnsw_impr" {
		indexType = "hnsw"
	}
	metaIndexPath = "../data/workerdata/meta_" + indexType + ".bin"

	// 判断这两个文件是否存在，如果有一个不存在，直接返回
	_, err := PathExists(metaIndexDataPath)
	if err != nil {
		return false, fmt.Errorf("meta index data file not found: %s, error: %v", metaIndexDataPath, ErrFileNotFound)
	}

	_, err = PathExists(metaIndexPath)
	if err != nil {
		return false, fmt.Errorf("meta index file not found: %s, error: %v", metaIndexPath, ErrFileNotFound)
	}

	metaIndexData, err := storage.LoadMetaIndexData(metaIndexDataPath, w.searchMethod)
	if err != nil {
		logger.Error("failed to load meta index data", zap.Error(err))
		return false, ErrLoadDataFailed
	}

	// 创建对应类型的索引
	_metaIndex, err := NewIndex(indexType)
	if err != nil {
		return false, err
	}

	// 加载索引数据
	err = _metaIndex.Load(metaIndexPath, int(metaIndexData.Dim), "l2")
	if err != nil {
		return false, ErrLoadDataFailed
	}

	logger.Info("load meta index data")
	// logger.Debug("load meta index data")
	w.MetaIndex = _metaIndex
	w.bucketNodeLabels = metaIndexData.BucketNodeLabels
	w.nearestPointsToCentroids = metaIndexData.NearestPointLabels
	w.bucketRadiuses = metaIndexData.BucketRadiuses
	w.bucketSizes = metaIndexData.BucketSizes
	w.totalNumVectors = metaIndexData.TotalNumVectors
	// logger.Debug("nearestPointsToCentroids", zap.Any("nearestPointsToCentroids", w.nearestPointsToCentroids))

	// 去除每一个维度的 -1，并调整每个bucket对应的node list大小
	for i := range w.bucketNodeLabels {
		// 找出非-1的节点
		validNodes := make([]uint32, 0)
		for _, nodeID := range w.bucketNodeLabels[i] {
			if nodeID != math.MaxUint32 { // -1 在uint32中是MaxUint32
				validNodes = append(validNodes, nodeID)
			}
		}
		// 更新为实际大小的slice
		w.bucketNodeLabels[i] = validNodes
	}
	// w.bucketCentroids = metaIndexData.Centroids

	ids := make([]uint32, metaIndexData.NumBuckets)
	for i := uint32(0); i < metaIndexData.NumBuckets; i++ {
		ids[i] = i
	}
	return true, nil
}

// InitNodeIndex 初始化节点索引
func (w *Worker) InitNodeIndex() error {
	nodeDataPath := fmt.Sprintf("../data/workerdata/%s.bin", w.ID)

	// 首先检查文件是否存在
	_, err := PathExists(nodeDataPath)
	if err != nil {
		return fmt.Errorf("node data file not found: %s, error: %v", nodeDataPath, ErrFileNotFound)
	}

	nodeData, err := storage.LoadNodeData(nodeDataPath)
	if err != nil {
		logger.Error("failed to load node data", zap.Error(err))
		return ErrLoadDataFailed
	}

	indexType := w.nodeIndexType
	if indexType == "hnsw_impr" {
		indexType = "hnsw"
	}
	startTime := time.Now()

	// Keep the small metadata fields needed later.
	w.absoluteNodeIndex = nodeData.AbsoluteLabelIndices
	w.BucketIndicesCount = nodeData.BucketIndicesCount
	logger.Debug("len of nodeData.BucketIndices", zap.Any("nodeData.BucketIndices", len(nodeData.BucketIndices)))
	logger.Debug("len of nodeData.AbsoluteLabelIndices", zap.Any("nodeData.AbsoluteLabelIndices", len(nodeData.AbsoluteLabelIndices)))
	logger.Debug("len of nodeData.Buckets", zap.Any("nodeData.Buckets", len(nodeData.Buckets)))

	// Build the IVF index from the node vectors FIRST, then drop the Go-side
	// vector materialization (nodeData.Buckets) BEFORE loading the HNSW graph.
	// Otherwise the three-way coexistence of (Go vectors + IVF copy + HNSW)
	// inflates peak load memory by ~one full vector copy (~30GB at 1B scale).
	// The search path is unchanged, so QPS/recall are unaffected.
	w.NodeIndex.IvfIndex = ivf.NewIndex(int(nodeData.Dim), nodeData.BucketIndices)
	err = w.NodeIndex.IvfIndex.AddItems(nodeData.Buckets)
	if err != nil {
		logger.Error("failed to add items to ivf index", zap.Error(err))
		return ErrAddItemsFailed
	}
	// IVF now owns its own copy of the vectors; release the Go copy and force it
	// back to the OS so the subsequent HNSW (C malloc) load does not stack on top.
	nodeData.Buckets = nil
	runtime.GC()
	debug.FreeOSMemory()

	// if indexType is ivf, we don't need to load graph-based index
	if indexType != "ivf" {
		nodeIndexPath := fmt.Sprintf("../data/workerdata/%s_%s.bin", w.ID, indexType)

		_, err = PathExists(nodeIndexPath)
		if err != nil {
			return fmt.Errorf("node index file not found: %s, error: %v", nodeIndexPath, ErrFileNotFound)
		}

		// load hnsw index
		_nodeIndex, err := NewIndex(indexType)
		if err != nil {
			return err
		}

		err = _nodeIndex.Load(nodeIndexPath, int(nodeData.Dim), "l2")
		if err != nil {
			logger.Error("failed to load node index", zap.Error(err))
			return ErrLoadDataFailed
		}
		w.NodeIndex.Index = _nodeIndex
	}

	buildTime := time.Since(startTime)
	// logger.Info("Node index built in", zap.Duration("buildTime", buildTime))
	logger.Debug("Node index built in", zap.Duration("buildTime", buildTime))

	return nil
}

// InitIndex 初始化工作节点的索引，包括nodeIndex和metaIndex
func (w *Worker) InitIndex() error {
	var err error
	if w.strategy == "similarity" {
		_, err = w.InitMetaIndex()
		if err != nil {
			return err
		}
		// logger.Info("Strategy: similarity, meta index initialized")
		logger.Debug("Strategy: similarity, meta index initialized")
	}
	err = w.InitNodeIndex()
	if err != nil {
		return err
	}
	// logger.Info("Node index initialized")
	logger.Debug("Node index initialized")
	return nil
}

// Start 启动工作节点的gRPC服务器
func (w *Worker) StartGRPCServer() {
	server := grpc.NewServer()
	pb.RegisterWorkerExtServiceServer(server, &WorkerServer{
		Worker: w,
	})

	pb.RegisterWorkerIntServiceServer(server, &WorkerServer{
		Worker: w,
	})

	lis, err := net.Listen("tcp", w.Address)
	if err != nil {
		panic(err)
	}

	// logger.Info("Worker gRPC server started at", zap.String("address", w.Address))
	logger.Debug("Worker gRPC server started at", zap.String("address", w.Address))
	if err := server.Serve(lis); err != nil {
		logger.Error("Failed to start worker gRPC server", zap.Error(err))
	}
}

func (w *Worker) RegisterToScheduler(ctx context.Context) error {
	// TODO: 完善逻辑：向调度器注册后，调度器返回全局工作节点信息
	client, err := rpc.GetSchedulerClient()
	if err != nil {
		return err
	}

	_, err = client.RegisterNode(ctx, &pb.RegisterNodeRequest{
		NodeId:  w.ID,
		Address: w.Address,
	})
	return err
}

func (w *Worker) AssignTasks(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case task, ok := <-w.taskChannel:
			if !ok {
				return
			}
			go func() {
				err := w.processTask(ctx, task)
				if err != nil {
					w.taskChannel <- task
				}
			}()
		}
	}
}

// Stop 停止工作节点
func (w *Worker) Stop() {
	// 关闭任务通道和停止通道
	close(w.taskChannel)
	close(w.stopCh)
	// 释放索引
	if w.NodeIndex != nil && w.NodeIndex.Index != nil {
		w.NodeIndex.Index.Unload()
	}
	if w.MetaIndex != nil {
		w.MetaIndex.Unload()
	}
}

// ReceiveTask 处理接收到的任务
func (w *Worker) ReceiveTask(ctx context.Context, task *definitions.Task) (taskID string, err error) {
	select {
	case w.taskChannel <- task:
		return task.ID, nil
	case <-ctx.Done():
		return "", ctx.Err()
	}

}

func (w *Worker) UseIVF(bucketsCount int) bool {
	if w.nodeIndexType == "ivf" {
		return true
	}
	return w.Mode == "distributed" && w.strategy == "similarity" && float64(bucketsCount) <= float64(w.ivfThreshold)
}

func (w *Worker) UseSearchWithBucketId(bucketsCount int) bool {
	return w.Mode == "distributed" && w.strategy == "similarity" && float64(bucketsCount) > 0
}

// InternalQuery 内部查询
// 返回：检索结果和距离
func (w *Worker) InternalQuery(ctx context.Context, query *definitions.Query) ([]uint32, []float32, error) {
	// TODO：根据召回率目标和topk确定检索参数
	topk := query.Topk
	// logger.Info("internalQuery of worker", zap.String("workerID", w.ID))
	logger.Debug("internalQuery of worker", zap.String("workerID", w.ID))
	atomic.AddInt32(&(w.ongoingInternalQueryCount), 1)
	atomic.AddInt32(&(w.totalInternalQueries), 1)

	// Initialize variables
	var labels []uint32
	var distances []float32
	var err error
	var internalStartingTime time.Time
	var internalQueryTime time.Duration

	useIVF := w.UseIVF(len(query.Buckets))
	logger.Debug("bucket num:", zap.Int("bucketNum", len(query.Buckets)))
	// Switch between ivf and hnsw
	if useIVF {
		internalStartingTime = time.Now()
		labels, distances, err = w.NodeIndex.IvfIndex.SearchKNN(query.Query, query.Buckets, topk)
		internalQueryTime = time.Since(internalStartingTime)
		logger.Debug("Using ivf index", zap.Int("total_buckets", len(query.Buckets)), zap.Int("topk", topk))
	} else {
		if w.NodeIndex.Index == nil {
			return nil, nil, errors.New("graph index is nil")
		}
		useSearchWithBucketId := w.UseSearchWithBucketId(len(query.Buckets))
		internalStartingTime = time.Now()
		// 选择第一个bucket的中心点作为起始点，目前只有vamana支持该方法
		if vIndex, ok := w.NodeIndex.Index.(*VamanaIndex); ok && useSearchWithBucketId {
			logger.Debug("Using vamana SearchWithBucketId")
			nearestCentroid := query.Buckets[len(query.Buckets)-1]
			labels, distances, err = vIndex.SearchWithBucketId(query.Query, nearestCentroid, topk)
		} else if w.nodeIndexType == "hnsw_impr" {
			logger.Debug("Using hnsw_impr SearchWithBucketId")
			logger.Debug("query.Buckets: ", zap.Any("buckets", query.Buckets))
			// logger.Debug("all labels", zap.Any("query", w.NodeIndex.GraphIndex.ShowAllLabels()))
			nearestCentroid := query.Buckets[len(query.Buckets)-1]
			entryPoint := w.nearestPointsToCentroids[nearestCentroid]
			labels, distances, err = w.NodeIndex.Index.SearchWithEntry(query.Query, entryPoint, topk)
		} else {
			logger.Debug("Using normal graph index Search")
			labels, distances, err = w.NodeIndex.Index.Search(query.Query, topk)
		}
		internalQueryTime = time.Since(internalStartingTime)
	}

	w.InternalQueryTimer.TimerLock.Lock()
	w.InternalQueryTimer.TotalTime += internalQueryTime
	w.InternalQueryTimer.TimerLock.Unlock()
	if err != nil {
		return nil, nil, err
	}
	// logger.Debug("labels: ", zap.Any("labels", labels))
	logger.Debug("distances: ", zap.Any("distances", distances))
	logger.Debug("Ongoing internal query count: ", zap.Int32("count", atomic.LoadInt32(&(w.ongoingInternalQueryCount))))
	atomic.AddInt32(&(w.ongoingInternalQueryCount), -1)
	postQueryStartingTime := time.Now()

	if len(labels) == 0 {
		logger.Warn("no results found")
		return nil, nil, nil
	}

	postQueryTime := time.Since(postQueryStartingTime)
	w.InternalQueryTimer.TimerLock.Lock()
	w.InternalQueryTimer.PostQueryTime += postQueryTime
	w.InternalQueryTimer.TimerLock.Unlock()
	return labels, distances, nil
}

// GetStatus 获取工作节点的状态
func (w *Worker) GetStatus() int {
	// 实现加权求和
	ongoingInternalQueryCount := atomic.LoadInt32(&w.ongoingInternalQueryCount)
	ongoingMetaCount := atomic.LoadInt32(&w.ongoingMetaQueryCount)
	ongoingAggregationCount := atomic.LoadInt32(&w.ongoingAggregationCount)
	pendingAggregationCount := atomic.LoadInt32(&w.pendingAggregationCount)
	pendingTaskCount := int32(len(w.taskChannel))
	return int(
		ongoingInternalQueryCount*ONGOING_INTERNAL_WEIGHT +
			ongoingMetaCount*ONGOING_META_WEIGHT +
			ongoingAggregationCount*ONGOING_AGGREGATION_WEIGHT +
			pendingAggregationCount*PENDING_AGGREGATION_WEIGHT +
			pendingTaskCount*PENDING_TASK_WEIGHT)
}

// sendHeartbeat 发送心跳
func (w *Worker) sendHeartbeat(ctx context.Context) error {
	client, err := rpc.GetSchedulerClient()
	if err != nil {
		return err
	}

	response, err := client.Heartbeat(ctx, &pb.HeartbeatRequest{
		NodeId: w.ID,
		Score:  int64(w.GetStatus()),
	})
	if err != nil {
		return err
	}

	// 更新节点表
	oldWorkerCount := len(w.GlobalWorkers)
	w.GlobalWorkers = response.NodeTable
	newWorkerCount := len(w.GlobalWorkers)

	if oldWorkerCount != newWorkerCount {
		// logger.Info("Node table updated",
		logger.Debug("Node table updated",
			zap.Int("oldCount", oldWorkerCount),
			zap.Int("newCount", newWorkerCount),
			zap.String("workerID", w.ID))
	}

	return nil
}

// StartHealthCheck 定期向调度器发送心跳
func (w *Worker) StartHealthCheck(ctx context.Context) {
	ticker := time.NewTicker(HB_INTERVAL)
	defer ticker.Stop()

	failedHeartbeats := 0

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			err := w.sendHeartbeat(ctx)
			if err != nil {
				logger.Error("Error sending heartbeat", zap.Error(err))
				failedHeartbeats++
				if failedHeartbeats >= MAX_FAILED_HEARTBEATS {
					logger.Error("Too many failed heartbeats. Exiting.")
					return
				}
			}
			// Reset the failed heartbeats counter
			failedHeartbeats = 0
		case <-w.stopCh:
			return
		}
	}
}

func (w *Worker) processNodeQuery(ctx context.Context, task *definitions.Task) error {
	atomic.AddInt32(&(w.pendingAggregationCount), 1)

	query := task.Query
	topk := task.Topk
	workerTopK := task.WorkerTopK

	var resultIds []uint32
	var distances []float32

	var wg sync.WaitGroup
	var mu sync.Mutex
	for workerID := range w.GlobalWorkers {
		wg.Add(1)
		go func(workerID string, query []float32, topk int64) {
			defer wg.Done()
			// logger.Info("send query to node", zap.String("workerID", workerID))
			logger.Debug("send query to node", zap.String("workerID", workerID))
			workerAddress := w.GlobalWorkers[workerID]

			client, err := rpc.GetWorkerIntClient(workerAddress)
			if err != nil {
				logger.Error("Node Query: error in get worker int client", zap.Error(err), zap.String("currentWorkerID", workerID), zap.String("workerAddress", workerAddress))
				return
			}

			result, err := client.Query(ctx, &pb.QueryRequest{
				TaskId:     task.ID,
				Query:      query,
				Topk:       topk,
				Goal:       task.RecallGoal,
				SearchType: definitions.ANN,
			})
			if err != nil {
				logger.Error("Node Query: error in query", zap.Error(err))
				return
			}
			// TODO: 可能要修改这部分的实现，现在是在每次得到结果的时候聚合
			mu.Lock()
			resultIds = append(resultIds, result.Result...)
			distances = append(distances, result.Distance...)
			mu.Unlock()
		}(workerID, query, workerTopK)
	}
	wg.Wait()

	atomic.AddInt32(&(w.pendingAggregationCount), -1)
	atomic.AddInt32(&(w.ongoingAggregationCount), 1)

	// aggregation
	aggregatedIds, aggregatedDistances := w.aggregate(resultIds, distances, topk)
	atomic.AddInt32(&(w.ongoingAggregationCount), -1)
	logger.Debug("aggregation result", zap.Any("aggregatedIds", aggregatedIds))
	logger.Debug("aggregation distances", zap.Any("aggregatedDistances", aggregatedDistances))

	// send result to client
	client, err := rpc.GetClientClient()
	if err != nil {
		return err
	}
	_, err = client.QueryTask(ctx, &pb.ClientTaskRequest{
		QueryId: task.QueryID,
		Result:  aggregatedIds,
		TaskId:  task.ID,
	})
	if err != nil {
		return err
	}
	schedulerClient, err := rpc.GetSchedulerClient()
	if err != nil {
		return err
	}
	_, err = schedulerClient.CompleteTask(ctx, &pb.TaskCompleteRequest{
		TaskId: task.ID,
	})
	if err != nil {
		logger.Error("error in complete task", zap.Error(err))
		return err
	}

	return nil
}

func (w *Worker) processMetaQuery(ctx context.Context, task *definitions.Task) error {
	atomic.AddInt32(&(w.ongoingMetaQueryCount), 1)
	atomic.AddInt32(&(w.pendingAggregationCount), 1)
	query := task.Query
	topk := task.Topk
	workerTopK := task.WorkerTopK

	// using meta_k instead of topk
	nodeDict, err := w.metaQuery(query, int64(w.metaK))
	atomic.AddInt32(&(w.ongoingMetaQueryCount), -1)

	if err != nil {
		atomic.AddInt32(&(w.pendingAggregationCount), -1)
		return err
	}

	// mock
	nodeDict = w.sortBuckets(nodeDict)
	time.Sleep(time.Millisecond * time.Duration(int((1-w.pruningLambda)*80)))

	var resultIds []uint32
	var distances []float32

	var wg sync.WaitGroup
	var mu sync.Mutex

	for nodeIndex, bucketIndicesAndDists := range nodeDict {
		bucketIndices := bucketIndicesAndDists.Buckets
		wg.Add(1)
		go func(nodeIndex uint32, bucketIndices []uint32) {
			defer wg.Done()
			// logger.Info("send query to node", zap.Uint32("nodeIndex", nodeIndex))
			logger.Debug("send query to node", zap.Uint32("nodeIndex", nodeIndex))
			currentWorkerID := FormatID(int(nodeIndex))
			workerAddress := w.GlobalWorkers[currentWorkerID]

			// debug: 原分片是按照哈希分片，如果分片正确，bucket%16应该等于nodeIndex
			// logger.Info("bucket indices", zap.Any("bucketIndices", bucketIndices))
			// logger.Info("nodeIndex", zap.Uint32("nodeIndex", nodeIndex))
			// logger.Info("bucket indices % 16", zap.Uint32("bucketIndices % 16", bucketIndices[0]%16))

			// logger.Info("Global workers", zap.Any("globalWorkers", w.GlobalWorkers))
			// client, err := rpc.GetWorkerIntClient(workerAddress)
			client, err := rpc.GetWorkerIntClient(workerAddress)
			if err != nil {
				logger.Error("Meta Query: error in get worker int client", zap.Error(err), zap.String("fromWorkerID", w.ID), zap.String("toWorkerID", currentWorkerID), zap.String("workerAddress", workerAddress))
				return
			}
			result, err := client.Query(ctx, &pb.QueryRequest{
				TaskId:     task.ID,
				Query:      query,
				Topk:       workerTopK,
				Buckets:    bucketIndices,
				Goal:       task.RecallGoal,
				SearchType: definitions.ANN,
			})
			if err != nil {
				logger.Error("Meta Query: error in query", zap.Error(err), zap.String("fromWorkerID", w.ID), zap.String("toWorkerID", currentWorkerID), zap.String("workerAddress", workerAddress))
				return
			}

			mu.Lock()
			resultIds = append(resultIds, result.Result...)
			distances = append(distances, result.Distance...)
			mu.Unlock()
		}(nodeIndex, bucketIndices)
	}
	wg.Wait()

	atomic.AddInt32(&(w.pendingAggregationCount), -1)
	atomic.AddInt32(&(w.ongoingAggregationCount), 1)

	// aggregation for topk
	aggregatedIds, aggregatedDistances := w.aggregate(resultIds, distances, topk)
	atomic.AddInt32(&(w.ongoingAggregationCount), -1)
	logger.Debug("aggregation result", zap.Any("aggregatedIds", aggregatedIds))
	logger.Debug("aggregation distances", zap.Any("aggregatedDistances", aggregatedDistances))

	// send result to client
	client, err := rpc.GetClientClient()
	if err != nil {
		return err
	}
	_, err = client.QueryTask(ctx, &pb.ClientTaskRequest{
		QueryId: task.QueryID,
		Result:  aggregatedIds,
		TaskId:  task.ID,
	})
	if err != nil {
		return err
	}
	schedulerClient, err := rpc.GetSchedulerClient()
	if err != nil {
		logger.Error("error in get scheduler client", zap.Error(err))
		return err
	}
	_, err = schedulerClient.CompleteTask(ctx, &pb.TaskCompleteRequest{
		TaskId: task.ID,
	})
	if err != nil {
		logger.Error("error in complete task", zap.Error(err))
		return err
	}

	return nil
}

func (w *Worker) ProcessMetaQueryHeuristicFiltering(ctx context.Context, task *definitions.Task) error {
	atomic.AddInt32(&(w.ongoingMetaQueryCount), 1)
	atomic.AddInt32(&(w.pendingAggregationCount), 1)
	query := task.Query
	topk := task.Topk
	workertopk := task.WorkerTopK

	// using meta_k instead of topk
	nodeDict, err := w.metaQuery(query, int64(w.metaK))
	atomic.AddInt32(&(w.ongoingMetaQueryCount), -1)

	if err != nil {
		atomic.AddInt32(&(w.pendingAggregationCount), -1)
		return err
	}

	nodeDict = w.pruneBuckets(nodeDict)

	var resultIds []uint32
	var distances []float32

	var wg sync.WaitGroup
	var mu sync.Mutex

	for nodeIndex, bucketIndicesAndDists := range nodeDict {
		bucketIndices := bucketIndicesAndDists.Buckets
		wg.Add(1)
		go func(nodeIndex uint32, bucketIndices []uint32) {
			defer wg.Done()
			logger.Info("send query to node", zap.Uint32("nodeIndex", nodeIndex))
			currentWorkerID := FormatID(int(nodeIndex))
			workerAddress := w.GlobalWorkers[currentWorkerID]
			logger.Debug("bucket indices", zap.Any("bucketIndices", bucketIndices))
			client, err := rpc.GetWorkerIntClient(workerAddress)
			if err != nil {
				logger.Error("error in get worker int client", zap.Error(err))
				return
			}
			result, err := client.Query(ctx, &pb.QueryRequest{
				TaskId:     task.ID,
				Query:      query,
				Topk:       workertopk,
				Buckets:    bucketIndices,
				Goal:       task.RecallGoal,
				SearchType: definitions.ANN,
			})
			if err != nil {
				logger.Error("error in query", zap.Error(err))
				return
			}

			mu.Lock()
			resultIds = append(resultIds, result.Result...)
			distances = append(distances, result.Distance...)
			mu.Unlock()
		}(nodeIndex, bucketIndices)
	}
	wg.Wait()

	atomic.AddInt32(&(w.pendingAggregationCount), -1)
	atomic.AddInt32(&(w.ongoingAggregationCount), 1)

	// aggregation for topk
	aggregatedIds, aggregatedDistances := w.aggregate(resultIds, distances, topk)
	atomic.AddInt32(&(w.ongoingAggregationCount), -1)
	logger.Debug("aggregation result", zap.Any("aggregatedIds", aggregatedIds))
	logger.Debug("aggregation distances", zap.Any("aggregatedDistances", aggregatedDistances))

	// send result to client
	client, err := rpc.GetClientClient()
	if err != nil {
		return err
	}
	_, err = client.QueryTask(ctx, &pb.ClientTaskRequest{
		QueryId: task.QueryID,
		Result:  aggregatedIds,
		TaskId:  task.ID,
	})
	if err != nil {
		return err
	}
	schedulerClient, err := rpc.GetSchedulerClient()
	if err != nil {
		logger.Error("error in get scheduler client", zap.Error(err))
		return err
	}
	_, err = schedulerClient.CompleteTask(ctx, &pb.TaskCompleteRequest{
		TaskId: task.ID,
	})
	if err != nil {
		logger.Error("error in complete task", zap.Error(err))
		return err
	}

	return nil
}

func (w *Worker) ProcessMetaQueryRadiusFiltering(ctx context.Context, task *definitions.Task) error {
	atomic.AddInt32(&(w.ongoingMetaQueryCount), 1)
	atomic.AddInt32(&(w.pendingAggregationCount), 1)
	query := task.Query
	topk := task.Topk
	workerTopK := task.WorkerTopK

	// using meta_k instead of topk
	nodeDict, err := w.metaQuery(query, int64(w.metaK))
	atomic.AddInt32(&(w.ongoingMetaQueryCount), -1)

	if err != nil {
		atomic.AddInt32(&(w.pendingAggregationCount), -1)
		return err
	}

	// sortbuckets by descending distances
	nodeDict = w.sortBuckets(nodeDict)
	nodeDict = w.pruneBuckets(nodeDict)

	_sendQueryToNode := func(nodeIndex uint32, bucketIndices []uint32) (*pb.QueryResponse, error) {
		logger.Debug("send query to node", zap.Uint32("nodeIndex", nodeIndex))
		currentWorkerID := FormatID(int(nodeIndex))
		workerAddress := w.GlobalWorkers[currentWorkerID]
		logger.Debug("bucket indices", zap.Any("bucketIndices", bucketIndices))
		client, err := rpc.GetWorkerIntClient(workerAddress)
		if err != nil {
			logger.Error("error in get worker int client", zap.Error(err))
			return nil, err
		}
		result, err := client.Query(ctx, &pb.QueryRequest{
			TaskId:     task.ID,
			Query:      query,
			Topk:       workerTopK,
			Buckets:    bucketIndices,
			Goal:       task.RecallGoal,
			SearchType: definitions.ANN,
		})
		return result, err
	}

	var resultIds []uint32
	var distances []float32

	var wg sync.WaitGroup
	var mu sync.Mutex
	nodeWithMostBuckets := -1
	currentMostBucketCount := -1
	for nodeIndex, bucketIndicesAndDists := range nodeDict {
		bucketIndices := bucketIndicesAndDists.Buckets
		if len(bucketIndices) > currentMostBucketCount {
			currentMostBucketCount = len(bucketIndices)
			nodeWithMostBuckets = int(nodeIndex)
		}
	}
	if nodeWithMostBuckets == -1 {
		return fmt.Errorf("no node with buckets found for query: %v", query)
	}

	logger.Debug("node with most buckets", zap.Int("nodeIndex", nodeWithMostBuckets), zap.Int("bucketCount", currentMostBucketCount))

	currentResult, err := _sendQueryToNode(uint32(nodeWithMostBuckets), nodeDict[uint32(nodeWithMostBuckets)].Buckets)
	if err != nil {
		logger.Error("error in query", zap.Error(err))
		return err
	}

	currentDistances := currentResult.Distance
	currentMaxDistance := float32(0)
	doFilter := false

	if len(currentDistances) >= int(topk) {
		doFilter = true
		for _, distance := range currentDistances {
			if distance > currentMaxDistance {
				currentMaxDistance = distance
			}
		}
	}

	logger.Debug("current distances for first node", zap.Any("currentDistances", currentDistances))
	logger.Debug("max distance in current distances", zap.Float32("maxDistance", currentMaxDistance))
	// 对maxDistance开根号
	currentMaxDistance = float32(math.Sqrt(float64(currentMaxDistance)))

	for nodeIndex, bucketIndicesAndDists := range nodeDict {
		bucketIndices := bucketIndicesAndDists.Buckets

		logger.Debug("processing node", zap.Uint32("nodeIndex", nodeIndex), zap.Any("bucketIndices", bucketIndices), zap.Any("bucketCentroidDists", bucketIndicesAndDists.Distances))

		wg.Add(1)
		go func(nodeIndex uint32, bucketIndices []uint32, bucketCentroidDists []float32, currentMaxDistance float32) {
			// 距离过滤：过滤掉桶中心距离加上桶半径大于currentMaxDistance的桶
			defer wg.Done()
			filteredBuckets := make([]uint32, 0, len(bucketIndices))
			filteredCount := 0
			if !doFilter {
				logger.Debug("haven't reached topk yet, not filtering")
				filteredBuckets = bucketIndices
			} else {
				for idx, bucket := range bucketIndices {
					bucketDist := float32(0)
					if idx < len(bucketCentroidDists) {
						bucketDist = float32(math.Sqrt(float64(bucketCentroidDists[idx])))
					} else {
						logger.Warn("bucket centroid distance not found", zap.Uint32("nodeIndex", nodeIndex), zap.Uint32("bucket", bucket), zap.Int("idx", idx))
					}
					bucketRadius := float32(0)
					if int(bucket) < len(w.bucketRadiuses) {
						bucketRadius = w.bucketRadiuses[bucket]
					}
					logger.Debug("bucket filtering", zap.Uint32("nodeIndex", nodeIndex), zap.Uint32("bucket", bucket), zap.Int("idx", idx), zap.Float32("bucketDist", bucketDist), zap.Float32("bucketRadius", bucketRadius), zap.Float32("currentMaxDistance", currentMaxDistance))
					// 计算到bucket的最小可能距离（如果query在bucket内，则为0）
					minPossibleDist := float32(math.Max(0, float64(bucketDist-bucketRadius)))
					if minPossibleDist <= currentMaxDistance {
						filteredBuckets = append(filteredBuckets, bucket)
						logger.Info("bucket passed filtering", zap.Uint32("nodeIndex", nodeIndex), zap.Uint32("bucket", bucket), zap.Float32("diff", bucketDist-bucketRadius), zap.Float32("currentMaxDistance", currentMaxDistance))
						// logger.Debug("bucket passed filtering", zap.Uint32("nodeIndex", nodeIndex), zap.Uint32("bucket", bucket), zap.Float32("diff", bucketDist-bucketRadius), zap.Float32("currentMaxDistance", currentMaxDistance))
					} else {
						filteredCount++
						logger.Info("bucket filtered out", zap.Uint32("nodeIndex", nodeIndex), zap.Uint32("bucket", bucket), zap.Float32("diff", bucketDist-bucketRadius), zap.Float32("currentMaxDistance", currentMaxDistance))
						// logger.Debug("bucket filtered out", zap.Uint32("nodeIndex", nodeIndex), zap.Uint32("bucket", bucket), zap.Float32("diff", bucketDist-bucketRadius), zap.Float32("currentMaxDistance", currentMaxDistance))
					}
				}
			}
			atomic.AddInt64(&w.TotalFilteredBucketCount, int64(filteredCount))
			atomic.AddInt64(&w.TotalBucketCount, int64(len(bucketIndices)))
			if len(filteredBuckets) == 0 {
				logger.Debug("no buckets left after filtering", zap.Uint32("nodeIndex", nodeIndex))
				return
			}
			result, err := _sendQueryToNode(nodeIndex, filteredBuckets)

			if err != nil {
				logger.Error("error in query", zap.Error(err), zap.Uint32("nodeIndex", nodeIndex))
				return
			}

			// TODO: 可能要修改这部分的实现，现在是在每次得到结果的时候聚合
			mu.Lock()
			resultIds = append(resultIds, result.Result...)
			distances = append(distances, result.Distance...)
			logger.Debug("appended results", zap.Uint32("nodeIndex", nodeIndex), zap.Int("resultCount", len(result.Result)), zap.Int("distanceCount", len(result.Distance)))
			mu.Unlock()
		}(nodeIndex, bucketIndices, bucketIndicesAndDists.Distances, currentMaxDistance)
	}
	wg.Wait()

	atomic.AddInt32(&(w.pendingAggregationCount), -1)
	atomic.AddInt32(&(w.ongoingAggregationCount), 1)

	// aggregation for topk
	aggregatedIds, aggregatedDistances := w.aggregate(resultIds, distances, topk)
	atomic.AddInt32(&(w.ongoingAggregationCount), -1)
	logger.Debug("aggregation result", zap.Any("aggregatedIds", aggregatedIds))
	logger.Debug("aggregation distances", zap.Any("aggregatedDistances", aggregatedDistances))

	// send result to client
	client, err := rpc.GetClientClient()
	if err != nil {
		return err
	}
	_, err = client.QueryTask(ctx, &pb.ClientTaskRequest{
		QueryId: task.QueryID,
		Result:  aggregatedIds,
		TaskId:  task.ID,
	})
	if err != nil {
		return err
	}
	schedulerClient, err := rpc.GetSchedulerClient()
	if err != nil {
		logger.Error("error in get scheduler client", zap.Error(err))
		return err
	}
	_, err = schedulerClient.CompleteTask(ctx, &pb.TaskCompleteRequest{
		TaskId: task.ID,
	})
	if err != nil {
		logger.Error("error in complete task", zap.Error(err))
		return err
	}

	return nil
}

// processTask 处理scheduler发来的查询任务
func (w *Worker) processTask(ctx context.Context, task *definitions.Task) error {
	switch task.Type {
	case definitions.TaskNodeQuery:
		// for naive distributed
		return w.processNodeQuery(ctx, task)
	case definitions.TaskMetaQuery:
		// for similarity distributed
		switch w.searchMethod {
		case "rf":
			return w.ProcessMetaQueryRadiusFiltering(ctx, task)
		case "hf", "hf-v2":
			return w.ProcessMetaQueryHeuristicFiltering(ctx, task)
		case "default":
			return w.processMetaQuery(ctx, task)
		default:
			logger.Error("unknown search method", zap.String("searchMethod", w.searchMethod))
			return fmt.Errorf("unknown search method: %s", w.searchMethod)
		}
	default:
		return ErrUnknownTaskType
	}
}

// aggregate 根据 distances 对 ids 和 distances 进行排序，并返回 topk 个元素
func (w *Worker) aggregate(ids []uint32, distances []float32, topk int64) ([]uint32, []float32) {
	// 如果 topk 大于数组长度，则取最小的值
	aggregationStartingTime := time.Now()
	n := len(ids)
	if int(topk) > n {
		topk = int64(n)
	}

	// 创建索引数组，记录原始索引
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}

	// 使用 sort.Slice 对 indices 进行排序，按照 distances 升序排序
	sort.Slice(indices, func(i, j int) bool {
		return distances[indices[i]] < distances[indices[j]]
	})

	// 生成排序后的 ids 和 distances，只取 topk
	sortedIDs := make([]uint32, topk)
	sortedDistances := make([]float32, topk)

	for i := int64(0); i < topk; i++ {
		sortedIDs[i] = ids[indices[i]]
		sortedDistances[i] = distances[indices[i]]
	}

	aggregationTime := time.Since(aggregationStartingTime)
	w.AggregationTimer.TimerLock.Lock()
	w.AggregationTimer.TotalTime += aggregationTime
	w.AggregationTimer.TimerLock.Unlock()

	return sortedIDs, sortedDistances
}

// metaQuery,对应python函数：`get_node_dict`, return: {node_index: [bucket_indices: np.ndarray]}
func (w *Worker) metaQuery(query []float32, topk int64) (map[uint32]struct {
	Buckets   []uint32
	Distances []float32
}, error) {
	atomic.AddInt32(&(w.totalMetaQueries), 1)
	startTime := time.Now()
	labels, distances, err := w.MetaIndex.Search(query, int(topk))
	if err != nil {
		return nil, err
	}
	metaQueryTime := time.Since(startTime)
	w.MetaQueryTimer.TimerLock.Lock()
	w.MetaQueryTimer.TotalTime += metaQueryTime
	w.MetaQueryTimer.TimerLock.Unlock()
	// logger.Info("labels length", zap.Int("labels length", len(labels)))
	logger.Debug("labels length", zap.Int("labels length", len(labels)))
	logger.Debug("meta query result: ", zap.Any("labels", labels))
	logger.Debug("meta query distances: ", zap.Any("distances", distances))
	// Guard: the meta index may return fewer than metaK results (e.g. an IVF
	// meta index returns at most nprobe*avg_cluster_size labels; on small
	// datasets with many clusters that can be < metaK). Truncate instead of
	// panicking on a slice out of range.
	if int64(len(labels)) > topk {
		labels = labels[:topk]
		distances = distances[:topk]
	}
	// 对labels去重
	uniqueLabels := unique(labels)
	startTime = time.Now()

	type nodeBuckets struct {
		Buckets   []uint32
		Distances []float32
	}
	nodeDict := make(map[uint32]*nodeBuckets)
	nodeFreq := make(map[uint32]int)

	// if metaIndexType is ivf, if means hash/hakes strategy
	if w.metaIndexType == "ivf" {
		numNodes := uint32(len(w.GlobalWorkers))
		// 检查节点表是否为空，如果为空则返回错误
		if numNodes == 0 {
			logger.Warn("GlobalWorkers is empty, cannot perform hash routing",
				zap.String("workerID", w.ID),
				zap.Int("globalWorkersCount", len(w.GlobalWorkers)))
			return nil, fmt.Errorf("GlobalWorkers is empty, cannot perform hash routing")
		}
		// 按照哈希的方式，发给对应的节点，然后聚合
		for _, label := range uniqueLabels {
			// 使用哈希函数计算label应该发送到哪个节点
			node := label % numNodes

			// 初始化节点字典
			if nodeDict[node] == nil {
				nodeDict[node] = &nodeBuckets{
					Buckets:   make([]uint32, 0),
					Distances: make([]float32, 0),
				}
			}

			// 将label添加到对应节点
			nodeDict[node].Buckets = append(nodeDict[node].Buckets, label)

			// 从原始labels和distances中找到对应的距离
			var labelDistance float32
			for i, origLabel := range labels {
				if origLabel == label {
					labelDistance = distances[i]
					break
				}
			}
			nodeDict[node].Distances = append(nodeDict[node].Distances, labelDistance)
		}

		// 返回哈希路由结果
		result := make(map[uint32]struct {
			Buckets   []uint32
			Distances []float32
		})
		for nodeID, buckets := range nodeDict {
			result[nodeID] = struct {
				Buckets   []uint32
				Distances []float32
			}{
				Buckets:   buckets.Buckets,
				Distances: buckets.Distances,
			}
		}
		return result, nil
	}

	// Single pass: count frequencies and assign buckets simultaneously
	for _, label := range uniqueLabels {
		possibleNodes := w.bucketNodeLabels[label]
		var bestNode uint32
		var lowestFreqNode uint32
		var highestFreqNode uint32
		bestFreq := 0
		lowestFreq := math.MaxInt32
		highFreq := 0

		// If only one possible node, directly assign the bucket to it, using the `bucketLabelToNodeID` function
		if len(possibleNodes) == 1 {
			bestNode = w.bucketLabelToNodeID(label)
		} else {
			// Find both highest and lowest frequency nodes while updating frequencies
			for _, nodeID := range possibleNodes {
				nodeFreq[nodeID]++
				currFreq := nodeFreq[nodeID]
				if currFreq >= highFreq {
					highestFreqNode = nodeID
					highFreq = currFreq
				}
				if currFreq < lowestFreq {
					lowestFreqNode = nodeID
					lowestFreq = currFreq
				}
			}

			// If bestFreq is above ivfThreshold, use the node with highest frequency
			if bestFreq >= w.ivfThreshold {
				bestNode = highestFreqNode
			} else {
				bestNode = lowestFreqNode
			}
		}

		// Find the index of label in labels to get the correct distance
		var dist float32
		for i, l := range labels {
			if l == label {
				dist = distances[i]
				break
			}
		}

		if _, ok := nodeDict[bestNode]; !ok {
			nodeDict[bestNode] = &nodeBuckets{
				Buckets:   make([]uint32, 0),
				Distances: make([]float32, 0),
			}
		}
		nodeDict[bestNode].Buckets = append(nodeDict[bestNode].Buckets, label)
		nodeDict[bestNode].Distances = append(nodeDict[bestNode].Distances, dist)
	}
	w.MetaQueryTimer.TimerLock.Lock()
	w.MetaQueryTimer.PostQueryTime += time.Since(startTime)
	w.MetaQueryTimer.TimerLock.Unlock()

	// Convert map[uint32]*nodeBuckets to map[uint32]struct{Buckets []uint32; Distances []float32}
	result := make(map[uint32]struct {
		Buckets   []uint32
		Distances []float32
	})
	for k, v := range nodeDict {
		result[k] = struct {
			Buckets   []uint32
			Distances []float32
		}{
			Buckets:   v.Buckets,
			Distances: v.Distances,
		}
	}
	return result, nil
}

// bucketLabelToNodeID, 目前仅在每一个 bucketLabel 只对应一个 nodeID 的情况下使用
func (w *Worker) bucketLabelToNodeID(bucketLabel uint32) (nodeID uint32) {
	nodeID = w.bucketNodeLabels[bucketLabel][0]
	return
}
