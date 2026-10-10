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
	MAX_FAILED_HEARTBEATS      = 3               // consecutive heartbeat failures before exit
	HB_INTERVAL                = time.Second * 3 // heartbeat interval
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
	ID              string  // worker id, formatted as worker-0
	Address         string  // listen address
	Mode            string  // singleNode or distributed
	strategy        string  // naive or similarity
	metaK           int     // number of buckets returned by the meta-index
	ivfThreshold    int     // bucket-count threshold for IVF selection
	metaIndexType   string  // meta-index type
	nodeIndexType   string  // local sub-index type
	searchMethod    string  // accepted for compatibility; every value uses the unpruned path
	pruningStrategy string  // configured pruning strategy; not applied on the query path yet
	pruningLambda   float64 // pruning retention rate; not applied on the query path yet

	// Indices
	NodeIndex                *IndexWrapper
	MetaIndex                Index
	bucketNodeLabels         [][]uint32
	nearestPointsToCentroids []uint32  // nearest vector id for each bucket centroid
	bucketRadiuses           []float32 // radius of each bucket
	bucketSizes              []uint32  // vector count of each bucket
	totalNumVectors          uint32    // number of vectors in the dataset
	absoluteNodeIndex        []int32   // global ids stored on this worker
	BucketIndicesCount       uint32
	// bucketCentroids    [][]float32

	// task management
	taskChannel chan *definitions.Task
	stopCh      chan struct{}

	// Counters (atmoic)
	ongoingInternalQueryCount int32 // access with atomic operations
	ongoingMetaQueryCount     int32 // access with atomic operations
	pendingAggregationCount   int32 // access with atomic operations
	ongoingAggregationCount   int32 // access with atomic operations
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

	TotalBucketCount         int64 // buckets considered by routing statistics
	TotalFilteredBucketCount int64 // buckets removed by routing statistics
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
		pruningStrategy:    config.Config.PruningStrategy,
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

// InitMetaIndex loads the replicated meta-index.
func (w *Worker) InitMetaIndex() (bool, error) {
	metaIndexDataPath := "../data/workerdata/meta_index_data.bin"
	var metaIndexPath string
	indexType := w.metaIndexType
	if indexType == "bf" || indexType == "hnsw_impr" {
		indexType = "hnsw"
	}
	metaIndexPath = "../data/workerdata/meta_" + indexType + ".bin"

	// Both files must exist.
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

	// Create the index implementation.
	_metaIndex, err := NewIndex(indexType)
	if err != nil {
		return false, err
	}

	// Load the index from disk.
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

	// Drop unassigned worker slots and shrink each bucket's node list.
	for i := range w.bucketNodeLabels {
		// Keep assigned workers only.
		validNodes := make([]uint32, 0)
		for _, nodeID := range w.bucketNodeLabels[i] {
			if nodeID != math.MaxUint32 { // stored -1 becomes MaxUint32
				validNodes = append(validNodes, nodeID)
			}
		}
		// Store the compacted list.
		w.bucketNodeLabels[i] = validNodes
	}
	// w.bucketCentroids = metaIndexData.Centroids

	ids := make([]uint32, metaIndexData.NumBuckets)
	for i := uint32(0); i < metaIndexData.NumBuckets; i++ {
		ids[i] = i
	}
	return true, nil
}

// InitNodeIndex loads this worker's local sub-index.
func (w *Worker) InitNodeIndex() error {
	nodeDataPath := fmt.Sprintf("../data/workerdata/%s.bin", w.ID)

	// The node data file must exist.
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

// InitIndex loads the meta-index and the local sub-index.
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

// StartGRPCServer starts the worker gRPC server.
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

// Stop shuts down the worker.
func (w *Worker) Stop() {
	// Close task and stop channels.
	close(w.taskChannel)
	close(w.stopCh)
	// Release indexes.
	if w.NodeIndex != nil && w.NodeIndex.Index != nil {
		w.NodeIndex.Index.Unload()
	}
	if w.MetaIndex != nil {
		w.MetaIndex.Unload()
	}
}

// ReceiveTask enqueues a task from the scheduler.
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

// InternalQuery runs the local sub-index search and returns ids and distances.
func (w *Worker) InternalQuery(ctx context.Context, query *definitions.Query) ([]uint32, []float32, error) {
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
		// After a descending distance sort, the last bucket is the nearest centroid.
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

// GetStatus returns a weighted load score.
func (w *Worker) GetStatus() int {
	// Weighted sum of in-flight and queued work.
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

// sendHeartbeat reports load and refreshes the worker table.
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

	// Refresh the worker table from the scheduler response.
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

// StartHealthCheck sends heartbeats until the worker stops.
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

	// TODO: apply angular bucket pruning before dispatch. hf uses this same unpruned path.
	nodeDict = w.sortBuckets(nodeDict)

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

// processTask handles a query assigned by the scheduler.
func (w *Worker) processTask(ctx context.Context, task *definitions.Task) error {
	switch task.Type {
	case definitions.TaskNodeQuery:
		// for naive distributed
		return w.processNodeQuery(ctx, task)
	case definitions.TaskMetaQuery:
		return w.processMetaQuery(ctx, task)
	default:
		return ErrUnknownTaskType
	}
}

// aggregate returns the top-k ids ordered by ascending distance.
func (w *Worker) aggregate(ids []uint32, distances []float32, topk int64) ([]uint32, []float32) {
	// Cap top-k at the number of candidates.
	aggregationStartingTime := time.Now()
	n := len(ids)
	if int(topk) > n {
		topk = int64(n)
	}

	// Index array used for an indirect sort.
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}

	// Sort by ascending distance.
	sort.Slice(indices, func(i, j int) bool {
		return distances[indices[i]] < distances[indices[j]]
	})

	// Keep the first top-k entries.
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

// metaQuery searches the meta-index and groups buckets by worker.
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
	// Drop duplicate bucket ids.
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
		// Hash routing needs a non-empty worker table.
		if numNodes == 0 {
			logger.Warn("GlobalWorkers is empty, cannot perform hash routing",
				zap.String("workerID", w.ID),
				zap.Int("globalWorkersCount", len(w.GlobalWorkers)))
			return nil, fmt.Errorf("GlobalWorkers is empty, cannot perform hash routing")
		}
		// Route each label by hash and aggregate later.
		for _, label := range uniqueLabels {
			// Hash the label onto a worker.
			node := label % numNodes

			// Create the worker entry.
			if nodeDict[node] == nil {
				nodeDict[node] = &nodeBuckets{
					Buckets:   make([]uint32, 0),
					Distances: make([]float32, 0),
				}
			}

			// Append the label.
			nodeDict[node].Buckets = append(nodeDict[node].Buckets, label)

			// Copy the distance that belongs to this label.
			var labelDistance float32
			for i, origLabel := range labels {
				if origLabel == label {
					labelDistance = distances[i]
					break
				}
			}
			nodeDict[node].Distances = append(nodeDict[node].Distances, labelDistance)
		}

		// Return the hash-routed buckets.
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

// bucketLabelToNodeID returns the first worker assigned to a bucket.
func (w *Worker) bucketLabelToNodeID(bucketLabel uint32) (nodeID uint32) {
	nodeID = w.bucketNodeLabels[bucketLabel][0]
	return
}

// sortBuckets orders each worker's buckets from farthest to nearest.
// The last bucket is the nearest centroid and is used as the graph entry.
func (w *Worker) sortBuckets(nodeDict map[uint32]struct {
	Buckets   []uint32
	Distances []float32
}) map[uint32]struct {
	Buckets   []uint32
	Distances []float32
} {
	result := make(map[uint32]struct {
		Buckets   []uint32
		Distances []float32
	}, len(nodeDict))
	for nodeID, data := range nodeDict {
		order := make([]int, len(data.Buckets))
		for i := range order {
			order[i] = i
		}
		sort.Slice(order, func(i, j int) bool {
			return data.Distances[order[i]] > data.Distances[order[j]]
		})
		buckets := make([]uint32, len(order))
		distances := make([]float32, len(order))
		for i, src := range order {
			buckets[i] = data.Buckets[src]
			distances[i] = data.Distances[src]
		}
		result[nodeID] = struct {
			Buckets   []uint32
			Distances []float32
		}{Buckets: buckets, Distances: distances}
	}
	return result
}
