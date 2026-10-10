package client

import (
	"context"
	"encoding/csv"
	"fmt"
	"net"
	"os"
	"sync"
	"time"

	"github.com/WHU-STAR/TITAN/internal/definitions"
	"github.com/WHU-STAR/TITAN/internal/rpc"
	"github.com/WHU-STAR/TITAN/internal/storage"
	"github.com/WHU-STAR/TITAN/pkg/logger"
	"github.com/WHU-STAR/TITAN/protocol/pb"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/types/known/emptypb"
)

type Client struct {
	address        string
	resultPool     *sync.Pool
	resultChannel  chan *Result
	QueryData      *storage.QueryData
	groundTruth    *storage.GroundTruth
	startTime      time.Time
	queryStartTime time.Time
	queryEndTime   time.Time
	CurrentMetaK   int
	CurrentThresh  int
	CurrentEF      int
	CurrentNProbe  int
	CurrentLambda  float32
	RunName        string
	RunEpochs      int

	ClassicTotalTopk  int
	ClassicWorkerTopk int

	latencyMu          sync.Mutex
	queryDispatchTimes map[uint32]time.Time
	queryLatencies     []float64
}
type Result struct {
	QueryID uint32
	Result  []uint32
	TaskID  string
}

func (r *Result) Reset() {
	r.QueryID = 0
	r.Result = nil
	r.TaskID = ""
}

func NewClient(address string) *Client {
	client := &Client{
		address:       address,
		RunEpochs:     10,
		CurrentLambda: 0.8,
		RunName:       "default_run",
		QueryData:     &storage.QueryData{},
		groundTruth:   &storage.GroundTruth{},
		resultChannel: make(chan *Result),
		resultPool: &sync.Pool{
			New: func() any {
				return &Result{}
			},
		},
	}
	return client
}

// LoadQuerySet loads the query vectors.
func (c *Client) LoadQuerySet(path string) error {
	var err error
	c.QueryData, err = storage.LoadQuerySet(path)
	if err != nil {
		return err
	}
	logger.Info("Query set loaded", zap.Uint32("dim", c.QueryData.Dim), zap.Uint32("num", c.QueryData.Num))
	return nil
}

// LoadGroundTruth loads the ground-truth neighbors.
func (c *Client) LoadGroundTruth(path string) error {
	var err error
	c.groundTruth, err = storage.LoadGroundTruth(path)
	if err != nil {
		return err
	}
	logger.Info("Ground truth loaded", zap.Uint32("num", c.groundTruth.Num))
	return nil
}

func (c *Client) IsServerInitialized() bool {
	schedulerClient, err := rpc.GetSchedulerClient()
	if err != nil {
		logger.Error("Failed to get scheduler client", zap.Error(err))
		return false
	}
	resp, err := schedulerClient.IsInitialized(context.Background(), &emptypb.Empty{})
	if err != nil {
		logger.Error("Failed to check if server is initialized", zap.Error(err))
		return false
	}
	return resp.IsInitialized
}

// Stop logs client shutdown.
func (c *Client) Stop() {
	logger.Info("Client stopped")
}

// StartGRPCServer serves result callbacks from workers.
func (c *Client) StartGRPCServer() {
	server := grpc.NewServer()
	pb.RegisterClientServiceServer(server, &ClientServer{client: c})

	lis, err := net.Listen("tcp", c.address)
	if err != nil {
		panic(err)
	}

	logger.Info("Client gRPC server started", zap.String("address", c.address))
	if err := server.Serve(lis); err != nil {
		logger.Error("Failed to start gRPC server", zap.Error(err))
	}
}

// StartQuery submits queries to the scheduler.
func (c *Client) StartQuery(count int, topk int, workerTopK int) {
	wg := sync.WaitGroup{}
	schedulerClient, err := rpc.GetSchedulerClient()
	if err != nil {
		logger.Error("Failed to get scheduler client", zap.Error(err))
	}
	logger.Info("Start sending query")
	c.ClassicTotalTopk = topk
	c.ClassicWorkerTopk = workerTopK

	c.startTime = time.Now()
	c.latencyMu.Lock()
	c.queryDispatchTimes = make(map[uint32]time.Time, count)
	c.queryLatencies = c.queryLatencies[:0]
	c.latencyMu.Unlock()

	for i := range count {
		query := c.QueryData.Queries[i]
		wg.Add(1)
		go func(queryID int, query []float32) {
			defer wg.Done()
			dispatchTime := time.Now()
			c.latencyMu.Lock()
			c.queryDispatchTimes[uint32(queryID)] = dispatchTime
			c.latencyMu.Unlock()

			_taskResponse, err := schedulerClient.ExecuteTask(context.Background(), &pb.ExecuteTaskRequest{
				QueryId:    uint32(queryID),
				Query:      query,
				Topk:       int64(topk),
				Workertopk: int64(workerTopK),
			})
			if err != nil {
				logger.Error("Failed to execute task", zap.Error(err))

			}
			logger.Debug("Task ID:", zap.String("Task ID", _taskResponse.TaskId))
		}(i, query)
	}
	wg.Wait()
	c.queryEndTime = time.Now()
	logger.Info("Finished sending queries", zap.Int("count", count))
}

// HandleResult collects results and computes recall.
func (c *Client) HandleResult(clientQueryCount int, skipperFunc func(recall float64)) definitions.RunResult {
	logger.Info("Start handling results")
	results := make(map[uint32][]uint32)
	totalRecall := 0.0
	processedQueries := 0
	// total time starts from zero
	schedulerClient, err := rpc.GetSchedulerClient()
	if err != nil {
		logger.Error("Failed to get scheduler client", zap.Error(err))
	}
	runResult := definitions.RunResult{}

	for result := range c.resultChannel {
		// logger.Debug("Received result:", zap.Any("result", result))
		results[result.QueryID] = result.Result
		logger.Debug("Query ID:", zap.Uint32("Query ID", result.QueryID))
		logger.Debug("Task ID:", zap.String("Task ID", result.TaskID))

		// Compute recall once every query has a result.
		if len(results) == clientQueryCount {
			runResult.TopK = c.ClassicTotalTopk
			runResult.WorkerTopK = c.ClassicWorkerTopk
			runResult.MetaK = c.CurrentMetaK
			runResult.Thresh = c.CurrentThresh
			runResult.EFSearch = c.CurrentEF
			runResult.NProbe = c.CurrentNProbe
			runResult.Lambda = float64(c.CurrentLambda)

			logger.Info("Processing batch of results", zap.Int("count", len(results)))
			endtime := time.Now()
			elapsedTime := endtime.Sub(c.startTime)
			elapsedQueryTime := c.queryEndTime.Sub(c.startTime)
			// Recall of each query.
			for queryID, queryResult := range results {
				groundTruth := c.groundTruth.GroundTruth[queryID]
				logger.Debug("Ground truth:", zap.Any("Ground truth", groundTruth))
				logger.Debug("Query result:", zap.Any("Query result", queryResult))
				recall := calculateRecall(groundTruth, queryResult)
				totalRecall += recall
				processedQueries++
				logger.Debug("Recall:", zap.Float64("Recall", recall))
			}

			// Average recall.
			avgRecall := totalRecall / float64(processedQueries)
			logger.Info(fmt.Sprintf("Average Recall@%d = %.4f", c.ClassicTotalTopk, avgRecall))

			logger.Info(fmt.Sprintf("Total elapsed time: %v", elapsedTime))
			qps := float64(clientQueryCount) / elapsedTime.Seconds()
			logger.Info("qps:", zap.Float64("qps", qps))
			logger.Info("Query elapsed time:", zap.Int64("elapsed time", elapsedQueryTime.Milliseconds()), zap.String("unit", "ms"))
			runResult.Recall = avgRecall
			skipperFunc(avgRecall)
			runResult.QPS = qps

			c.latencyMu.Lock()
			latencies := definitions.SortedCopy(c.queryLatencies)
			c.latencyMu.Unlock()
			if len(latencies) > 0 {
				runResult.AvgTime = definitions.Mean(latencies)
				runResult.P50Time = definitions.Percentile(latencies, 50)
				runResult.P95Time = definitions.Percentile(latencies, 95)
				runResult.P99Time = definitions.Percentile(latencies, 99)
			} else {
				runResult.AvgTime = elapsedQueryTime.Seconds() / float64(clientQueryCount)
			}

			logger.Info("Latency (s)",
				zap.Float64("avg", runResult.AvgTime),
				zap.Float64("p50", runResult.P50Time),
				zap.Float64("p95", runResult.P95Time),
				zap.Float64("p99", runResult.P99Time),
			)

			err = runResult.WriteToCsv("../results/" + c.RunName + ".csv")
			if err != nil {
				logger.Error("Failed to write result to CSV", zap.Error(err))
			}
			if err := c.writeLatencyDetailsCsv("../results/"+c.RunName+"_latency.csv", latencies); err != nil {
				logger.Error("Failed to write latency details CSV", zap.Error(err))
			}
			ctx := context.Background()
			// Meta-index query count.
			workerMetaMsg, err := schedulerClient.GetWorkerTotalMetaCount(ctx, &emptypb.Empty{})
			if err != nil {
				logger.Error("Failed to get worker meta count", zap.Error(err))
			}
			logger.Info("Worker meta query total:", zap.Int32("total", workerMetaMsg.Total))
			logger.Info("Worker meta query counts:", zap.Any("counts", workerMetaMsg.WorkerCounts))

			// Local-index query count.
			workerInternalMsg, err := schedulerClient.GetWorkerTotalInternalCount(ctx, &emptypb.Empty{})
			if err != nil {
				logger.Error("Failed to get worker internal count", zap.Error(err))
			}
			logger.Info("Worker internal query total:", zap.Int32("total", workerInternalMsg.Total))
			logger.Info("Worker internal query counts:", zap.Any("counts", workerInternalMsg.WorkerCounts))

			workersBucketsMsg, err := schedulerClient.GetWorkerIntegerProperty(ctx, &pb.GetWorkerIntegerPropertyRequest{QueryType: int32(definitions.GET_TOTAL_BUCKET_COUNT)})
			if err != nil {
				logger.Error("Failed to get worker total bucket count", zap.Error(err))
			}
			workersFilteredBucketsMsg, err := schedulerClient.GetWorkerIntegerProperty(ctx, &pb.GetWorkerIntegerPropertyRequest{QueryType: int32(definitions.GET_TOTAL_FILTERED_BUCKET_COUNT)})
			if err != nil {
				logger.Error("Failed to get worker filtered bucket count", zap.Error(err))
			}

			workersBucketsTotal := 0
			for _, count := range workersBucketsMsg.WorkerValues {
				workersBucketsTotal += int(count)
			}

			workersFilteredBucketsTotal := 0
			for _, count := range workersFilteredBucketsMsg.WorkerValues {
				workersFilteredBucketsTotal += int(count)
			}

			filteredBucketsRatio := float64(workersFilteredBucketsTotal) / float64(workersBucketsTotal)
			logger.Info("Worker total buckets:", zap.Int("total", workersBucketsTotal))
			logger.Info("Worker filtered buckets:", zap.Int("total", workersFilteredBucketsTotal))
			logger.Info("Filtered buckets ratio:", zap.Float64("ratio", filteredBucketsRatio))

			break
		}

		// Return the result object to the pool.
		result.Reset()
		c.resultPool.Put(result)
	}
	logger.Info("Finished handling results")
	return runResult

}

// ProcessQueryResult records one query result.
func (c *Client) ProcessQueryResult(ctx context.Context, req *pb.ClientTaskRequest) {
	queryID := uint32(req.QueryId)
	now := time.Now()
	c.latencyMu.Lock()
	if start, ok := c.queryDispatchTimes[queryID]; ok {
		c.queryLatencies = append(c.queryLatencies, now.Sub(start).Seconds())
	}
	c.latencyMu.Unlock()

	result := c.resultPool.Get().(*Result)
	result.QueryID = queryID
	result.Result = req.Result
	result.TaskID = req.TaskId
	c.resultChannel <- result
}

func (c *Client) writeLatencyDetailsCsv(csvPath string, sortedLatencies []float64) error {
	if len(sortedLatencies) == 0 {
		return nil
	}
	file, err := os.OpenFile(csvPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()
	if err := writer.Write([]string{"query_index", "latency_ms"}); err != nil {
		return err
	}
	for i, latency := range sortedLatencies {
		if err := writer.Write([]string{
			fmt.Sprintf("%d", i),
			fmt.Sprintf("%.4f", latency*1000),
		}); err != nil {
			return err
		}
	}
	return nil
}

// calculateRecall returns |intersection| / |ground truth|.
func calculateRecall(gt []uint32, result []uint32) float64 {
	if len(result) < len(gt) {
		gt = gt[:len(result)]
	}

	set := make(map[uint32]struct{})
	for _, v := range gt {
		set[v] = struct{}{}
	}

	hits := 0
	for _, v := range result {
		if _, exists := set[v]; exists {
			hits++
		}
	}
	return float64(hits) / float64(len(gt))
}

// SetLambda pushes the pruning lambda to every worker.
func (c *Client) SetLambda(lambda float32) error {
	schedulerClient, err := rpc.GetSchedulerClient()
	c.CurrentLambda = lambda
	if err != nil {
		return err
	}
	_, err = schedulerClient.SetPruningLambda(context.Background(), &pb.SetLambdaRequest{Lambda: lambda})
	return err
}

// SetEFSearch pushes efSearch to every worker.
func (c *Client) SetEFSearch(ef int) error {
	schedulerClient, err := rpc.GetSchedulerClient()
	c.CurrentEF = ef
	if err != nil {
		return err
	}
	_, err = schedulerClient.SetEFSearch(context.Background(), &pb.SetEFSearchRequest{EfSearch: int32(ef)})
	return err
}

// SetThreshold pushes the IVF threshold to every worker.
func (c *Client) SetThreshold(thresh int) error {
	c.CurrentThresh = thresh
	schedulerClient, err := rpc.GetSchedulerClient()
	if err != nil {
		return err
	}
	_, err = schedulerClient.SetThreshold(context.Background(), &pb.SetThresholdRequest{Threshold: int32(thresh)})
	return err
}

// SetMetaK pushes metaK to every worker.
func (c *Client) SetMetaK(metaK int) error {
	c.CurrentMetaK = metaK
	schedulerClient, err := rpc.GetSchedulerClient()
	if err != nil {
		return err
	}
	_, err = schedulerClient.SetMetaK(context.Background(), &pb.SetMetaKRequest{MetaK: int32(metaK)})
	return err
}

// SetNProbe pushes nprobe to every worker.
func (c *Client) SetNProbe(nProbe int) error {
	c.CurrentNProbe = nProbe
	schedulerClient, err := rpc.GetSchedulerClient()
	if err != nil {
		return err
	}
	_, err = schedulerClient.SetNProbe(context.Background(), &pb.SetNProbeRequest{NProbe: int32(nProbe)})
	return err
}
