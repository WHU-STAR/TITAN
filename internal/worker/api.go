package worker

import (
	"context"
	"errors"
	"sync/atomic"

	"github.com/WHU-STAR/TITAN/internal/definitions"
	pb "github.com/WHU-STAR/TITAN/protocol/pb"
	"google.golang.org/protobuf/types/known/emptypb"
)

type WorkerServer struct {
	pb.UnimplementedWorkerExtServiceServer
	pb.UnimplementedWorkerIntServiceServer
	Worker           *Worker
	totalMetaQueries int32
}

func NewWorkerServer(worker *Worker) *WorkerServer {
	return &WorkerServer{Worker: worker}
}

func (w *WorkerServer) ProcessTask(ctx context.Context, req *pb.TaskProcessRequest) (*emptypb.Empty, error) {
	// 处理任务的具体逻辑
	_, err := w.Worker.ReceiveTask(ctx, &definitions.Task{
		ID:         req.TaskId,
		QueryID:    req.QueryId,
		Query:      req.Query,
		Type:       req.TaskType,
		Topk:       req.TopK,
		WorkerTopK: req.WorkerTopK,
	})
	atomic.AddInt32(&w.totalMetaQueries, 1)

	if err != nil {
		return nil, err
	}

	return &emptypb.Empty{}, nil
}

func (w *WorkerServer) Query(ctx context.Context, req *pb.QueryRequest) (*pb.QueryResponse, error) {
	result, distances, err := w.Worker.InternalQuery(ctx, &definitions.Query{
		ID:         req.TaskId,
		Query:      req.Query,
		Topk:       int(req.Topk),
		Buckets:    req.Buckets,
		Goal:       req.Goal,
		SearchType: req.SearchType,
	})

	if err != nil {
		return nil, err
	}
	return &pb.QueryResponse{
		TaskId:   req.TaskId,
		Result:   result,
		Distance: distances,
	}, nil
}

func (w *WorkerServer) UpdateNodeTable(ctx context.Context, req *pb.UpdateNodeTableRequest) (*pb.UpdateNodeTableResponse, error) {
	// 更新节点表
	// TODO: 也许要加锁
	node_table := req.NodeTable
	w.Worker.GlobalWorkers = node_table
	response := &pb.UpdateNodeTableResponse{
		Success: true,
		Message: "Successfully updated node table",
	}
	return response, nil
}

func (w *WorkerServer) GetScore(ctx context.Context, req *emptypb.Empty) (*pb.GetScoreResponse, error) {
	status := w.Worker.GetStatus()
	response := &pb.GetScoreResponse{
		Score: int64(status),
	}
	return response, nil
}

func (w *WorkerServer) GetMetaQueryTotal(ctx context.Context, req *emptypb.Empty) (*pb.GetQueryTotalResponse, error) {
	res := atomic.LoadInt32(&w.totalMetaQueries)
	response := &pb.GetQueryTotalResponse{
		Total: res,
	}
	return response, nil
}

func (w *WorkerServer) GetInternalQueryTotal(ctx context.Context, req *emptypb.Empty) (*pb.GetQueryTotalResponse, error) {
	res := atomic.LoadInt32(&w.Worker.totalInternalQueries)
	response := &pb.GetQueryTotalResponse{
		Total: res,
	}
	return response, nil
}

func (w *WorkerServer) GetAvgHops(ctx context.Context, req *pb.GetAvgHopsRequest) (*pb.GetAvgHopsResponse, error) {
	// For now, only HNSW supports getting average hops
	switch req.QueryType {
	case 0:
		if hnswIndex, ok := w.Worker.MetaIndex.(*HNSWIndex); ok {
			res := hnswIndex.index.GetAvgHops()
			return &pb.GetAvgHopsResponse{Num: float32(res)}, nil
		}
	case 1:
		if w.Worker.NodeIndex != nil && w.Worker.NodeIndex.Index != nil {
			res := w.Worker.NodeIndex.Index.GetAvgHops()
			return &pb.GetAvgHopsResponse{Num: float32(res)}, nil
		}
	}
	return &pb.GetAvgHopsResponse{Num: 0}, nil
}

func (w *WorkerServer) GetAvgDistComputations(ctx context.Context, req *pb.GetAvgDistComputationsRequest) (*pb.GetAvgDistComputationsResponse, error) {
	// For now, only HNSW supports getting average distance computations
	switch req.QueryType {
	case 0:
		if hnswIndex, ok := w.Worker.MetaIndex.(*HNSWIndex); ok {
			res := hnswIndex.index.GetAvgDistComputations()
			return &pb.GetAvgDistComputationsResponse{Num: float32(res)}, nil
		}
	case 1:
		if w.Worker.NodeIndex != nil && w.Worker.NodeIndex.Index != nil {
			res := w.Worker.NodeIndex.Index.GetAvgDistComputations()
			return &pb.GetAvgDistComputationsResponse{Num: float32(res)}, nil
		}
	}
	return &pb.GetAvgDistComputationsResponse{Num: 0}, nil
}

func (w *WorkerServer) GetTimeAvg(ctx context.Context, req *pb.GetQueryTimeRequest) (*pb.GetQueryTimeResponse, error) {
	// 获取查询时间
	querytype := req.QueryType
	var res int64
	var count int32
	switch querytype {
	case 0:
		res = w.Worker.MetaQueryTimer.TotalTime.Milliseconds()
		count = atomic.LoadInt32(&w.totalMetaQueries)
	case 1:
		res = w.Worker.MetaQueryTimer.PostQueryTime.Milliseconds()
		count = atomic.LoadInt32(&w.totalMetaQueries)
	case 10:
		res = w.Worker.InternalQueryTimer.TotalTime.Milliseconds()
		count = atomic.LoadInt32(&w.Worker.totalInternalQueries)
	case 11:
		res = w.Worker.InternalQueryTimer.PostQueryTime.Milliseconds()
		count = atomic.LoadInt32(&w.Worker.totalInternalQueries)
	case 20:
		res = w.Worker.AggregationTimer.TotalTime.Milliseconds()
		count = atomic.LoadInt32(&w.totalMetaQueries)
	default:
		return nil, errors.New("invalid query type")
	}
	var avg float32
	if count == 0 {
		avg = 0
	} else {
		avg = float32(res) / float32(count)
	}

	response := &pb.GetQueryTimeResponse{Avg: avg}
	return response, nil
}

func (w *WorkerServer) GetIntegerValue(ctx context.Context, req *pb.GetIntegerValueRequest) (*pb.GetScoreResponse, error) {
	// 获取查询时间
	querytype := req.QueryType
	var res int64
	switch querytype {
	case int32(definitions.GET_TOTAL_BUCKET_COUNT):
		res = w.Worker.TotalBucketCount
	case int32(definitions.GET_TOTAL_FILTERED_BUCKET_COUNT):
		res = w.Worker.TotalFilteredBucketCount
	default:
		return nil, errors.New("invalid query type")
	}
	response := &pb.GetScoreResponse{Score: res}
	return response, nil
}

// 更新节点状态
func (w *WorkerServer) SetThreshold(ctx context.Context, req *pb.SetThresholdRequest) (*pb.SetterResponse, error) {
	w.Worker.ivfThreshold = int(req.Threshold)
	return &pb.SetterResponse{
		Success: true,
		Message: "ivfThreshold updated successfully",
	}, nil
}

func (w *WorkerServer) SetMetaK(ctx context.Context, req *pb.SetMetaKRequest) (*pb.SetterResponse, error) {
	w.Worker.metaK = int(req.MetaK)
	return &pb.SetterResponse{
		Success: true,
		Message: "MetaK updated successfully",
	}, nil
}

func (w *WorkerServer) SetEFSearch(ctx context.Context, req *pb.SetEFSearchRequest) (*pb.SetterResponse, error) {
	w.Worker.NodeIndex.Index.SetEFSearch(int(req.EfSearch))
	return &pb.SetterResponse{
		Success: true,
		Message: "EFSearch updated successfully",
	}, nil
}

// set nprobe for meta index
func (w *WorkerServer) SetNProbe(ctx context.Context, req *pb.SetNProbeRequest) (*pb.SetterResponse, error) {
	// if use naive, it will be nil
	if w.Worker.MetaIndex != nil {
		w.Worker.MetaIndex.SetNProbe(int(req.NProbe))
	}
	return &pb.SetterResponse{
		Success: true,
		Message: "NProbe updated successfully",
	}, nil
}

// SetLambda(ctx context.Context, in *SetLambdaRequest, opts ...grpc.CallOption) (*SetterResponse, error)
func (w *WorkerServer) SetLambda(ctx context.Context, req *pb.SetLambdaRequest) (*pb.SetterResponse, error) {
	w.Worker.pruningLambda = float64(req.Lambda)
	return &pb.SetterResponse{
		Success: true,
		Message: "Lambda updated successfully",
	}, nil

}

func (w *WorkerServer) GetNodeTableSize(ctx context.Context, req *emptypb.Empty) (*pb.GetNodeTableSizeResponse, error) {
	size := int32(len(w.Worker.GlobalWorkers))
	return &pb.GetNodeTableSizeResponse{
		Size: size,
	}, nil
}
