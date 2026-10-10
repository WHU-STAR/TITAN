package scheduler

import (
	"context"

	"github.com/WHU-STAR/TITAN/internal/definitions"
	pb "github.com/WHU-STAR/TITAN/protocol/pb"
	"google.golang.org/protobuf/types/known/emptypb"
)

type SchedulerServer struct {
	pb.UnimplementedSchedulerServiceServer
	scheduler *Scheduler
}

func (s *SchedulerServer) ExecuteTask(ctx context.Context, req *pb.ExecuteTaskRequest) (*pb.ExecuteTaskResponse, error) {
	taskID, err := s.scheduler.ExecuteTask(ctx, req.Query, req.Topk, req.Workertopk, req.QueryId)
	return &pb.ExecuteTaskResponse{
		TaskId: taskID,
	}, err
}

func (s *SchedulerServer) RegisterNode(ctx context.Context, req *pb.RegisterNodeRequest) (*pb.RegisterNodeResponse, error) {
	s.scheduler.RegisterNode(req.NodeId, req.Address)
	return &pb.RegisterNodeResponse{
		Success: true,
		Message: "Node registered successfully",
	}, nil
}

func (s *SchedulerServer) CompleteTask(ctx context.Context, req *pb.TaskCompleteRequest) (*pb.TaskCompleteResponse, error) {
	err := s.scheduler.CompleteTask(ctx, req.TaskId)
	if err != nil {
		return nil, err
	}

	return nil, nil
}

func (s *SchedulerServer) Heartbeat(ctx context.Context, req *pb.HeartbeatRequest) (*pb.HeartbeatResponse, error) {
	s.scheduler.Heartbeat(ctx, req)
	// 简单的心跳处理
	return &pb.HeartbeatResponse{
		Alive:     true,
		NodeTable: s.scheduler.GetNodeTable(),
	}, nil
}

func (s *SchedulerServer) GetWorkerTotalMetaCount(ctx context.Context, req *emptypb.Empty) (*pb.GetWorkerTotalQueryCountResponse, error) {
	total, counts, err := s.scheduler.GetWorkerTotal(ctx, "meta")
	if err != nil {
		return nil, err
	}
	return &pb.GetWorkerTotalQueryCountResponse{
		Total:        total,
		WorkerCounts: counts,
	}, nil
}

func (s *SchedulerServer) GetWorkerTotalInternalCount(ctx context.Context, req *emptypb.Empty) (*pb.GetWorkerTotalQueryCountResponse, error) {
	total, counts, err := s.scheduler.GetWorkerTotal(ctx, "internal")
	if err != nil {
		return nil, err
	}
	return &pb.GetWorkerTotalQueryCountResponse{
		Total:        total,
		WorkerCounts: counts,
	}, nil
}

func (s *SchedulerServer) GetWorkerTotalTimeAvg(ctx context.Context, req *pb.GetWorkerQueryTimeRequest) (*pb.GetWorkerQueryTimeResponse, error) {
	avg, workeravgs, err := s.scheduler.GetWorkerTime(ctx, req.QueryType)
	if err != nil {
		return nil, err
	}
	return &pb.GetWorkerQueryTimeResponse{
		Avg:        avg,
		WorkerAvgs: workeravgs,
	}, nil
}

func (s *SchedulerServer) GetWorkerTotalQueryHops(ctx context.Context, req *pb.GetWorkerQueryHopsRequest) (*pb.GetWorkerQueryHopsResponse, error) {
	avg, workeravgs, err := s.scheduler.GetWorkerHops(ctx, req.QueryType)
	if err != nil {
		return nil, err
	}
	return &pb.GetWorkerQueryHopsResponse{
		Avg:        avg,
		WorkerAvgs: workeravgs,
	}, nil
}

func (s *SchedulerServer) GetWorkerTotalQueryDistAvg(ctx context.Context, req *pb.GetWorkerQueryDistAvgRequest) (*pb.GetWorkerQueryDistAvgResponse, error) {
	avg, workeravgs, err := s.scheduler.GetWorkerDistAvg(ctx, req.QueryType)
	if err != nil {
		return nil, err
	}
	return &pb.GetWorkerQueryDistAvgResponse{
		Avg:        avg,
		WorkerAvgs: workeravgs,
	}, nil
}

func (s *SchedulerServer) GetWorkerIntegerProperty(ctx context.Context, req *pb.GetWorkerIntegerPropertyRequest) (*pb.GetWorkerIntegerPropertyResponse, error) {
	value, err := s.scheduler.GetWorkerIntegerProperty(ctx, req.QueryType)
	if err != nil {
		return nil, err
	}
	return &pb.GetWorkerIntegerPropertyResponse{
		WorkerValues: value,
	}, nil
}

func (s *SchedulerServer) SetPruningLambda(ctx context.Context, req *pb.SetLambdaRequest) (*pb.SetterResponse, error) {
	err := s.scheduler.SetWorkersFloatVal(ctx, definitions.SET_PRUNING_LAMBDA, req.Lambda)
	if err != nil {
		return &pb.SetterResponse{
			Success: false,
			Message: err.Error(),
		}, nil
	}
	return &pb.SetterResponse{
		Success: true,
		Message: "Lambda updated successfully",
	}, nil
}

func (s *SchedulerServer) SetEFSearch(ctx context.Context, req *pb.SetEFSearchRequest) (*pb.SetterResponse, error) {
	err := s.scheduler.SetWorkersIntVal(ctx, definitions.SET_EF_SEARCH, req.EfSearch)
	if err != nil {
		return &pb.SetterResponse{
			Success: false,
			Message: err.Error(),
		}, nil
	}
	return &pb.SetterResponse{
		Success: true,
		Message: "EF updated successfully",
	}, nil
}

func (s *SchedulerServer) SetThreshold(ctx context.Context, req *pb.SetThresholdRequest) (*pb.SetterResponse, error) {
	err := s.scheduler.SetWorkersIntVal(ctx, definitions.SET_THRESHOLD, req.Threshold)
	if err != nil {
		return &pb.SetterResponse{
			Success: false,
			Message: err.Error(),
		}, nil
	}
	return &pb.SetterResponse{
		Success: true,
		Message: "Threshold updated successfully",
	}, nil
}

func (s *SchedulerServer) SetMetaK(ctx context.Context, req *pb.SetMetaKRequest) (*pb.SetterResponse, error) {
	err := s.scheduler.SetWorkersIntVal(ctx, definitions.SET_META_K, req.MetaK)
	if err != nil {
		return &pb.SetterResponse{
			Success: false,
			Message: err.Error(),
		}, nil
	}
	return &pb.SetterResponse{
		Success: true,
		Message: "MetaK updated successfully",
	}, nil
}

func (s *SchedulerServer) SetNProbe(ctx context.Context, req *pb.SetNProbeRequest) (*pb.SetterResponse, error) {
	err := s.scheduler.SetWorkersIntVal(ctx, definitions.SET_NPROBE, req.NProbe)
	if err != nil {
		return &pb.SetterResponse{
			Success: false,
			Message: err.Error(),
		}, nil
	}
	return &pb.SetterResponse{
		Success: true,
		Message: "NProbe updated successfully",
	}, nil
}

func (s *SchedulerServer) IsInitialized(ctx context.Context, req *emptypb.Empty) (*pb.IsInitializedResponse, error) {
	// TODO: read from server
	isInitialized := s.scheduler.IsInitialized()

	return &pb.IsInitializedResponse{
		IsInitialized: isInitialized,
	}, nil

}
