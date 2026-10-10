package scheduler

import (
	"context"
	"errors"
	"net"
	"sync"
	"time"

	"github.com/WHU-STAR/TITAN/internal/definitions"
	"github.com/WHU-STAR/TITAN/internal/rpc"
	"github.com/WHU-STAR/TITAN/pkg/logger"
	pb "github.com/WHU-STAR/TITAN/protocol/pb"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/types/known/emptypb"
)

/*
TODO:
1. current available node
2. send index data query
3. worker: send task to other workers after meta data query
*/

var ErrNoAvailableNode = errors.New("no available node")

const HEARTBEAT_TIMEOUT = time.Second * 24

type Scheduler struct {
	address     string
	strategy    string
	num_workers int
	nodes       map[string]*definitions.Node
	nodeskey    []string
	nodesLock   sync.RWMutex

	tasks    chan *definitions.Task       // 任务队列
	taskMap  map[string]*definitions.Task // 任务状态映射
	taskLock sync.RWMutex

	ctx        context.Context    // 用于控制goroutine的生命周期
	cancelFunc context.CancelFunc // 用于取消所有goroutine
	nodeOffset int32
}

func NewScheduler(address string, strategy string, num_workers int) *Scheduler {
	ctx, cancel := context.WithCancel(context.Background())
	s := &Scheduler{
		address:     address,
		strategy:    strategy,
		num_workers: num_workers,
		nodes:       make(map[string]*definitions.Node),
		tasks:       make(chan *definitions.Task, 100000), // 缓冲区大小可配置
		taskMap:     make(map[string]*definitions.Task),
		ctx:         ctx,
		cancelFunc:  cancel,
		nodeOffset:  0,
	}
	// 启动心跳监控
	go s.StartHealthCheck(ctx)

	// 启动任务处理
	go s.AssignTasks(ctx)
	return s
}

func (s *Scheduler) StartHealthCheck(ctx context.Context) {
	ticker := time.NewTicker(HEARTBEAT_TIMEOUT)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			now := time.Now()
			s.nodesLock.Lock()
			for nodeID, node := range s.nodes {
				if now.Sub(node.LastPing) > HEARTBEAT_TIMEOUT {
					logger.Info("Node is offline", zap.String("nodeID", nodeID))
					delete(s.nodes, nodeID)
					if err := rpc.GetConnectionManager().CloseConnection(node.ID); err != nil {
						logger.Error("Failed to close connection to node", zap.String("nodeID", node.ID), zap.Error(err))
					}
				}
			}
			s.nodesLock.Unlock()
		}
	}
}

func (s *Scheduler) Stop() {
	logger.Info("Stopping scheduler")
	// close task queue
	close(s.tasks)

	// close goroutines
	s.cancelFunc()

	// close all connections
	connManager := rpc.GetConnectionManager()
	connManager.CloseAllConnections()

}

func (s *Scheduler) StartGRPCServer() {
	server := grpc.NewServer()
	pb.RegisterSchedulerServiceServer(server, &SchedulerServer{
		scheduler: s,
	})

	lis, err := net.Listen("tcp", s.address)
	if err != nil {
		panic(err)
	}
	logger.Info("Scheduler gRPC server started at", zap.String("address", s.address))
	if err := server.Serve(lis); err != nil {
		logger.Error("Failed to start gRPC server", zap.Error(err))
	}
}

func (s *Scheduler) RegisterNode(nodeID string, address string) error {
	logger.Info("Registering node", zap.String("nodeID", nodeID), zap.String("address", address))

	s.nodesLock.Lock()
	defer s.nodesLock.Unlock()
	s.nodes[nodeID] = &definitions.Node{
		ID:       nodeID,
		Address:  address,
		Score:    0,
		LastPing: time.Now(),
	}
	rpcNewNodesMap := make(map[string]string)

	// 向节点更新节点表
	nodes_table := make(map[string]string)
	s.nodeskey = make([]string, 0)
	for k, v := range s.nodes {
		nodes_table[k] = v.Address
		s.nodeskey = append(s.nodeskey, k)
		rpcNewNodesMap[k] = v.Address
	}
	for _, node := range s.nodes {
		client, err := rpc.GetWorkerExtClient(node.Address)
		if err != nil {
			logger.Error("Failed to connect to node", zap.String("nodeID", node.ID),

				zap.String("address", node.Address), zap.Error(err))
			continue
		}
		_, err = client.UpdateNodeTable(context.Background(), &pb.UpdateNodeTableRequest{
			NodeTable: rpcNewNodesMap,
		})
		if err != nil {
			logger.Error("Failed to update node table", zap.String("nodeID", node.ID),
				zap.String("address", node.Address), zap.Error(err))
			continue
		}
	}

	logger.Info("Nodes table:", zap.Any("nodes_table", nodes_table))
	return nil

}

func (s *Scheduler) GetNodeTable() map[string]string {
	nodes_table := make(map[string]string)
	for k, v := range s.nodes {
		nodes_table[k] = v.Address
	}
	return nodes_table
}

// AssignTasks 持续运行，将任务分配给可用节点
func (s *Scheduler) AssignTasks(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case task, ok := <-s.tasks:
			if !ok {
				return
			}
			if err := s.assignTask(ctx, task); err != nil {
				// 分配失败，重新入队
				select {
				case s.tasks <- task:
				case <-ctx.Done():
					return
				}
			}
		}
	}
}

// assignTask 将任务分配给可用节点
func (s *Scheduler) assignTask(ctx context.Context, task *definitions.Task) error {
	// s.QueryNodeStatus(ctx)
	// node := s.GetAvailableNode()
	// randomly pick a node from s.nodes:
	s.nodesLock.RLock()
	if len(s.nodes) == 0 {
		s.nodesLock.RUnlock()
		return ErrNoAvailableNode
	}
	randomNodeID := s.nodeOffset % int32(len(s.nodes))
	s.nodeOffset++
	nodeID := s.nodeskey[randomNodeID]
	node := s.nodes[nodeID]
	s.nodesLock.RUnlock()
	if node == nil {
		logger.Error("No available node found", zap.String("nodeID", nodeID))
		return ErrNoAvailableNode
	}

	// 更新任务状态
	s.taskLock.Lock()
	task.Status = definitions.TaskStatusAssigned
	task.NodeID = node.ID

	// 根据策略确定任务类型
	if s.strategy == "similarity" {
		task.Type = definitions.TaskMetaQuery
	} else {
		task.Type = definitions.TaskNodeQuery
	}
	s.taskLock.Unlock()

	// 异步发送任务到工作节点
	go s.sendTaskToNode(ctx, task, node)

	return nil
}

// sendTaskToNode 通过gRPC发送任务到工作节点
func (s *Scheduler) sendTaskToNode(ctx context.Context, task *definitions.Task, node *definitions.Node) {
	//创建 WorkerService 客户端
	client, err := rpc.GetWorkerExtClient(node.Address)
	if err != nil {
		logger.Error("Failed to connect to node", zap.String("nodeID", node.ID), zap.Error(err))
		return
	}

	// 准备将任务传送给工作节点
	taskRequest := &pb.TaskProcessRequest{
		TaskId:     task.ID,
		TaskType:   task.Type,
		QueryId:    task.QueryID,
		Query:      task.Query,
		TopK:       task.Topk,
		WorkerTopK: task.WorkerTopK,
	}

	//向工作节点发送任务
	_, err = client.ProcessTask(ctx, taskRequest)
	if err != nil {
		logger.Error("Failed to send task to node", zap.String("nodeID", node.ID), zap.Error(err))
		return
	}
}

// GetAvailableNode 获取可用节点
func (s *Scheduler) GetAvailableNode() *definitions.Node {
	s.nodesLock.RLock()
	defer s.nodesLock.RUnlock()
	minimumScore := -1
	var selectedNode *definitions.Node
	if len(s.nodes) == 0 {
		return nil
	}
	for _, node := range s.nodes {
		if minimumScore < 0 || node.Score < minimumScore {
			selectedNode = node
			minimumScore = node.Score
		}
	}
	return selectedNode
}

func (s *Scheduler) QueryNodeStatus(ctx context.Context) {
	logger.Info("Querying node status")
	wg := sync.WaitGroup{}
	for _, node := range s.nodes {
		wg.Add(1)
		go func(node *definitions.Node) {
			client, err := rpc.GetWorkerExtClient(node.Address)
			if err != nil {
				logger.Error("Failed to connect to node", zap.String("nodeID", node.ID), zap.Error(err))
				return
			}
			result, err := client.GetScore(ctx, &emptypb.Empty{})
			if err != nil {
				logger.Error("Failed to query node status", zap.String("nodeID", node.ID), zap.Error(err))
				return
			}
			logger.Info("Node status", zap.String("nodeID", node.ID), zap.Int64("score", result.Score))
			s.UpdateNodeStatus(node.ID, result.Score)
			wg.Done()
		}(node)
	}
	wg.Wait()
}

// UpdateNodeStatus 更新节点状态
func (s *Scheduler) UpdateNodeStatus(nodeID string, score int64) {
	s.nodesLock.Lock()
	defer s.nodesLock.Unlock()

	if node, exists := s.nodes[nodeID]; exists {
		node.Score = int(score)
		node.LastPing = time.Now()
	}
}

// Heartbeat 处理节点心跳
func (s *Scheduler) Heartbeat(ctx context.Context, req *pb.HeartbeatRequest) {
	logger.Info("Heartbeat from", zap.String("nodeID", req.NodeId), zap.Int64("score", req.Score))
	s.UpdateNodeStatus(req.NodeId, req.Score)
}

// ExecuteTask 处理查询任务
func (s *Scheduler) ExecuteTask(ctx context.Context, query []float32, topK int64, workerTopK int64, queryID uint32) (taskID string, err error) {
	// TODO: 目前假设一个请求里只有一个query
	task := &definitions.Task{
		ID:         generateTaskID(),
		Query:      query,
		Topk:       topK,
		WorkerTopK: workerTopK,
		Status:     definitions.TaskStatusPending,
		CreatedAt:  time.Now(),
		QueryID:    queryID,
	}

	logger.Debug("Received Task", zap.String("taskID", task.ID))
	// 保存任务状态
	s.taskLock.Lock()
	s.taskMap[task.ID] = task
	s.taskLock.Unlock()

	// 将任务加入队列
	select {
	case s.tasks <- task:
		return task.ID, nil
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

func (s *Scheduler) CompleteTask(ctx context.Context, taskID string) error {
	logger.Debug("Completing Task", zap.String("taskID", taskID))
	s.taskLock.Lock()
	delete(s.taskMap, taskID)
	s.taskLock.Unlock()
	logger.Debug("Task completed", zap.String("taskID", taskID))
	return nil
}

func generateTaskID() string {
	return time.Now().Format("20060102150405") + "-" + randomString(6)
}

func randomString(n int) string {
	const letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	b := make([]byte, n)
	for i := range b {
		b[i] = letters[time.Now().UnixNano()%int64(len(letters))]
	}
	return string(b)
}

func (s *Scheduler) SetWorkersFloatVal(ctx context.Context, valType int32, val float32) error {
	var errList []error
	wg := sync.WaitGroup{}
	for _, node := range s.nodes {
		wg.Add(1)
		go func(node *definitions.Node) {
			defer wg.Done()
			client, err := rpc.GetWorkerExtClient(node.Address)
			if err != nil {
				errList = append(errList, err)
				return
			}
			switch valType {
			case definitions.SET_PRUNING_LAMBDA: // pruning lambda
				_, err = client.SetLambda(ctx, &pb.SetLambdaRequest{Lambda: val})
			default:
				err = errors.New("unknown valType")
			}
			if err != nil {
				errList = append(errList, err)
			}
		}(node)
	}
	wg.Wait()
	if len(errList) > 0 {
		return errList[0]
	}
	return nil
}

func (s *Scheduler) SetWorkersIntVal(ctx context.Context, valType int32, val int32) error {
	var errList []error
	wg := sync.WaitGroup{}
	for _, node := range s.nodes {
		wg.Add(1)
		go func(node *definitions.Node) {
			defer wg.Done()
			client, err := rpc.GetWorkerExtClient(node.Address)
			if err != nil {
				errList = append(errList, err)
				return
			}
			switch valType {
			case definitions.SET_EF_SEARCH: // ef search
				_, err = client.SetEFSearch(ctx, &pb.SetEFSearchRequest{EfSearch: val})
			case definitions.SET_THRESHOLD: // threshold
				_, err = client.SetThreshold(ctx, &pb.SetThresholdRequest{Threshold: val})
			case definitions.SET_META_K: // meta_k
				_, err = client.SetMetaK(ctx, &pb.SetMetaKRequest{MetaK: val})
			case definitions.SET_NPROBE: // nprobe
				_, err = client.SetNProbe(ctx, &pb.SetNProbeRequest{NProbe: val})
			default:
				err = errors.New("unknown valType")
			}
			if err != nil {
				errList = append(errList, err)
			}
		}(node)
	}
	wg.Wait()
	if len(errList) > 0 {
		return errList[0]
	}
	return nil
}

func (s *Scheduler) GetWorkerTotal(ctx context.Context, countType string) (int32, []int32, error) {
	result := make([]int32, 0)
	reslock := sync.Mutex{}
	wg := sync.WaitGroup{}
	if !(countType == "meta" || countType == "internal") {
		return 0, nil, errors.New("countType must be 'meta' or 'internal'")
	}
	for _, node := range s.nodes {
		wg.Add(1)
		go func(node *definitions.Node) {
			client, err := rpc.GetWorkerExtClient(node.Address)
			if err != nil {
				logger.Error("Failed to connect to node", zap.String("nodeID", node.ID), zap.Error(err))
				return
			}
			workerqueryres := &pb.GetQueryTotalResponse{}
			switch countType {
			case "meta":
				workerqueryres, err = client.GetMetaQueryTotal(ctx, &emptypb.Empty{})
			case "internal":
				workerqueryres, err = client.GetInternalQueryTotal(ctx, &emptypb.Empty{})
			}
			if err != nil {
				logger.Error("Failed to query node status", zap.String("nodeID", node.ID), zap.Error(err))
				return
			}
			reslock.Lock()
			result = append(result, workerqueryres.Total)
			reslock.Unlock()
			wg.Done()
		}(node)
	}
	wg.Wait()

	total := int32(0)
	for _, v := range result {
		total += v
	}

	return total, result, nil
}

func (s *Scheduler) GetWorkerHops(ctx context.Context, queryType uint32) (float32, []float32, error) {
	result := make([]float32, 0)
	reslock := sync.Mutex{}
	wg := sync.WaitGroup{}

	for _, node := range s.nodes {
		wg.Add(1)
		go func(node *definitions.Node) {
			client, err := rpc.GetWorkerExtClient(node.Address)
			if err != nil {
				logger.Error("Failed to connect to node", zap.String("nodeID", node.ID), zap.Error(err))
				return
			}
			workerqueryres, err := client.GetAvgHops(ctx, &pb.GetAvgHopsRequest{
				QueryType: queryType,
			})
			if err != nil {
				logger.Error("Failed to query node status", zap.String("nodeID", node.ID), zap.Error(err))
				return
			}
			reslock.Lock()
			result = append(result, workerqueryres.Num)
			reslock.Unlock()
			wg.Done()
		}(node)
	}
	wg.Wait()

	total := float32(0)
	for _, v := range result {
		total += v
	}
	total = total / float32(len(result))

	return float32(total), result, nil
}

func (s *Scheduler) GetWorkerDistAvg(ctx context.Context, queryType uint32) (float32, []float32, error) {
	result := make([]float32, 0)
	reslock := sync.Mutex{}
	wg := sync.WaitGroup{}

	for _, node := range s.nodes {
		wg.Add(1)
		go func(node *definitions.Node) {
			client, err := rpc.GetWorkerExtClient(node.Address)
			if err != nil {
				logger.Error("Failed to connect to node", zap.String("nodeID", node.ID), zap.Error(err))
				return
			}
			workerqueryres, err := client.GetAvgDistComputations(ctx, &pb.GetAvgDistComputationsRequest{
				QueryType: queryType,
			})
			if err != nil {
				logger.Error("Failed to query node status", zap.String("nodeID", node.ID), zap.Error(err))
				return
			}
			reslock.Lock()
			result = append(result, workerqueryres.Num)
			reslock.Unlock()
			wg.Done()
		}(node)
	}
	wg.Wait()

	total := float32(0)
	for _, v := range result {
		total += float32(v)
	}
	total = total / float32(len(result))

	return float32(total), result, nil
}

func (s *Scheduler) GetWorkerTime(ctx context.Context, timeType int32) (float32, []float32, error) {
	result := make([]float32, 0)
	reslock := sync.Mutex{}
	wg := sync.WaitGroup{}

	for _, node := range s.nodes {
		wg.Add(1)
		go func(node *definitions.Node) {
			client, err := rpc.GetWorkerExtClient(node.ID)
			if err != nil {
				logger.Error("Failed to connect to node", zap.String("nodeID", node.ID), zap.Error(err))
				return
			}
			workerqueryres, err := client.GetTimeAvg(ctx, &pb.GetQueryTimeRequest{
				QueryType: int32(timeType),
			})
			if err != nil {
				logger.Error("Failed to query node status", zap.String("nodeID", node.ID), zap.Error(err))
				return
			}
			reslock.Lock()
			result = append(result, workerqueryres.Avg)
			reslock.Unlock()
			wg.Done()
		}(node)
	}
	wg.Wait()

	total := float32(0)
	for _, v := range result {
		total += v
	}
	total = total / float32(len(result))
	return total, result, nil
}

func (s *Scheduler) GetWorkerIntegerProperty(ctx context.Context, queryType int32) ([]int32, error) {
	result := make([]int32, 0)
	wg := sync.WaitGroup{}
	reslock := sync.Mutex{}
	for _, node := range s.nodes {
		wg.Add(1)
		go func(node *definitions.Node) {
			defer wg.Done()
			client, err := rpc.GetWorkerExtClient(node.Address)
			if err != nil {
				logger.Error("Failed to connect to node", zap.String("nodeID", node.ID), zap.Error(err))
				return
			}
			resp, err := client.GetIntegerValue(ctx, &pb.GetIntegerValueRequest{QueryType: queryType})
			if err != nil {
				logger.Error("Failed to query node integer property", zap.String("nodeID", node.ID), zap.Error(err))
				return
			}
			reslock.Lock()
			result = append(result, int32(resp.Score))
			reslock.Unlock()
		}(node)
	}
	wg.Wait()
	return result, nil
}

// check if all nodes are initialized
func (s *Scheduler) IsInitialized() bool {
	s.nodesLock.RLock()
	defer s.nodesLock.RUnlock()
	return len(s.nodes) == s.num_workers
}
