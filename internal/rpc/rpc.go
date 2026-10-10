package rpc

import (
	"errors"
	"sync"

	"github.com/WHU-STAR/TITAN/config"
	"github.com/WHU-STAR/TITAN/protocol/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

var (
	connManager = &ConnectionManager{
		conns: make(map[string]*grpc.ClientConn),
	}
)

type ConnectionManager struct {
	sync.RWMutex
	conns map[string]*grpc.ClientConn
}

// GetConnectionManager returns the global connection manager instance
func GetConnectionManager() *ConnectionManager {
	return connManager
}

// GetConnection returns an existing connection or creates a new one
func (m *ConnectionManager) GetConnection(address string) (*grpc.ClientConn, error) {
	m.RLock()
	if conn, exists := m.conns[address]; exists {
		m.RUnlock()
		return conn, nil
	}
	m.RUnlock()

	// Create new connection
	m.Lock()
	defer m.Unlock()

	// Double check after acquiring write lock
	if conn, exists := m.conns[address]; exists {
		return conn, nil
	}

	conn, err := grpc.NewClient(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}

	m.conns[address] = conn
	return conn, nil
}

// CloseConnection closes and removes a connection
func (m *ConnectionManager) CloseConnection(address string) error {
	m.Lock()
	defer m.Unlock()

	if conn, exists := m.conns[address]; exists {
		delete(m.conns, address)
		return conn.Close()
	}
	return nil
}

// CloseAllConnections closes all connections
func (m *ConnectionManager) CloseAllConnections() {
	m.Lock()
	defer m.Unlock()

	for address, conn := range m.conns {
		conn.Close()
		delete(m.conns, address)
	}
}

// GetSchedulerClientConn returns a gRPC client connection to the scheduler
func GetSchedulerClientConn() (conn *grpc.ClientConn, err error) {
	address := config.Config.GetSchedulerAddress()
	if address == "" {
		return nil, errors.New("empty scheduler address provided")
	}
	return connManager.GetConnection(address)
}

// GetWorkerClientConn returns a gRPC client connection to the worker with the given ID
func GetWorkerClientConn(address string) (conn *grpc.ClientConn, err error) {
	if address == "" {
		return nil, errors.New("empty worker address provided")
	}
	return connManager.GetConnection(address)
}

// GetClientClientConn returns a gRPC client connection to the client with the given address
func GetClientClientConn() (conn *grpc.ClientConn, err error) {
	address := config.Config.GetClientAddress()
	if address == "" {
		return nil, errors.New("empty client address provided")
	}
	return connManager.GetConnection(address)
}

// GetSchedulerClient returns a SchedulerServiceClient
func GetSchedulerClient() (pb.SchedulerServiceClient, error) {
	conn, err := GetSchedulerClientConn()
	if err != nil {
		return nil, err
	}
	return pb.NewSchedulerServiceClient(conn), nil
}

// GetClientClient returns a ClientServiceClient
func GetClientClient() (pb.ClientServiceClient, error) {
	conn, err := GetClientClientConn()
	if err != nil {
		return nil, err
	}
	return pb.NewClientServiceClient(conn), nil
}

// GetWorkerExtClient returns a WorkerExtServiceClient for the given address
func GetWorkerExtClient(workerAddress string) (pb.WorkerExtServiceClient, error) {
	conn, err := GetWorkerClientConn(workerAddress)
	if err != nil {
		return nil, err
	}
	return pb.NewWorkerExtServiceClient(conn), nil
}

// GetWorkerIntClient returns a WorkerIntServiceClient for the given address
func GetWorkerIntClient(workerAddress string) (pb.WorkerIntServiceClient, error) {
	conn, err := GetWorkerClientConn(workerAddress)
	if err != nil {
		return nil, err
	}
	return pb.NewWorkerIntServiceClient(conn), nil
}
