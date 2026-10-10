package test

import (
	"context"
	"log"
	"net"

	"github.com/WHU-STAR/TITAN/protocol/pb"
	"google.golang.org/grpc"
)

type TestServer struct {
	pb.UnimplementedTestDelayServiceServer
}

func NewTestServer() *TestServer {
	return &TestServer{}
}

func (t *TestServer) TestDelay(ctx context.Context, req *pb.TestDelayRequest) (*pb.TestDelayResponse, error) {
	return &pb.TestDelayResponse{
		Message: "Hello",
	}, nil

}

func (t *TestServer) StartGRPCServer() {
	server := grpc.NewServer()
	pb.RegisterTestDelayServiceServer(server, t)
	lis, err := net.Listen("tcp", "127.0.0.1:50051")

	if err != nil {
		panic(err)
	}

	if err := server.Serve(lis); err != nil {
		log.Fatal("Failed to start worker gRPC server")
	}

}
