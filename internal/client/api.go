package client

import (
	"context"

	"github.com/golang/protobuf/ptypes/empty"
	pb "github.com/WHU-STAR/TITAN/protocol/pb"
)

type ClientServer struct {
	pb.UnimplementedClientServiceServer
	client *Client
}

func (c *ClientServer) QueryTask(ctx context.Context, req *pb.ClientTaskRequest) (*empty.Empty, error) {
	c.client.ProcessQueryResult(ctx, req)
	return &empty.Empty{}, nil
}
