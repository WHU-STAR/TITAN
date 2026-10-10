#!/bin/bash
cd "$(dirname "$0")/.."
protoc \
  --go_out=. --go_opt=module=github.com/WHU-STAR/TITAN \
  --go-grpc_out=. --go-grpc_opt=module=github.com/WHU-STAR/TITAN \
  -I protocol/proto \
  protocol/proto/*.proto
