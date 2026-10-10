#!/bin/bash

protoc --go_out=.. --go-grpc_out=.. ./protocol/proto/*.proto -I ./protocol/proto
