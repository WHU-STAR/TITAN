# Makefile
all: clean grpc build image logrotate

pwd := $(shell pwd)
NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

.PHONY: build
build: engine worker scheduler client

.PHONY: logrotate
logrotate:
	mkdir -p config/logrotate/
	awk -v p="$(pwd)" 'NR==1{$$0=p "/log/*/*.log {"} 1' config/logrotate/logrotate.conf.template > config/logrotate/logrotate.conf

.PHONY: client
client:
	@echo "building client"
	@go build -o ./bin/client ./cmd/client/main.go

.PHONY: worker
worker:
	@echo "building worker"
	@go build -o ./bin/worker ./cmd/worker/main.go

.PHONY: scheduler
scheduler:
	@echo "building scheduler"
	@go build -o ./bin/scheduler ./cmd/scheduler/main.go

.PHONY: engine
engine:
	@echo "building engine"
	@cd internal/engine && mkdir -p build && cd build && cmake .. && make -j$(NPROC)

.PHONY: pybind
pybind:
	@echo "building pybind"
	@cd internal/engine/index/vamana/pybinding && pip install .

.PHONY: image
image:
	@echo "building image"
	@docker build -t titan .

.PHONY: clean
clean:
	@echo "cleaning..."
	@rm -rf ./internal/engine/build
	@rm -rf ./bin/*

.PHONY: grpc
grpc:
	@echo "building grpc"
	@sh ./script/rpc_gen.sh

.PHONY: config
config:
	@echo "building config"
	@sh ./script/generate_config.sh
