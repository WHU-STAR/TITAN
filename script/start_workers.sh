#!/bin/bash

# Build worker
pushd .. && make worker
popd

if [ -z "$BASE_PORT" ]; then
  BASE_PORT=50050
fi

BASE_CORE=0

while getopts "c:l:" opt; do
  case $opt in
  c)
    BASE_CORE="$OPTARG"
    ;;
  \?)
    echo "无效参数: -$OPTARG" >&2
    exit 1
    ;;
  esac
done

echo "BASE_CORE: $BASE_CORE"

echo "BASE_PORT: $BASE_PORT"

# Kill all workers before starting
# sh kill_workers.sh

MODE=distributed

if [ -z "$num_workers" ]; then
  num_workers=16
fi


# Base ip for workers
BASE_IP="localhost"
dataTarget=$(readlink ../data/workerdata)

if [[ "x$session_id" == "x" ]]; then
  echo "Error: session_id is not set"
  exit 1
fi

if [[ -z "$dataTarget" ]]; then
  echo "Error: workerdata is not linked to any data directory"
  exit 1
fi

naiveArg=""

if [[ $dataTarget == *"naive"* ]]; then
  naiveArg="--strategy=naive"
fi

# Binary dir
BINARY_DIR="../bin/"
cd $BINARY_DIR
# sh ../logrotate.sh &
# echo $! > ../pids/$session_id/logrotate.pid

if [[ -d "../pids/$session_id" ]]; then
  echo "Session directory already exists，killing existing workers"
  sh ../script/kill_workers.sh "$session_id"
fi

# Create pid directory if it doesn't exist
mkdir -p ../pids/$session_id

# Create log directory if it doesn't exist
mkdir -p ../log/$session_id

# Clear all existing log files
rm -f ../log/$session_id/worker-*.log

# Clear previous log file
echo "Starting workers at $(date)" >"../log/$session_id/all_workers.log"

# Clear individual worker logs
for ((i = 0; i < num_workers; i++)); do
  WORKER_ID="worker-$i"
  echo "Starting worker $WORKER_ID at $(date)" >"../log/$session_id/$WORKER_ID.log"
done

if [ -z "$coreCount" ]; then
  coreCount=16
fi

# Start workers
for ((i = 0; i < num_workers; i++)); do
  WORKER_PORT=$((BASE_PORT + i + 1))
  WORKER_ID="worker-$i"

  # 设置 CPU 绑定逻辑
  if ((coreCount <= num_workers)); then
    # 核心数少于 worker 数，轮转分配单个核心
    CPU_CORE=$((BASE_CORE + (i % coreCount)))
    CPU_BINDING=$CPU_CORE
  else
    # 核心数多于等于 worker 数，均匀分配多核
    CORES_PER_WORKER=$((coreCount / num_workers))
    START_CORE=$((BASE_CORE + i * CORES_PER_WORKER))
    END_CORE=$((START_CORE + CORES_PER_WORKER - 1))
    CPU_BINDING="${START_CORE}-${END_CORE}"
  fi

  echo "Worker port: $WORKER_PORT"
  CMD="taskset -c $CPU_BINDING ./worker --id \"$WORKER_ID\" --address \"$BASE_IP:$WORKER_PORT\" --mode \"$MODE\" $naiveArg 2>&1 | tee -a \"../log/$session_id/$WORKER_ID.log\" >>\"../log/$session_id/all_workers.log\" &"
  echo "CMD: $CMD"
  eval $CMD

  echo $! >"../pids/$session_id/$WORKER_ID.pid"
done

echo "Started $num_workers workers"
echo "Individual worker logs are in log/$session_id/worker-*.log"
echo "Combined log is in log/$session_id/all_workers.log"
echo "To follow all workers output: tail -f log/$session_id/all_workers.log"
echo "To follow a specific worker: tail -f log/$session_id/worker-N.log"
echo "To stop a specific worker: kill \$(cat pids/worker-N.pid)"

# Follow the combined log output
tail -f "../log/$session_id/all_workers.log"
