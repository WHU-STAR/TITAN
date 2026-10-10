#!/bin/bash
# THIS SCRIPT IS ONLY FOR MACOS

# Build worker
make worker

# Parse command line arguments
for arg in "$@"; do
    case $arg in
    DEBUG=*)
        DEBUG="${arg#*=}"
        shift
        ;;
    esac
done

# Kill all workers before starting
sh ./kill_workers.sh

# Debug mode flag (if not set by command line)
DEBUG=${DEBUG:-false}

MODE=distributed

# Number of workers to start
NUM_WORKERS=16

# Base port for workers
BASE_PORT=50050

# Base ip for workers
BASE_IP="localhost"
dataTarget=$(readlink ../data/workerdata)

if [[ -z "$dataTarget" ]]; then
    echo "Error: workerdata is not linked to any data directory"
    exit 1
fi

naiveArg=""

if [[ $dataTarget == *"naive"* ]]; then
    naiveArg="--strategy=naive"
else
    naiveArg=""
fi

# Binary dir
BINARY_DIR="../bin/"
cd $BINARY_DIR

# Create pid directory if it doesn't exist
mkdir -p ../pids

# Create log directory if it doesn't exist
mkdir -p ../log

# Clear all existing log files
rm -f ../log/worker-*.log

# Clear previous log file
echo "Starting workers at $(date)" >"../log/all_workers.log"

# Clear individual worker logs
for ((i = 0; i < NUM_WORKERS; i++)); do
    WORKER_ID="worker-$i"
    echo "Starting worker $WORKER_ID at $(date)" >"../log/$WORKER_ID.log"
done


if [ -z "$coreCount" ]; then
    coreCount=16
fi

# Start workers
for ((i = 0; i < NUM_WORKERS; i++)); do
    WORKER_PORT=$((BASE_PORT + i + 1))
    WORKER_ID="worker-$i"

    # 设置 CPU 绑定逻辑
    if (( coreCount <= NUM_WORKERS )); then
        # 核心数少于 worker 数，轮转分配单个核心
        CPU_CORE=$((i % coreCount))
        CPU_BINDING=$CPU_CORE
    else
        # 核心数多于等于 worker 数，均匀分配多核
        CORES_PER_WORKER=$((coreCount / NUM_WORKERS))
        START_CORE=$((i * CORES_PER_WORKER))
        END_CORE=$((START_CORE + CORES_PER_WORKER - 1))
        CPU_BINDING="${START_CORE}-${END_CORE}"
    fi

    echo "Starting worker $WORKER_ID on port $WORKER_PORT using CPU(s) $CPU_BINDING"

    if [ "$DEBUG" = "true" ]; then
        echo "Starting worker $WORKER_ID in debug mode"
        LOG_LEVEL=DEBUG ./worker --id "$WORKER_ID" --address "$BASE_IP:$WORKER_PORT" --mode "$MODE" $naiveArg 2>&1 | tee -a "../log/$WORKER_ID.log" >>"../log/all_workers.log" &
    else
        echo "Starting worker $WORKER_ID in normal mode"
        ./worker --id "$WORKER_ID" --address "$BASE_IP:$WORKER_PORT" --mode "$MODE" $naiveArg 2>&1 | tee -a "../log/$WORKER_ID.log" >>"../log/all_workers.log" &
    fi

    echo $! >"../pids/$WORKER_ID.pid"
done



echo "Started $NUM_WORKERS workers"
echo "Individual worker logs are in log/worker-*.log"
echo "Combined log is in logs/all_workers.log"
echo "To follow all workers output: tail -f log/all_workers.log"
echo "To follow a specific worker: tail -f log/worker-N.log"
echo "To stop a specific worker: kill \$(cat pids/worker-N.pid)"

# Follow the combined log output
tail -f "../log/all_workers.log"
