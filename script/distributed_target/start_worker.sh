#!/bin/bash
################################################################################
#      This script is meant to be run under its direct parent directory.
################################################################################

TITAN_CONFIG_BUILDER=default_distributed
NAIVE_ARG=""
while getopts "b:i:p:a:s:n" opt; do
  case $opt in
  b)
    export TITAN_CONFIG_BUILDER="$OPTARG"
    ;;
  i)
    WORKER_ID="$OPTARG"
    ;;
  p)
    WORKER_PORT="$OPTARG"
    ;;
  a)
    WORKER_IP="$OPTARG"
    ;;
  s)
    SESSION_ID="$OPTARG"
    ;;
  n)
    NAIVE_ARG="--strategy=naive"
    ;;
  \?)
    echo "无效参数: -$OPTARG" >&2
    exit 1
    ;;
  esac

done

if [[ -z "$WORKER_ID" ]]; then
  echo "Error: WORKER_ID is not set"
  exit 1
fi


if [[ -z "$WORKER_IP" ]]; then
  echo "Error: WORKER_IP is not set"
  exit 1
fi

if [[ -z "$WORKER_PORT" ]]; then
  echo "Error: WORKER_PORT is not set"
  exit 1
fi

if [[ -z "$SESSION_ID" ]]; then
  echo "Error: SESSION_ID is not set"
  exit 1
fi

cd ../../

BINARY_DIR="./bin/"
cd $BINARY_DIR

PID_DIR="../pids/$SESSION_ID"

if [[ ! -d "$PID_DIR" ]]; then
  echo "Error: PID_DIR $PID_DIR does not exist"
  exit 1
fi

./worker \
  --id="$WORKER_ID" \
  --address="$WORKER_IP:$WORKER_PORT" \
  --mode=distributed \
  $NAIVE_ARG \
  2>&1 > >(tee -a "../log/$SESSION_ID/$WORKER_ID.log" \
  >>"../log/$SESSION_ID/all_workers.log") &

PID=$!
echo $PID >"$PID_DIR/${WORKER_ID}.pid"

echo "PID: $PID"
echo "Description: "
ps -p $PID -o cmd=
disown
