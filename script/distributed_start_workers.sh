#!/bin/bash

CWD=$(pwd)
cd ..; make worker
cd $CWD


if [ -z LOG_LEVEL ]; then
  LOG_LEVEL=INFO
fi
MODE=distributed
REMOTE_DIR="${REMOTE_DIR:-/data/titan/script/distributed_target}"
WORKER_BIN="worker"
pwd

if [ -z "$BASE_PORT" ]; then
  echo "Error: BASE_PORT is not set"
  exit 1
fi

if [ -z "$num_workers" ]; then
  num_workers=16
fi

# 获取 data 链接目标
dataTarget=$(readlink ../data/workerdata)

if [[ "x$session_id" == "x" ]]; then
  echo "Error: session_id is not set"
  exit 1
fi

if [[ -d "../pids/$session_id" ]]; then
  echo "Session directory already exists, please kill existing workers first and purge files first."
  exit 1
fi


# if [[ -d "../log/$session_id" ]]; then
#   echo "Error: log directory for session_id already exists, please kill existing workers first and purge files first."
#   exit 1
# fi

OLDUMASK=$(umask)
umask 000
PIDDIR="../pids/$session_id"
mkdir -p "$PIDDIR"
chmod 777 "$PIDDIR"
# make sure that all servers can write to the pid directory

if [[ -z "$TITAN_CONFIG_BUILDER" ]]; then
  TITAN_CONFIG_BUILDER=default_distributed
fi

echo "dataTarget: $dataTarget"
if [[ -z "$dataTarget" ]]; then
  echo "Error: workerdata is not linked to any data directory"
  exit 1
fi

naiveArg=""
if [[ $dataTarget == *"naive"* ]]; then
  naiveArg="-n"
fi

LOGDIR="../log/$session_id"
mkdir -p "$LOGDIR"
chmod 777 "$LOGDIR"

umask "$OLDUMASK"

sleep 30			# 确保worker在scheduler之后启动

echo "Starting distributed workers at $(date)" >../log/all_workers.log

# 启动 worker
for ((i = 0; i < num_workers; i++)); do
  WORKER_ID="worker-$i"
  HOSTNAME=`./generate_host_name.sh $i`
  WORKER_PORT=$((BASE_PORT + i + 1))
  BASE_IP=$(getent hosts "$HOSTNAME" | awk '{ print $1 }')
  LOGFILE="../log/${WORKER_ID}.log"

  echo "Starting $WORKER_ID on $HOSTNAME:$WORKER_PORT"

  # 执行远程命令并将远程 PID 文件内容保存到本地
  CMD="cd $REMOTE_DIR && ./start_worker.sh -b $TITAN_CONFIG_BUILDER -a $BASE_IP -i $WORKER_ID -p $WORKER_PORT -s $session_id $naiveArg 2>&1"
  echo "$CMD"
  ssh -n "$target_user@$HOSTNAME" "$CMD" &
  disown
done


echo "Started $num_workers distributed workers"
echo "Log files in ../log/$session_id/worker-*.log"
echo "To stop a worker: kill \$(cat ../pids/worker-XXXX.pid)"
# 合并日志（本地 stdout）
tail -f ../log/$session_id/all_workers.log
