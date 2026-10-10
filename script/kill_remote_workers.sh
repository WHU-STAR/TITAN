#!/bin/bash

NUM_WORKERS=16
target_user=$USER
REMOTE_DIR="/data/vcluster-system"
while getopts ":s:w:u:" opt; do
  case $opt in
  s)
    session_id="$OPTARG"
    ;;
  w)
    NUM_WORKERS="$OPTARG"
    ;;
  u)
    target_user="$OPTARG"
    ;;
  \?)
    echo "无效参数: -$OPTARG" >&2
    exit 1
    ;;
  esac
done

if [ -z "$session_id" ]; then
  if ! command -v fzf >/dev/null 2>&1; then
    echo "fzf 未安装，请先安装 fzf"
    exit 1
  fi
  session_id=$(find ../pids -mindepth 1 -maxdepth 1 -type d | sed 's|../pids/||' | fzf --prompt="选择要停止的 session: ")
  if [ -z "$session_id" ]; then
    echo "未选择 session，退出"
    exit 0
  fi

fi

for ((i = 0; i < NUM_WORKERS; i++)); do
    WORKER_ID="worker-$i"
    HOSTNAME=`./generate_host_name.sh $i`
    PID_FILE="../pids/${session_id}/${WORKER_ID}.pid"

    if [[ -f "$PID_FILE" ]]; then
        REMOTE_PID=$(cat "$PID_FILE")
        echo "Killing remote worker $WORKER_ID on $HOSTNAME with PID $REMOTE_PID"
        kill_command="kill -9 $REMOTE_PID && rm -f ${REMOTE_DIR}/pids/${WORKER_ID}.pid"
        # echo $kill_command
        # ssh -n "${target_user}@${HOSTNAME}" $kill_command
        ssh -n "${target_user}@${HOSTNAME}" "killall worker"
        # TODO: may need to wait for the process to be really killed
        if [ $? -eq 0 ]; then
            echo "$WORKER_ID on $HOSTNAME killed successfully"
        else
            echo "Failed to kill $WORKER_ID on $HOSTNAME or worker already stopped"
        fi
        rm -f "$PID_FILE"
    else
        echo "No PID file found for $WORKER_ID"
    fi
done

rm -rf "../pids/$session_id"

echo "All remote workers killed"

