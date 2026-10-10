#!/bin/bash
VCLUSTER_CONFIG_BUILDER=default_distributed
runName="default_run"

session_id=0
while [ -d "../pids/$session_id" ]; do
  session_id=$((session_id + 1))
done

target_user=$USER
num_workers=16
BASE_PORT=50050

PROMPTSTOP="-p"
do_wait=1 # 0表示等待窗口关闭，1表示不等待

# 使用 getopts 解析选项
while getopts "r:n:u:c:w:p:s" opt; do
  case $opt in
  r)
    VCLUSTER_CONFIG_BUILDER="$OPTARG"
    ;;
  n)
    runName="$OPTARG"
    ;;
  u)
    target_user="$OPTARG"
    ;;
  c)
    client_user="$OPTARG"
    ;;
  w)
    num_workers="$OPTARG"
    ;;

    
  p)
    BASE_PORT="$OPTARG"
    ;;
  s)
    do_wait=0
    PROMPTSTOP=""
    ;;
  \?)
    echo "无效参数: -$OPTARG" >&2
    ;;
  esac
done

if [ -z "$client_user" ]; then
  client_user=$target_user
fi

num_workers=$num_workers \
  PROMPTSTOP=$PROMPTSTOP \
  client_user=$client_user \
  target_user=$target_user \
  session_id=$session_id \
  runName=$runName \
  BASE_PORT=$BASE_PORT \
  VCLUSTER_CONFIG_BUILDER=$VCLUSTER_CONFIG_BUILDER \
  TMUXIFIER_LAYOUT_PATH=$(pwd)/../.tmuxifier \
  tmuxifier load-window tmux_distributed_runner_window_layout



if [ $do_wait -eq 0 ]; then
  echo "等待tmux窗口关闭..."
  window_name="runner"
  while tmux list-windows -F '#W' 2>/dev/null | grep -q "^${window_name}\$"; do
    sleep 1
  done
  echo "tmux 窗口已关闭，实验完成"
fi
