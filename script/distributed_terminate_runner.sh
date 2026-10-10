#!/bin/env bash

# 用于在tmux会话中终止runner的脚本，包含杀死workers和关闭tmux窗口

PROMPT=false
TARGET_USER=$USER
NUM_WORKERS=16
while getopts "pu:w:" opt; do
  case $opt in
  p)
    PROMPT=true
    ;;
  u)
    TARGET_USER="$OPTARG"
    ;;
  w)
    NUM_WORKERS="$OPTARG"
    ;;
  \?)
    echo "无效参数: -$OPTARG" >&2
    exit 1
    ;;
  esac
done

if [[ $PROMPT == true ]]; then
  read -p "Terminate the runner? (y/N): " confirm
  confirm=${confirm,,}
  if [[ $confirm != "y" ]]; then
    echo "Termination cancelled."
    exit 0
  fi
fi

if [[ -z "$SESSION_ID" ]]; then
  echo "Error: SESSION_ID is not set."
  exit 1
fi

./kill_remote_workers.sh -s $SESSION_ID -u $TARGET_USER -w $NUM_WORKERS

tmux kill-window -t "runner"
