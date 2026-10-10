#!/bin/env bash

# 用于在tmux会话中终止runner的脚本，包含杀死workers和关闭tmux窗口

PROMPT=false

while getopts "p" opt; do
  case $opt in
  p)
    PROMPT=true
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

./kill_workers.sh $SESSION_ID

tmux kill-window -t "runner-$SESSION_ID"
