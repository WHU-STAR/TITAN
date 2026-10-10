#!/bin/bash

# Kill all workers using their pid files
# 选择 session 子目录
if [ $# -eq 0 ]; then
    if ! command -v fzf >/dev/null 2>&1; then
        echo "fzf 未安装，请先安装 fzf"
        exit 1
    fi
    session_dir=$(find ../pids -mindepth 1 -maxdepth 1 -type d | sed 's|../pids/||' | fzf --prompt="选择要停止的 session: ")
    if [ -z "$session_dir" ]; then
        echo "未选择 session，退出"
        exit 0
    fi
else
    session_dir="$1"
    if [ ! -d "../pids/$session_dir" ]; then
        echo "指定的 session 不存在: $session_dir"
        exit 1
    fi
fi

for pid_file in ../pids/$session_dir/worker-*.pid; do
    if [ -f "$pid_file" ]; then
        worker_id=$(basename "$pid_file" .pid)
        echo "Stopping $worker_id..."
        if kill $(cat "$pid_file") 2>/dev/null; then
            wait $(cat "$pid_file") 2>/dev/null
            rm "$pid_file"
            echo "$worker_id stopped"
        else
            echo "Failed to stop $worker_id or process already stopped"
            rm "$pid_file"
        fi
    fi
done

# 删除 session 目录
rm -rf ../pids/$session_dir

echo "All workers stopped"
