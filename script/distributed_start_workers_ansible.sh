#!/bin/bash
set -euo pipefail

# 构建二进制
make worker

# 读取/准备运行时所需信息
dataTarget=$(readlink ./data/workerdata || true)
if [[ -z "${dataTarget:-}" ]]; then
    echo "Error: workerdata is not linked to any data directory"
    exit 1
fi

# 默认变量（可被环境覆盖）
export TITAN_CONFIG_BUILDER=${TITAN_CONFIG_BUILDER:-default_distributed}
export BASE_PORT=${BASE_PORT:-50050}

# 使用 Ansible 停止旧实例（幂等）
ANSIBLE_INVENTORY="../ansible/inventory.ini"
ANSIBLE_PLAY_START="../ansible/start_workers.yml"
ANSIBLE_PLAY_STOP="../ansible/stop_workers.yml"

if [[ -f "$ANSIBLE_PLAY_STOP" ]]; then
  ansible-playbook -i "$ANSIBLE_INVENTORY" "$ANSIBLE_PLAY_STOP" --forks 50 || true
fi

# 使用 Ansible 启动分布式 workers（systemd --user）
ansible-playbook -i "$ANSIBLE_INVENTORY" "$ANSIBLE_PLAY_START" --forks 50 \
  -e titan_config_builder="$TITAN_CONFIG_BUILDER" \
  -e base_port="$BASE_PORT"

echo "Started distributed workers via Ansible."
