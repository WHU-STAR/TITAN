#!/bin/bash

# Build client

runName="default_run"
# Get data paths
pushd ..
make client
popd
paths=($(bash ../data/workerdata/get_data_path.sh))
query_set_path=$(realpath "${paths[0]}")
ground_truth_path=$(realpath "${paths[1]}")

while getopts "r:u:" opt; do
  case $opt in
  r)
    runName="$OPTARG"
    ;;
  u)
    target_user="$OPTARG"
    ;;

  \?)
    echo "无效参数: -$OPTARG" >&2
    ;;
  esac
done

if [[ -z "$TITAN_CONFIG_BUILDER" ]]; then
  TITAN_CONFIG_BUILDER=default_distributed
fi

if [[ -z "$target_user" ]]; then
  target_user=worker
fi

# Build the client binary if it doesn't exist
go build -o ../bin/client ../cmd/client

CLIENT_IP="vsys-client"
REMOTE_DIR="/data/titan/bin" # worker 程序所在路径
CLIENT_BIN="client"

# Start the client with default parameters

# cd bin
# ./client start --top_k=10 --delay=0 --run_name="$runName" --query_set_path="$query_set_path" --ground_truth_path="$ground_truth_path"
echo "starting client..."
CMD="cd $REMOTE_DIR && TITAN_CONFIG_BUILDER=$TITAN_CONFIG_BUILDER ./$CLIENT_BIN start --run_name=$runName --query_set_path=$query_set_path --ground_truth_path=$ground_truth_path"

ssh -n "$target_user@$CLIENT_IP" "$CMD"

exit 0
