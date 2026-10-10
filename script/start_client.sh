#!/bin/bash

# Build client

runName="default_run"

cwd=$(pwd)
# Get data paths
cd ..
pwd
make client
cd "$cwd"

paths=($(bash ../data/workerdata/get_data_path.sh))
query_set_path=$(realpath "${paths[0]}")
ground_truth_path=$(realpath "${paths[1]}")
while getopts "r:" opt; do
    case $opt in
    r)
        runName="$OPTARG"
        ;;
    \?)
        echo "无效参数: -$OPTARG" >&2
        ;;
    esac
done

# Build the client binary if it doesn't exist
go build -o ../bin/client ../cmd/client

# Start the client with default parameters
cd ../bin
./client start \
    --run_name="$runName" \
    --query_set_path="$query_set_path" \
    --ground_truth_path="$ground_truth_path"
