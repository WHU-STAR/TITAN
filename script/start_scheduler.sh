#!/bin/bash

#############################
### Make sure pwd is on #####
####### ./bin        ########
#############################

# Build scheduler
pushd .. && make scheduler
popd

echo "make done"

dataTarget=$(readlink ../data/workerdata)

BINARY_DIR="../bin/"
cd $BINARY_DIR

if [ -z "$num_workers" ]; then
  num_workers=16
fi

BINARY_PATH="./scheduler"

# Check if binary exists
if [ ! -f "$BINARY_PATH" ]; then
  echo "Error: Scheduler binary not found at $BINARY_PATH"
  echo "Please build the project first"
  exit 1
fi

# Ensure binary is executable
chmod +x "$BINARY_PATH"

# Create log directory if it doesn't exist
mkdir -p ../log

# Start the scheduler
echo "Starting scheduler..."

if [[ -z "$dataTarget" ]]; then
  echo "Error: workerdata is not linked to any data directory"
  exit 1
fi

naiveArg=""

if [[ $dataTarget == *"naive"* ]]; then
  naiveArg="--strategy=naive"
else
  naiveArg=""
fi

echo "num_workers: $num_workers"


"$BINARY_PATH" "$naiveArg" "-w$num_workers"
