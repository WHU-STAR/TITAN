#!/usr/bin/env bash
cd ../data
if [ -n "$1" ]; then
  selected="$1"
else
  selected=$(find . -name "*worker_data*" | fzf)
fi
if [ -z "$selected" ]; then
  echo "No selection made. Exiting."
  exit 1
fi
if [ -L workerdata ]; then
  rm workerdata
fi
ln -s $selected workerdata

if [ ! -e workerdata ]; then
  echo "Error: workerdata symlink is invalid."
  exit 1
fi
echo "Successfully changed data to $selected"
