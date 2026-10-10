#!/bin/bash

START=${1:-50050}
END=${2:-50100}

for PORT in $(seq $START $END); do
    if ss -ltn | grep -q ":$PORT "; then
        echo "Port $PORT is LISTENING"
    fi
done
