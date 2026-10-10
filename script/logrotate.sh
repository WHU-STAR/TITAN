#!/bin/bash

CONFIG_FILE="../config/logrotate/logrotate.conf"
STATUS_FILE="../log/logrotate.status"
INTERVAL=120

echo "Starting logrotate daemon at $(date)"

while true; do
    echo "Running logrotate at $(date)"
    /usr/sbin/logrotate -vs "$STATUS_FILE" "$CONFIG_FILE" 2>&1
    sleep "$INTERVAL"
done
