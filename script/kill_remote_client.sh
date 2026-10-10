#!/bin/bash
ssh -n "worker@vsys-client" "kill -9 \`pgrep -f './client start'\`"
