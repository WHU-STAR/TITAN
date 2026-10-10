#!/bin/bash
cd "$(dirname "$0")/.."

fileExists=false
if [ -f runconfig.yaml ]; then
    fileExists=true
fi

generateFile=false

if [ "$fileExists" = true ]; then
    echo "runconfig.yaml already exists."
    read -p "Overwrite it? (y/n): " choice
    if [ "$choice" = "y" ]; then
        generateFile=true
    else
        echo "Keeping the existing runconfig.yaml."
    fi
else
    generateFile=true
fi

if [ "$generateFile" = true ]; then
    echo "Creating runconfig.yaml..."
    cp config/runconfig_default.yaml runconfig.yaml
fi
