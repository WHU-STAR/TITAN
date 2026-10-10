#!/bin/bash


fileExists=false
if [ -f ../runconfig.yaml ]; then
    fileExists=true
fi

generateFile=false

# if fileExists
if [ "$fileExists" = true ]; then
    echo "runconfig.yaml file already exists."
    read -p "Do you wish to overwrite it? (y/n): " choice
    if [ "$choice" = "y" ]; then
        generateFile=true
    else
        echo "Keeping the existing runconfig.yaml file."
        echo "Bye bye."
    fi
else
    generateFile=true
fi


if [ "$generateFile" = true ]; then
    echo "Creating runconfig.yaml file..."
    cp ../config/runconfig_default.yaml ../runconfig.yaml
fi
