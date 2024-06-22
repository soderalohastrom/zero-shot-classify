#!/bin/bash

# Change to the directory containing the download_model.py script
cd /app/model_cache

# Check if the model cache directory exists
if [ ! -d "/model_cache" ]; then
    echo "Creating model cache directory..."
    mkdir -p /model_cache
fi

# Run the download_model.py script
python download_model.py