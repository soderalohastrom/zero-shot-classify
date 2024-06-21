#!/bin/bash

# Install the required dependencies
pip install -r requirements.txt

# Ensure the mount directory exists
mkdir -p /model_cache

# Run the download_model.sh script
bash /model_cache/download_model.sh
