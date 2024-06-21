#!/bin/bash

# Install the required dependencies
pip install -r requirements.txt

# Make the download_model.sh script executable
chmod +x ./model_cache/download_model.sh

# Run the download_model.sh script
./model_cache/download_model.sh
