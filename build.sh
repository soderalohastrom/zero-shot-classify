#!/bin/bash

# Install the required dependencies
pip install -r requirements.txt

# Download and cache the model
python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='MoritzLaurer/bge-m3-zeroshot-v2.0')"