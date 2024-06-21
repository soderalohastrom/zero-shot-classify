import os
from transformers import AutoTokenizer, AutoModel

model_name = "MoritzLaurer/bge-m3-zeroshot-v2.0"
cache_dir = "/model_cache"  # Use the absolute path

print("Downloading and caching the model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
print("Model and tokenizer downloaded and cached successfully.")
