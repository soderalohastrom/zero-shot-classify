import os
from transformers import AutoTokenizer, AutoModel

model_name = "MoritzLaurer/bge-m3-zeroshot-v2.0"
# model_name = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
# model_name = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
# model_name = "zeroshot/gte-small-quant"
# model_name = "DAMO-NLP-SG/zero-shot-classify-SSTuning-base"

cache_dir = "/model_cache"  # Use the absolute path

print("Downloading and caching the model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
print("Model and tokenizer downloaded and cached successfully.")
