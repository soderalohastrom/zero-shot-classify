import os
from transformers import AutoTokenizer, AutoModel

# model_name = "MoritzLaurer/bge-m3-zeroshot-v2.0"
model_name = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
# model_name = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
# model_name = "DAMO-NLP-SG/zero-shot-classify-SSTuning-base"
# model_name = "facebook/bart-large-mnli"

cache_dir = "/model_cache"  # Use the absolute path

def model_exists(model_name, cache_dir):
    # Check if both model and tokenizer files exist
    model_path = os.path.join(cache_dir, "models--" + model_name.replace("/", "--"))
    return os.path.exists(model_path)

if model_exists(model_name, cache_dir):
    print(f"Model {model_name} already exists in cache. Skipping download.")
else:
    print(f"Downloading and caching the model and tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    print("Model and tokenizer downloaded and cached successfully.")

print("Model is ready for use.")