import modal
from modal import Image

stub = modal.Stub("mistral-fine-tuning-deepeval")

fine_tune_img = Image.debian_slim(python_version="3.11.7").pip_install(
    "protobuf==4.25.1",
    "datasets==2.17.1",
    "transformers==4.38.1",
    "trl==0.7.11",
    "deepeval==0.20.74",
    "peft==0.8.2",
    "bitsandbytes",
    "accelerate",
)
