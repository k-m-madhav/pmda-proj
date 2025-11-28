from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image
# import requests

# DINOv3 is a Gated model. Need HF token for access
# https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m/discussions/8
# You'll need transformers version 4.56.0 and above for DINOv3
print("Loading DINOv2 model from Hugging Face...")

# Load model and processor
PRETRAINED_MODEL_NAME = "facebook/dinov2-base" # dinov3-vit7b16-pretrain-lvd1689m
image_processor = AutoImageProcessor.from_pretrained(
    PRETRAINED_MODEL_NAME, use_fast=True
)
model = AutoModel.from_pretrained(PRETRAINED_MODEL_NAME)

print("Model loaded successfully!")
print(f"Model type: {type(model)}")
print(f"Model device: {next(model.parameters()).device}")

# Test with a sample image
print("\nTesting with a sample image from the RaiSem19 dataset...")
# URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(URL, stream=True).raw)
rs19_sample_image = Image.open("data/railsem19/jpgs/rs19_val/rs00001.jpg")

# Get original image size
width, height = rs19_sample_image.size

# Display the size
print(f"Original Image size: {width} x {height} pixels") # 1920 x 1080

# Process image
# The processor converts your PIL/numpy image into the format needed by the model
# The "return_tensors" parameter specifies which tensor library format to return
# "pt" is for PyTorch
inputs = image_processor(
    images=rs19_sample_image,
    return_tensors="pt",
    size={"height": 518, "width": 518}
).to(model.device)
print(f"\nInput shape: {inputs['pixel_values'].shape}")
# torch.Size([1, 3, 518, 518])
# [batch_size, channels, width, height]

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Get the features
features = outputs.last_hidden_state
print(f"Output feature shape: {features.shape}") # torch.Size([1, 1370, 768])
# [batch_size, num_tokens, feature_dimensions]
# num_tokens = (518*518) / (14*14) + 1 = 1370, where 14x14 is the patch size and
# 1 is a special classification token addedd by the ViT (Vision Transformer) at the start
# Think of it as dividing the actual image into multiple patches of size 14x14
# Analogy: Each patch is treated like a "word" in NLP transformers
print(f"Feature dimensions: {features.shape[-1]}")

print("\nDINOv2 is working correctly!")
print("Ready for rail track segmentation!")
