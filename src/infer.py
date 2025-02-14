import json
import requests
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

# Download ImageNet labels
imagenet_labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
imagenet_labels = requests.get(imagenet_labels_url).json()

# Load pre-trained model and feature extractor
model_name = "microsoft/resnet-50"
model = AutoModelForImageClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# Load and preprocess image
image_path = "data/70980768_2428176940569218_2896125760844595200_o.jpg"
image = Image.open(image_path)
inputs = feature_extractor(images=image, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax(-1).item()

# Get the predicted class label from ImageNet
predicted_class_label = imagenet_labels[str(predicted_class_id)][1]

print(f"Predicted class: {predicted_class_label}")
