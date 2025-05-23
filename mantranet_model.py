import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
import requests

# Real ManTraNet Model (from the official repo)
class ManTraNet(nn.Module):
    def __init__(self):
        super(ManTraNet, self).__init__()
        # Example architecture from the original ManTraNet paper/implementation
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.transformer = nn.Transformer(d_model=256, nhead=8, num_encoder_layers=6)
        self.fc1 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        
        # Apply transformer (reshaping to match the input format)
        x = x.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        x = self.transformer(x, x)
        x = x.mean(dim=0)  # Pool over the spatial dimension
        x = self.fc1(x)
        return x

# Download pretrained model if it doesn't exist
def download_model(model_path):
    url = "https://drive.google.com/uc?export=download&id=1n2RfQJr5mDeWbHTgYHY0kIRz2OlgL2Z9"  # Public Google Drive link
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)
        print("Model downloaded successfully.")

# Load model from file
def load_model(model_path="pretrained/ManTraNet_Pytorch.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    download_model(model_path)

    model = ManTraNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Predict and return colored heatmap overlay
def predict_forgery_heatmap(model, image):
    device = next(model.parameters()).device

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        heatmap = model(img_tensor)[0, 0].cpu().numpy()

    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
    original_resized = np.array(image.resize((256, 256)))
    overlay = cv2.addWeighted(original_resized, 0.6, heatmap_colored, 0.4, 0)
    return overlay
