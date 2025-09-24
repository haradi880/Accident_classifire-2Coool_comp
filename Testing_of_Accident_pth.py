"""
make sure you have accident_model.pth in same folder
if you dont have run the Accident_model_treaning.py file first
then change the path of the classifier model path MODEL_PATH = "accident_model.pth"  # make sure this is in same folder 
and the video path video_path = "Video.mp4"   # replace with any video path
this code will predict the video is accident or no incident

This code is for testing the accident detection model.

"""


import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

# ------------------------
# Config
# ------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 16
MODEL_PATH = "accident_model.pth"  # make sure this is in same folder

# ------------------------
# Label map
# ------------------------
label_map = {0: "No Incident", 1: "Near Collision", 2: "Collision"}
# ------------------------
# Model Definition
# ------------------------
class VideoClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.fc = nn.Identity()
        self.base = base
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):  # x: [B,T,C,H,W]
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.base(x)
        feats = feats.view(B, T, 512).mean(1)
        out = self.fc(feats)
        return out

# ------------------------
# Frame preprocessing
# ------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

def sample_frames(frames, num_frames=NUM_FRAMES):
    if len(frames) == 0:
        return [torch.zeros(3,224,224)]*num_frames
    if len(frames) < num_frames:
        frames += [frames[-1]]*(num_frames - len(frames))
    idxs = torch.linspace(0, len(frames)-1, steps=num_frames, dtype=torch.long)
    return [frames[i] for i in idxs]

# ------------------------
# Load Model
# ------------------------
model = VideoClassifier().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ------------------------
# Predict Function
# ------------------------
def predict(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    frames = sample_frames(frames)
    frames = [transform(f) for f in frames]
    video_tensor = torch.stack(frames).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(video_tensor)
        pred = out.argmax(1).item()

    return label_map[pred]

# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    video_path = "Video/525.mp4"   # replace with any video path
    result = predict(video_path)
    print(f"Prediction: {result}")
