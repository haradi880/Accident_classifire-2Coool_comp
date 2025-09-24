"""
* The data *

you well need a data to trean the accident model.
the data you can fine on kaggle it was opensource data.
the link is below:
https://www.kaggle.com/datasets/haradibots/accident-data-sets-cnn-vary-small

* The model *
    ACCIDENT_MODEL_PATH = "accident_model.pth" 
    if you dont wanna train the model you can download the model from here;
    https://www.kaggle.com/models/haradibots/1st-model_accident_classifire/PyTorch/default/1
    and put it in the same folder where this code is.

it was the data of the traffic accident.it have 2 directories:
1. Accident
2. No incident

Each directory have 60 dashcam video clips.

you have to do is download the data and then trean the model whith any gpu like T4 or P100.


* What the model do? *

i have trean the model to detect the accident from the dashcam video clips.
it take the video as input and then it will predict the accident or no incident.
Video clips in input then output = 0 or 1 / accident or no incident.


The model is a binary classification model.

Code is below:
"""


#--------- First we have to import our libraries ---------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import os, cv2, random
import numpy as np

# --------- then we have to do some configrations ---------
ACCIDENT_PATH = "/kaggle/input/accident-data-sets-cnn-vary-small/Accident"
NO_INCIDENT_PATH = "/kaggle/input/accident-data-sets-cnn-vary-small/NO incident"
NUM_FRAMES = 16   
IMG_SIZE = 112
BATCH_SIZE = 4
EPOCHS = 12
LR = 1e-4


# --------- and next you we have to load and some prossesing of the data ---------
class AccidentVideoDataset(Dataset):
    def __init__(self, accident_dir, no_incident_dir, transform=None, num_frames=16):
        self.samples = []
        self.transform = transform
        self.num_frames = num_frames

        for f in os.listdir(accident_dir):
            self.samples.append((os.path.join(accident_dir, f), 1))
        for f in os.listdir(no_incident_dir):
            self.samples.append((os.path.join(no_incident_dir, f), 0))

    def __len__(self):
        return len(self.samples)

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frames.append(frame)
        cap.release()

        # Sample fixed number of frames
        if len(frames) >= self.num_frames:
            idxs = np.linspace(0, len(frames)-1, self.num_frames).astype(int)
            frames = [frames[i] for i in idxs]
        else:  # pad with last frame
            while len(frames) < self.num_frames:
                frames.append(frames[-1])

        frames = np.stack(frames)  # (T, H, W, C)
        frames = torch.from_numpy(frames).permute(3,0,1,2)  # (C,T,H,W)
        frames = frames.float() / 255.0
        return frames

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        video = self._load_video(path)
        if self.transform:
            video = self.transform(video)
        return video, label

# --------- choose the model we have to use  ---------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torchvision.models.video.r3d_18(weights="KINETICS400_V1")
model.fc = nn.Linear(model.fc.in_features, 2)  # binary classification
model = model.to(device)

# --------- then the treaning of the model ---------
dataset = AccidentVideoDataset(ACCIDENT_PATH, NO_INCIDENT_PATH, num_frames=NUM_FRAMES)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for videos, labels in dataloader:
        videos, labels = videos.to(device), torch.tensor(labels).to(device)
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

# --------- finally we have to save the model ---------
torch.save(model.state_dict(), "accident_classifier.pth")
print("✅ Model saved as accident_classifier.pth")
