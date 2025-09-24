"""
This project combines YOLOv8 object detection with a custom accident classifier
to automatically analyze driving videos and detect traffic incidents.

How it works:
1. **Object Tracking** – YOLOv8 detects and tracks vehicles, pedestrians,
   bicycles, and animals in each video. Unique IDs are assigned so objects
   aren’t double-counted.
2. **Accident Classification** – A ResNet-based video classifier looks at
   sampled frames from each clip and determines whether there was:
      - No incident
      - A near collision
      - A collision
3. **Dynamic Incident Detection** – If an incident is predicted, the system
   estimates the most likely frame where it occurs by measuring how close
   objects move relative to each other.
4. **Traffic Context** – Before the incident, traffic density around the ego
   vehicle is described (empty road, light, moderate, or heavy traffic).
5. **Results Export** – For each video, the pipeline outputs a structured CSV
   file containing:
      - The video ID
      - Incident frame
      - Incident type and severity
      - Ego-vehicle involvement
      - Object counts (vehicles, pedestrians, animals, bicycles)
      - Narrative captions before and after the incident

In short, the pipeline produces both *quantitative data* (counts, labels, frames)
and a *narrative description* of the scene, making it useful for traffic analysis,
autonomous vehicle research, and safety event detection.
"""


""" 
the model 
    ACCIDENT_MODEL_PATH = "accident_model.pth" 
    if you dont wanna train the model you can download the model from here;
    https://www.kaggle.com/models/haradibots/1st-model_accident_classifire/PyTorch/default/1
    and put it in the same folder where this code is.

the data 
    you can download the fdata from here;
    https://www.kaggle.com/datasets/haradibots/video-clips-of-accidents-for-traning-r3d-model
    this is a big data 661 video clips in total 11+ GB of data.

"""

import os
import cv2
import csv
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from collections import defaultdict
from tqdm import tqdm

# ------------------------------
# Config
# ------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES_ACCIDENT = 24
ACCIDENT_MODEL_PATH = "accident_model.pth"
VIDEO_DIR = "Video"  # directory containing input videos
OUTPUT_CSV = "submission.csv"

ALLOWED_CLASSES = {"car", "truck", "bus", "person", "dog", "cat", "cow", "bicycle"}

# ------------------------------
# Label maps
# ------------------------------
incident_label_map = {0: "No incident", 1: "Near Collision", 2: "Collision"}
crash_severity_map = {
    0: "0. No Crash",
    1: "3. Possible crash, low severity",
    2: "4. Other cars collided with person/car/object but ego-car is ok"
}
incident_type_map = {
    0: "no incident",
    1: "near miss with another vehicle",
    2: "vehicle drives into another vehicle"
}

# ------------------------------
# Accident Classifier
# ------------------------------
class VideoClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.fc = nn.Identity()
        self.base = base
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.base(x)
        feats = feats.view(B, T, 512).mean(1)
        out = self.fc(feats)
        return out

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

def sample_frames(frames, num_frames=NUM_FRAMES_ACCIDENT):
    if len(frames) == 0:
        return [torch.zeros(3,224,224)]*num_frames
    if len(frames) < num_frames:
        frames += [frames[-1]]*(num_frames - len(frames))
    idxs = torch.linspace(0, len(frames)-1, steps=num_frames, dtype=torch.long)
    return [frames[i] for i in idxs]

# Load accident model
accident_model = VideoClassifier().to(DEVICE)
accident_model.load_state_dict(torch.load(ACCIDENT_MODEL_PATH, map_location=DEVICE))
accident_model.eval()

def classify_incident(frames):
    frames = sample_frames(frames)
    frames = [transform(f) for f in frames]
    video_tensor = torch.stack(frames).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = accident_model(video_tensor)
        pred = out.argmax(1).item()
    return pred

# ------------------------------
# Dynamic Incident Frame Finder
# ------------------------------
def find_incident_frame(yolo_results, distance_threshold=50):
    incident_frame = 0
    max_near_objects = 0

    for i, result in enumerate(yolo_results):
        if not result.boxes:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        # Compute all pairwise distances
        count = 0
        for j in range(len(boxes)):
            for k in range(j+1, len(boxes)):
                x1, y1, x2, y2 = boxes[j]
                x1b, y1b, x2b, y2b = boxes[k]
                cx1, cy1 = (x1+x2)/2, (y1+y2)/2
                cx2, cy2 = (x1b+x2b)/2, (y1b+y2b)/2
                dist = ((cx1-cx2)**2 + (cy1-cy2)**2)**0.5
                if dist < distance_threshold:
                    count += 1
        # Count objects suddenly appearing or moving close
        if count > max_near_objects:
            max_near_objects = count
            incident_frame = i

    # If no incident detected, fallback to middle frame
    if max_near_objects == 0:
        incident_frame = len(yolo_results)//2
    return incident_frame


# ------------------------------
# YOLO Model
# ------------------------------
yolo = YOLO("yolov8m.pt").to(DEVICE)

# ------------------------------
# Traffic Caption Generator
# ------------------------------
def traffic_caption(counts):
    if counts["vehicles"] >= 10:
        return "Ego-car is driving in heavy traffic."
    elif 5 <= counts["vehicles"] <= 9:
        return "Ego-car is driving with moderate traffic."
    elif 1 <= counts["vehicles"] <= 4:
        return "Ego-car is driving with light traffic."
    else:
        return "Ego-car is driving on an empty road."

# ------------------------------
# Video Processing
# ------------------------------
videos = [v for v in os.listdir(VIDEO_DIR) if v.endswith(".mp4")]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "video","Incident window start frame","Incident Detection","Crash Severity",
        "Ego-car involved","Label","Number of Bicyclists/Scooters",
        "Number of animals involved","Number of pedestrians involved",
        "Number of vehicles involved (excluding ego-car)",
        "Caption Before Incident","Reason of Incident"
    ])

    for vid in tqdm(videos, desc="Processing videos"):
        video_path = os.path.join(VIDEO_DIR, vid)
        unique_objects = defaultdict(set)
        all_frames = []
        all_results = []

        # YOLO tracking
        for result in yolo.track(source=video_path, stream=True, tracker="bytetrack.yaml", conf=0.6, verbose=False, show=False):
            frame = result.orig_img
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame_rgb)
            all_results.append(result)

            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else None
                name = yolo.names[cls]

                if conf < 0.6 or name not in ALLOWED_CLASSES or track_id is None:
                    continue
                unique_objects[name].add(track_id)

        # Accident classification
        pred = classify_incident(all_frames)

        # Dynamic incident frame only if incident detected
        incident_window = find_incident_frame(all_results) if pred != 0 else 0

        # Aggregate counts
        vehicle_count = len(unique_objects["car"] | unique_objects["truck"] | unique_objects["bus"])
        vehicle_count = max(vehicle_count, 1)  # Include ego-car
        counts = {
            "vehicles": vehicle_count,
            "pedestrians": len(unique_objects["person"]),
            "animals": len(unique_objects["dog"] | unique_objects["cat"] | unique_objects["cow"]),
            "bicycles": len(unique_objects["bicycle"])
        }

        # Ego involvement
        ego_involved = 1 if pred == 2 else 0

        # Dynamic captions
        before_caption = traffic_caption(counts)
        after_caption_map = {
            0: "No accident occurred.",
            1: "Other vehicles collided near ego-car.",
            2: "A collision involving another vehicle was detected."
        }
        after_caption = after_caption_map[pred]

        # Write CSV
        writer.writerow([
            vid.replace(".mp4",""), incident_window,
            incident_label_map[pred],
            crash_severity_map[pred],
            ego_involved,
            incident_type_map[pred],
            counts["bicycles"], counts["animals"], counts["pedestrians"], counts["vehicles"],
            before_caption, after_caption
        ])

print(f"✅ Processing done. CSV saved at {OUTPUT_CSV}")