"""

This script uses a YOLOv8 model to detect and track objects in a video, focusing on
vehicles, pedestrians, and a few animals. It assigns each object a unique ID, keeps
track of its movement across frames, and determines whether it is approaching,
moving away, or shifting left/right relative to an ego vehicle positioned at the
bottom center of the frame.

As the video is processed:
- Objects are annotated with bounding boxes, IDs, and movement directions.
- A running count of unique detections is maintained.
- Movement trajectories are analyzed to produce a narrative summary.

The output includes:
1. An annotated video with all detections and directions overlaid.
2. A final report summarizing how many objects of each class were observed and
   how they moved in relation to the ego vehicle.

This tool can be extended for traffic analysis, autonomous driving research, or
general scene understanding where movement trends are important.

First install the ultralytics package if you haven't already:
!pip install ultralytics

"""


import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
import torch

# ------------------------------
# Config
# ------------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8m.pt").to(device)

ALLOWED_CLASSES = {"car", "truck", "bus", "person", "dog", "cat", "cow"}

# Tracking
unique_objects = defaultdict(set)
trajectories = defaultdict(lambda: deque(maxlen=5))
object_directions = defaultdict(dict)

video_path = "Video/525.mp4"
cap = cv2.VideoCapture(video_path)
width, height = int(cap.get(3)), int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

ego_x, ego_y = width // 2, height   # ego car reference point

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

# ------------------------------
# Process video
# ------------------------------
for result in model.track(
    source=video_path,
    stream=True,
    tracker="bytetrack.yaml",
    conf=0.6,   # stricter confidence
    verbose=False,
    show=False
):

    frame = result.orig_img  # get the current frame

    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        track_id = int(box.id[0]) if box.id is not None else None
        name = model.names[cls]

        if conf < 0.6 or name not in ALLOWED_CLASSES or track_id is None:
            continue

        unique_objects[name].add(track_id)

        # Center point of detection
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Save trajectory
        trajectories[track_id].append((cx, cy))
        direction = ""

        if len(trajectories[track_id]) >= 2:
            (px, py) = trajectories[track_id][0]
            (lx, ly) = trajectories[track_id][-1]
            dx, dy = lx - px, ly - py

            # Direction analysis
            if abs(dx) > abs(dy):
                direction = "→ right" if dx > 0 else "← left"
            else:
                if dy > 0:
                    direction = "↓ approaching"
                else:
                    direction = "↑ moving away"

            object_directions[name][track_id] = direction

        # ----------------- Draw on frame -----------------
        color = (0, 255, 0) if name in ["car", "truck", "bus"] else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} #{track_id} {direction}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # Write annotated frame to output video
    out.write(frame)

cap.release()
out.release()

# ------------------------------
# Final Summary
# ------------------------------
print("\nFinal Video Summary:")
for name, ids in unique_objects.items():
    print(f"{name}: {len(ids)}")

# Aggregate movement summary
summary_directions = defaultdict(lambda: defaultdict(int))
for name, objs in object_directions.items():
    for _, direction in objs.items():
        summary_directions[name][direction] += 1

print("\nTraffic Narrative:")
print("Ego car is driving in traffic.")

for name, count in unique_objects.items():
    n = len(count)
    if n == 0:
        continue

    directions = summary_directions[name]
    if directions:
        dir_text = ", ".join([f"{v} {k}" for k, v in directions.items()])
        print(f"Detected {n} {name}(s), with movements: {dir_text}.")
    else:
        print(f"Detected {n} {name}(s).")
