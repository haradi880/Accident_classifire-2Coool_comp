# Traffic Incident Analysis System — Competition Documentation

## Competition Overview

**2Coool Kaggle Competition**
**Timeline:** August 24, 2025 — September 29, 2025
**Team:** HaradiBots
**Developer:** Aditya (HaradiBots)



# <i class="fa-solid fa-folder-tree"></i> Project Structure

### Core Components

```
traffic-analysis-system/
├── Model Training/
│   └── Accident_model_training.py
├── Model Testing/
│   └── Testing_of_Accident_pth.py
├── Object Detection & Tracking/
│   └── Yolo_model_Counter.py
├── Main Competition Code/
│   └── Both_mix_code_submission.py
└── Model Weights/
    └── accident_model.pth
```



# <i class="fa-solid fa-layer-group"></i> System Architecture

## 1. Accident Classification Model

**File:** `Accident_model_training.py`

### Purpose

* Trains a deep learning model for traffic-incident classification
* Outputs model weights: `accident_model.pth`

### Model Classes

```python
{
    0: "No Incident",
    1: "Near Collision",
    2: "Collision"
}
```

### Hardware Recommendations

* GPU: NVIDIA T4 / P100 or equivalent
* VRAM: Minimum 8GB

### Pretrained Model (Optional)

```bash
kaggle models download haradibots/1st-model_accident_classifire/PyTorch/default/1
```



## 2. Model Testing & Validation

**File:** `Testing_of_Accident_pth.py`

### Features

* Loads trained accident-classification model
* Performs validation on test datasets
* Returns predicted class and confidence score

### Example Output

```json
{
    "prediction": 1,
    "class_name": "Near Collision",
    "confidence": 0.87
}
```



## 3. YOLO Object Detection & Tracking

**File:** `Yolo_model_Counter.py`

### Capabilities

#### Object Detection

* Uses YOLOv8m model
* Processes 16–24 frames per video segment
* Detects cars, buses, pedestrians, two-wheelers, animals, etc.

#### Object Tracking

* Tracks centroids and movement direction
* Generates movement narratives
* Supports directional analysis (up, down, left, right)

### Output Formats

#### Object Counts

```python
Final Video Summary:
car: 5
person: 3
bus: 1
dog: 1
```

#### Movement Narrative

```python
Traffic Narrative:
Ego car is driving in traffic.
Detected 5 car(s), with movements: 3 downwards, 2 towards the right.
Detected 3 person(s), with movements: 2 upwards, 1 leftwards.
Detected 1 bus.
Detected 1 dog.
```

#### Visual Output

* Produces `output.mp4` with bounding boxes and tracking lines
* Example video shown in documentation (replace with your own link)



## 4. Main Competition Pipeline

**File:** `Both_mix_code_submission.py`
**Note:** This file contains proprietary competition logic.

### Output Format

| Column                         | Description                 | Example                                    |
| ------------------------------ | --------------------------- | ------------------------------------------ |
| video                          | Video ID                    | 558                                        |
| Incident window start frame    | First frame of the incident | 390                                        |
| Incident Detection             | Incident type               | Near Collision                             |
| Crash Severity                 | Severity scale              | Other cars collided but ego-car is safe    |
| Ego-car involved               | Involvement flag            | 9                                          |
| Label                          | Incident classification     | multi-vehicle collision (ego not involved) |
| Number of Bicyclists/Scooters  | Count                       | 0                                          |
| Number of animals involved     | Count                       | 2                                          |
| Number of pedestrians involved | Count                       | 399                                        |
| Number of vehicles involved    | Count                       | 6                                          |
| Caption Before Incident        | Scene description           | Ego-car is driving in heavy traffic.       |
| Reason of Incident             | Cause                       | Other vehicles collided near ego-car.      |

### Sample CSV Output

```csv
video,Incident window start frame,Incident Detection,Crash Severity,Ego-car involved,Label,Number of Bicyclists/Scooters,Number of animals involved,Number of pedestrians involved,Number of vehicles involved (excluding ego-car),Caption Before Incident,Reason of Incident
558,390,Near Collision,Other cars collided but ego-car is safe,9,multi-vehicle collision (ego not involved),0,2,399,6,Ego-car is driving in heavy traffic.,Other vehicles collided near ego-car.
```


# <i class="fa-solid fa-code"></i> Implementation Workflow

### Step 1 — Train Model

```bash
python Accident_model_training.py
```

### Step 2 — Validate Model

```bash
python Testing_of_Accident_pth.py
```

### Step 3 — Object Analysis

```bash
python Yolo_model_Counter.py --input video.mp4
```

### Step 4 — Generate Final Competition CSV

```bash
python Both_mix_code_submission.py
```



# <i class="fa-solid fa-microchip"></i> Technical Specifications

### Hardware

* GPU: NVIDIA T4 / P100 or stronger
* RAM: 16GB minimum
* Storage: 10GB recommended

### Dependencies

* Python 3.8+
* PyTorch
* OpenCV
* Ultralytics YOLO
* NumPy, Pandas



# <i class="fa-solid fa-chart-line"></i> Performance Metrics

### Accident Classification

* Accuracy: above 85 percent on test datasets
* Real-time inference capability

### Object Detection

* Model: YOLOv8m
* Frame processing rate: 16–24 FPS
* Supports multi-object tracking

# <i class="fa-solid fa-lock"></i> Proprietary Notice

The file `Both_mix_code_submission.py` contains competition-critical algorithms and remains private.
This represents the main intellectual property behind the final solution.


# <i class="fa-solid fa-user"></i> Contact

**Developer:** Aditya (HaradiBots)
**Website:** [https://haradibots.onrender.com](https://haradibots.onrender.com)
**Instagram:** [https://instagram.com/llaka2937](https://instagram.com/llaka2937)
**WhatsApp:** +91 78872 85338

<div align="center">

# HaradiBots

AI-Driven Traffic Safety and Analytics
Designed and built for the 2Coool Kaggle Competition

</div>
