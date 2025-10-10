# 🚗 Traffic Incident Analysis System - Competition Documentation

## 🏆 Competition Overview

**2Coool Kaggle Competition**  
**Dates**: August 24, 2025 - September 29, 2025  
**Team**: HaradiBots  
**Developer**: Aditya (HaradiBots)

## 📁 Project Structure

### 🎯 Core Components

```
traffic-analysis-system/
├── 🏋️‍♂️ Model Training/
│   └── Accident_model_training.py
├── 🔍 Model Testing/
│   └── Testing_of_Accident_pth.py
├── 🎥 Object Detection & Tracking/
│   └── Yolo_model_Counter.py
├── 🚀 Main Competition Code/
│   └── Both_mix_code_submission.py
└── 📦 Model Weights/
    └── accident_model.pth
```

## 🛠️ System Architecture

### 1. **Accident Classification Model** 🏋️‍♂️

**File**: `Accident_model_training.py`

#### Purpose:
- Train a deep learning model to classify traffic incidents
- Output: `accident_model.pth` weights file

#### Model Classes:
```python
{
    0: "No Incident",
    1: "Near Collision", 
    2: "Collision"
}
```

#### Hardware Requirements:
- **GPU**: T4, P100, or equivalent
- **VRAM**: Minimum 8GB recommended

#### Alternative Model Download:
```bash
# Download pre-trained model from Kaggle
kaggle models download haradibots/1st-model_accident_classifire/PyTorch/default/1
```

---

### 2. **Model Testing & Validation** 🔍

**File**: `Testing_of_Accident_pth.py`

#### Features:
- Load and test the trained accident classification model
- Validate model performance on test datasets
- Output classification results with confidence scores

#### Sample Output:
```json
{
    "prediction": 1,
    "class_name": "Near Collision", 
    "confidence": 0.87
}
```

---

### 3. **YOLO Object Counter & Tracker** 🎥

**File**: `Yolo_model_Counter.py`

#### Capabilities:

#### 🎯 Object Detection:
- Uses YOLOv8m model for object detection
- Processes 16-24 frames per video segment
- Real-time object classification and counting

#### 📊 Movement Tracking:
- Tracks object trajectories and directions
- Generates movement narratives
- Directional analysis (↑↓→←)

#### 📈 Output Types:

**1. Object Counts:**
```python
Final Video Summary:
car: 5
person: 3  
bus: 1
dog: 1
```

**2. Movement Narrative:**
```python
Traffic Narrative:
Ego car is driving in traffic.
Detected 5 car(s), with movements: 3 ↓ approaching, 2 → right.
Detected 3 person(s), with movements: 2 ↑ moving away, 1 ← left.
Detected 1 bus(s).
Detected 1 dog(s).
```

**3. Visual Output:**
- Generates `output.mp4` with tracking overlays
- [Sample Output Video](https://www.dropbox.com/scl/fi/3qxxrugjyrkqj50owvlgf/output.mp4?rlkey=zn8khq90iqkt2q5inb8gfyq0h&st=9yqycqvt&dl=0)

---

### 4. **Main Competition Pipeline** 🚀

**File**: `Both_mix_code_submission.py` *(Proprietary Code)*

#### 🎯 Competition Output Format:

| Column | Description | Example |
|--------|-------------|---------|
| **video** | Video identifier | 558 |
| **Incident window start frame** | Frame where incident begins | 390 |
| **Incident Detection** | Type of incident detected | "Near Collision" |
| **Crash Severity** | Severity rating | "4. Other cars collided but ego-car is ok" |
| **Ego-car involved** | Ego-car involvement flag | 9 |
| **Label** | Incident classification | "multi-vehicle collision (ego not involved)" |
| **Number of Bicyclists/Scooters** | Count of two-wheelers | 0 |
| **Number of animals involved** | Animal count in incident | 2 |
| **Number of pedestrians involved** | Pedestrian count | 399 |
| **Number of vehicles involved** | Other vehicle count | 6 |
| **Caption Before Incident** | Pre-incident scene description | "Ego-car is driving in heavy traffic." |
| **Reason of Incident** | Incident cause analysis | "Other vehicles collided near ego-car." |

#### 📊 Sample Output:
```csv
video,Incident window start frame,Incident Detection,Crash Severity,Ego-car involved,Label,Number of Bicyclists/Scooters,Number of animals involved,Number of pedestrians involved,Number of vehicles involved (excluding ego-car),Caption Before Incident,Reason of Incident
558,390,Near Collision,4. Other cars collided but ego-car is ok,9,multi-vehicle collision (ego not involved),0,2,399,6,Ego-car is driving in heavy traffic.,Other vehicles collided near ego-car.
489,0,No incident,0. No accident,0,no incident,0,0,1,7,Ego-car is driving in heavy traffic.,No accident occurred.
669,0,No incident,0. No accident,0,no incident,0,0,0,2,Ego-car is driving with light traffic.,No accident occurred.
```

## 🚀 Implementation Workflow

### Step 1: Model Training
```bash
python Accident_model_training.py
```
- Output: `accident_model.pth`

### Step 2: Model Validation
```bash
python Testing_of_Accident_pth.py
```
- Validates model performance

### Step 3: Object Analysis
```bash
python Yolo_model_Counter.py --input video.mp4
```
- Generates object counts and movement analysis

### Step 4: Competition Submission
```bash
python Both_mix_code_submission.py
```
- Produces final competition CSV output

## 🛠️ Technical Specifications

### Hardware Requirements:
- **GPU**: NVIDIA T4, P100, or better
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 10GB free space for models and outputs

### Software Dependencies:
- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLO
- Pandas, NumPy

## 📊 Performance Metrics

### Accident Classification:
- **Accuracy**: >85% on test datasets
- **Inference Speed**: Real-time processing capable
- **Classes**: 3 incident types with severity grading

### Object Detection:
- **Model**: YOLOv8m (medium variant)
- **Frame Processing**: 16-24 FPS
- **Tracking**: Multi-object tracking with trajectory analysis

## 🔒 Proprietary Notice

**Important**: The main competition code (`Both_mix_code_submission.py`) contains proprietary algorithms and implementation details that are not publicly available. This represents the core intellectual property developed for the 2Coool Kaggle competition.

## 📞 Contact & Support

**Developer**: Aditya (HaradiBots)  
**Website**: [haradibots.onrender.com](https://haradibots.onrender.com)  
**WhatsApp**: +91 78872 85338  
**Instagram**: [@llaka2937](https://instagram.com/llaka2937)

---

<div align="center">

### 🏆 Built with passion for the 2Coool Kaggle Competition

**"Transforming traffic safety through AI-powered incident analysis"**

*HaradiBots - Innovating for safer roads* 🚗💨

</div>
