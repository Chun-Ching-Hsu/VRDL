# NYCU Computer Vision 2025 Spring - HW1

**Student ID**: 313554033  
**Name**: 許晴鈞

---

## 📘 Introduction

This project implements a 100-class image classification model based on a pretrained **ResNet50** architecture, using transfer learning. Key features include:

- Customized classifier with dropout
- Rich data augmentation pipeline (e.g., color jitter, rotation, erasing)
- Class weighting to handle dataset imbalance
- Mixed precision training (AMP)
- Early stopping & model checkpointing
- Per-class accuracy logging

---

## ⚙️ Environment Setup (with Conda)

1. Create a new conda environment:
   ```bash
   conda create -n cv_hw1 python=3.9 -y
   conda activate cv_hw1
2.  Install required packages:
```python
pip install -r requirements.txt
```
3.  Download the dataset and place it in the ./data folder with the following structure:
```
data/
  ├── train/
  ├── val/
  └── test/ 
```
## 🧠 How to Run
Train the model using:
``` python
python train.py
```
The model will be saved under ./model/model_ResNet50

Training/validation logs and per-class accuracy CSV will also be saved

##  📈 Performance Snapshot

|Metric|	Value
|-----|------|
Best |Val Accuracy|	85.3%
Epoch| Achieved	|25
Model |Parameters|	~23.5M

Worst-performing classes (example):
```python
Class 10 - Accuracy: 0.33
Class 52 - Accuracy: 0.33
Class 86 - Accuracy: 0.33
```