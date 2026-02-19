# 🧬 Project 5 — Convolutional Neural Network (CNN) for Breast Cancer Classification Using Histopathological Images

---

## 📌 Overview

This project develops and evaluates Convolutional Neural Network (CNN) models for classifying breast cancer histopathological images into benign and malignant categories.

Approximately 8,000 microscopic images are used to train deep learning models that learn spatial morphological patterns associated with cancer pathology.

Two CNN configurations are trained and compared:

- Custom 3-block VGG-style CNN (VGG3)
- Extended CNN model (labeled VGG16 in script)

This project introduces medical imaging AI into the translational oncology workflow.

---

## 🎯 Objective

Build a binary classifier to distinguish:

- **Benign breast tissue**
- **Malignant breast cancer tissue**

### Input
RGB histopathological images resized to 200 × 200 pixels.

### Output
Binary classification:
- 0 → Benign
- 1 → Malignant

---

## 🧪 Dataset

**Dataset Type:** Histopathological microscopy images  
**Total Images:** ~8,000  
**Image Size (resized):** 200 × 200 × 3  

### Required Folder Structure


⚠️ Image data is not included in this repository due to size limitations.

---

## 🧠 Model Architecture

### 1️⃣ Custom VGG3 CNN

Architecture:

- Conv2D (32 filters, 3×3, ReLU, He initialization)
- MaxPooling2D (2×2)
- Dropout (0.2)
- Conv2D (64 filters, 3×3, ReLU)
- MaxPooling2D
- Dropout (0.2)
- Conv2D (128 filters, 3×3, ReLU)
- MaxPooling2D
- Flatten
- Dense (128 units, ReLU)
- Dropout (0.5)
- Dense (1 unit, Sigmoid)

### 2️⃣ VGG16 Model (as implemented)

Note: The VGG16 model defined in the script uses the same architecture structure as VGG3 rather than a pretrained VGG16 network.

---

### Training Configuration

- Loss Function: Binary Crossentropy
- Optimizer: SGD  
  - Learning rate = 0.001  
  - Momentum = 0.9  
- Batch Size: 64
- Epochs: 20
- Evaluation Metric: Accuracy

---

## 🛠 Implementation Details

### Data Augmentation

Images are augmented using `ImageDataGenerator`:

- Rescaling: 1./255
- Width shift range: 10%
- Height shift range: 10%
- Zoom range: 2
- Rotation range: 90 degrees
- Horizontal flip
- Vertical flip

This improves model generalization.

---

### Visualization

- Random sample images plotted from each class
- Training & validation loss curves
- Training & validation accuracy curves

Diagnostic plots are saved as:

