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
```text
dataset/
│
├── benign/
└── malignant/
```

⚠️ Due to size constraints, image data is not included in this repository.

---

## 🧠 Model Architecture

The model is implemented using TensorFlow / Keras.

### CNN Architecture

- Input Layer (image input)
- Convolutional Layer + ReLU
- MaxPooling Layer
- Convolutional Layer + ReLU
- MaxPooling Layer
- Flatten Layer
- Dense Layer (Fully connected)
- Output Layer (Sigmoid activation)

### Training Configuration

- Loss: Binary Crossentropy
- Optimizer: Adam
- Batch size: Defined in script
- Epochs: Defined in script
- Validation split applied

---

## 🛠 Implementation Details

- Image preprocessing and resizing
- Normalization of pixel values
- Train/test split
- Data augmentation (if applied)
- Model training and evaluation
- Accuracy tracking during training
- Model saving for later inference

Libraries used:

- tensorflow / keras
- numpy
- matplotlib
- os

---

## 📁 Project Structure
```text
Project5_BreastCancer_CNN/
│
├── Machine_Learning_Project_5_-_Developing_A_Convolutional_Neural_Network_CNN_for_Classifying_Breast_Cancer_Based_on_8000_Histopathological_Images.py
├── dataset/ (not included)
├── figures/
├── saved_model/
└── README.md
```

---

## ▶️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install tensorflow numpy matplotlib
### 2️⃣ Prepare Dataset

Ensure dataset directory is structured as:
```text
dataset/
│
├── benign/
└── malignant/
```
### 3️⃣ Train the Model
```python
python Machine_Learning_Project_5_-_Developing_A_Convolutional_Neural_Network_CNN_for_Classifying_Breast_Cancer_Based_on_8000_Histopathological_Images.py
```
### 📊 Output

The script generates:

Model summary

Training accuracy

Validation accuracy

Final test accuracy

Saved trained model

If implemented, training curves may be displayed:

Accuracy vs Epoch

Loss vs Epoch

### 📈 Results

The CNN model learns spatial tissue morphology patterns and demonstrates strong performance in distinguishing malignant from benign tissue samples.

Deep learning significantly outperforms traditional handcrafted feature methods in medical image classification tasks.

---

## 🔬 Scientific Context

Histopathological image analysis is critical in:

Cancer diagnosis

Tumor grading

Pathology workflow automation

AI-assisted diagnostic systems

Digital pathology research

CNN-based models have become foundational in computational pathology and medical imaging AI.

This project demonstrates how deep learning can assist in automated cancer detection pipelines.

---

## ⚠️ Limitations

No external validation dataset

No transfer learning (e.g., ResNet, EfficientNet)

No interpretability analysis (Grad-CAM)

No cross-validation

Potential class imbalance issues

Dataset size moderate (8,000 images)

---

## 🚀 Future Improvements

Implement transfer learning (ResNet50, EfficientNet)

Add data augmentation pipeline

Use early stopping and learning rate scheduling

Add ROC curve and AUROC evaluation

Implement Grad-CAM for interpretability

Perform cross-validation

Deploy as web-based inference tool

Integrate multimodal modeling (image + genomics)

---

## 🧬 Translational Relevance

This project supports:

AI-assisted pathology workflows

Automated cancer screening systems

Precision oncology diagnostics

Digital pathology research platforms

Clinical decision support development
