# 🧬 Project 1 — Deep Learning-based Identification of Prostate Cancer Using TCGA RNA-seq

## 📌 Overview

This project implements a deep neural network (DNN) classifier to distinguish prostate cancer samples using bulk RNA-seq gene expression data from The Cancer Genome Atlas (TCGA) Prostate Adenocarcinoma (PRAD) cohort.

The goal is to demonstrate how deep learning can be applied to transcriptomic data for cancer classification in a translational research context.

---

## 🎯 Objective

Build a binary classifier that predicts:

- **Tumor (Primary solid tumor)**
- **Normal (Solid tissue normal)**

using gene expression features derived from TCGA RNA-seq data.

---

## 🧪 Dataset

- Source: TCGA PRAD
- Data Type: Bulk RNA-seq gene expression
- Features: ~577 genes (as defined in the input dataset)
- Labels:
  - `Primary solid Tumor`
  - `Solid Tissue Normal`

⚠️ **Note**: Raw patient-level data is not included in this repository.
Users must download TCGA PRAD expression data from:

- GDC Data Portal  
  https://portal.gdc.cancer.gov/

---

## 🧠 Model Architecture

The classifier is implemented using TensorFlow / Keras.

### Neural Network Structure

- Input Layer: 577 features
- Dense Layer (256 units, ReLU)
- Dropout (0.5)
- Dense Layer (128 units, ReLU)
- Dropout (0.5)
- Output Layer (1 unit, Sigmoid activation)

### Loss Function
Binary Crossentropy

### Optimizer
Adam

### Evaluation Metric
Accuracy

---

## 🛠 Implementation Details

### Preprocessing
- Label encoding (Tumor = 1, Normal = 0)
- Gene expression values used as numeric input features

### Training
- Batch size: 32
- Epochs: 20
- Model trained on full dataset

### Output
- Training accuracy
- Model evaluation accuracy
- Session saved using `dill`

---

## 📁 Project Structure

Project1_PRAD_TCGA_DNN/
- Machine_Learning_Project_1_Deep_Learning-based_Identification_of_Prostate_Cancer_using_TCGA_RNA_seq.py
- README.md
- PRAD_TCGA_RNA_seq.txt
- PRAD_TCGA_Types.txt


---


---

## ▶️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install tensorflow pandas numpy dill

python Machine_Learning_Project_1_Deep_Learning-based_Identification_of_Prostate_Cancer_using_TCGA_RNA_seq.py



