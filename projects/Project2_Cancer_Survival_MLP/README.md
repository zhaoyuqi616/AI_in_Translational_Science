# 🧬 Project 2 — Multilayer Perceptron for Breast Cancer Survival Prediction

## 📌 Overview

This project implements a Multilayer Perceptron (MLP) neural network to predict 5-year survival outcomes in breast cancer patients using the Haberman Survival Dataset.

It demonstrates how neural networks can be applied to structured clinical data for prognostic modeling in translational oncology.

---

## 🎯 Objective

Predict whether a patient survived at least 5 years after breast cancer surgery using clinical features.

### Binary Classification Task

- 1 → Survived 5 years or longer  
- 0 → Died within 5 years  

---

## 🧪 Dataset

**Haberman Survival Dataset**

- Source: UCI Machine Learning Repository
- Total samples: 306
- Clinical Features:
  1. Age at time of operation
  2. Year of operation
  3. Number of positive axillary lymph nodes

Target variable:
- Survival status (binary)

Dataset file included.
haberman.csv

---

## 🧠 Model Architecture

The classifier is implemented using TensorFlow / Keras.

### Neural Network Structure

- Input Layer: 3 clinical features
- Dense Layer (16 units, ReLU)
- Dense Layer (8 units, ReLU)
- Output Layer (1 unit, Sigmoid activation)

### Loss Function
Binary Crossentropy

### Optimizer
Adam

### Evaluation Metrics
- Accuracy
- Classification Report

---

## 🛠 Implementation Details

### Preprocessing
- Label encoding (Class 1 → 1, Class 2 → 0)
- Feature scaling (if applied in your script)
- Train-test split

### Training
- Batch size: 32
- Epochs: (as defined in script)
- Validation split included

### Evaluation
- Test set accuracy
- Model performance summary

---

## 📁 Project Structure
Project2_BreastCancer_Survival_MLP/
│
├── Machine_Learning_Project_2_Develop_A_Multilayer_Perceptron_Model_for_Predicting_Breast_Cancer_Survival.py
├── haberman.csv
├── README.md

---

## ▶️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
```
### 2️⃣ Run the Script
```
python Machine_Learning_Project_2_Develop_A_Multilayer_Perceptron_Model_for_Predicting_Breast_Cancer_Survival.py
```
### 📊 Expected Output

The script prints:

Model summary

Training accuracy

Test accuracy

Classification report

---

## 🔬 Scientific Context

Survival prediction models are foundational in:

Prognostic modeling

Risk stratification

Personalized treatment planning

Translational oncology studies

Although the Haberman dataset is small, it serves as a clean example of applying neural networks to clinical outcome prediction.

----

## ⚠️ Limitations

Small dataset (306 samples)

Limited feature space (3 features)

No cross-validation

No survival analysis modeling (e.g., Cox regression)

Does not handle class imbalance explicitly

---

## 🚀 Future Improvements

Planned upgrades:

Stratified k-fold cross-validation

ROC curve + AUROC

Confusion matrix visualization

Class imbalance handling

Comparison with Logistic Regression and Random Forest

SHAP interpretability

Survival modeling (Cox proportional hazards)

Larger clinical datasets (e.g., TCGA survival data)

----

## 🧬 Translational Relevance

This project demonstrates how neural networks can be applied to:

Clinical survival datasets

Risk prediction modeling

Translational decision-support research

Early-stage AI prototyping for healthcare
