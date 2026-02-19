# 🧬 Project 4 — Machine Learning-Based Breast Cancer Subtype Classification Using PAM50 Gene Expression (Python Implementation)

---

## 📌 Overview

This project implements supervised machine learning models in Python to classify breast cancer molecular subtypes using PAM50 gene expression data from TCGA BRCA.

The objective is to evaluate the performance of classical machine learning algorithms in subtype prediction and compare their predictive capabilities within a reproducible Python workflow.

This project serves as a Python-based counterpart to Project 3 (R implementation).

---

## 🎯 Objective

Perform multi-class classification of breast cancer tumors into molecular subtypes using PAM50 gene expression biomarkers.

### Input
Expression levels of 50 PAM50 genes.

### Output
Predicted molecular subtype:

- Luminal A
- Luminal B
- HER2-enriched
- Basal-like
- Normal-like

---

## 🧪 Dataset

**Source:** TCGA Breast Invasive Carcinoma (BRCA)

**Files Used:**
- `BRCA_PAM50_Expression.txt` — Gene expression matrix
- `BRCA_Subtypes.txt` — Sample subtype labels
- `PAM50_Genes.txt` — PAM50 gene list

**Data Type:** Bulk RNA-seq gene expression  
**Features:** 50 biomarker genes  
**Target Variable:** Molecular subtype

⚠️ Raw TCGA patient-level data is not redistributed in this repository.

---

## 🧠 Model Architecture

The following machine learning models are implemented in Python:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- K-Nearest Neighbors (KNN)
- (Optional) Gradient Boosting

### Implementation Framework

- scikit-learn
- pandas
- numpy

### Evaluation Strategy

- Train/test split
- Accuracy comparison
- Multi-class classification evaluation

---

## 🛠 Implementation Details

- Data preprocessing and subtype label alignment
- Feature filtering restricted to PAM50 genes
- Train/test dataset split
- Standardization of features (if applied)
- Multi-model training and evaluation
- Comparative performance reporting

---

## 📁 Project Structure

