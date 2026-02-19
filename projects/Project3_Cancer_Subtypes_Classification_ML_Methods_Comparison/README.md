# 🧬 Project 3 — Comparison of Five Machine Learning Methods for Breast Cancer Subtyping Using PAM50 Gene Expression

---

## 📌 Overview

This project compares five machine learning algorithms for classifying breast cancer molecular subtypes using PAM50 gene expression biomarkers derived from TCGA BRCA.

The goal is to evaluate how different classification models perform in predicting molecular subtypes, a key task in precision oncology and translational cancer research.

---

## 🎯 Objective

Perform multi-class classification of breast cancer tumors into molecular subtypes based on PAM50 gene expression.

### Input
Expression values of PAM50 biomarker genes.

### Output
Predicted breast cancer subtype:
- Luminal A
- Luminal B
- HER2-enriched
- Basal-like
- Normal-like

---

## 🧪 Dataset

**Source:** TCGA Breast Invasive Carcinoma (BRCA)

**Data Files:**
- `BRCA_PAM50_Expression.txt` — Expression matrix
- `BRCA_Subtypes.txt` — Subtype labels
- `PAM50_Genes.txt` — List of PAM50 genes
- `BRCA_Data_Retrieval.R` — Data preprocessing script

**Data Type:** Bulk RNA-seq gene expression  
**Features:** 50 PAM50 biomarker genes  
**Target Variable:** Molecular subtype

⚠️ Raw TCGA data is not redistributed in this repository.

---

## 🧠 Model Architecture

Five supervised learning models are implemented and compared:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- K-Nearest Neighbors (KNN)
- Naive Bayes (if included)

### Evaluation Strategy

- Train/test split
- Accuracy comparison
- Performance benchmarking

---

## 🛠 Implementation Details

- Implemented in R
- Data preprocessing and subtype alignment
- Feature filtering restricted to PAM50 genes
- Model training using standard ML libraries
- Performance comparison across algorithms

---

## 📁 Project Structure

