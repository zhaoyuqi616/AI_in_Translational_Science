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
```text
Project4_BRCA_Subtype_Python/
│
├── Machine_Learning_Project_4_-_Comparison_of_Machine_Learning_Methods_in_Subtyping_Breast_Cancer_Tumors_Based_on_Gene_Expression_of_Biomarkers.py
├── BRCA_PAM50_Expression.txt
├── BRCA_Subtypes.txt
├── PAM50_Genes.txt
├── README.md
```

---

## ▶️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
### 2️⃣ Execute the Script
```bash
python Machine_Learning_Project_4_-_Comparison_of_Machine_Learning_Methods_in_Subtyping_Breast_Cancer_Tumors_Based_on_Gene_Expression_of_Biomarkers.py
```
### 📊 Output

The script generates:

Accuracy for each model

Performance comparison summary

Best-performing classifier identification
### 📈 Results

The comparative evaluation allows:

Identification of the most robust classifier

Analysis of linear vs non-linear model performance

Benchmarking of classical ML methods for biomarker-driven subtype prediction

Tree-based models and SVM are expected to perform strongly due to non-linear gene expression patterns.

---

## 🔬 Scientific Context

Breast cancer molecular subtyping is fundamental for:

Treatment selection

HER2-targeted therapy decisions

Hormone receptor-based stratification

Prognosis assessment

Precision oncology frameworks

The PAM50 biomarker panel is clinically validated and widely used for subtype classification.

This project demonstrates how AI models can be used to reproduce and benchmark subtype classification pipelines.

---

## ⚠️ Limitations

Limited feature space (PAM50 only)

No hyperparameter tuning

No cross-validation

No external validation cohort

No interpretability analysis

Does not integrate clinical or multi-omics data

---

## 🚀 Future Improvements

Stratified k-fold cross-validation

Hyperparameter tuning via GridSearchCV

ROC curves per subtype

Confusion matrix visualization

SHAP feature importance analysis

Deep learning model comparison

Multi-omics integration

Model deployment workflow

---

## 🧬 Translational Relevance

This project supports:

AI-assisted molecular subtype classification

Biomarker-driven oncology modeling

Translational cancer research pipelines

Precision medicine strategy development
