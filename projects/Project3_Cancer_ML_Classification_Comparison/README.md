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
```text
Project3_BRCA_Subtype_Comparison/
│
├── Machine_Learning_Project_3_-_Comparison_of_Five_Machine_Learning_Methods_in_Subtyping_Breast_Cancer_Tumors_Based_on_Gene_Expression_of_Biomarkers.R
├── BRCA_Data_Retrieval.R
├── BRCA_PAM50_Expression.txt
├── BRCA_Subtypes.txt
├── PAM50_Genes.txt
├── README.md
```

---

## ▶️ How to Run

### 1️⃣ Install Required R Packages

```r
install.packages(c("caret", "e1071", "randomForest", "class"))
```
### 2️⃣ Run the Script
```r
source("Machine_Learning_Project_3_-_Comparison_of_Five_Machine_Learning_Methods_in_Subtyping_Breast_Cancer_Tumors_Based_on_Gene_Expression_of_Biomarkers.R")
```
### 📊 Output

The script outputs:

Model accuracy for each algorithm

Performance comparison table

Best-performing classifier identification

---

## 📈 Results

The project compares predictive performance across models to determine:

Which algorithm best captures subtype-specific gene expression patterns

The relative robustness of linear vs non-linear classifiers

Feasibility of biomarker-based subtype prediction

Random Forest and SVM are expected to perform strongly due to non-linear decision boundaries.

---

## 🔬 Scientific Context

Breast cancer molecular subtyping is foundational in:

Treatment stratification

Prognosis estimation

Hormone receptor targeting

HER2-targeted therapy decisions

Precision oncology

PAM50 biomarkers are widely used clinically to classify breast tumors.

This project demonstrates how AI models can replicate and potentially enhance subtype prediction.

---

## ⚠️ Limitations

Limited feature set (PAM50 only)

No cross-validation

No hyperparameter tuning

No external validation cohort

No model interpretability analysis

Does not integrate multi-omics data

---

## 🚀 Future Improvements

Stratified k-fold cross-validation

Hyperparameter tuning (Grid Search)

ROC curves for each subtype

Confusion matrix visualization

SHAP-based feature importance

Expansion to full transcriptome

Deep learning classifier comparison

Integration with clinical variables

---

## 🧬 Translational Relevance

This work supports:

AI-assisted molecular subtype prediction

Biomarker-driven oncology modeling

Translational cancer research workflows

Precision medicine decision support systems
