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
│
├── Machine_Learning_Project_1_Deep_Learning-based_Identification_of_Prostate_Cancer_using_TCGA_RNA_seq.py
├── README.md
└── PRAD_TCGA_RNA_seq.txt
└── PRAD_TCGA_Types.txt


---

## ▶️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install tensorflow pandas numpy dill

2️⃣ Place Dataset

Ensure the TCGA PRAD RNA-seq dataset CSV is located in the working directory.

Modify the file path inside the script if necessary.

3️⃣ Run Training
python Machine_Learning_Project_1_Deep_Learning-based_Identification_of_Prostate_Cancer_using_TCGA_RNA_seq.py

📊 Expected Output

The script prints:

Training accuracy

Evaluation accuracy

Example:

Accuracy: 0.94

🔬 Scientific Context

Deep learning models applied to bulk RNA-seq data can:

Identify cancer vs normal tissue signatures

Enable biomarker discovery

Support translational oncology modeling

Provide groundwork for multi-omics integration

This project demonstrates a foundational workflow for applying neural networks to transcriptomic cancer datasets.

⚠️ Limitations

This initial implementation:

Trains and evaluates on the same dataset

Does not include cross-validation

Does not include external validation

Does not perform feature scaling

Does not assess AUROC

Future versions will improve robustness and reproducibility.

🚀 Future Improvements

Planned enhancements:

Train/validation/test split

Standardization of gene expression

Early stopping

ROC curve + AUROC

Confusion matrix

SHAP-based model interpretability

Cross-cohort validation

Model export for deployment

🧬 Translational Relevance

This workflow illustrates how AI models can be integrated into:

Precision oncology pipelines

Biomarker discovery workflows

Predictive modeling frameworks

Clinical decision support research

📜 License

MIT License

🤝 Author

AI in Translational Science
Focused on AI-driven precision medicine, single-cell analysis, and generative biology.


---

# 💡 Important Recommendation

Your current implementation is a **great learning start**, but if you want this project to impress biotech/AI hiring managers, I strongly recommend:

- Adding train/test split
- Adding AUROC
- Adding proper scaling
- Saving model weights instead of dill session
- Including a ROC curve image in README

If you'd like, I can next:

- 🔥 Rewrite the code into a “production-ready v2”
- 📊 Add ROC + confusion matrix figure
- 🧠 Add SHAP interpretability
- 🎯 Upgrade this to industry-grade quality

Just tell me the level you want:
- Learning project
- Industry portfolio
- Research-paper style
- Generative AI extension
