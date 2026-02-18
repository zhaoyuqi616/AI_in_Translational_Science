# 🧬 AI in Translational Science  

**Building reproducible AI workflows for genomics, single-cell biology, and precision medicine.**

This repository documents my structured journey applying machine learning and generative AI to real biomedical datasets, with a focus on:

- Cancer transcriptomics (TCGA, bulk RNA-seq)
- Single-cell & spatial omics
- Biomarker discovery
- Predictive modeling for translational research
- Generative models in biology

All projects emphasize:
- Reproducibility
- Clean code structure
- Proper validation
- Transparent evaluation
- Biomedical interpretability

---

## 📌 Repository Structure


---

## 🚀 Projects

| Project | Topic | Dataset | Methods | Status |
|----------|--------|---------|----------|---------|
| **Project 1** | Deep Learning for Prostate Cancer Classification | TCGA PRAD RNA-seq | Keras DNN, Early Stopping, AUROC | ✅ Completed |
| Project 2 | (Planned) TCGA BRCA Subtype Classification | TCGA BRCA | Baseline ML + DNN | ⏳ In Progress |
| Project 3 | (Planned) Single-Cell Embedding Learning | Public scRNA-seq | Autoencoders / Contrastive Learning | 🧠 Design Phase |
| Project 4 | (Planned) Generative Model for Cancer Subtypes | TCGA PAM50 | Conditional VAE | 🔬 Planned |

---

## 🧪 Example: Project 1 – Prostate Cancer (TCGA PRAD)

**Goal**  
Build a deep neural network classifier to distinguish prostate cancer samples using RNA-seq gene expression profiles.

**Highlights**
- Train/validation split
- Standardization of gene expression
- Early stopping
- AUROC + Accuracy evaluation
- Model + scaler saved for reproducibility

**Why this matters**  
Bulk RNA-seq classification pipelines are foundational for:
- Biomarker discovery
- Translational oncology modeling
- AI-guided therapeutic stratification

📂 See: `projects/Project1_PRAD_TCGA_DNN/`

---

## 🧠 Learning Roadmap

### Phase 1 — Foundations
- Classical ML (Logistic Regression, Random Forest, XGBoost)
- Cross-validation & evaluation metrics
- Feature importance & interpretability

### Phase 2 — Deep Learning
- Feedforward networks for transcriptomics
- Autoencoders for dimensionality reduction
- Model calibration & uncertainty

### Phase 3 — Generative AI in Biology
- Variational Autoencoders
- Conditional VAEs for subtype modeling
- Diffusion models (future direction)

### Phase 4 — Single-Cell & Spatial AI
- Embedding learning
- Cell type classification
- Graph neural networks

---

## 🔬 Technical Stack

- Python
- TensorFlow / Keras
- PyTorch (planned expansion)
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn

---

## ⚠️ Data Usage Notice

No raw patient-level TCGA data is stored in this repository.

Scripts assume users download data from:
- TCGA / GDC Data Portal
- Public GEO datasets

All pipelines are designed to be reproducible using publicly accessible datasets.

---

## 📈 Future Directions

- Model interpretability (SHAP, Integrated Gradients)
- Multi-omics integration
- Foundation models for cellular state prediction
- LLM-assisted biomarker hypothesis generation
- Cloud training workflows

---

## 🤝 Purpose

This repository serves as:

- A structured AI learning record
- A reproducible portfolio for industry positions
- A sandbox for translational AI experiments
- A foundation for future academic/biotech collaborations

---

## 📬 Contact

Open to collaboration in:
- Computational oncology
- Single-cell AI
- Generative biology models
- Translational biomarker discovery
