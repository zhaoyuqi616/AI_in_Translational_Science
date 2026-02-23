# 🧬 Project 7 — Conditional VAE for BRCA PAM50 Subtype Representation Learning and Subtype-Conditioned Generation

---

## 📌 Overview

This project builds a **Conditional Variational Autoencoder (cVAE)** to learn a compact latent representation of **TCGA BRCA** gene expression (PAM50 genes) and generate **subtype-conditioned** synthetic expression profiles.

Unlike standard classifiers (Projects 3–4), this project focuses on **generative modeling + representation learning**, which are core concepts behind modern biological foundation models.

---

## 🎯 Objective

### Goals
1. Learn a latent embedding of PAM50 expression that captures major biological variation.
2. Condition the generative model on molecular subtype labels (Basal, LumA, LumB, HER2, Normal).
3. Generate **synthetic gene expression profiles** conditioned on a chosen subtype.

### Input
- PAM50 expression vector (50 genes) per sample
- Subtype label (one-hot)

### Output
- Reconstructed expression (denoised reconstruction)
- Latent embeddings (μ)
- Generated expression profiles for a requested subtype

---

## 🧪 Dataset

### Files (not committed)
Place these in `data/`:

- `BRCA_PAM50_Expression.txt` — expression matrix (genes × samples OR similar)
- `BRCA_Subtypes.txt` — sample subtype labels

Expected format:
- Expression: rows = genes, columns = samples (script transposes to samples × genes)
- Subtypes: columns include `Patients` and `Subtypes` (or first two columns sample/label)

⚠️ Do not commit patient-level data. Keep `data/` in `.gitignore`.

---

## 🧠 Model Architecture

### Conditional VAE (cVAE)

**Encoder**: takes `[x, y]` where  
- `x` = standardized gene expression vector  
- `y` = one-hot subtype label  

Encoder outputs:
- latent mean `μ`
- latent log-variance `logσ²`

**Reparameterization trick**:
- `z = μ + ε * σ`

**Decoder**: takes `[z, y]` and reconstructs `x_hat`

### Loss
- Reconstruction loss: **MSE** (works well for standardized expression)
- KL divergence regularization
- Total: `loss = recon + β * KL`

---

## 🛠 Implementation Details

- Train/val/test split with stratification by subtype
- Standardization fitted on training set only
- Early stopping based on validation loss
- Saves:
  - best model checkpoint (`outputs/models/best_cvae.pt`)
  - training history (`outputs/metrics/train_history.json`)
  - latent embeddings + PCA/t-SNE arrays (`outputs/metrics/*.npy`)
  - generated samples (`outputs/metrics/generated_<Subtype>_n<k>.csv`)

---

## 📁 Project Structure

```text
projects/Project7_BRCA_cVAE_PAM50/
├── README.md              # Project-specific documentation
├── requirements.txt       # Managed Python dependencies
├── .gitignore             # Ensures sensitive/large data is not tracked
├── data/                  # Local data storage (excluded from Git)
│   └── (NOT COMMITTED)    # BRCA_PAM50_Expression.txt, BRCA_Subtypes.txt
├── src/                   # Core functional logic and model architecture
│   ├── __init__.py
│   ├── config.py          # Hyperparameters and directory paths
│   ├── data.py            # Data loading and preprocessing pipelines
│   ├── model.py           # CVAE architecture implementation
│   ├── train.py           # Model training procedures
│   ├── eval.py            # Performance evaluation scripts
│   ├── sample.py          # Latent space sampling and generation
│   └── utils.py           # Reusable helper functions
├── notebooks/             # Exploratory data analysis and visualization
│   └── 01_latent_space_visualization.ipynb
├── outputs/               # Directory for generated artifacts
│   ├── models/            # Serialized model weights (.pth or .h5)
│   ├── figures/           # Generated plots and visualization results
│   └── metrics/           # Log files and performance statistics
└── scripts/               # DevOps and automation tasks
    └── run_all.sh         # Shell script to automate the full pipeline
```

---

## ▶️ How to Run

---

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
### 2️⃣ Put data files into datta/
data/BRCA_PAM50_Expression.txt
data/BRCA_Subtypes.txt
### 3️⃣ Train cVAE
python -m src.train
### 4️⃣ Extract latent embeddings (μ) and save PCA/t-SNE arrays
python -m src.eval
### 5️⃣ Generate subtype-conditioned expression profiles
python -m src.sample
```

---

## 📊 Output

After running, you should have:

outputs/models/best_cvae.pt (best checkpoint)

outputs/metrics/train_history.json (loss curves)

outputs/metrics/latent_mu.npy (latent means)

outputs/metrics/pca2.npy, outputs/metrics/tsne2.npy (2D embeddings)

outputs/metrics/generated_Basal_n10.csv (example subtype-conditioned samples)

---

## 🔬 Scientific Context

cVAEs are useful in translational genomics because they can:

Learn denoised latent representations of expression profiles

Capture continuous biological variability beyond discrete subtypes

Enable controlled generation (e.g., subtype-conditioned profiles)

Provide embeddings for downstream tasks (clustering, trajectory, prediction)

This is a stepping stone toward modern biological foundation model ideas (representation learning, generative priors).

---

## ⚠️ Limitations

This cVAE is trained only on PAM50 genes (small feature space)

Generated expression is synthetic and may not preserve all biological constraints

No external cohort validation included

No explicit batch correction included (could be added via covariates)

---

## 🚀 Future Improvements

Expand from PAM50 to larger gene sets (e.g., Hallmarks / variable genes)

Add covariates (batch, purity, clinical features) as conditioning inputs

Add latent space evaluation (silhouette score by subtype)

Add downstream tasks using latent embeddings (classification, survival)

Compare against scVI-like frameworks for deeper generative modeling

---


