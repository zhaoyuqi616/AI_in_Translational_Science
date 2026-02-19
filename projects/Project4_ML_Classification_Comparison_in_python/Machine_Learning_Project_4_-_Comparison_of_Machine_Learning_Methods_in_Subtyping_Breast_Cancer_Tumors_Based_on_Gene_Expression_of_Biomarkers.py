# Machine Learning Project 4 (Python version)
# - Loads BRCA PAM50 expression + subtype labels
# - Extracts ERBB2/ESR1/PGR, log1p transforms
# - Splits train/validation (stratified 80/20)
# - Trains 5 classifiers with 10-fold CV
# - Compares CV accuracy
# - Fits SVM (RBF) and evaluates on validation with confusion matrix
# - Saves a lightweight "workspace" (data + results) to disk

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import matplotlib.pyplot as plt


# -----------------------------
# 1) Load data
# -----------------------------
expr_path = "BRCA_PAM50_Expression.txt"
subtype_path = "BRCA_Subtypes.txt"

# Expression is assumed: rows = genes, columns = samples, with first column as rownames (genes).
# If your file is different, see notes below.
BRCA_PAM50_Expression = pd.read_csv(expr_path, sep=",", header=0, index_col=0)

# In R: colnames gsub("\\.","-", colnames)
BRCA_PAM50_Expression.columns = BRCA_PAM50_Expression.columns.str.replace(".", "-", regex=False)

# Load subtypes
BRCA_Subtypes = pd.read_csv(subtype_path, sep=",", header=0)

# Expect a column named "Patients" and "Subtypes" (matching your R code).
# If your file uses different names, adjust here:
patients_col = "Patients"
subtypes_col = "Subtypes"

# -----------------------------
# 2) Make sure samples align
# -----------------------------
expr_samples = BRCA_PAM50_Expression.columns.astype(str)
subtype_patients = BRCA_Subtypes[patients_col].astype(str)

# Reorder subtypes to match expression column order (safer than just checking equality)
# This is the Python equivalent of ensuring identical sample ordering.
subtype_map = BRCA_Subtypes.set_index(patients_col)[subtypes_col]
missing_in_subtypes = [s for s in expr_samples if s not in subtype_map.index]
if missing_in_subtypes:
    raise ValueError(
        f"{len(missing_in_subtypes)} expression samples are missing in BRCA_Subtypes: "
        f"{missing_in_subtypes[:10]}{'...' if len(missing_in_subtypes) > 10 else ''}"
    )

y_all = subtype_map.loc[expr_samples].astype("category")

# -----------------------------
# 3) Select biomarkers and preprocess
# -----------------------------
Subtype_Biomarkers = ["ERBB2", "ESR1", "PGR"]
missing_genes = [g for g in Subtype_Biomarkers if g not in BRCA_PAM50_Expression.index]
if missing_genes:
    raise ValueError(f"Missing biomarkers in expression matrix rows: {missing_genes}")

# R: Biomarkers_Expression <- BRCA_PAM50_Expression[Subtype_Biomarkers,]
# then transpose to samples x genes
X_all = BRCA_PAM50_Expression.loc[Subtype_Biomarkers, :].T.copy()

# R: log1p on the first 3 columns (genes)
X_all[Subtype_Biomarkers] = np.log1p(X_all[Subtype_Biomarkers].astype(float))

# Combine like R data frame (optional)
Biomarkers_Expression = X_all.copy()
Biomarkers_Expression["Subtypes"] = y_all.values  # categorical labels


# -----------------------------
# 4) Train/Validation split (80/20 stratified)
# -----------------------------
X = Biomarkers_Expression[Subtype_Biomarkers].values
y = Biomarkers_Expression["Subtypes"].astype(str).values  # sklearn likes strings

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.20,
    random_state=600,
    stratify=y
)

# -----------------------------
# 5) Basic EDA summaries (rough equivalents)
# -----------------------------
train_df = pd.DataFrame(X_train, columns=Subtype_Biomarkers)
train_df["Subtypes"] = y_train

print("\nData types:")
print(train_df.dtypes)

print("\nHead (first 5 rows):")
print(train_df.head())

print("\nClass levels:")
print(sorted(pd.unique(y_train)))

print("\nClass distribution (train):")
freq = pd.Series(y_train).value_counts().sort_index()
pct = (freq / freq.sum() * 100).round(2)
print(pd.DataFrame({"freq": freq, "percentage": pct}))

print("\nSummary statistics (train features):")
print(train_df[Subtype_Biomarkers].describe())


# -----------------------------
# 6) Plots (optional)
# -----------------------------
# Boxplots for each feature
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, g in enumerate(Subtype_Biomarkers):
    axes[i].boxplot(train_df[g].values, vert=True)
    axes[i].set_title(g)
plt.tight_layout()
plt.show()

# Class breakdown bar plot
pd.Series(y_train).value_counts().plot(kind="bar", figsize=(6, 4))
plt.title("Training class counts")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Simple scatter matrix (rough analogue to featurePlot ellipse/box/density)
pd.plotting.scatter_matrix(train_df[Subtype_Biomarkers], figsize=(8, 8), diagonal="kde")
plt.tight_layout()
plt.show()


# -----------------------------
# 7) 10-fold CV model comparison (Accuracy)
# -----------------------------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=600)

models = {
    "lda": Pipeline([("scaler", StandardScaler()), ("clf", LinearDiscriminantAnalysis())]),
    "cart": Pipeline([("clf", DecisionTreeClassifier(random_state=600))]),
    "knn": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
    "svm": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=False, random_state=600))]),
    "rf": Pipeline([("clf", RandomForestClassifier(n_estimators=500, random_state=600))]),
}

cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    cv_results[name] = scores
    print(f"{name}: mean={scores.mean():.4f}, sd={scores.std():.4f}")

# Compare CV accuracy via boxplot (rough equivalent to caret dotplot)
plt.figure(figsize=(7, 4))
plt.boxplot([cv_results[k] for k in cv_results.keys()], labels=list(cv_results.keys()))
plt.title("10-fold CV Accuracy Comparison")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()


# -----------------------------
# 8) Fit the "best" model (SVM like your R code) and evaluate on validation
# -----------------------------
best_model = models["svm"]
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_val)

print("\nValidation Accuracy:", accuracy_score(y_val, y_pred))

labels = sorted(pd.unique(y))  # fixed label order
cm = confusion_matrix(y_val, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])

print("\nConfusion Matrix (validation):")
print(cm_df)

print("\nClassification report (validation):")
print(classification_report(y_val, y_pred, labels=labels))


# -----------------------------
# 9) Save a lightweight "workspace"
# -----------------------------
# Similar intent to save.image(...) in R: save the important objects/results.
import joblib

os.makedirs("ml_project3_outputs", exist_ok=True)

# Save the fitted model
joblib.dump(best_model, "ml_project3_outputs/svm_model.joblib")

# Save CV results and confusion matrix
pd.DataFrame({k: v for k, v in cv_results.items()}).to_csv("ml_project3_outputs/cv_accuracy_scores.csv", index=False)
cm_df.to_csv("ml_project3_outputs/validation_confusion_matrix.csv")

# Save processed dataset
Biomarkers_Expression.to_csv("ml_project3_outputs/biomarkers_expression_processed.csv", index=True)

print("\nSaved outputs under: ml_project3_outputs/")

