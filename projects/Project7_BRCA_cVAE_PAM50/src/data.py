import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

PAM50_ORDER = ["Basal", "LumA", "LumB", "HER2", "Normal"]

class OmicsDataset(Dataset):
    def __init__(self, x: np.ndarray, y_onehot: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y_onehot, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def load_pam50_expression(expr_path: str) -> pd.DataFrame:
    """
    Expected format (common in TCGA course projects):
      - rows: genes
      - columns: samples
    This returns a DataFrame with samples as rows.
    """
    df = pd.read_csv(expr_path, sep=",", header=0)
    # If first column is gene names, set it as index automatically
    if df.columns[0].lower() in ("gene", "genes", "symbol", "hgnc_symbol", "id"):
        df = df.set_index(df.columns[0])
    # transpose: samples x genes
    return df.T

def load_subtypes(subtype_path: str) -> pd.Series:
    df = pd.read_csv(subtype_path, sep=",", header=0)
    # Expect columns like Patients, Subtypes (common in your earlier projects)
    # If different, adjust here
    if "Patients" in df.columns and "Subtypes" in df.columns:
        s = pd.Series(df["Subtypes"].values, index=df["Patients"].values)
    else:
        # fallback: first col = sample, second col = label
        s = pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0].values)
    return s

def align_and_prepare(expr_df: pd.DataFrame, subtype_s: pd.Series):
    # Align by sample IDs
    common = expr_df.index.intersection(subtype_s.index)
    expr_df = expr_df.loc[common].copy()
    subtype_s = subtype_s.loc[common].copy()

    # Basic cleanup: ensure subtype labels match expected set
    subtype_s = subtype_s.astype(str).str.strip()
    return expr_df, subtype_s

def one_hot_subtypes(subtype_s: pd.Series, classes=PAM50_ORDER):
    class_to_i = {c:i for i,c in enumerate(classes)}
    y = np.zeros((len(subtype_s), len(classes)), dtype=np.float32)
    for r, lab in enumerate(subtype_s.values):
        if lab not in class_to_i:
            raise ValueError(f"Unknown subtype label '{lab}'. Expected one of {classes}.")
        y[r, class_to_i[lab]] = 1.0
    return y

def make_splits_and_loaders(X: np.ndarray, Y: np.ndarray, batch_size: int, seed: int, val_frac: float, test_frac: float):
    # split: train / temp
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=(val_frac + test_frac), random_state=seed, stratify=Y.argmax(1)
    )
    # split temp into val/test
    rel_test = test_frac / (val_frac + test_frac)
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=rel_test, random_state=seed, stratify=Y_temp.argmax(1)
    )

    train_ds = OmicsDataset(X_train, Y_train)
    val_ds   = OmicsDataset(X_val, Y_val)
    test_ds  = OmicsDataset(X_test, Y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return (train_loader, val_loader, test_loader), (X_train, X_val, X_test), (Y_train, Y_val, Y_test)

def standardize_train_val_test(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    return scaler, X_train_s, X_val_s, X_test_s
