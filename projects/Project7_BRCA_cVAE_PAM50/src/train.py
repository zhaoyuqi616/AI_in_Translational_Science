import os
import json
import torch
from tqdm import tqdm

from .config import Config
from .utils import set_seed, ensure_dirs, device
from .data import (
    load_pam50_expression, load_subtypes, align_and_prepare, one_hot_subtypes,
    make_splits_and_loaders, standardize_train_val_test
)
from .model import ConditionalVAE, vae_loss

def train(cfg: Config):
    set_seed(cfg.seed)
    ensure_dirs(cfg.out_dir)
    dev = device()

    expr = load_pam50_expression(cfg.expr_path)
    sub  = load_subtypes(cfg.subtype_path)
    expr, sub = align_and_prepare(expr, sub)

    Y = one_hot_subtypes(sub)
    X = expr.values

    # Splits first, then scale using train only
    (train_loader, val_loader, test_loader), (X_tr, X_va, X_te), _ = make_splits_and_loaders(
        X, Y, cfg.batch_size, cfg.seed, cfg.val_frac, cfg.test_frac
    )
    scaler, X_tr_s, X_va_s, X_te_s = standardize_train_val_test(X_tr, X_va, X_te)

    # Replace dataset tensors with standardized values (simple approach)
    train_loader.dataset.x = torch.tensor(X_tr_s, dtype=torch.float32)
    val_loader.dataset.x   = torch.tensor(X_va_s, dtype=torch.float32)
    test_loader.dataset.x  = torch.tensor(X_te_s, dtype=torch.float32)

    x_dim = X.shape[1]
    y_dim = Y.shape[1]

    model = ConditionalVAE(x_dim, y_dim, cfg.hidden_dim, cfg.latent_dim, cfg.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_path = os.path.join(cfg.out_dir, "models", "best_cvae.pt")
    patience = 0

    history = {"train_total": [], "train_recon": [], "train_kl": [],
               "val_total": [], "val_recon": [], "val_kl": []}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tr_tot = tr_rec = tr_kl = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False):
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            x_hat, mu, logvar = model(x, y)
            loss, recon, kl = vae_loss(x, x_hat, mu, logvar, cfg.beta_kl)
            loss.backward()
            opt.step()
            tr_tot += loss.item()
            tr_rec += recon.item()
            tr_kl  += kl.item()

        ntr = len(train_loader)
        tr_tot /= ntr; tr_rec /= ntr; tr_kl /= ntr

        model.eval()
        va_tot = va_rec = va_kl = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(dev), y.to(dev)
                x_hat, mu, logvar = model(x, y)
                loss, recon, kl = vae_loss(x, x_hat, mu, logvar, cfg.beta_kl)
                va_tot += loss.item()
                va_rec += recon.item()
                va_kl  += kl.item()
        nva = len(val_loader)
        va_tot /= nva; va_rec /= nva; va_kl /= nva

        history["train_total"].append(tr_tot); history["train_recon"].append(tr_rec); history["train_kl"].append(tr_kl)
        history["val_total"].append(va_tot);   history["val_recon"].append(va_rec);   history["val_kl"].append(va_kl)

        print(f"Epoch {epoch:03d} | train total {tr_tot:.4f} (recon {tr_rec:.4f}, kl {tr_kl:.4f})"
              f" | val total {va_tot:.4f} (recon {va_rec:.4f}, kl {va_kl:.4f})")

        if va_tot < best_val - 1e-5:
            best_val = va_tot
            patience = 0
            torch.save({
                "model_state": model.state_dict(),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "genes": expr.columns.tolist(),
                "classes": ["Basal","LumA","LumB","HER2","Normal"],
                "config": cfg.__dict__,
            }, best_path)
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch}. Best val loss: {best_val:.4f}")
                break

    # save history
    with open(os.path.join(cfg.out_dir, "metrics", "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved best model to: {best_path}")
    return best_path

if __name__ == "__main__":
    best = train(Config())
