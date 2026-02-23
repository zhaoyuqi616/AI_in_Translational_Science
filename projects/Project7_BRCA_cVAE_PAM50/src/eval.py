import os
import numpy as np
import torch

from .utils import device
from .config import Config
from .data import load_pam50_expression, load_subtypes, align_and_prepare, one_hot_subtypes
from .model import ConditionalVAE

def load_checkpoint(path, x_dim, y_dim):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    model = ConditionalVAE(x_dim, y_dim, cfg["hidden_dim"], cfg["latent_dim"], cfg.get("dropout", 0.1))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mean = np.array(ckpt["scaler_mean"], dtype=np.float32)
    scale = np.array(ckpt["scaler_scale"], dtype=np.float32)
    classes = ckpt["classes"]
    return model, mean, scale, classes

def main(cfg: Config, ckpt_path: str):
    expr = load_pam50_expression(cfg.expr_path)
    sub  = load_subtypes(cfg.subtype_path)
    expr, sub = align_and_prepare(expr, sub)
    Y = one_hot_subtypes(sub)
    X = expr.values.astype(np.float32)

    model, mean, scale, classes = load_checkpoint(ckpt_path, X.shape[1], Y.shape[1])
    dev = device()
    model = model.to(dev)

    Xs = (X - mean) / scale
    x = torch.tensor(Xs, dtype=torch.float32).to(dev)
    y = torch.tensor(Y, dtype=torch.float32).to(dev)

    with torch.no_grad():
        mu, logvar = model.encode(x, y)
        Z = mu.cpu().numpy()

    # quick embeddings (save arrays; plotting can be done in notebook)
    out = os.path.join(cfg.out_dir, "metrics", "latent_mu.npy")
    np.save(out, Z)
    np.save(os.path.join(cfg.out_dir, "metrics", "labels_onehot.npy"), Y)
    np.save(os.path.join(cfg.out_dir, "metrics", "labels_idx.npy"), Y.argmax(1))


if __name__ == "__main__":
    cfg = Config()
    ckpt = os.path.join(cfg.out_dir, "models", "best_cvae.pt")
    main(cfg, ckpt)
