import os
import numpy as np
import torch

from .config import Config
from .utils import device
from .model import ConditionalVAE

PAM50_ORDER = ["Basal", "LumA", "LumB", "HER2", "Normal"]

def onehot(label: str):
    y = np.zeros((1, len(PAM50_ORDER)), dtype=np.float32)
    y[0, PAM50_ORDER.index(label)] = 1.0
    return y

def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    mean = np.array(ckpt["scaler_mean"], dtype=np.float32)
    scale = np.array(ckpt["scaler_scale"], dtype=np.float32)
    genes = ckpt["genes"]
    classes = ckpt["classes"]
    model = ConditionalVAE(len(genes), len(classes), cfg["hidden_dim"], cfg["latent_dim"], cfg.get("dropout", 0.1))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, mean, scale, genes, classes, cfg

def main(subtype="Basal", n=10):
    cfg = Config()
    ckpt_path = os.path.join(cfg.out_dir, "models", "best_cvae.pt")
    model, mean, scale, genes, classes, mcfg = load_checkpoint(ckpt_path)
    dev = device()
    model = model.to(dev)

    y = np.repeat(onehot(subtype), repeats=n, axis=0)
    y_t = torch.tensor(y, dtype=torch.float32).to(dev)

    z = torch.randn((n, mcfg["latent_dim"]), dtype=torch.float32).to(dev)

    with torch.no_grad():
        x_hat = model.decode(z, y_t).cpu().numpy()

    # de-standardize to original scale
    X_gen = x_hat * scale + mean

    out = os.path.join(cfg.out_dir, "metrics", f"generated_{subtype}_n{n}.csv")
    import pandas as pd
    pd.DataFrame(X_gen, columns=genes).to_csv(out, index=False)
    print(f"Saved generated samples to: {out}")

if __name__ == "__main__":
    main(subtype="Basal", n=10)
