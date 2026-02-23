from dataclasses import dataclass

@dataclass
class Config:
    # Data
    expr_path: str = "data/BRCA_PAM50_Expression.txt"
    subtype_path: str = "data/BRCA_Subtypes.txt"
    gene_id_col: str = None  # set to "gene" if your expression file has a gene column
    seed: int = 123

    # Train
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 200
    weight_decay: float = 1e-5
    val_frac: float = 0.15
    test_frac: float = 0.15
    early_stop_patience: int = 20

    # Model
    hidden_dim: int = 128
    latent_dim: int = 16
    beta_kl: float = 1.0  # KL weight (beta-VAE style)
    dropout: float = 0.1

    # Outputs
    out_dir: str = "outputs"
