import os
import argparse
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    csv_path: str = "training_set_pixel_size_and_HC.csv"
    images_dir: str = "training_set"
    seed: int = 42

    img_size: int = 224
    batch_size: int = 32
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4

    val_ratio: float = 0.2
    num_workers: int = 0  # Windows: keep 0 to avoid multiprocessing issues


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Data
# -----------------------------
def load_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def infer_columns(df: pd.DataFrame) -> tuple[str, str]:
    lower = {c.lower(): c for c in df.columns}

    id_col = None
    for cand in ["id", "image", "img", "filename", "file", "index"]:
        if cand in lower:
            id_col = lower[cand]
            break
    if id_col is None:
        id_col = df.columns[0]

    target_col = None
    for cand in ["hc", "head circumference (mm)", "head_circumference"]:
        if cand.lower() in lower:
            target_col = lower[cand.lower()]
            break
    if target_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found to use as target.")
        target_col = numeric_cols[-1]

    return id_col, target_col


def build_paths_and_labels(
    df: pd.DataFrame,
    images_dir: str,
    id_col: str,
    target_col: str
) -> pd.DataFrame:

    images_dir = os.path.abspath(images_dir)
    df = df.copy()
    df["image_id"] = df[id_col].astype(str).str.strip()

    def resolve_img_path(image_id: str) -> str:
        candidates = [
            image_id,
            f"{image_id}.png",
            f"{image_id}.jpg",
            f"{image_id}.jpeg",
        ]
        for name in candidates:
            p = os.path.join(images_dir, name)
            if os.path.isfile(p):
                return p
        return ""

    df["img_path"] = df["image_id"].apply(resolve_img_path)
    df["hc"] = df[target_col].astype(float)

    exists = df["img_path"].str.len() > 0
    missing = (~exists).sum()
    if missing > 0:
        print(f"[WARN] {missing} images listed in CSV are missing on disk.")
        df = df[exists].reset_index(drop=True)

    return df[["image_id", "img_path", "hc"]]


def train_val_split(df: pd.DataFrame, val_ratio: float, seed: int):
    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_val = int(len(df) * val_ratio)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


class HCDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(row["img_path"]).convert("L")
        img = img.resize((self.img_size, self.img_size))

        x = np.asarray(img, dtype=np.float32) / 255.0
        x = np.expand_dims(x, axis=0)

        y = np.float32(row["hc"])
        return torch.from_numpy(x), torch.tensor([y], dtype=torch.float32)


# -----------------------------
# Model
# -----------------------------
class SimpleCNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.features(x))


def mae(pred, y):
    return torch.mean(torch.abs(pred - y))


# -----------------------------
# Train / Eval
# -----------------------------
def run_epoch(model, loader, optimizer, device, train):
    model.train() if train else model.eval()

    total_loss, total_mae, n = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = nn.functional.l1_loss(pred, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        bsz = x.size(0)
        total_loss += loss.item() * bsz
        total_mae += mae(pred, y).item() * bsz
        n += bsz

    return total_loss / n, total_mae / n


def train_model(cfg: Config):
    set_seed(cfg.seed)
    device = get_device()
    print("Device:", device)

    df_raw = load_table(cfg.csv_path)
    id_col, target_col = infer_columns(df_raw)
    df = build_paths_and_labels(df_raw, cfg.images_dir, id_col, target_col)

    train_df, val_df = train_val_split(df, cfg.val_ratio, cfg.seed)

    train_loader = DataLoader(HCDataset(train_df, cfg.img_size),
                              batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(HCDataset(val_df, cfg.img_size),
                            batch_size=cfg.batch_size, shuffle=False)

    model = SimpleCNNRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_mae = run_epoch(model, train_loader, optimizer, device, True)
        va_loss, va_mae = run_epoch(model, val_loader, optimizer, device, False)
        print(f"Epoch {epoch:02d} | train_MAE={tr_mae:.3f} | val_MAE={va_mae:.3f}")

        if va_mae < best_val:
            best_val = va_mae
            torch.save(model.state_dict(), "best_model.pt")

    print("Best val MAE:", best_val)


# -----------------------------
# Exploration
# -----------------------------
def explore(cfg: Config):
    df_raw = load_table(cfg.csv_path)
    id_col, target_col = infer_columns(df_raw)
    df = build_paths_and_labels(df_raw, cfg.images_dir, id_col, target_col)

    print("CSV columns:", list(df_raw.columns))
    print("Using id_col:", id_col, "| target_col:", target_col)
    print("Rows (after filtering missing images):", len(df))

    print("\nHC summary:")
    print(df["hc"].describe())

    if len(df) < 3:
        raise ValueError("DataFrame is empty or contains fewer than 3 rows before sampling")

    sample = df.sample(3, random_state=cfg.seed)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (_, row) in zip(axes, sample.iterrows()):
        img = Image.open(row["img_path"]).convert("L")
        ax.imshow(img, cmap="gray")
        ax.set_title(f'ID {row["image_id"]} | HC={row["hc"]:.1f}')
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["explore", "train"], default="explore")
    p.add_argument("--csv_path", default="training_set_pixel_size_and_HC.csv")
    p.add_argument("--images_dir", default="training_set")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(csv_path=args.csv_path, images_dir=args.images_dir)

    if args.mode == "explore":
        explore(cfg)
    else:
        train_model(cfg)


if __name__ == "__main__":
    main()
