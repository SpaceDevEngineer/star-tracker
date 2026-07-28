"""
train.py — Star detection training pipeline for TESS star tracker.

Pipeline:
    PNG images + JSON labels  →  512×512 tiles  →  Gaussian heatmaps
    U-Net  →  predicted heatmap  →  centroids  →  metrics

Usage:
    python train.py --data-dir /path/to/dataset_tess_full --epochs 50
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from scipy import ndimage
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TILE_SIZE    = 512
HEATMAP_SIGMA = 1.5   # sharper Gaussian peaks → model learns confident detections
MIN_STARS    = 3      # skip tiles with fewer stars than this
STAR_WEIGHT  = 20.0   # lower weight → better precision/recall balance

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
class TESSDataset(Dataset):
    """
    Reads PNG + JSON label pairs, tiles each image into TILE_SIZE×TILE_SIZE
    patches, and generates a Gaussian heatmap from star pixel coordinates.

    Split by IMAGE (not by tile) is done outside this class to prevent
    data leakage between train/val/test.
    """

    def __init__(self, pairs, tile_size=TILE_SIZE, sigma=HEATMAP_SIGMA,
                 min_stars=MIN_STARS, augment=False):
        self.tile_size = tile_size
        self.sigma     = sigma
        self.augment   = augment

        self.tiles = []  # (img_path, [(x,y), ...] local coords, row, col)
        for img_path, lbl_path in pairs:
            with open(lbl_path) as f:
                label = json.load(f)

            h, w = label["image_shape"]  # [H, W]
            stars = label["stars"]       # [{x, y, magnitude, hipparcos_id}, ...]

            for row in range(0, h - tile_size + 1, tile_size):
                for col in range(0, w - tile_size + 1, tile_size):
                    local = [
                        (s["x"] - col, s["y"] - row)
                        for s in stars
                        if col <= s["x"] < col + tile_size
                        and row <= s["y"] < row + tile_size
                    ]
                    if len(local) >= min_stars:
                        self.tiles.append((img_path, local, row, col))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        img_path, local_stars, row, col = self.tiles[idx]

        img = np.array(Image.open(img_path), dtype=np.float32) / 255.0
        tile = img[row:row + self.tile_size, col:col + self.tile_size]

        heatmap = self._make_heatmap(local_stars)

        if self.augment:
            tile, heatmap = self._augment(tile, heatmap)

        return (
            torch.tensor(tile[None]),     # (1, H, W)
            torch.tensor(heatmap[None]),  # (1, H, W)
        )

    def _make_heatmap(self, local_stars):
        hm = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
        for x, y in local_stars:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < self.tile_size and 0 <= yi < self.tile_size:
                hm[yi, xi] = 1.0
        hm = ndimage.gaussian_filter(hm, sigma=self.sigma)
        if hm.max() > 0:
            hm /= hm.max()
        return hm

    def _augment(self, tile, heatmap):
        if np.random.rand() > 0.5:
            tile    = np.fliplr(tile).copy()
            heatmap = np.fliplr(heatmap).copy()
        if np.random.rand() > 0.5:
            tile    = np.flipud(tile).copy()
            heatmap = np.flipud(heatmap).copy()
        k = np.random.randint(0, 4)
        tile    = np.rot90(tile, k).copy()
        heatmap = np.rot90(heatmap, k).copy()
        return tile, heatmap


def split_pairs(data_dir, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Returns (train_pairs, val_pairs, test_pairs).
    Split is done by IMAGE so tiles from the same image never appear in two splits.
    """
    images_dir = Path(data_dir) / "images"
    labels_dir = Path(data_dir) / "labels"

    images = sorted(images_dir.glob("*.png"))
    pairs  = [
        (img, labels_dir / (img.stem + ".json"))
        for img in images
        if (labels_dir / (img.stem + ".json")).exists()
    ]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))

    n_test = int(len(pairs) * test_ratio)
    n_val  = int(len(pairs) * val_ratio)

    test_pairs  = [pairs[i] for i in idx[:n_test]]
    val_pairs   = [pairs[i] for i in idx[n_test:n_test + n_val]]
    train_pairs = [pairs[i] for i in idx[n_test + n_val:]]

    return train_pairs, val_pairs, test_pairs


# ---------------------------------------------------------------------------
# 2. Model — 4-level U-Net with skip connections
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()
        c = base_ch
        self.enc1 = DoubleConv(1,    c)
        self.enc2 = DoubleConv(c,    c*2)
        self.enc3 = DoubleConv(c*2,  c*4)
        self.enc4 = DoubleConv(c*4,  c*8)
        self.bot  = DoubleConv(c*8,  c*16)

        self.pool = nn.MaxPool2d(2)

        self.up4  = nn.ConvTranspose2d(c*16, c*8,  2, stride=2)
        self.dec4 = DoubleConv(c*16, c*8)

        self.up3  = nn.ConvTranspose2d(c*8,  c*4,  2, stride=2)
        self.dec3 = DoubleConv(c*8,  c*4)

        self.up2  = nn.ConvTranspose2d(c*4,  c*2,  2, stride=2)
        self.dec2 = DoubleConv(c*4,  c*2)

        self.up1  = nn.ConvTranspose2d(c*2,  c,    2, stride=2)
        self.dec1 = DoubleConv(c*2,  c)

        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bot(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.head(d1))


# ---------------------------------------------------------------------------
# 3. Loss
# ---------------------------------------------------------------------------
def weighted_mse(pred, target, star_weight=STAR_WEIGHT):
    w = torch.ones_like(target)
    w[target > 0.05] = star_weight
    return (w * (pred - target) ** 2).mean()


def focal_heatmap_loss(pred, target, alpha=2.0, beta=4.0, pos_threshold=0.9, pos_weight=50.0):
    """
    CenterNet-style focal loss adapted for soft Gaussian heatmap targets.
    pos_weight balances pos vs neg contribution (star pixels are <0.01% of image).
    """
    pos_mask   = (target >= pos_threshold).float()
    neg_weight = (1.0 - target).clamp(0.0, 1.0) ** beta

    pos_loss = -torch.log(pred.clamp(min=1e-6))    * (1 - pred) ** alpha * pos_mask
    neg_loss = -torch.log((1-pred).clamp(min=1e-6)) * pred ** alpha * neg_weight * (1 - pos_mask)

    n_pos = pos_mask.sum().clamp(min=1)
    n_neg = (1 - pos_mask).sum().clamp(min=1)
    # normalise pos and neg separately, then balance with pos_weight
    return pos_loss.sum() / n_pos * pos_weight + neg_loss.sum() / n_neg


def compute_loss(pred, target, loss_type, star_weight):
    if loss_type == "focal":
        return focal_heatmap_loss(pred, target)
    return weighted_mse(pred, target, star_weight)


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------
def extract_centroids(heatmap_np, threshold=0.3, min_area=3):
    """(H,W) numpy → (N,2) xy centroids."""
    binary = heatmap_np > threshold
    if not binary.any():
        return np.zeros((0, 2))
    labeled, n = ndimage.label(binary)
    if n == 0:
        return np.zeros((0, 2))
    sizes = np.bincount(labeled.ravel())
    keep  = np.where(sizes >= min_area)[0]
    keep  = keep[keep != 0]
    if len(keep) == 0:
        return np.zeros((0, 2))
    yx = ndimage.center_of_mass(heatmap_np, labeled, keep.tolist())
    return np.array([(cx, cy) for (cy, cx) in yx], dtype=np.float32)


def match_centroids(pred_xy, gt_xy, threshold_px=5.0):
    """Greedy nearest-neighbour matching. Returns (n_matched, n_gt, n_pred)."""
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return 0, len(gt_xy), len(pred_xy)
    dist = cdist(pred_xy, gt_xy)
    n_matched = 0
    for _ in range(min(len(pred_xy), len(gt_xy))):
        if dist.min() > threshold_px:
            break
        pi, gi = np.unravel_index(dist.argmin(), dist.shape)
        n_matched += 1
        dist[pi, :] = np.inf
        dist[:, gi] = np.inf
    return n_matched, len(gt_xy), len(pred_xy)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.3, match_px=5.0):
    """Run model on a DataLoader, return loss + centroid precision/recall/F1."""
    model.eval()
    total_loss = matched = gt_total = pred_total = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        total_loss += weighted_mse(preds, masks).item()

        for pred, gt in zip(preds.cpu().numpy(), masks.cpu().numpy()):
            pred_c = extract_centroids(pred.squeeze(), threshold)
            gt_c   = extract_centroids(gt.squeeze(),   threshold)
            m, ng, np_ = match_centroids(pred_c, gt_c, match_px)
            matched    += m
            gt_total   += ng
            pred_total += np_

    n = len(loader)
    precision = matched / max(pred_total, 1)
    recall    = matched / max(gt_total,   1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-6)

    return {
        "loss":      total_loss / n,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }


# ---------------------------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_pairs, val_pairs, test_pairs = split_pairs(args.data_dir)
    print(f"Images — train: {len(train_pairs)}  val: {len(val_pairs)}  test: {len(test_pairs)}")

    train_ds = TESSDataset(train_pairs, augment=True,  sigma=args.sigma)
    val_ds   = TESSDataset(val_pairs,   augment=False, sigma=args.sigma)
    test_ds  = TESSDataset(test_pairs,  augment=False, sigma=args.sigma)
    print(f"Tiles  — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    model     = UNet(base_ch=args.base_ch).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_val_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for images, masks in train_dl:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = compute_loss(model(images), masks, args.loss, args.star_weight)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # --- validate ---
        val_metrics = evaluate(model, val_dl, device)
        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_metrics["f1"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"precision={val_metrics['precision']:.3f}  "
            f"recall={val_metrics['recall']:.3f}  "
            f"F1={val_metrics['f1']:.3f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"  ✓ saved best model (val_f1={best_val_f1:.4f})")

    # --- test ---
    print("\n--- Test set evaluation ---")
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    test_metrics = evaluate(model, test_dl, device)
    print(
        f"test_loss={test_metrics['loss']:.4f}  "
        f"precision={test_metrics['precision']:.3f}  "
        f"recall={test_metrics['recall']:.3f}  "
        f"F1={test_metrics['f1']:.3f}"
    )

    _save_plots(history, out_dir)
    _save_samples(model, test_dl, device, out_dir)

    return test_metrics


# ---------------------------------------------------------------------------
# 6. Plots & sample visualisation
# ---------------------------------------------------------------------------
def _save_plots(history, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["val_f1"], label="val F1", color="green")
    axes[1].set_title("Validation F1 (centroid detection)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=120)
    plt.close()
    print(f"Saved training curves → {out_dir / 'training_curves.png'}")


@torch.no_grad()
def _save_samples(model, loader, device, out_dir, n_samples=4):
    model.eval()
    images, masks = next(iter(loader))
    images = images.to(device)
    preds  = model(images).cpu()

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples * 4))
    for i in range(min(n_samples, len(images))):
        img_np  = images[i].cpu().squeeze().numpy()
        pred_np = preds[i].squeeze().numpy()
        mask_np = masks[i].squeeze().numpy()

        centroids = extract_centroids(pred_np)

        axes[i, 0].imshow(img_np,  cmap="gray"); axes[i, 0].set_title("Input")
        axes[i, 1].imshow(pred_np, cmap="hot");  axes[i, 1].set_title(f"Predicted ({len(centroids)} stars)")
        axes[i, 2].imshow(mask_np, cmap="hot");  axes[i, 2].set_title("Ground truth")

        if len(centroids):
            axes[i, 1].scatter(centroids[:, 0], centroids[:, 1], s=15, c="cyan", marker="+")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "sample_predictions.png", dpi=120)
    plt.close()
    print(f"Saved sample predictions → {out_dir / 'sample_predictions.png'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   required=True,
                   help="Folder with images/ and labels/ sub-dirs.")
    p.add_argument("--out-dir",    default="Results/unet_run1",
                   help="Where to save model weights and plots.")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch-size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--base-ch",    type=int,   default=32,
                   help="U-Net base channels (32 = ~7M params, 16 = ~1.7M).")
    p.add_argument("--workers",    type=int,   default=4)
    p.add_argument("--sigma",      type=float, default=HEATMAP_SIGMA,
                   help="Gaussian sigma for heatmap targets.")
    p.add_argument("--star-weight",type=float, default=STAR_WEIGHT,
                   help="Weight for star pixels in weighted MSE loss.")
    p.add_argument("--loss",       default="wmse", choices=["wmse", "focal"],
                   help="Loss function: wmse (weighted MSE) or focal (CenterNet-style).")
    p.add_argument("--scheduler",  default="plateau", choices=["plateau", "cosine"],
                   help="LR scheduler: plateau (ReduceLROnPlateau) or cosine (CosineAnnealing).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
