"""
train_hrnet.py — HRNet training for star heatmap detection.

IDENTICAL conditions to U-Net training (Code/Model_train_code/train.py):
  - Same dataset / split / tiling
  - Same loss (weighted MSE, star_weight=20)
  - Same scheduler (cosine)
  - Same sigma (1.5) for heatmap targets
  - Same metrics (F1 at threshold)
  - Same optimizer (Adam, lr=1e-3)
  - Same epochs (50)

Only the model architecture changes (HRNet vs U-Net).
This allows direct apples-to-apples comparison.

Usage:
    python train.py \
        --data-dir   ~/star_tracker/Data/dataset_tess \
        --out-dir    ~/star_tracker/Results/hrnet_run1 \
        --epochs     50

Outputs:
    {out-dir}/best_model.pt         — best checkpoint by val_F1
    {out-dir}/training_curves.png   — loss / F1 curves
    {out-dir}/results.csv           — per-epoch metrics
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Reuse U-Net's dataset, loss, eval — IDENTICAL conditions
UNET_CODE = Path(__file__).parent.parent / "Model_train_code"
sys.path.insert(0, str(UNET_CODE))
from train import (
    TESSDataset,
    split_pairs,
    compute_loss,
    evaluate,
)

from hrnet_model import HRNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",     required=True)
    ap.add_argument("--out-dir",      required=True)
    ap.add_argument("--epochs",       type=int,   default=50)
    ap.add_argument("--batch-size",   type=int,   default=8)
    ap.add_argument("--workers",      type=int,   default=2)
    ap.add_argument("--sigma",        type=float, default=1.5)
    ap.add_argument("--star-weight",  type=float, default=20.0)
    ap.add_argument("--loss",         choices=["wmse", "focal"], default="wmse")
    ap.add_argument("--scheduler",    choices=["cosine", "plateau"], default="cosine")
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--base-ch",      type=int,   default=32)
    ap.add_argument("--threshold-eval", type=float, default=0.3,
                    help="Threshold for F1 during training (eval_threshold.py finds best later)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data (identical to U-Net) ----
    train_pairs, val_pairs, test_pairs = split_pairs(args.data_dir)
    print(f"Images — train: {len(train_pairs)}  val: {len(val_pairs)}  test: {len(test_pairs)}")

    train_ds = TESSDataset(train_pairs, augment=True,  sigma=args.sigma)
    val_ds   = TESSDataset(val_pairs,   augment=False, sigma=args.sigma)
    test_ds  = TESSDataset(test_pairs,  augment=False, sigma=args.sigma)
    print(f"Tiles  — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    # ---- Model (the only difference vs U-Net training) ----
    model = HRNet(base_ch=args.base_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: HRNet (base_ch={args.base_ch})")
    print(f"Model parameters: {n_params:,}")

    # ---- Optimizer & scheduler (identical to U-Net) ----
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5)

    # ---- Resume from checkpoint if exists ----
    last_ckpt = out_dir / "last_checkpoint.pt"
    best_val_f1 = -1.0
    history = []
    start_epoch = 0
    if last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt["best_val_f1"]
        history = ckpt["history"]
        print(f"\nResumed from epoch {start_epoch} (best val F1 so far: {best_val_f1:.3f})\n")

    # ---- Training loop (identical to U-Net) ----
    print(f"\n{'Epoch':>5} {'Train loss':>11} {'Val loss':>10} {'Val P':>7} "
          f"{'Val R':>7} {'Val F1':>7} {'LR':>10}")
    print("-" * 70)

    for epoch in range(start_epoch, args.epochs):
        # ---- Train ----
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for images, masks in train_dl:
            images = images.to(device)
            masks  = masks.to(device)
            preds  = model(images)
            loss = compute_loss(preds, masks, args.loss,
                                star_weight=args.star_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1
        train_loss = train_loss_sum / max(n_batches, 1)

        # ---- Validate ----
        val_metrics = evaluate(
            model, val_dl, device,
            threshold=args.threshold_eval, match_px=5.0,
        )
        val_loss = val_metrics["loss"]
        val_p    = val_metrics["precision"]
        val_r    = val_metrics["recall"]
        val_f1   = val_metrics["f1"]

        # ---- LR scheduling ----
        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_f1)

        cur_lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch+1:>5}  {train_loss:>10.5f}  {val_loss:>9.5f}  "
              f"{val_p:>6.3f}  {val_r:>6.3f}  {val_f1:>6.3f}  {cur_lr:>10.2e}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_precision": val_p,
            "val_recall":    val_r,
            "val_f1":        val_f1,
            "lr": cur_lr,
        })

        # ---- Save best ----
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), out_dir / "best_model.pt")

        # ---- Save checkpoint EVERY epoch for resume capability ----
        torch.save({
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "best_val_f1":  best_val_f1,
            "history":      history,
        }, out_dir / "last_checkpoint.pt")

    # ---- Final test evaluation ----
    print(f"\nLoading best model (val_F1={best_val_f1:.3f}) for final test eval...")
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    test_metrics = evaluate(
        model, test_dl, device, threshold=args.threshold_eval, match_px=5.0)
    print(f"Test:  loss={test_metrics['loss']:.5f}  "
          f"precision={test_metrics['precision']:.3f}  "
          f"recall={test_metrics['recall']:.3f}  "
          f"F1={test_metrics['f1']:.3f}")

    # ---- Save results ----
    import csv
    with open(out_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # ---- Plot training curves ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = [h["epoch"] for h in history]
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"]   for h in history], label="val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [h["val_precision"] for h in history], label="P")
    axes[1].plot(epochs, [h["val_recall"]    for h in history], label="R")
    axes[1].plot(epochs, [h["val_f1"]        for h in history],
                 label="F1", linewidth=2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    axes[1].set_title(f"HRNet val metrics (best F1={best_val_f1:.3f})")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=120)
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
