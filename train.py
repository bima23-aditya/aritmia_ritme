"""
Training + evaluasi 1D-CNN cabang ritme.

Contoh:
  python train.py --data data/rhythm_130hz.npz --epochs 40 --out results
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from model import RhythmCNN
from preprocessing import TARGET_FS, config_dict

CLASS_NAMES = {"SR": "Sinus normal", "SB": "Sinus brady",
               "ST": "Sinus tachy", "AFIB": "AFib"}


# ---------------------------------------------------------------------------
# Augmentasi
# CATATAN: JANGAN memakai time-stretch / resampling acak. Meregangkan sinyal
# mengubah denyut jantung, sehingga label SR/SB/ST bisa menjadi salah.
# Hanya kanal EKG (indeks 0) yang diberi noise/skala; kanal tachogram RR
# (indeks 1) dibiarkan apa adanya -- itu justru fitur yang ingin kita
# pertahankan keakuratannya, bukan sumber variasi untuk diperkuat modelnya.
# ---------------------------------------------------------------------------
def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = x.copy()
    ecg = x[0] * rng.uniform(0.8, 1.2)                             # skala amplitudo
    ecg = ecg + rng.normal(0, rng.uniform(0.01, 0.05), size=ecg.shape)  # noise putih
    t = np.arange(ecg.shape[-1]) / TARGET_FS                       # baseline wander sisa
    ecg = ecg + rng.uniform(0, 0.1) * np.sin(2 * np.pi * rng.uniform(0.1, 0.5) * t
                                             + rng.uniform(0, 2 * np.pi))
    x[0] = ecg
    return x.astype(np.float32)


class ECGDataset(Dataset):
    def __init__(self, X, y, train=False, seed=0):
        self.X, self.y, self.train = X, y, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.train:
            x = augment(x, self.rng)
        return torch.from_numpy(x), torch.tensor(self.y[i])


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, preds, labels = 0.0, [], []
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            preds.append(logits.argmax(1).cpu().numpy())
            labels.append(yb.cpu().numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    return total_loss / len(labels), f1_score(labels, preds, average="macro"), preds, labels


def plot_confusion(cm, names, path):
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.0%})", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label sebenarnya")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(hist, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(hist["train_loss"], label="train")
    axes[0].plot(hist["val_loss"], label="validasi")
    axes[0].set_title("Loss")
    axes[1].plot(hist["train_f1"], label="train")
    axes[1].plot(hist["val_f1"], label="validasi")
    axes[1].set_title("Macro F1")
    for a in axes:
        a.set_xlabel("Epoch")
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/rhythm_130hz.npz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = np.load(args.data, allow_pickle=False)
    X, y = data["X"], data["y"]
    classes = [str(c) for c in data["classes"]]
    names = [CLASS_NAMES.get(c, c) for c in classes]
    print(f"Data: {X.shape}, device: {device}")

    # Split 70/15/15, stratified. Di Chapman versi figshare setiap rekaman
    # berasal dari pasien berbeda, sehingga split ini sudah inter-patient.
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=args.seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=args.seed)
    np.savez(os.path.join(args.out, "split.npz"), train=tr, val=va, test=te)
    print(f"Train {len(tr)} | Val {len(va)} | Test {len(te)}")

    train_dl = DataLoader(ECGDataset(X[tr], y[tr], train=True, seed=args.seed),
                          batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(ECGDataset(X[va], y[va]), batch_size=args.batch)
    test_dl = DataLoader(ECGDataset(X[te], y[te]), batch_size=args.batch)

    # Class weight untuk menangani ketidakseimbangan (SB jauh lebih banyak)
    counts = np.bincount(y[tr], minlength=len(classes))
    weights = torch.tensor(len(tr) / (len(classes) * counts), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    model = RhythmCNN(n_classes=len(classes), in_channels=X.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.5, patience=3)

    hist = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_f1, wait = -1.0, 0
    ckpt_path = os.path.join(args.out, "rhythm_cnn_best.pt")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        trl, trf, _, _ = run_epoch(model, train_dl, device, criterion, optimizer)
        vl, vf, _, _ = run_epoch(model, val_dl, device, criterion)
        scheduler.step(vf)
        for k, v in zip(hist, (trl, vl, trf, vf)):
            hist[k].append(v)
        flag = ""
        if vf > best_f1:
            best_f1, wait, flag = vf, 0, " *"
            torch.save({"model_state": model.state_dict(), "classes": classes,
                        "class_names": names, "preproc": config_dict(),
                        "in_channels": X.shape[1],
                        "epoch": ep, "val_macro_f1": vf}, ckpt_path)
        else:
            wait += 1
        print(f"Epoch {ep:3d} | loss {trl:.4f}/{vl:.4f} | macro-F1 {trf:.3f}/{vf:.3f} "
              f"| {time.time() - t0:.1f}s{flag}")
        if wait >= args.patience:
            print(f"Early stopping (tidak membaik selama {args.patience} epoch)")
            break

    # ---------------- Evaluasi akhir pada test set ----------------
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, test_f1, preds, labels = run_epoch(model, test_dl, device, criterion)

    report = classification_report(labels, preds, target_names=names, digits=3)
    cm = confusion_matrix(labels, preds)
    print("\n=== HASIL TEST SET ===")
    print(report)
    print("Confusion matrix (baris = label, kolom = prediksi):\n", cm)

    with open(os.path.join(args.out, "test_report.txt"), "w") as f:
        f.write(f"Checkpoint terbaik: epoch {ckpt['epoch']} "
                f"(val macro-F1 {ckpt['val_macro_f1']:.3f})\n\n{report}\n{cm}\n")
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(hist, f, indent=1)
    plot_confusion(cm, names, os.path.join(args.out, "confusion_matrix.png"))
    plot_history(hist, os.path.join(args.out, "training_curve.png"))
    print(f"\nSemua hasil tersimpan di folder '{args.out}/'")


if __name__ == "__main__":
    main()
