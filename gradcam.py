"""
Grad-CAM untuk 1D-CNN.

Dipakai dua cara:
  1. Sebagai library (backend real-time):
       cam = GradCAM1D(model)
       heatmap, pred, prob = cam(x)   # x: np.ndarray (1300,)
  2. Sebagai skrip untuk melihat contoh dari test set:
       python gradcam.py --data data/rhythm_130hz.npz --ckpt results/rhythm_cnn_best.pt
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from model import RhythmCNN
from preprocessing import TARGET_FS


class GradCAM1D:
    def __init__(self, model: RhythmCNN):
        self.model = model.eval()
        self._acts, self._grads = None, None
        layer = model.target_layer
        layer.register_forward_hook(lambda m, i, o: setattr(self, "_acts", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "_grads", go[0]))

    def __call__(self, x: np.ndarray, target: int | None = None):
        """
        x: (n_channels, window_len) -- lihat preprocessing.preprocess_window_2ch.
        Juga menerima array 1-D (window_len,) untuk checkpoint lama 1-kanal.
        Mengembalikan (heatmap[0..1] sepanjang input, kelas prediksi, probabilitas).
        target=None -> jelaskan kelas yang diprediksi.
        """
        device = next(self.model.parameters()).device
        xt = torch.as_tensor(x, dtype=torch.float32, device=device)
        xt = xt.unsqueeze(0) if xt.ndim == 2 else xt.view(1, 1, -1)
        logits = self.model(xt)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        cls = int(probs.argmax()) if target is None else target

        self.model.zero_grad()
        logits[0, cls].backward()

        weights = self._grads.mean(dim=2, keepdim=True)          # (1, C, 1)
        cam = F.relu((weights * self._acts).sum(dim=1, keepdim=True))  # (1, 1, 81)
        cam = F.interpolate(cam, size=xt.shape[-1], mode="linear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
        return cam, cls, probs


def load_model(ckpt_path: str, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    # in_channels: pakai yang tersimpan di checkpoint; untuk checkpoint lama (sebelum
    # kanal tachogram ditambahkan) yang tidak punya field ini, simpulkan dari bentuk
    # bobot Conv1d pertama supaya tetap bisa dimuat (model lama = 1 kanal EKG saja).
    in_channels = ckpt.get("in_channels")
    if in_channels is None:
        in_channels = ckpt["model_state"]["features.0.0.weight"].shape[1]
    model = RhythmCNN(n_classes=len(ckpt["classes"]), in_channels=in_channels)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


def plot_examples(data_path, ckpt_path, split_path, out_path, per_class=2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model, ckpt = load_model(ckpt_path)
    names = ckpt["class_names"]
    data = np.load(data_path)
    X, y = data["X"], data["y"]
    test_idx = np.load(split_path)["test"]
    cam = GradCAM1D(model)
    rng = np.random.default_rng(0)

    rows = []
    for c in range(len(names)):
        cand = test_idx[y[test_idx] == c]
        rows += list(rng.choice(cand, size=min(per_class, len(cand)), replace=False))

    t = np.arange(X.shape[-1]) / TARGET_FS
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 1.9 * len(rows)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, rows):
        ecg = X[i, 0] if X.ndim == 3 else X[i]  # kanal EKG saja untuk ditampilkan
        heat, pred, prob = cam(X[i])
        ax.imshow(heat[None, :], aspect="auto", cmap="Reds", alpha=0.55, vmin=0, vmax=1,
                  extent=[t[0], t[-1], ecg.min() - 0.5, ecg.max() + 0.5])
        ax.plot(t, ecg, color="black", lw=0.8)
        ok = "benar" if pred == y[i] else "SALAH"
        ax.set_title(f"Label: {names[y[i]]}  |  Prediksi: {names[pred]} "
                     f"({prob[pred]:.0%}, {ok})", fontsize=10, loc="left")
        ax.set_yticks([])
    axes[-1].set_xlabel("Waktu (detik)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Contoh Grad-CAM tersimpan: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/rhythm_130hz.npz")
    ap.add_argument("--ckpt", default="results/rhythm_cnn_best.pt")
    ap.add_argument("--split", default="results/split.npz")
    ap.add_argument("--out", default="results/gradcam_examples.png")
    ap.add_argument("--per_class", type=int, default=2)
    a = ap.parse_args()
    plot_examples(a.data, a.ckpt, a.split, a.out, a.per_class)
