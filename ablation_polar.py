"""
Ablation kanal untuk model 2-kanal (EKG + tachogram RR) pada rekaman Polar.

Untuk setiap jendela 10 detik, model dijalankan 3 kali:
  asli    : [EKG, tachogram]
  rr_flat : tachogram diganti RR KONSTAN (= rata-rata RR jendela itu, artinya
            ritme teratur sempurna dengan HR yang sama) -- seberapa besar
            prediksi bergantung pada ketidakteraturan RR?
  ecg_off : kanal EKG di-nol-kan, hanya tachogram -- apa yang bisa diputuskan
            model dari RR saja?

Contoh:
  python ablation_polar.py --file ecg_main_mobile-legends2.csv \
      --ckpt results_2ch_rrfix/rhythm_cnn_best.pt --out results_polar_rrfix
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from gradcam import load_model
from preprocessing import (TARGET_FS, WINDOW_LEN, preprocess_window, rr_norm_of,
                           rr_tachogram)
from test_polar import detect_inverted, estimate_hr, load_polar, to_uniform_130


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="results_polar")
    ap.add_argument("--step", type=float, default=5.0)
    ap.add_argument("--skip_start", type=float, default=5.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ecg, t, fs_est, _ = load_polar(args.file)
    ecg = to_uniform_130(ecg, t, fs_est)[int(args.skip_start * TARGET_FS):]
    if detect_inverted(ecg):
        ecg = -ecg

    model, ck = load_model(args.ckpt)
    classes = ck["classes"]
    norm = rr_norm_of(ck.get("preproc"))
    print(f"[i] Checkpoint: {args.ckpt} | normalisasi tachogram: {norm}")

    rows, variants = [], {"asli": [], "rr_flat": [], "ecg_off": []}
    step = int(args.step * TARGET_FS)
    for s in range(0, len(ecg) - WINDOW_LEN + 1, step):
        xw = preprocess_window(ecg[s:s + WINDOW_LEN], TARGET_FS)
        hr, peaks, cv = estimate_hr(xw)
        tach = rr_tachogram(xw, peaks, norm=norm)
        # RR konstan: z-score dari konstanta = 0; skala tetap = rata-rata tachogram
        flat = np.full_like(tach, 0.0 if norm == "zscore" else tach.mean())
        variants["asli"].append(np.stack([xw, tach]))
        variants["rr_flat"].append(np.stack([xw, flat]))
        variants["ecg_off"].append(np.stack([np.zeros_like(xw), tach]))
        rows.append({"start_s": round(s / TARGET_FS + args.skip_start, 1),
                     "hr_bpm": round(hr, 1), "rr_cv": round(cv, 3)})

    df = pd.DataFrame(rows)
    with torch.no_grad():
        for name, xs in variants.items():
            prob = torch.softmax(model(torch.from_numpy(np.stack(xs))), 1).numpy()
            df[f"pred_{name}"] = [classes[i] for i in prob.argmax(1)]
            df[f"pAFIB_{name}"] = prob[:, classes.index("AFIB")].round(3)

    tag = os.path.splitext(os.path.basename(os.path.dirname(args.ckpt) or args.ckpt))[0]
    path = os.path.join(args.out, f"ablation_{tag}.csv")
    df.to_csv(path, index=False)

    print(f"\n=== {len(df)} jendela | RR-CV median {df.rr_cv.median():.3f} ===")
    for name in variants:
        vc = df[f"pred_{name}"].value_counts()
        print(f"  {name:8s}: " + ", ".join(f"{c} {vc.get(c, 0)}" for c in classes)
              + f" | mean p_AFIB {df[f'pAFIB_{name}'].mean():.3f}")
    d = (df.pAFIB_asli - df.pAFIB_rr_flat).abs()
    print(f"  |p_AFIB asli - rr_flat| rata-rata {d.mean():.3f}, maks {d.max():.3f}")
    print(f"Tersimpan: {path}")


if __name__ == "__main__":
    main()
