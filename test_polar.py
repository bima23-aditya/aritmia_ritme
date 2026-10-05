"""
Uji model ritme pada rekaman Polar H10.

Mendukung:
  - Polar Sensor Logger (.txt/.csv, pemisah ';', kolom "ecg [uV]")
  - CSV/TXT umum dengan satu kolom EKG (dengan atau tanpa kolom waktu)

Contoh:
  python test_polar.py --file rekaman_polar.txt --ckpt results/rhythm_cnn_best.pt
  python test_polar.py --file rekaman_polar.txt --ckpt results/rhythm_cnn_best.pt --polarity invert

Output (folder --out):
  polar_windows.csv      hasil per jendela 10 detik
  polar_timeline.png     label, confidence, dan HR sepanjang rekaman
  polar_gradcam.png      contoh jendela dengan heatmap Grad-CAM
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gradcam import GradCAM1D, load_model
from preprocessing import (TARGET_FS, WINDOW_LEN, detect_r_peaks, preprocess_window,
                           remove_baseline, rr_norm_of, rr_tachogram)

COLORS = ["#378ADD", "#1D9E75", "#EF9F27", "#E24B4A"]  # SR, SB, ST, AFIB


# ---------------------------------------------------------------------------
# 1. Membaca file
# ---------------------------------------------------------------------------
def load_polar(path: str):
    df = pd.read_csv(path, sep=None, engine="python")
    cols = {c.lower(): c for c in df.columns}

    ecg_col = next((cols[c] for c in cols if "ecg" in c), None)
    if ecg_col is None:
        num = df.select_dtypes("number").columns
        ecg_col = num[-1]
        print(f"[!] Kolom 'ecg' tidak ditemukan, memakai kolom '{ecg_col}'")
    ecg = df[ecg_col].to_numpy(dtype=np.float64)

    # Cari kolom waktu untuk mengestimasi sampling rate sebenarnya
    t = None
    for key, scale in (("sensor timestamp", 1e-9), ("[ns]", 1e-9), ("[ms]", 1e-3)):
        c = next((cols[k] for k in cols if key in k), None)
        if c is not None and pd.api.types.is_numeric_dtype(df[c]):
            t = (df[c].to_numpy(dtype=np.float64) - df[c].iloc[0]) * scale
            break

    mask = np.isfinite(ecg)
    ecg = ecg[mask]
    t = t[mask] if t is not None else None
    fs_est = (len(ecg) - 1) / (t[-1] - t[0]) if t is not None and t[-1] > t[0] else None
    return ecg, t, fs_est, ecg_col


def to_uniform_130(ecg, t, fs_est):
    """Jika fs meleset > 0,5 Hz, interpolasi ke grid 130 Hz berdasarkan timestamp."""
    if t is None or fs_est is None or abs(fs_est - TARGET_FS) <= 0.5:
        return ecg
    print(f"[i] Sampling rate terukur {fs_est:.2f} Hz -> diinterpolasi ke {TARGET_FS} Hz")
    grid = np.arange(0, t[-1], 1 / TARGET_FS)
    return np.interp(grid, t, ecg)


# ---------------------------------------------------------------------------
# 2. Polaritas dan HR independen
# ---------------------------------------------------------------------------
def detect_inverted(ecg) -> bool:
    """Jika puncak negatif jauh lebih dominan dari positif, sinyal kemungkinan terbalik."""
    x = remove_baseline(ecg, TARGET_FS)
    return -np.percentile(x, 0.5) > 1.2 * np.percentile(x, 99.5)


def estimate_hr(xw):
    """HR sederhana dari R-peak pada jendela yang sudah diproses (z-score).
    Pakai detektor yang SAMA dengan training/backend (preprocessing.detect_r_peaks)
    supaya kanal tachogram yang diberikan ke model konsisten."""
    peaks = detect_r_peaks(xw, TARGET_FS)
    if len(peaks) < 3:
        return np.nan, peaks, np.nan
    rr = np.diff(peaks) / TARGET_FS
    cv = rr.std() / rr.mean()          # koefisien variasi RR (ketidakteraturan)
    return 60.0 / rr.mean(), peaks, cv


def hr_expected(hr):
    if not np.isfinite(hr):
        return "?"
    return "SB" if hr < 60 else "ST" if hr > 100 else "SR"


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--ckpt", default="results/rhythm_cnn_best.pt")
    ap.add_argument("--out", default="results_polar")
    ap.add_argument("--step", type=float, default=5.0, help="geser jendela (detik)")
    ap.add_argument("--skip_start", type=float, default=5.0,
                    help="abaikan detik awal (sensor biasanya belum stabil)")
    ap.add_argument("--polarity", choices=["auto", "normal", "invert"], default="auto")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # --- Baca data ---
    ecg, t, fs_est, col = load_polar(args.file)
    print(f"[i] Kolom EKG: '{col}', {len(ecg)} sampel"
          + (f", fs terukur {fs_est:.2f} Hz" if fs_est else ", fs diasumsikan 130 Hz"))
    ecg = to_uniform_130(ecg, t, fs_est)
    ecg = ecg[int(args.skip_start * TARGET_FS):]
    dur = len(ecg) / TARGET_FS
    print(f"[i] Durasi dianalisis: {dur:.1f} detik")
    if len(ecg) < WINDOW_LEN:
        raise SystemExit("Rekaman kurang dari 10 detik setelah skip_start.")

    inverted = detect_inverted(ecg) if args.polarity == "auto" else args.polarity == "invert"
    if inverted:
        ecg = -ecg
    print(f"[i] Polaritas: {'DIBALIK' if inverted else 'normal'} (mode {args.polarity})")

    # --- Model ---
    model, ck = load_model(args.ckpt)
    classes, names = ck["classes"], ck["class_names"]
    cam = GradCAM1D(model)

    # --- Jendela geser ---
    step = int(args.step * TARGET_FS)
    rows, cache = [], []
    for s in range(0, len(ecg) - WINDOW_LEN + 1, step):
        xw = preprocess_window(ecg[s:s + WINDOW_LEN], TARGET_FS)
        hr, peaks, cv = estimate_hr(xw)
        tach = rr_tachogram(xw, peaks, norm=rr_norm_of(ck.get("preproc")))
        heat, pred, prob = cam(np.stack([xw, tach]))
        exp = hr_expected(hr)
        rows.append({
            "start_s": round(s / TARGET_FS + args.skip_start, 1),
            "pred": classes[pred], "conf": round(float(prob[pred]), 3),
            **{f"p_{c}": round(float(p), 3) for c, p in zip(classes, prob)},
            "hr_bpm": round(hr, 1) if np.isfinite(hr) else None,
            "rr_cv": round(cv, 3) if np.isfinite(cv) else None,
            "hr_rule": exp,
            "agree": (exp == classes[pred]) if classes[pred] != "AFIB" else None,
        })
        cache.append((xw, heat, pred, prob, peaks))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "polar_windows.csv"), index=False)

    # --- Ringkasan ---
    print(f"\n=== {len(df)} jendela (10 s, geser {args.step:g} s) ===")
    for c, n in zip(classes, names):
        k = (df["pred"] == c).sum()
        print(f"  {n:14s}: {k:4d} jendela ({k / len(df):.0%})")
    print(f"  Confidence rata-rata: {df['conf'].mean():.2f}")
    if df["hr_bpm"].notna().any():
        print(f"  HR (deteksi R-peak): {df['hr_bpm'].min():.0f}-{df['hr_bpm'].max():.0f} bpm, "
              f"median {df['hr_bpm'].median():.0f}")
    sub = df[df["agree"].notna()]
    if len(sub):
        print(f"  Kesesuaian model vs aturan HR (non-AFib): {sub['agree'].mean():.0%}")
    n_af = (df["pred"] == "AFIB").sum()
    if n_af:
        print(f"  [!] {n_af} jendela diprediksi AFib -> periksa rr_cv di CSV "
              f"(sinus biasanya < 0,10) dan lihat gambar Grad-CAM")

    plot_timeline(df, classes, names, os.path.join(args.out, "polar_timeline.png"))
    plot_gradcam(df, cache, names, os.path.join(args.out, "polar_gradcam.png"))
    print(f"\nHasil tersimpan di '{args.out}/'")


# ---------------------------------------------------------------------------
# 4. Visualisasi
# ---------------------------------------------------------------------------
def plot_timeline(df, classes, names, path):
    tc = df["start_s"] + 5  # tengah jendela
    fig, ax = plt.subplots(3, 1, figsize=(12, 6.5), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1.2, 1.2]})
    for i, c in enumerate(classes):
        m = df["pred"] == c
        ax[0].scatter(tc[m], np.full(m.sum(), i), c=COLORS[i], s=40, label=names[i])
    ax[0].set_yticks(range(len(names)), names)
    ax[0].set_title("Prediksi per jendela", loc="left", fontsize=10)
    ax[0].set_ylim(-0.5, len(names) - 0.5)

    ax[1].plot(tc, df["conf"], "-o", ms=3, color="#534AB7")
    ax[1].axhline(0.8, ls="--", lw=0.8, color="gray")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("Confidence (garis putus = 0,8)", loc="left", fontsize=10)

    ax[2].plot(tc, df["hr_bpm"], "-o", ms=3, color="#D85A30")
    for y in (60, 100):
        ax[2].axhline(y, ls="--", lw=0.8, color="gray")
    ax[2].set_title("HR dari deteksi R-peak (garis putus = 60 dan 100 bpm)",
                    loc="left", fontsize=10)
    ax[2].set_xlabel("Waktu (detik)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_gradcam(df, cache, names, path, max_n=6):
    # Pilih: jendela paling yakin per kelas + jendela paling tidak yakin
    chosen = []
    for c in df["pred"].unique():
        chosen.append(df[df["pred"] == c]["conf"].idxmax())
    chosen += list(df["conf"].nsmallest(max_n).index)
    chosen = list(dict.fromkeys(chosen))[:max_n]

    t = np.arange(WINDOW_LEN) / TARGET_FS
    fig, axes = plt.subplots(len(chosen), 1, figsize=(12, 2.2 * len(chosen)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, chosen):
        xw, heat, pred, prob, peaks = cache[i]
        r = df.loc[i]
        top = xw.max() + 1.2  # ruang ekstra untuk label interval RR
        ax.imshow(heat[None, :], aspect="auto", cmap="Reds", alpha=0.55, vmin=0, vmax=1,
                  extent=[t[0], t[-1], xw.min() - 0.5, top])
        ax.plot(t, xw, color="black", lw=0.8)
        ax.plot(t[peaks], xw[peaks], "v", color="#185FA5", ms=4)
        label_y = xw.max() + 0.6  # baris sejajar untuk semua label, tidak mengikuti tinggi denyut
        for a, b in zip(peaks[:-1], peaks[1:]):
            rr_ms = (b - a) / TARGET_FS * 1000
            ax.text((t[a] + t[b]) / 2, label_y, f"{rr_ms:.0f} ms",
                    fontsize=6.5, ha="center", color="#534AB7")
        ax.set_ylim(xw.min() - 0.5, top)
        hr_txt = f"{r['hr_bpm']:.0f} bpm" if pd.notna(r["hr_bpm"]) else "HR ?"
        ax.set_title(f"t={r['start_s']:.0f}s  |  Prediksi: {names[pred]} ({prob[pred]:.0%})"
                     f"  |  {hr_txt}, RR-CV {r['rr_cv']}", fontsize=10, loc="left")
        ax.set_yticks([])
    axes[-1].set_xlabel("Waktu dalam jendela (detik)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
