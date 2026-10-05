"""Membaca file rekaman Polar H10 (Polar Sensor Logger atau CSV umum)."""
import numpy as np
import pandas as pd

from preprocessing import TARGET_FS


def load_polar(path: str):
    """Mengembalikan (ecg_uV, waktu_detik atau None, fs_terukur atau None, nama_kolom)."""
    df = pd.read_csv(path, sep=None, engine="python")
    cols = {c.lower(): c for c in df.columns}

    ecg_col = next((cols[c] for c in cols if "ecg" in c), None)
    if ecg_col is None:
        ecg_col = df.select_dtypes("number").columns[-1]
        print(f"[!] Kolom 'ecg' tidak ditemukan, memakai kolom '{ecg_col}'")
    ecg = df[ecg_col].to_numpy(dtype=np.float64)

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


def load_polar_130(path: str) -> np.ndarray:
    """Praktis: baca file dan pastikan hasilnya 130 Hz."""
    ecg, t, fs_est, _ = load_polar(path)
    return to_uniform_130(ecg, t, fs_est)
