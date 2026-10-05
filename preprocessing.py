"""
Preprocessing sinyal EKG untuk cabang RITME (jendela 10 detik).

PENTING: modul ini dipakai di DUA tempat dengan kode yang sama persis:
  1. Saat menyiapkan dataset (Chapman-Shaoxing, 500 Hz -> 130 Hz)
  2. Saat inferensi real-time dari Polar H10 (sudah 130 Hz)
Jangan membuat versi preprocessing kedua di tempat lain.

Urutan:
  resample -> baseline removal (cascaded median) -> DWT denoising -> z-score
"""
from math import gcd

import numpy as np
import pywt
from scipy.signal import find_peaks, medfilt, resample_poly

# ---------------------------------------------------------------------------
# Konfigurasi (disimpan juga di checkpoint model agar training = inferensi)
# ---------------------------------------------------------------------------
TARGET_FS = 130            # sampling rate Polar H10
WINDOW_SEC = 10            # panjang jendela ritme
WINDOW_LEN = TARGET_FS * WINDOW_SEC  # 1300 sampel

MEDIAN_1_MS = 200          # menghapus QRS dan gelombang P
MEDIAN_2_MS = 600          # menghapus gelombang T -> sisa = baseline wander
WAVELET = "db4"
DWT_LEVEL = 4
DWT_DENOISE_LEVELS = (1, 2)  # D1 (32,5-65 Hz) dan D2 (16,3-32,5 Hz) saja

RPEAK_MIN_RR_SEC = 0.27      # HR maksimum ~220 bpm; dipakai bersama training & real-time
RPEAK_HEIGHT_FRAC = 0.4      # ambang tinggi puncak R relatif terhadap persentil ke-99,5

# Normalisasi kanal tachogram dengan skala TETAP (bukan z-score per jendela).
# Z-score per jendela membuat std tachogram = 1 untuk SEMUA kelas, sehingga
# variasi RR +-15 ms (sinus) dan +-150 ms (AFib) terlihat sama besar oleh model.
# Dengan skala tetap, 15 ms -> 0,075 dan 150 ms -> 0,75: besarnya
# ketidakteraturan RR (dan level HR) tetap terbawa ke model.
RR_CENTER_MS = 800.0
RR_SCALE_MS = 200.0
RR_NORM = "fixed"            # "zscore" hanya untuk checkpoint lama (results_2ch)


def _odd(n: int) -> int:
    """Median filter membutuhkan panjang kernel ganjil."""
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


def resample(x: np.ndarray, fs_in: int, fs_out: int = TARGET_FS) -> np.ndarray:
    """Resampling polyphase (sudah termasuk filter anti-aliasing)."""
    if fs_in == fs_out:
        return x.astype(np.float64)
    g = gcd(fs_in, fs_out)
    return resample_poly(x, fs_out // g, fs_in // g).astype(np.float64)


def remove_baseline(x: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """Cascaded median filter 200 ms -> 600 ms, lalu sinyal - baseline."""
    k1 = _odd(MEDIAN_1_MS / 1000 * fs)   # 27 sampel pada 130 Hz
    k2 = _odd(MEDIAN_2_MS / 1000 * fs)   # 79 sampel pada 130 Hz
    baseline = medfilt(medfilt(x, k1), k2)
    return x - baseline


def dwt_denoise(x: np.ndarray) -> np.ndarray:
    """
    Soft thresholding pada koefisien detail frekuensi tinggi saja.
    D3, D4, dan A4 dibiarkan utuh agar QRS dan gelombang P tidak tumpul.
    """
    coeffs = pywt.wavedec(x, WAVELET, level=DWT_LEVEL, mode="symmetric")
    # coeffs = [A4, D4, D3, D2, D1]
    d1 = coeffs[-1]
    sigma = np.median(np.abs(d1)) / 0.6745                 # estimasi noise (MAD)
    thr = sigma * np.sqrt(2 * np.log(len(x)))               # threshold universal
    for lvl in DWT_DENOISE_LEVELS:
        idx = -lvl                                          # D1 -> -1, D2 -> -2
        coeffs[idx] = pywt.threshold(coeffs[idx], thr, mode="soft")
    y = pywt.waverec(coeffs, WAVELET, mode="symmetric")
    return y[: len(x)]


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalisasi per jendela (bukan global) agar bisa dipakai real-time."""
    return (x - x.mean()) / (x.std() + eps)


def preprocess_window(x: np.ndarray, fs_in: int) -> np.ndarray:
    """
    Pipeline lengkap untuk satu jendela 10 detik.
    Output: array float32 dengan panjang WINDOW_LEN (1300).
    """
    x = np.asarray(x, dtype=np.float64)
    x = resample(x, fs_in, TARGET_FS)
    x = remove_baseline(x, TARGET_FS)
    x = dwt_denoise(x)
    x = zscore(x)

    # Pastikan panjang tepat 1300 (potong / pad kecil jika ada selisih pembulatan)
    if len(x) >= WINDOW_LEN:
        x = x[:WINDOW_LEN]
    else:
        x = np.pad(x, (0, WINDOW_LEN - len(x)), mode="edge")
    return x.astype(np.float32)


def detect_r_peaks(xw: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """Deteksi puncak R sederhana pada jendela yang sudah di-z-score. Dipakai
    bersama oleh training (kanal tachogram) dan real-time (backend/analysis.py)
    supaya keduanya konsisten -- JANGAN duplikasi logika ini di tempat lain."""
    height = max(1.5, RPEAK_HEIGHT_FRAC * np.percentile(xw, 99.5))
    peaks, _ = find_peaks(xw, height=height, distance=int(RPEAK_MIN_RR_SEC * fs))
    return peaks


def rr_norm_of(preproc: dict | None) -> str:
    """Normalisasi tachogram yang dipakai checkpoint. Checkpoint lama tidak
    menyimpan field ini dan dilatih dengan z-score per jendela."""
    return (preproc or {}).get("rr_norm", "zscore")


def rr_tachogram(xw: np.ndarray, peaks: np.ndarray | None = None,
                 fs: int = TARGET_FS, norm: str = RR_NORM) -> np.ndarray:
    """Kanal ke-2 untuk model: periode RR sesaat (ms), diinterpolasi linear ke
    setiap sampel, selaras sampel dengan kanal EKG. Tujuannya memberi model
    akses EKSPLISIT ke keteraturan irama (ciri utama AFib), bukan hanya
    tersirat lewat morfologi gelombang.

    norm="fixed"  : (RR - RR_CENTER_MS) / RR_SCALE_MS -- amplitudo variasi RR
                    dipertahankan (default, lihat catatan di RR_CENTER_MS).
    norm="zscore" : z-score per jendela, HANYA untuk checkpoint lama."""
    if peaks is None:
        peaks = detect_r_peaks(xw, fs)
    n = len(xw)
    if len(peaks) < 2:
        return np.zeros(n, dtype=np.float32)
    rr = np.diff(peaks) / fs * 1000.0                      # ms
    centers = (peaks[:-1] + peaks[1:]) / 2.0
    tach = np.interp(np.arange(n), centers, rr, left=rr[0], right=rr[-1])
    if norm == "zscore":
        return zscore(tach).astype(np.float32)
    return ((tach - RR_CENTER_MS) / RR_SCALE_MS).astype(np.float32)


def preprocess_window_2ch(x: np.ndarray, fs_in: int) -> np.ndarray:
    """Pipeline lengkap + kanal tachogram, untuk model 2-kanal (EKG + RR).
    Output: array float32 (2, WINDOW_LEN)."""
    xw = preprocess_window(x, fs_in)
    peaks = detect_r_peaks(xw)
    tach = rr_tachogram(xw, peaks)
    return np.stack([xw, tach]).astype(np.float32)


def config_dict() -> dict:
    """Disimpan di checkpoint agar backend real-time memakai parameter yang sama."""
    return {
        "target_fs": TARGET_FS,
        "window_sec": WINDOW_SEC,
        "median_ms": [MEDIAN_1_MS, MEDIAN_2_MS],
        "wavelet": WAVELET,
        "dwt_level": DWT_LEVEL,
        "dwt_denoise_levels": list(DWT_DENOISE_LEVELS),
        "normalization": "zscore_per_window",
        "input_channels": 2,
        "channel_names": ["ecg", "rr_tachogram"],
        "rr_norm": RR_NORM,
        "rr_center_ms": RR_CENTER_MS,
        "rr_scale_ms": RR_SCALE_MS,
    }
