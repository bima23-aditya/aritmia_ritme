"""
Analisis pendukung klasifikasi: deteksi R-peak, HR, polaritas, dan SQI.

Catatan: deteksi R-peak di sini versi sederhana (find_peaks pada sinyal yang
sudah diproses). Cukup untuk HR per jendela 10 detik. Pan-Tompkins lengkap
akan dibutuhkan nanti untuk cabang denyut (PAC/PVC).
"""
import numpy as np
from scipy.stats import kurtosis

from preprocessing import TARGET_FS, detect_r_peaks, remove_baseline

HR_RANGE = (30, 220)  # rentang fisiologis
MIN_KURTOSIS = 4.0    # EKG bersih "runcing"; noise Gaussian ~3
MIN_TEMPLATE_CORR = 0.8  # kemiripan tiap QRS dengan denyut rata-rata

N_SEGMENTS = 5              # bagi jendela 10 s jadi ~2 s untuk cek burst lokal
MAX_SEGMENT_STD_RATIO = 2.0  # std segmen terburuk vs terbaik; artefak gerak melokal di waktu
MIN_SEGMENT_KURTOSIS = 4.0   # sama seperti MIN_KURTOSIS, tapi per segmen bukan rata-rata jendela


def detect_inverted(raw: np.ndarray) -> bool:
    """True jika puncak negatif jauh lebih dominan (sensor terpasang terbalik)."""
    x = remove_baseline(np.asarray(raw, dtype=np.float64), TARGET_FS)
    return -np.percentile(x, 0.5) > 1.2 * np.percentile(x, 99.5)


def analyze_rpeaks(xw: np.ndarray) -> dict:
    """xw: jendela yang sudah melalui preprocess_window (z-score)."""
    peaks = detect_r_peaks(xw, TARGET_FS)
    out = {"peaks": peaks, "heights": xw[peaks], "hr": None, "rr_cv": None,
           "rr_ms": []}
    if len(peaks) >= 2:
        rr = np.diff(peaks) / TARGET_FS
        out["rr_ms"] = np.round(rr * 1000, 1).tolist()  # interval tiap pasang R-R berurutan
    if len(peaks) >= 3:
        out["hr"] = float(60.0 / rr.mean())
        out["rr_cv"] = float(rr.std() / rr.mean())
    return out


def template_corr(xw: np.ndarray, peaks: np.ndarray, half: int = 26) -> float:
    """Rata-rata korelasi tiap denyut (+-200 ms) terhadap template denyut rata-rata.
    Aman untuk AFib: bentuk QRS tetap konsisten, yang tidak teratur hanya jaraknya."""
    segs = [xw[p - half:p + half] for p in peaks if p - half >= 0 and p + half <= len(xw)]
    if len(segs) < 3:
        return 0.0
    s = np.asarray(segs)
    tpl = s.mean(axis=0)
    return float(np.mean([np.corrcoef(seg, tpl)[0, 1] for seg in s]))


def segment_burst(xw: np.ndarray, n_segments: int = N_SEGMENTS) -> tuple[float, float]:
    """Cek burst/putus singkat (artefak gerak dada) yang tersamarkan oleh rata-rata
    seluruh jendela 10 s: bagi jadi beberapa segmen pendek dan bandingkan amplitudo
    serta kurtosis antar-segmen. EKG bersih relatif stasioner antar-segmen; satu
    segmen dengan lonjakan/putus amplitudo menandakan noise lokal, bukan AFib asli
    (yang bentuk QRS-nya tetap konsisten, hanya jaraknya yang tidak teratur)."""
    L = len(xw) // n_segments
    segs = [xw[i * L:(i + 1) * L] for i in range(n_segments)]
    stds = np.array([s.std() for s in segs])
    std_ratio = float(stds.max() / max(stds.min(), 1e-6))
    # segmen yang nyaris konstan membuat kurtosis tidak terdefinisi (NaN); itu sendiri
    # tanda segmen mati/putus, jadi perlakukan sebagai nilai terendah (gagal ambang).
    kurts = [float(kurtosis(s, fisher=False)) if s.std() > 1e-6 else 0.0 for s in segs]
    return std_ratio, min(kurts)


def signal_quality(raw_window: np.ndarray, xw: np.ndarray, rp: dict) -> tuple[str, list[str], dict]:
    """
    SQI berbasis aturan. Mengembalikan ("baik" | "buruk", alasan, nilai metrik).
    Sengaja TIDAK menolak RR yang tidak teratur, karena itu justru ciri AFib.
    Ambang dikalibrasi pada data sintetis; kalibrasi ulang dengan rekaman H10 asli.
    """
    reasons = []
    k = float(kurtosis(xw, fisher=False))
    tc = template_corr(xw, rp["peaks"])
    std_ratio, min_seg_k = segment_burst(xw)
    if np.std(raw_window) < 20:                      # satuan uV
        reasons.append("sinyal datar, kemungkinan strap tidak menempel")
    if rp["hr"] is None or not (HR_RANGE[0] <= rp["hr"] <= HR_RANGE[1]):
        reasons.append("R-peak tidak terdeteksi dengan wajar")
    if k < MIN_KURTOSIS:
        reasons.append(f"bentuk sinyal menyerupai noise (kurtosis {k:.1f})")
    if tc < MIN_TEMPLATE_CORR:
        reasons.append(f"bentuk denyut tidak konsisten, kemungkinan artefak gerak (korelasi {tc:.2f})")
    if std_ratio > MAX_SEGMENT_STD_RATIO or min_seg_k < MIN_SEGMENT_KURTOSIS:
        reasons.append("terdapat lonjakan/putus sesaat dalam jendela, kemungkinan gerakan dada "
                       f"(rasio std antar-segmen {std_ratio:.1f})")
    return ("baik" if not reasons else "buruk"), reasons, {"kurtosis": round(k, 2),
                                                           "template_corr": round(tc, 3),
                                                           "segment_std_ratio": round(std_ratio, 2),
                                                           "segment_min_kurtosis": round(min_seg_k, 2)}
