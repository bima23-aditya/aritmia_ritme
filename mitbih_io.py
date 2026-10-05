"""
Membaca rekaman format WFDB (MIT-BIH Arrhythmia Database: .hea, .dat, .atr).

Label ritme di MIT-BIH ada di anotasi bertanda '+' dengan aux_note seperti:
  (N     = ritme sinus normal       (AFIB = atrial fibrilasi
  (AFL   = atrial flutter           (B    = bigemini ventrikel
  (T     = trigemini ventrikel      (VT   = takikardia ventrikel
  (SBR   = sinus bradikardia        (SVTA = takikardia supraventrikular
  ... dan beberapa lainnya.
"""
from pathlib import Path

import numpy as np
import wfdb

# Ritme MIT-BIH -> label model kita. "SINUS" = SR/SB/ST (ditentukan dari HR).
RHYTHM_MAP = {"(N": "SINUS", "(SBR": "SINUS", "(AFIB": "AFIB"}


def read_mitbih(record_path: str, lead: str = "MLII"):
    """
    record_path: path TANPA ekstensi, misalnya 'data/mitdb/201'.
    Mengembalikan (ecg_uV, fs, lead_terpakai, daftar_segmen).
    Segmen: dict(start, end, rhythm) dalam satuan sampel.
    """
    record_path = str(Path(record_path).with_suffix(""))
    rec = wfdb.rdrecord(record_path)
    names = list(rec.sig_name)
    ch = names.index(lead) if lead in names else 0   # rekaman 114: MLII di kanal ke-2
    ecg = rec.p_signal[:, ch].astype(np.float64)
    if (rec.units[ch] or "").lower() == "mv":
        ecg = ecg * 1000.0                            # mV -> uV (sama dengan Polar)
    ecg = np.nan_to_num(ecg)

    ann = wfdb.rdann(record_path, "atr")
    changes = [(s, (a or "").strip("\x00 ").strip())
               for s, sym, a in zip(ann.sample, ann.symbol, ann.aux_note) if sym == "+"]
    segments = []
    for i, (start, rhythm) in enumerate(changes):
        end = changes[i + 1][0] if i + 1 < len(changes) else len(ecg)
        if end > start:
            segments.append({"start": int(start), "end": int(end), "rhythm": rhythm})
    return ecg, int(rec.fs), names[ch], segments


def describe_segments(segments, fs):
    """Ringkasan total durasi per ritme (detik)."""
    total = {}
    for s in segments:
        total[s["rhythm"]] = total.get(s["rhythm"], 0) + (s["end"] - s["start"]) / fs
    return dict(sorted(total.items(), key=lambda kv: -kv[1]))
