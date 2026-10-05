"""
Menyiapkan dataset Chapman-Shaoxing (versi figshare, 10.646 pasien) untuk
cabang ritme: Sinus Normal (SR), Sinus Bradikardia (SB), Sinus Takikardia (ST),
dan Atrial Fibrilasi (AFIB).

Struktur folder yang diharapkan (hasil unduhan dari figshare):
  data/chapman/
    Diagnostics.xlsx
    ECGData/               <- folder hasil ekstrak ECGData.zip (sinyal MENTAH)
      MUSE_20180111_155115_19000.csv
      ...

Kita memakai sinyal MENTAH (ECGData), bukan ECGDataDenoised, karena sinyal
Polar H10 juga mentah dan harus melewati preprocessing yang sama.

Contoh pemakaian:
  python prepare_chapman.py --data_dir data/chapman --lead II --out data/rhythm_130hz.npz
"""
import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing import WINDOW_LEN, config_dict, preprocess_window_2ch

FS_CHAPMAN = 500
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Label asli Chapman -> label kita. Ritme lain (AF/flutter, SVT, SA, dll) diabaikan.
LABEL_MAP = {"SR": "Sinus normal", "SB": "Sinus bradikardia",
             "ST": "Sinus takikardia", "AFIB": "Atrial fibrilasi"}
CLASS_ORDER = ["SR", "SB", "ST", "AFIB"]


def read_lead(csv_path: str, lead: str) -> np.ndarray:
    """Membaca satu lead dari CSV Chapman, tahan terhadap ada/tidaknya header."""
    with open(csv_path, "r") as f:
        first = f.readline()
    has_header = any(c.isalpha() for c in first)
    df = pd.read_csv(csv_path, header=0 if has_header else None)
    if has_header and lead in df.columns:
        return df[lead].to_numpy(dtype=np.float64)
    return df.iloc[:, LEADS.index(lead)].to_numpy(dtype=np.float64)


def is_valid(x: np.ndarray) -> bool:
    """Tolak rekaman kosong, flat, atau berisi NaN (ada beberapa di Chapman)."""
    if len(x) < FS_CHAPMAN * 9 or not np.all(np.isfinite(x)):
        return False
    return np.std(x) > 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/chapman")
    ap.add_argument("--signal_dir", default="ECGData",
                    help="subfolder sinyal mentah di dalam data_dir")
    ap.add_argument("--lead", default="II", choices=LEADS)
    ap.add_argument("--out", default="data/rhythm_130hz.npz")
    args = ap.parse_args()

    diag = pd.read_excel(os.path.join(args.data_dir, "Diagnostics.xlsx"))
    diag = diag[diag["Rhythm"].isin(CLASS_ORDER)].reset_index(drop=True)
    print("Jumlah rekaman per kelas (sebelum validasi):")
    print(diag["Rhythm"].value_counts().to_string(), "\n")

    X, y, names, vrate = [], [], [], []
    skipped = 0
    for _, row in tqdm(diag.iterrows(), total=len(diag), desc="Preprocessing"):
        fname = str(row["FileName"])
        if not fname.endswith(".csv"):
            fname += ".csv"
        path = os.path.join(args.data_dir, args.signal_dir, fname)
        if not os.path.exists(path):
            skipped += 1
            continue
        raw = read_lead(path, args.lead)
        if not is_valid(raw):
            skipped += 1
            continue
        X.append(preprocess_window_2ch(raw[: FS_CHAPMAN * 10], FS_CHAPMAN))
        y.append(CLASS_ORDER.index(row["Rhythm"]))
        names.append(fname)
        vrate.append(row.get("VentricularRate", np.nan))

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    assert X.shape[1:] == (2, WINDOW_LEN)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out, X=X, y=y, names=np.array(names),
        ventricular_rate=np.array(vrate, dtype=np.float32),
        classes=np.array(CLASS_ORDER), lead=args.lead,
        preproc=str(config_dict()),
    )
    print(f"\nTersimpan: {args.out}")
    print(f"Bentuk X: {X.shape}, dilewati: {skipped}")
    for i, c in enumerate(CLASS_ORDER):
        print(f"  {c:5s} ({LABEL_MAP[c]}): {(y == i).sum()}")


if __name__ == "__main__":
    main()
