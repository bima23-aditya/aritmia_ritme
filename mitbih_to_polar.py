"""
Konversi rekaman MIT-BIH ke format file Polar (130 Hz, uV) agar bisa diputar
lewat mode replay di backend, seolah-olah berasal dari Polar H10.

Contoh:
  # Lihat dulu daftar ritme dalam rekaman
  python mitbih_to_polar.py --record data/mitdb/201 --list

  # Ambil segmen AFib terpanjang (tanpa sambungan buatan)
  python mitbih_to_polar.py --record data/mitdb/201 --rhythm AFIB --out demo/mitdb_201_afib.txt

  # Ambil rentang waktu tertentu (detik), mis. untuk melihat transisi sinus -> AFib
  python mitbih_to_polar.py --record data/mitdb/202 --start 600 --end 900 --out demo/mitdb_202.txt

Selain file sinyal, dibuat juga <out>_labels.csv berisi segmen ritme sebenarnya
(dalam detik relatif terhadap awal file) sebagai kunci jawaban.
"""
import argparse
import os
from math import gcd

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from mitbih_io import describe_segments, read_mitbih
from preprocessing import TARGET_FS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True, help="path tanpa ekstensi, mis. data/mitdb/201")
    ap.add_argument("--list", action="store_true", help="tampilkan segmen ritme lalu keluar")
    ap.add_argument("--rhythm", help="ambil segmen terpanjang dari ritme ini, mis. AFIB atau N")
    ap.add_argument("--start", type=float, help="detik awal (alternatif --rhythm)")
    ap.add_argument("--end", type=float, help="detik akhir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ecg, fs, lead, segs = read_mitbih(a.record)
    rec_name = os.path.basename(a.record)
    print(f"Rekaman {rec_name}: {len(ecg) / fs / 60:.1f} menit, {fs} Hz, lead {lead}")
    print("Total durasi per ritme:")
    for r, d in describe_segments(segs, fs).items():
        print(f"  {r:8s} {d / 60:6.1f} menit")

    if a.list:
        print("\nSegmen (detik):")
        for s in segs:
            print(f"  {s['start'] / fs:8.1f} - {s['end'] / fs:8.1f}  {s['rhythm']}")
        return

    if a.rhythm:
        key = "(" + a.rhythm.strip("(").upper()
        cand = [s for s in segs if s["rhythm"].upper() == key]
        if not cand:
            raise SystemExit(f"Tidak ada ritme {key} di rekaman {rec_name}")
        best = max(cand, key=lambda s: s["end"] - s["start"])
        i0, i1 = best["start"], best["end"]
    else:
        i0 = int((a.start or 0) * fs)
        i1 = int(a.end * fs) if a.end else len(ecg)
    if (i1 - i0) / fs < 10:
        print(f"[!] Potongan hanya {(i1 - i0) / fs:.1f} detik, kurang dari satu jendela 10 detik")

    g = gcd(fs, TARGET_FS)
    y = resample_poly(ecg[i0:i1], TARGET_FS // g, fs // g)
    t_ms = np.arange(len(y)) / TARGET_FS * 1000

    out = a.out or f"{rec_name}_{a.rhythm or 'potongan'}.txt"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("timestamp [ms];ecg [uV]\n")
        for t, v in zip(t_ms, y):
            f.write(f"{t:.1f};{v:.0f}\n")

    # Kunci jawaban: segmen ritme yang beririsan dengan potongan
    rows = []
    for s in segs:
        a0, a1 = max(s["start"], i0), min(s["end"], i1)
        if a1 > a0:
            rows.append({"start_s": round((a0 - i0) / fs, 2), "end_s": round((a1 - i0) / fs, 2),
                         "rhythm": s["rhythm"]})
    lab = os.path.splitext(out)[0] + "_labels.csv"
    pd.DataFrame(rows).to_csv(lab, index=False)

    print(f"\nTersimpan: {out} ({len(y) / TARGET_FS:.1f} detik, 130 Hz)")
    print(f"Kunci jawaban: {lab}")
    for r in rows:
        print(f"  {r['start_s']:7.1f} - {r['end_s']:7.1f} s  {r['rhythm']}")


if __name__ == "__main__":
    main()
