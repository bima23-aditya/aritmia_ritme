"""
Evaluasi eksternal model ritme pada MIT-BIH Arrhythmia Database.

Jendela 10 detik (geser 5 s) diambil hanya jika SELURUHNYA berada dalam satu
segmen ritme yang dikenal:
  (AFIB        -> label AFIB
  (N, (SBR     -> label SINUS (SR/SB/ST dibedakan dengan aturan HR)
Ritme lain (flutter, bigemini, VT, dll.) diabaikan karena bukan kelas model.

Contoh:
  python eval_mitbih.py --data_dir data/mitdb --ckpt results/rhythm_cnn_best.pt
  python eval_mitbih.py --data_dir data/mitdb --records all
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "backend"))

from analysis import analyze_rpeaks, signal_quality  # noqa: E402
from gradcam import load_model  # noqa: E402
from mitbih_io import RHYTHM_MAP, read_mitbih  # noqa: E402
from preprocessing import preprocess_window, rr_norm_of, rr_tachogram  # noqa: E402

# Rekaman MIT-BIH Arrhythmia yang mengandung AFib
AFIB_RECORDS = ["201", "202", "203", "210", "219", "221", "222"]


def hr_rule(hr):
    if hr is None:
        return None
    return "SB" if hr < 60 else "ST" if hr > 100 else "SR"


def evaluate_record(path, model, classes, step_sec, rr_norm="fixed"):
    import torch
    ecg, fs, lead, segs = read_mitbih(path)
    win, step = 10 * fs, int(step_sec * fs)
    rows, batch = [], []
    for s in segs:
        truth = RHYTHM_MAP.get(s["rhythm"])
        if truth is None:
            continue
        for i in range(s["start"], s["end"] - win + 1, step):
            raw = ecg[i:i + win]
            xw = preprocess_window(raw, fs)
            rp = analyze_rpeaks(xw)
            sqi, reasons, _ = signal_quality(raw, xw, rp)
            tach = rr_tachogram(xw, rp["peaks"], norm=rr_norm)
            batch.append(np.stack([xw, tach]))
            rows.append({"record": Path(path).name, "t_start": round(i / fs, 1),
                         "truth": truth, "hr": rp["hr"], "rr_cv": rp["rr_cv"],
                         "sqi": sqi, "sqi_reason": "; ".join(reasons)})
    if not rows:
        return pd.DataFrame()
    with torch.no_grad():
        x = torch.from_numpy(np.stack(batch))  # (N, 2, win) -- sudah termasuk kanal RR
        prob = torch.softmax(model(x), dim=1).numpy()
    df = pd.DataFrame(rows)
    df["pred"] = [classes[k] for k in prob.argmax(1)]
    df["conf"] = prob.max(1).round(3)
    df["p_afib"] = prob[:, classes.index("AFIB")].round(3)
    df["hr_rule"] = [hr_rule(h) for h in df["hr"]]
    return df


def report(df, title):
    af, si = df[df.truth == "AFIB"], df[df.truth == "SINUS"]
    print(f"\n--- {title} ---")
    if len(af):
        sens = (af.pred == "AFIB").mean()
        print(f"  AFib : {len(af):5d} jendela | terdeteksi AFib (sensitivitas) = {sens:.1%}")
        wrong = af[af.pred != "AFIB"].pred.value_counts()
        if len(wrong):
            print(f"         salah diprediksi sebagai: {dict(wrong)}")
    if len(si):
        fp = (si.pred == "AFIB").mean()
        print(f"  Sinus: {len(si):5d} jendela | keliru dianggap AFib = {fp:.1%} "
              f"(spesifisitas {1 - fp:.1%})")
        ok = si[si.pred != "AFIB"]
        if len(ok) and ok.hr_rule.notna().any():
            print(f"         kesesuaian SR/SB/ST dengan aturan HR = "
                  f"{(ok.pred == ok.hr_rule).mean():.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/mitdb")
    ap.add_argument("--records", default="afib",
                    help="'afib' (7 rekaman AFib), 'all', atau daftar: 201,202,100")
    ap.add_argument("--ckpt", default="results/rhythm_cnn_best.pt")
    ap.add_argument("--step", type=float, default=5.0)
    ap.add_argument("--out", default="results_mitbih.csv")
    a = ap.parse_args()

    d = Path(a.data_dir)
    if a.records == "afib":
        recs = AFIB_RECORDS
    elif a.records == "all":
        recs = sorted(p.stem for p in d.glob("*.hea"))
    else:
        recs = [r.strip() for r in a.records.split(",")]
    recs = [r for r in recs if (d / f"{r}.hea").exists()]
    if not recs:
        raise SystemExit(f"Tidak ada file .hea yang cocok di {d}")

    model, ck = load_model(a.ckpt)
    classes = ck["classes"]
    parts = []
    for r in recs:
        df = evaluate_record(str(d / r), model, classes, a.step,
                             rr_norm_of(ck.get("preproc")))
        if len(df):
            parts.append(df)
            af = df[df.truth == "AFIB"]
            print(f"Rekaman {r}: {len(df):4d} jendela"
                  + (f" | AFib {len(af):4d}, terdeteksi {(af.pred == 'AFIB').mean():.0%}"
                     if len(af) else "")
                  + f" | sinyal buruk {(df.sqi == 'buruk').mean():.0%}")
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(a.out, index=False)

    report(df, "SEMUA jendela (tanpa SQI)")
    good = df[df.sqi == "baik"]
    report(good, f"Hanya jendela lolos SQI ({len(good)}/{len(df)})")
    print(f"\nDetail per jendela: {a.out}")


if __name__ == "__main__":
    main()
