"""
Prosesor real-time untuk cabang ritme.

Dua jalur terpisah:
  1. Jalur TAMPILAN : filter bandpass kausal (0,5-40 Hz) per chunk -> dikirim
                      langsung ke dashboard. Ringan dan tanpa jeda.
  2. Jalur ANALISIS : setiap `step_sec`, ambil 10 detik terakhir -> pipeline
                      preprocessing yang SAMA dengan training -> model + Grad-CAM.
"""
import hashlib
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from analysis import analyze_rpeaks, detect_inverted, signal_quality
from gradcam import GradCAM1D, load_model
from preprocessing import TARGET_FS, WINDOW_LEN, preprocess_window, rr_norm_of, rr_tachogram

POLARITY_DECIDE_SEC = 5   # polaritas otomatis ditentukan dari 5 detik pertama
STABLE_N = 2              # label resmi berubah setelah muncul N jendela berturut-turut


class DisplayFilter:
    """Bandpass Butterworth kausal dengan state, aman diproses per chunk."""

    def __init__(self):
        self.sos = butter(2, [0.5, 40], btype="band", fs=TARGET_FS, output="sos")
        self.zi = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.zi is None:
            self.zi = sosfilt_zi(self.sos) * x[0]
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y


class RhythmProcessor:
    def __init__(self, ckpt_path: str, step_sec: float = 2.0,
                 conf_threshold: float = 0.8, polarity: str = "auto"):
        model, ck = load_model(ckpt_path)
        self.cam = GradCAM1D(model)
        self.classes, self.names = ck["classes"], ck["class_names"]
        # Checkpoint lama (sebelum kanal tachogram) hanya 1 kanal (EKG saja);
        # deteksi dari bobot model, bukan diasumsikan, supaya tidak crash diam-diam.
        self.in_channels = model.features[0][0].weight.shape[1]
        self.rr_norm = rr_norm_of(ck.get("preproc"))
        self.model_info = {
            "checkpoint": Path(ckpt_path).name,
            "sha256": hashlib.sha256(Path(ckpt_path).read_bytes()).hexdigest(),
            "epoch": ck.get("epoch"), "val_macro_f1": ck.get("val_macro_f1"),
            "classes": self.classes, "preproc": ck.get("preproc"),
            "step_sec": step_sec, "conf_threshold": conf_threshold,
        }
        self.step = int(step_sec * TARGET_FS)
        self.conf_threshold = conf_threshold
        self.polarity_mode = polarity
        self.reset()

    def reset(self):
        self.buf = np.zeros(0)
        self.n_total = 0            # jumlah sampel sejak sesi dimulai
        self.since_last = 0
        self.inverted = {"auto": None, "normal": False, "invert": True}[self.polarity_mode]
        self.display = DisplayFilter()
        self.stable_label, self.stable_name = None, None
        self._run_label, self._run_count = None, 0

    # ------------------------------------------------------------------ input
    def push(self, chunk: np.ndarray) -> dict:
        """Menerima chunk mentah (uV). Mengembalikan pesan 'ecg' untuk dashboard."""
        t0 = self.n_total / TARGET_FS
        self.n_total += len(chunk)
        self.since_last += len(chunk)
        self.buf = np.concatenate([self.buf, chunk])[-WINDOW_LEN * 2:]

        if self.inverted is None and self.n_total >= POLARITY_DECIDE_SEC * TARGET_FS:
            self.inverted = detect_inverted(self.buf)

        sign = -1.0 if self.inverted else 1.0
        shown = self.display(chunk) * sign
        return {"type": "ecg", "t0": round(t0, 4), "fs": TARGET_FS,
                "samples": np.round(shown, 1).tolist()}

    def due(self) -> bool:
        return len(self.buf) >= WINDOW_LEN and self.since_last >= self.step

    # --------------------------------------------------------------- analisis
    def analyze(self) -> dict:
        """Berat (~5-10 ms). Dipanggil lewat thread agar event loop tidak tersendat."""
        self.since_last = 0
        raw = self.buf[-WINDOW_LEN:].copy()
        if self.inverted:
            raw = -raw
        t_end = self.n_total / TARGET_FS
        t_start = t_end - WINDOW_LEN / TARGET_FS

        xw = preprocess_window(raw, TARGET_FS)
        rp = analyze_rpeaks(xw)
        sqi, reasons, sqi_metrics = signal_quality(raw, xw, rp)

        msg = {
            "type": "rhythm", "t_start": round(t_start, 3), "t_end": round(t_end, 3),
            "sqi": sqi, "sqi_reasons": reasons, "sqi_metrics": sqi_metrics,
            "hr": round(rp["hr"], 1) if rp["hr"] else None,
            "rr_cv": round(rp["rr_cv"], 3) if rp["rr_cv"] is not None else None,
            "rr_ms": rp["rr_ms"],  # interval R-R berurutan (ms), selaras dengan r_peaks
            "polarity": "terbalik" if self.inverted else "normal",
            "window": np.round(xw, 3).tolist(),
            "r_peaks": rp["peaks"].tolist(),
        }
        if sqi == "buruk":
            # Jangan klasifikasi sinyal buruk: lebih baik diam daripada alarm palsu
            msg.update(label=None, label_name="Sinyal buruk", conf=None,
                       probs=None, uncertain=True, cam=None)
            self._run_label, self._run_count = None, 0
            return self._with_stable(msg)

        if self.in_channels == 2:
            tach = rr_tachogram(xw, rp["peaks"], norm=self.rr_norm)
            model_input = np.stack([xw, tach])
        else:
            model_input = xw  # checkpoint lama, 1 kanal (EKG saja)
        heat, pred, prob = self.cam(model_input)
        conf = float(prob[pred])
        msg.update(
            label=self.classes[pred], label_name=self.names[pred], conf=round(conf, 3),
            probs={c: round(float(p), 3) for c, p in zip(self.classes, prob)},
            uncertain=conf < self.conf_threshold,
            cam=np.round(heat, 3).tolist(),
        )
        if self.classes[pred] == self._run_label:
            self._run_count += 1
        else:
            self._run_label, self._run_count = self.classes[pred], 1
        if self._run_count >= STABLE_N:
            self.stable_label, self.stable_name = self.classes[pred], self.names[pred]
        return self._with_stable(msg)

    def _with_stable(self, msg: dict) -> dict:
        """label       = hasil jendela ini (bisa berkedip)
        label_stable = status resmi untuk dashboard & alarm (butuh STABLE_N berturut-turut)"""
        msg["label_stable"], msg["label_stable_name"] = self.stable_label, self.stable_name
        return msg
