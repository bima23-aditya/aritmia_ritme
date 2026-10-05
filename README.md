# Aritmia Ritme — Pemantauan Ritme Jantung Real-time dengan Polar H10

Sistem klasifikasi ritme jantung dari EKG satu lead (strap dada **Polar H10**, 130 Hz)
memakai **1D-CNN + Grad-CAM**. Sinyal dianalisis per jendela 10 detik, ditampilkan
langsung di dashboard web, lalu dirangkum menjadi **laporan PDF** per sesi.

Kelas ritme:

| Kode | Ritme |
|---|---|
| `SR` | Sinus normal |
| `SB` | Sinus bradikardia |
| `ST` | Sinus takikardia |
| `AFIB` | Atrial fibrilasi |

```
Polar H10 (BLE) ─┐                                        ┌─► Dashboard web (WebSocket)
                 ├─► buffer 10 s ─► preprocessing ─► CNN ─┤
File rekaman ────┘    (tiap 2 s)    + SQI + HR + RR       └─► sessions/<id>.json ─► reports/<id>.pdf
   (replay)                         + Grad-CAM
```

> **Catatan:** hasil sistem ini adalah pemantauan otomatis, **bukan diagnosis medis**.

---

## Cara menjalankan

### 1. Siapkan environment (Python 3.10+)

```bash
git clone https://github.com/bima23-aditya/aritmia_ritme.git
cd aritmia_ritme

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Jalankan backend + dashboard

Jalankan dari folder utama `aritmia_ritme/`, bukan dari dalam `backend/`, karena path
model, `sessions/`, dan `reports/` dihitung relatif dari folder ini.

```bash
python backend/server.py --ckpt results_2ch_rrfix/rhythm_cnn_best.pt
```

Tanpa `--ckpt`, server memakai `results/rhythm_cnn_best.pt` (model lama 1 kanal).
Checkpoint lain di folder `results*/` juga bisa dipilih langsung dari dashboard.

Opsi lain: `--host`, `--port` (default `127.0.0.1:8000`), `--step` (interval analisis,
default 2 detik), `--polarity auto|normal|invert`.

### 3. Buka dashboard

Buka **http://127.0.0.1:8000**, lalu pilih salah satu sumber data:

- **Replay (tanpa perangkat):** isi path file rekaman, misalnya
  `demo/mitdb_202_afib.txt` (contoh AFib) atau `demo/mitdb_232_sb.txt` (contoh sinus
  bradikardia), lalu klik mulai. Isi kecepatan `5` atau `10` untuk memutar lebih cepat.
- **Polar H10 (real-time):** pasang strap dengan elektroda dibasahi dan pastikan strap
  **tidak sedang terhubung ke HP/aplikasi lain**. Komputer perlu Bluetooth LE. Lalu klik
  mulai Polar H10 (pemindaian maksimal 15 detik).

Saat sesi dihentikan, server otomatis menyimpan:
- `sessions/<id>.json`: hasil semua jendela, ringkasan, dan log perubahan ritme
- `reports/<id>.pdf`: laporan pemantauan (bisa diunduh dari dashboard)

Laporan PDF juga bisa dibuat ulang secara manual:

```bash
python backend/report.py sessions/S-20261005-022528.json --out reports
```

---

## Struktur folder

```
aritmia_ritme/
├── preprocessing.py        # SATU-SATUNYA sumber preprocessing (training = real-time)
├── model.py                # Arsitektur RhythmCNN (1 atau 2 kanal input)
├── prepare_chapman.py      # Dataset Chapman-Shaoxing -> .npz
├── train.py                # Training, early stopping, evaluasi test set
├── gradcam.py              # Grad-CAM 1D + visualisasi contoh
├── polar_io.py             # Pembaca file rekaman Polar H10 (Polar Sensor Logger / CSV)
├── test_polar.py           # Uji offline model pada rekaman Polar
├── ablation_polar.py       # Ablasi kanal (EKG vs tachogram RR) pada rekaman Polar
├── mitbih_io.py            # Pembaca format WFDB (MIT-BIH)
├── mitbih_to_polar.py      # Konversi rekaman MIT-BIH -> format file Polar (untuk replay)
├── eval_mitbih.py          # Evaluasi eksternal pada MIT-BIH Arrhythmia Database
├── index.html              # Prototipe desain dashboard (mockup statis, tidak dipakai server)
├── PANDUAN_RETRAIN_RR.md   # Latar belakang & langkah retraining model 2 kanal
├── requirements.txt
│
├── backend/                # Server real-time (detail: backend/README.md)
│   ├── server.py           # FastAPI: REST + WebSocket + manajemen sesi
│   ├── processor.py        # Buffer 10 s, klasifikasi berkala, label stabil
│   ├── sources.py          # PolarH10Source (BLE/PMD) & ReplaySource
│   ├── analysis.py         # R-peak, HR, polaritas otomatis, Signal Quality Index
│   ├── report.py           # Generator laporan PDF
│   └── static/index.html   # Dashboard web yang disajikan server
│
├── data/                   # Dataset hasil preprocessing (.npz)
│   ├── rhythm_130hz.npz            # 1 kanal (EKG)
│   ├── rhythm_130hz_2ch.npz        # 2 kanal (EKG + tachogram RR, z-score)
│   └── rhythm_130hz_2ch_rrfix.npz  # 2 kanal, tachogram RR normalisasi tetap
│
├── results/                # Model 1 kanal (baseline)
├── results_2ch/            # Model 2 kanal, tachogram z-score per jendela
├── results_2ch_rrfix/      # Model 2 kanal, tachogram normalisasi tetap (direkomendasikan)
├── results_polar*/         # Hasil uji/ablasi pada rekaman Polar
├── demo/                   # Rekaman MIT-BIH yang sudah dikonversi untuk replay
├── sessions/               # Log sesi pemantauan (JSON)
└── reports/                # Laporan PDF per sesi
```

### Versi model

| Folder | Input | Epoch terbaik | Val macro-F1 |
|---|---|---|---|
| `results/` | EKG saja | 38 | 0,973 |
| `results_2ch/` | EKG + tachogram RR (z-score per jendela) | 18 | 0,971 |
| `results_2ch_rrfix/` | EKG + tachogram RR (normalisasi tetap) | 23 | 0,972 |

Kanal tachogram RR ditambahkan karena model 1 kanal sempat memprediksi AFib pada ritme
yang RR-nya teratur. Penjelasan lengkapnya ada di `PANDUAN_RETRAIN_RR.md`.
Pada `rrfix`, amplitudo variasi RR tidak lagi dihapus oleh z-score per jendela.
Laporan per kelas tiap model ada di `results*/test_report.txt`.

---

## Melatih ulang model (opsional)

Model siap pakai sudah tersedia di `results*/`. Langkah ini hanya perlu dilakukan jika
ingin melatih ulang dari awal.

1. **Unduh dataset Chapman-Shaoxing** dari figshare
   (https://figshare.com/collections/ChapmanECG/4560497): `Diagnostics.xlsx` dan
   `ECGData.zip` (sinyal **mentah**, bukan *Denoised*). Susun seperti ini:
   ```
   data/chapman/
     Diagnostics.xlsx
     ECGData/MUSE_20180111_155115_19000.csv ...
   ```
2. **Preprocessing** (sekitar 11 menit di CPU):
   ```bash
   python prepare_chapman.py --data_dir data/chapman --lead II --out data/rhythm_130hz_2ch_rrfix.npz
   ```
3. **Training:**
   ```bash
   python train.py --data data/rhythm_130hz_2ch_rrfix.npz --epochs 40 --out results_2ch_rrfix
   ```
   Output: `rhythm_cnn_best.pt`, `test_report.txt`, `confusion_matrix.png`,
   `training_curve.png`, `history.json`, `split.npz`.
4. **Visualisasi Grad-CAM:**
   ```bash
   python gradcam.py --data data/rhythm_130hz_2ch_rrfix.npz \
       --ckpt results_2ch_rrfix/rhythm_cnn_best.pt --split results_2ch_rrfix/split.npz \
       --out results_2ch_rrfix/gradcam_examples.png
   ```

## Evaluasi tambahan

```bash
# Uji offline pada rekaman Polar H10 (output: polar_windows.csv, timeline, Grad-CAM)
python test_polar.py --file rekaman_polar.txt --ckpt results_2ch_rrfix/rhythm_cnn_best.pt --out results_polar_rrfix

# Ablasi kanal: seberapa besar prediksi bergantung pada kanal RR vs EKG
python ablation_polar.py --file rekaman_polar.txt --ckpt results_2ch_rrfix/rhythm_cnn_best.pt --out results_polar_rrfix

# Evaluasi eksternal MIT-BIH (letakkan file .hea/.dat/.atr di data/mitdb/)
python eval_mitbih.py --data_dir data/mitdb --records afib --ckpt results_2ch_rrfix/rhythm_cnn_best.pt

# Membuat file replay dari rekaman MIT-BIH
python mitbih_to_polar.py --record data/mitdb/202 --rhythm AFIB --out demo/mitdb_202_afib.txt
```

Format rekaman yang didukung untuk replay maupun `test_polar.py`: keluaran aplikasi
**Polar Sensor Logger** (`.txt`/`.csv`, kolom `ecg [uV]`), atau CSV/TXT umum dengan satu
kolom EKG. Kalau ada kolom waktu, sinyal otomatis di-resample ke 130 Hz.

---

## Mekanisme pengurangan alarm palsu

1. **Signal Quality Index (SQI):** jendela dengan sinyal datar, R-peak tidak wajar,
   kurtosis rendah, korelasi template QRS rendah, atau lonjakan lokal akibat gerakan
   ditandai **"Sinyal buruk"** dan tidak diklasifikasi.
2. **Confidence:** prediksi dengan keyakinan < 0,8 ditandai `uncertain`.
3. **Label stabil:** status ritme resmi baru berubah setelah label yang sama muncul
   **2 jendela berturut-turut**.

## Catatan penting

- Preprocessing (`preprocessing.py`) dipakai bersama oleh training dan backend. Jangan
  membuat versi kedua di tempat lain. Konfigurasinya juga disimpan di checkpoint.
- Jangan memakai augmentasi *time-stretch*: meregangkan sinyal mengubah denyut jantung
  sehingga label SR/SB/ST menjadi salah.
- Polar H10 hanya memberi satu lead dada. Kelainan morfologi seperti bundle branch block
  tidak dapat dibedakan secara andal, dan sistem hanya mencakup 4 kelas ritme di atas.
- Format pesan WebSocket dan endpoint REST dijelaskan di [`backend/README.md`](backend/README.md).
