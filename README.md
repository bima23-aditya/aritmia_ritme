# Tahap 1 — Klasifikasi ritme (jendela 10 detik)

Kelas: Sinus normal (SR), Sinus bradikardia (SB), Sinus takikardia (ST), Atrial fibrilasi (AFIB).
Model: 1D-CNN + Grad-CAM. Input: 1 lead, 10 detik, 130 Hz (1300 sampel).

## Isi folder

| File | Fungsi |
|---|---|
| `preprocessing.py` | Resample → cascaded median → DWT db4 → z-score. **Dipakai juga nanti untuk data Polar H10.** |
| `prepare_chapman.py` | Membaca dataset Chapman-Shaoxing dan menyimpannya sebagai `.npz` |
| `model.py` | Arsitektur `RhythmCNN` (~238 ribu parameter) |
| `train.py` | Training, early stopping, evaluasi test set, confusion matrix |
| `gradcam.py` | Kelas `GradCAM1D` + skrip visualisasi contoh |

## Langkah

### 1. Instal library
```bash
pip install -r requirements.txt
```
Di Google Colab, PyTorch sudah terpasang; cukup `pip install PyWavelets openpyxl`.

### 2. Unduh dataset
Dari koleksi figshare **ChapmanECG** (https://figshare.com/collections/ChapmanECG/4560497), unduh:
- `Diagnostics.xlsx`
- `ECGData.zip` (sinyal **mentah**, bukan yang *Denoised*)

Susun seperti ini:
```
data/chapman/
  Diagnostics.xlsx
  ECGData/
    MUSE_20180111_155115_19000.csv
    ...
```

### 3. Siapkan data
```bash
python prepare_chapman.py --data_dir data/chapman --lead II --out data/rhythm_130hz.npz
```

### 4. Training
```bash
python train.py --data data/rhythm_130hz.npz --epochs 40 --out results
```
Output di `results/`: `rhythm_cnn_best.pt`, `test_report.txt`, `confusion_matrix.png`,
`training_curve.png`, `split.npz`.

### 5. Lihat penjelasan Grad-CAM
```bash
python gradcam.py --data data/rhythm_130hz.npz --ckpt results/rhythm_cnn_best.pt
```

## Catatan penting
- **Jangan menambahkan augmentasi time-stretch.** Meregangkan sinyal mengubah denyut
  jantung sehingga label SR/SB/ST menjadi salah.
- Laporkan **precision, recall, F1 per kelas**, bukan hanya akurasi.
- Checkpoint menyimpan konfigurasi preprocessing (`ckpt["preproc"]`). Backend real-time
  wajib memakai `preprocessing.preprocess_window()` yang sama.
- Pembanding lead: coba juga `--lead I`, lalu pilih yang morfologinya paling mirip
  dengan rekaman Polar H10 Anda.
