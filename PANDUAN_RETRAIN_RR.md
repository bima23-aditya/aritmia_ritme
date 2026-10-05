# Panduan retraining: kanal tachogram RR

## Masalah yang diperbaiki

Pada laporan `S-20261002-052232.pdf`, model mengklasifikasikan satu jendela sebagai
**AFib dengan keyakinan 97%**, padahal interval RR pada jendela itu sendiri
(669, 654, 646, 638, 631, 615, 608, 615, 608, 600, 585, 585, 577, 577, 577 ms)
memberi **RR-CV 0,047** -- lebih teratur daripada ambang batas sinus normal (0,10)
yang disebutkan di laporan itu sendiri. Artinya model mengenali AFib dari
**morfologi/tekstur gelombang**, bukan dari **ketidakteraturan RR** yang
seharusnya jadi kriteria utama AFib secara klinis.

Akar masalahnya ada di arsitektur `RhythmCNN` (`model.py`): reseptive field tiap
unit di lapisan konvolusi terakhir hanya ~0,76 detik (kira-kira satu denyut), dan
Global Average Pooling (GAP) yang menyatukan 81 langkah waktu itu **tidak punya
informasi posisi**. Jadi secara struktural, jaringan tidak bisa mengukur jarak
antar-puncak-R yang berjauhan -- ia hanya bisa merata-ratakan "seberapa mirip
AFib bentuk lokalnya" di tiap denyut. Ini membuatnya rentan menangkap *shortcut*
dari dataset latih (Chapman, 12-lead klinis) yang tidak selalu berlaku pada
sinyal satu-lead dari strap dada Polar H10.

## Solusi: kanal tachogram sebagai input eksplisit

Model sekarang menerima **2 kanal** per jendela 10 detik, bukan 1:

1. **Kanal EKG** (seperti sebelumnya): sinyal yang sudah melalui
   resample -> baseline removal -> DWT denoising -> z-score.
2. **Kanal tachogram RR** (baru): periode RR sesaat (ms) dari deret puncak R
   yang sama, diinterpolasi linear ke setiap sampel lalu di-z-score. Kanal ini
   memberi jaringan akses *langsung* ke pola ketidakteraturan irama di setiap
   titik waktu, melengkapi kanal EKG yang membawa informasi bentuk gelombang.

Semua logika baru ada di `preprocessing.py` (satu-satunya sumber preprocessing,
dipakai training maupun real-time, sesuai aturan lama di proyek ini):

- `detect_r_peaks(xw, fs)` -- deteksi puncak R (logika sama persis dengan versi
  lama di `backend/analysis.py`, hanya dipindah supaya dipakai bersama).
- `rr_tachogram(xw, peaks, fs)` -- bangun kanal ke-2 dari puncak R.
- `preprocess_window_2ch(x, fs_in)` -- gabungan keduanya, output `(2, 1300)`.

## File yang berubah

| File | Perubahan |
|---|---|
| `preprocessing.py` | Tambah `detect_r_peaks`, `rr_tachogram`, `preprocess_window_2ch` |
| `model.py` | `RhythmCNN` sekarang `in_channels=2` secara default |
| `prepare_chapman.py` | Pakai `preprocess_window_2ch`, output `X` berbentuk `(N, 2, 1300)` |
| `train.py` | Dataset & augmentasi disesuaikan untuk 2 kanal; augmentasi noise/skala **hanya** menyentuh kanal EKG (kanal RR dibiarkan bersih); checkpoint menyimpan `in_channels` |
| `gradcam.py` | `GradCAM1D` & `load_model` menerima input 1 atau 2 kanal; `load_model` menyimpulkan `in_channels` dari bobot jika checkpoint lama tidak punya field itu |
| `backend/analysis.py` | `analyze_rpeaks` memakai `detect_r_peaks` bersama (tidak duplikasi logika lagi) |
| `backend/processor.py` | Mendeteksi `in_channels` dari model yang dimuat, otomatis membangun kanal tachogram saat model 2-kanal |
| `eval_mitbih.py`, `test_polar.py` | Ikut membangun kanal tachogram sebelum memanggil model |

**Kompatibilitas mundur:** checkpoint lama (`results/rhythm_cnn_best.pt`, 1 kanal)
masih bisa dimuat dan dipakai di backend tanpa error -- `load_model` dan
`RhythmProcessor` otomatis mendeteksi jumlah kanal dari bobot model. Tapi tentu
saja checkpoint lama **tidak** mendapat manfaat perbaikan ini sampai dilatih ulang.

## Data sudah disiapkan

Dataset 2-kanal sudah dibuat dari `data/chapman/` dan tersimpan di:

```
data/rhythm_130hz_2ch.npz   (9.063 jendela, bentuk X = (9063, 2, 1300), 0 dilewati)
```

Rincian per kelas: SR 1826, SB 3889, ST 1568, AFIB 1780 -- sama persis dengan
jumlah di `Diagnostics.xlsx`, tidak ada rekaman yang gagal diproses.

(dibuat dengan `python prepare_chapman.py --data_dir data/chapman --lead II --out data/rhythm_130hz_2ch.npz`,
makan waktu 11m23s untuk 9.063 rekaman di CPU). Anda tidak perlu mengulang
langkah ini kecuali ingin mencoba lead lain (`--lead I`, dst).

## Langkah retraining

```bash
cd aritmia_ritme
python train.py --data data/rhythm_130hz_2ch.npz --epochs 40 --out results_2ch
```

Perkiraan waktu di CPU (4 core, tanpa GPU): **~30-45 menit** untuk 40 epoch
(early stopping bisa membuatnya lebih cepat; patience default 8 epoch).
Jalankan di background kalau mau lanjut kerja lain:

```bash
nohup python train.py --data data/rhythm_130hz_2ch.npz --epochs 40 --out results_2ch \
  > train_2ch.log 2>&1 &
```

Output yang dihasilkan di `results_2ch/`:
- `rhythm_cnn_best.pt` -- checkpoint terbaik (dipakai backend)
- `test_report.txt` -- precision/recall/F1 per kelas pada test set Chapman
- `confusion_matrix.png`, `training_curve.png`, `split.npz`

## Verifikasi perbaikan setelah training

### 1. Akurasi dasar tidak boleh turun jauh dari baseline lama
Baseline lama (1 kanal, lihat `results/test_report.txt`): macro F1 0,973,
akurasi 97,9%. Cek `results_2ch/test_report.txt` -- seharusnya setara atau lebih
baik (kanal tachogram memberi informasi tambahan, bukan mengurangi).

### 2. Cek apakah model sekarang "dengar" RR-CV, bukan cuma morfologi
Jalankan ulang evaluasi MIT-BIH (data independen, bukan Chapman):

```bash
python eval_mitbih.py --data_dir data/mitdb --records afib --ckpt results_2ch/rhythm_cnn_best.pt
```

Bandingkan sensitivitas/spesifisitas AFib dengan hasil lama (`results_mitbih.csv`).
Yang penting bukan cuma angka sensitivitas naik/turun, tapi: pada
`results_mitbih.csv` yang baru, cek kolom `rr_cv` untuk baris-baris yang
diprediksi AFib -- seharusnya mayoritas punya `rr_cv` tinggi (>0,10), bukan
rendah seperti kasus di laporan `S-20261002-052232.pdf`.

### 3. Uji ulang pada rekaman Polar H10 yang pernah salah klasifikasi
```bash
python test_polar.py --file ecg_main_mobile-legends2.csv --ckpt results_2ch/rhythm_cnn_best.pt
```
atau jalankan replay lewat `backend/server.py` seperti biasa, lalu buat ulang
laporan dari rekaman yang sama (kalau raw signal-nya masih ada) dan bandingkan
window AFib yang dulu salah: apakah sekarang confidence-nya turun / label
berubah jadi sinus, dan apakah window yang *benar-benar* AFib (RR-CV tinggi)
tetap terdeteksi.

### 4. Lihat Grad-CAM
```bash
python gradcam.py --data data/rhythm_130hz_2ch.npz --ckpt results_2ch/rhythm_cnn_best.pt --split results_2ch/split.npz
```
Secara kualitatif, untuk contoh AFib, sorotan panas seharusnya tersebar merata
mengikuti pola jarak antar-denyut (bukan cuma menumpuk di satu-dua denyut
dengan bentuk "aneh").

## Pakai checkpoint baru di backend

Tidak perlu ubah kode apa pun -- `backend/server.py` menerima path checkpoint
lewat `--ckpt`:

```bash
python backend/server.py --ckpt results_2ch/rhythm_cnn_best.pt
```

`RhythmProcessor` otomatis mendeteksi bahwa checkpoint ini 2-kanal dan membangun
kanal tachogram tiap jendela secara real-time (lihat `backend/processor.py`).
Atau, kalau sudah puas dengan hasilnya, ganti saja `results_2ch/rhythm_cnn_best.pt`
ke `results/rhythm_cnn_best.pt` (path default yang dipakai server tanpa `--ckpt`).
