# Tahap 2 — Backend real-time

```
Sumber data             Pemrosesan (tiap 2 s)              Keluaran
PolarH10Source (BLE) ─┐  buffer 10 s → preprocessing →    WebSocket /ws
ReplaySource (file) ──┴► model + Grad-CAM + HR + SQI  ──► sessions/<id>.json
                         + label stabil
```

## Struktur
| File | Fungsi |
|---|---|
| `server.py` | FastAPI: REST, WebSocket, manajemen sesi, simpan sesi ke JSON |
| `processor.py` | Buffer 10 s, filter tampilan kausal, klasifikasi berkala, label stabil |
| `sources.py` | `PolarH10Source` (BLE/PMD) dan `ReplaySource` (mode demo) |
| `analysis.py` | R-peak, HR, polaritas otomatis, SQI (kurtosis + korelasi template) |
| `static/index.html` | Halaman uji sederhana (bukan dashboard final) |

Modul `preprocessing.py`, `model.py`, `gradcam.py`, dan `polar_io.py` di folder induk dipakai
langsung, sehingga pipeline real-time identik dengan training.

## Menjalankan
```bash
pip install fastapi "uvicorn[standard]" bleak
cd aritmia_ritme
python backend/server.py --ckpt results/rhythm_cnn_best.pt
```
Buka http://127.0.0.1:8000, lalu:
- **Replay:** isi path file rekaman (misalnya rekaman Mobile Legends Anda), klik "Mulai replay".
  Isi kecepatan `5` atau `10` untuk memutar lebih cepat saat pengujian.
- **Polar H10:** pakai strap (elektroda dibasahi), pastikan **tidak terhubung ke HP/aplikasi
  lain**, lalu klik "Mulai Polar H10". Pemindaian berlangsung maksimal 15 detik.

## Format pesan WebSocket
```jsonc
// Tiap chunk (~0,5 s): sinyal untuk grafik live (sudah difilter 0,5-40 Hz)
{"type": "ecg", "t0": 12.35, "fs": 130, "samples": [...]}

// Tiap 2 s: hasil analisis jendela 10 s terakhir
{"type": "rhythm", "t_start": 2.35, "t_end": 12.35,
 "label": "SR", "label_name": "Sinus normal",          // hasil jendela ini
 "label_stable": "SR", "label_stable_name": "Sinus normal", // status resmi (2x berturut-turut)
 "conf": 0.97, "uncertain": false, "probs": {...},
 "hr": 84.2, "rr_cv": 0.041, "sqi": "baik", "sqi_reasons": [], "sqi_metrics": {...},
 "polarity": "normal", "window": [1300 angka], "cam": [1300 angka], "r_peaks": [...]}

// Perubahan status sesi
{"type": "status", "state": "running" | "stopped", "error": null, "saved": "sessions/...json"}
```
Dashboard sebaiknya menampilkan **`label_stable_name`** sebagai status utama, dan
`label_name` hanya sebagai informasi tambahan. Jika `sqi == "buruk"`, tampilkan
"Sinyal buruk" beserta `sqi_reasons`, bukan hasil klasifikasi.

## Tiga lapis pertahanan terhadap alarm palsu
1. **SQI**: menolak sinyal datar, R-peak tidak wajar, kurtosis < 4, korelasi template < 0,8,
   serta lonjakan/putus amplitudo lokal antar-segmen (burst gerak dada singkat yang
   tersamarkan oleh rata-rata seluruh jendela 10 detik — lihat `segment_burst()` di `analysis.py`).
2. **Confidence**: `uncertain = true` jika confidence < 0,8.
3. **Label stabil**: status resmi baru berubah setelah label sama muncul 2 jendela berturut-turut.

## Catatan penting
- **Bagian BLE belum diuji pada perangkat fisik** (hanya parser frame yang diuji dengan data
  buatan). Jika gagal terhubung, lihat log server. Di macOS, `address` berupa UUID, bukan MAC.
- **Kalibrasi SQI:** ambang saat ini dikalibrasi pada data sintetis. Jalankan replay rekaman
  H10 asli Anda; `n_bad_signal` di file sesi seharusnya mendekati 0 untuk rekaman saat duduk
  tenang. Ubah konstanta di `analysis.py` jika banyak jendela bersih ikut ditolak.
- Waktu (`t0`, `t_end`) dihitung dari jumlah sampel sejak sesi dimulai, bukan jam dinding.
- File `sessions/<id>.json` berisi ringkasan dan log kejadian sesi. File inilah bahan
  laporan PDF pada tahap berikutnya.
