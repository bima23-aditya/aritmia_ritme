"""
1D-CNN untuk klasifikasi ritme dari jendela 10 detik (2 x 1300 sampel:
kanal EKG + kanal tachogram RR, lihat preprocessing.preprocess_window_2ch).

Desain yang sengaja dibuat ramah Grad-CAM:
  - Blok konvolusi terakhir (self.target_layer) TIDAK diikuti pooling,
    sehingga resolusi waktunya masih 81 langkah (~0,12 s per langkah).
  - Global Average Pooling (GAP) sebelum classifier. GAP juga membantu model
    "menghitung" jumlah denyut dalam jendela, yang penting untuk SB/ST.

Catatan desain -- kenapa ada kanal tachogram:
  Reseptive field tiap unit di lapisan konvolusi terakhir hanya ~0,76 detik
  (satu denyut), dan GAP meratakan 81 langkah waktu itu TANPA informasi
  posisi. Dengan hanya kanal EKG mentah, model tidak punya cara langsung
  mengukur jarak antar-puncak-R yang berjauhan -- ia hanya bisa meng-"hitung"
  rata-rata bentuk lokal tiap denyut. Ini membuatnya rentan mengenali AFib
  dari morfologi/tekstur baseline (ciri khas dataset Chapman), bukan dari
  ketidakteraturan RR yang sebenarnya jadi kriteria klinis AFib. Kanal
  tachogram memberi akses eksplisit ke pola waktu antar-denyut di setiap
  sampel, sehingga GAP bisa meratakan "seberapa bervariasi periode RR di
  sekitar sini" -- bukan cuma "seberapa mirip AFib bentuk lokalnya".
"""
import torch
import torch.nn as nn


def conv_block(c_in, c_out, k, pool=True):
    layers = [
        nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2, bias=False),
        nn.BatchNorm1d(c_out),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool1d(2))
    return nn.Sequential(*layers)


class RhythmCNN(nn.Module):
    def __init__(self, n_classes: int = 4, dropout: float = 0.3, in_channels: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_channels, 32, 7),  # 1300 -> 650
            conv_block(32, 64, 7),     # 650  -> 325
            conv_block(64, 128, 5),    # 325  -> 162
            conv_block(128, 128, 5),   # 162  -> 81
            conv_block(128, 256, 3, pool=False),  # 81 -> 81  (target Grad-CAM)
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, n_classes))

    @property
    def target_layer(self):
        return self.features[-1]

    def forward(self, x):            # x: (batch, 2, 1300) -- [ekg, rr_tachogram]
        f = self.features(x)         # (batch, 256, 81)
        return self.classifier(self.gap(f).squeeze(-1))


if __name__ == "__main__":
    m = RhythmCNN()
    out = m(torch.randn(2, 2, 1300))
    n_params = sum(p.numel() for p in m.parameters())
    print("Output:", tuple(out.shape), "| Parameter:", f"{n_params:,}")
