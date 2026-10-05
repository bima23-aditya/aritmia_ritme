"""
Generator laporan PDF per sesi.

Input : sessions/<id>.json (dibuat otomatis oleh server saat sesi berhenti)
Output: reports/<id>.pdf

Dipakai otomatis oleh server, atau manual:
  python backend/report.py sessions/S-20260923-101500.json --out reports/
"""
import argparse
import io
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

FS = 130
ORDER = ["SR", "SB", "ST", "AFIB"]
NAMES = {"SR": "Sinus normal", "SB": "Sinus bradikardia",
         "ST": "Sinus takikardia", "AFIB": "Atrial fibrilasi"}
COLORS = {"SR": "#378ADD", "SB": "#1D9E75", "ST": "#EF9F27", "AFIB": "#E24B4A"}
INK = colors.HexColor("#2C2C2A")
MUTED = colors.HexColor("#73726C")
LINE = colors.HexColor("#D3D1C7")
SOFT = colors.HexColor("#F1EFE8")


# ---------------------------------------------------------------------------
# Utilitas
# ---------------------------------------------------------------------------
def mmss(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def dec(x: float, n: int = 2) -> str:
    """Angka desimal gaya Indonesia (koma)."""
    return f"{x:.{n}f}".replace(".", ",")


def fmt_dt(iso: str | None) -> str:
    if not iso:
        return "-"
    return datetime.fromisoformat(iso).strftime("%d-%m-%Y %H:%M:%S")


BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus",
         "September", "Oktober", "November", "Desember"]


def fmt_date(iso: str | None) -> str:
    if not iso:
        return "-"
    d = datetime.fromisoformat(iso)
    return f"{d.day} {BULAN[d.month - 1]} {d.year}"


def durasi_teks(sec: float) -> str:
    m, s = divmod(max(0, int(round(sec))), 60)
    return f"{m} menit {s} detik" if m else f"{s} detik"


def pct(x: float) -> str:
    return dec(x * 100, 1) + "%"


def fig_to_image(fig, width_mm: float) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width_mm * mm
    img.drawHeight = width_mm * mm * ratio
    return img


def afib_episodes(windows: list[dict]) -> list[tuple[float, float]]:
    """Episode AFib = rangkaian jendela berturut-turut dengan label STABIL = AFIB."""
    eps, cur = [], None
    for w in windows:
        if w.get("label_stable") == "AFIB" and w.get("sqi") == "baik":
            cur = [w["t_start"], w["t_end"]] if cur is None else [cur[0], w["t_end"]]
        elif cur is not None:
            eps.append(tuple(cur))
            cur = None
    if cur is not None:
        eps.append(tuple(cur))
    return eps


# ---------------------------------------------------------------------------
# Grafik
# ---------------------------------------------------------------------------
def timeline_figure(windows: list[dict]):
    t = np.array([w["t_end"] for w in windows])
    fig, ax = plt.subplots(3, 1, figsize=(9, 4.6), sharex=True,
                           gridspec_kw={"height_ratios": [0.55, 1.3, 0.9]})
    # Pita status ritme (label stabil)
    for w in windows:
        lab = w.get("label_stable") if w.get("sqi") == "baik" else None
        c = COLORS.get(lab, "#B4B2A9") if lab else "#D3D1C7"
        ax[0].axvspan(w["t_end"] - 2, w["t_end"], color=c, lw=0)
    ax[0].set_yticks([])
    ax[0].set_title("Status ritme (abu-abu = belum stabil / sinyal buruk)", loc="left", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in ORDER]
    ax[0].legend(handles, [NAMES[k] for k in ORDER], ncol=4, fontsize=7.5, frameon=False,
                 loc="upper right", bbox_to_anchor=(1, 1.9))

    hr = np.array([w["hr"] if w.get("hr") else np.nan for w in windows], dtype=float)
    ax[1].plot(t, hr, color="#D85A30", lw=1.2)
    for y in (60, 100):
        ax[1].axhline(y, ls="--", lw=0.7, color="#888780")
    ax[1].set_ylabel("HR (bpm)", fontsize=8)
    ax[1].set_title("Laju jantung per jendela 10 detik (garis putus: 60 dan 100 bpm)",
                    loc="left", fontsize=9)

    conf = np.array([w["conf"] if w.get("conf") is not None else np.nan for w in windows])
    ax[2].plot(t, conf, color="#534AB7", lw=1)
    ax[2].axhline(0.8, ls="--", lw=0.7, color="#888780")
    bad = [w["t_end"] for w in windows if w.get("sqi") == "buruk"]
    if bad:
        ax[2].scatter(bad, np.full(len(bad), 0.05), marker="|", color="#A32D2D", s=60,
                      label="sinyal buruk")
        ax[2].legend(fontsize=7, frameon=False, loc="lower right")
    ax[2].set_ylim(0, 1.05)
    ax[2].set_ylabel("Confidence", fontsize=8)
    ax[2].set_xlabel("Waktu sejak sesi dimulai (detik)", fontsize=8)
    for a in ax:
        a.tick_params(labelsize=7.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    return fig


def example_figure(ex: dict):
    x = np.asarray(ex["window"])
    t = np.arange(len(x)) / FS
    top = x.max() + 1.6  # ruang ekstra di atas untuk label interval RR
    fig, ax = plt.subplots(figsize=(9, 2.1))
    if ex.get("cam"):
        ax.imshow(np.asarray(ex["cam"])[None, :], aspect="auto", cmap="Reds", alpha=0.55,
                  vmin=0, vmax=1, extent=[t[0], t[-1], x.min() - 0.4, top])
    ax.plot(t, x, color="#2C2C2A", lw=0.8)
    pk = np.asarray(ex.get("r_peaks") or [], dtype=int)
    if len(pk):
        ax.plot(t[pk], x[pk] + 0.35, "v", color="#185FA5", ms=3.5)
    rr = ex.get("rr_ms") or []
    label_y = x.max() + 1.0  # baris sejajar untuk semua label, tidak mengikuti tinggi denyut
    for i in range(min(len(rr), len(pk) - 1)):
        tm = (t[pk[i]] + t[pk[i + 1]]) / 2
        ax.text(tm, label_y, f"{rr[i]:.0f} ms", fontsize=6.5, ha="center", color="#534AB7")
    ax.set_yticks([])
    ax.set_ylim(x.min() - 0.4, top)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Detik dalam jendela", fontsize=7.5)
    ax.tick_params(labelsize=7.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Dokumen
# ---------------------------------------------------------------------------
def build_styles():
    ss = getSampleStyleSheet()
    base = dict(fontName="Helvetica", textColor=INK)
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold",
                                fontSize=16, leading=20, textColor=INK, spaceAfter=6),
        "h": ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=11.5, leading=15,
                            textColor=INK, spaceBefore=12, spaceAfter=5),
        "body": ParagraphStyle("b", **base, fontSize=9.5, leading=13.5),
        "small": ParagraphStyle("sm", **{**base, "textColor": MUTED}, fontSize=8, leading=11),
        "cell": ParagraphStyle("c", **base, fontSize=9, leading=12),
    }


def table(data, widths, header=True, align_right=()):
    tb = Table(data, colWidths=[w * mm for w in widths])
    st = [("FONT", (0, 0), (-1, -1), "Helvetica", 9),
          ("TEXTCOLOR", (0, 0), (-1, -1), INK),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
          ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE)]
    for c in align_right:
        st.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    if header:
        st += [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
               ("BACKGROUND", (0, 0), (-1, 0), SOFT),
               ("LINEBELOW", (0, 0), (-1, 0), 0.8, MUTED)]
    tb.setStyle(TableStyle(st))
    return tb


def afib_text(windows: list[dict], eps: list[tuple[float, float]]) -> str:
    """AFib hanya dinyatakan terdeteksi bila status ritme STABIL (episode); jendela tunggal
    disebut meragukan supaya tidak terbaca sebagai temuan."""
    if eps:
        return (f"<b>Terdeteksi: {len(eps)} episode</b>, total sekitar "
                f"{durasi_teks(sum(b - a for a, b in eps))}")
    af = [w for w in windows if w.get("label") == "AFIB"]
    if not af:
        return "<b>Tidak terdeteksi</b>"
    conf = max(w["conf"] for w in af)
    return (f"<b>Tidak terdeteksi</b> ({len(af)} jendela tunggal berlabel AFib tidak "
            f"terkonfirmasi, keyakinan tertinggi {conf:.0%})")


def generate_report(session_json: str, out_dir: str = "reports") -> Path:
    data = json.loads(Path(session_json).read_text())
    ses, summ, windows = data["session"], data["summary"], data["windows"]
    examples = data.get("examples", {})
    S = build_styles()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pdf_path = out / f"{ses['id']}.pdf"
    generated = datetime.now()

    valid = [w for w in windows if w.get("label")]
    n_all, n_valid = len(windows), len(valid)
    dur = windows[-1]["t_end"] if windows else 0
    counts = {k: sum(1 for w in valid if w["label"] == k) for k in ORDER}
    dominant = max(counts, key=counts.get) if n_valid else None
    eps = afib_episodes(windows)
    af_unconfirmed = counts["AFIB"] > 0 and not eps

    story = [Paragraph("LAPORAN PEMANTAUAN RITME JANTUNG", S["title"])]

    # --- Identitas sesi ---
    info = [["ID Sesi", ses["id"], "ID Pengguna", ses.get("patient_id", "ANONIM")],
            ["Tanggal", fmt_date(ses.get("started_at")), "Durasi", durasi_teks(dur)],
            ["Mulai", fmt_dt(ses.get("started_at")), "Selesai", fmt_dt(ses.get("ended_at"))]]
    tb = table(info, [28, 57, 28, 57], header=False)
    tb.setStyle(TableStyle([("FONT", (0, 0), (0, -1), "Helvetica-Bold", 9),
                            ("FONT", (2, 0), (2, -1), "Helvetica-Bold", 9),
                            ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
                            ("TEXTCOLOR", (2, 0), (2, -1), MUTED)]))
    story.append(tb)

    # --- Hasil pemantauan ---
    story.append(Paragraph("Hasil Pemantauan", S["h"]))
    if n_valid == 0:
        story.append(Paragraph("Tidak ada jendela dengan kualitas sinyal yang layak dianalisis.",
                               S["body"]))
    else:
        has_hr = summ.get("hr_median") is not None
        rows = [["Informasi", "Hasil"],
                ["Ritme dominan", Paragraph(f"<b>{NAMES[dominant]}</b>", S["cell"])],
                ["Deteksi AFib", Paragraph(afib_text(windows, eps), S["cell"])],
                ["Laju jantung", Paragraph(
                    f"<b>{summ['hr_min']:.0f}–{summ['hr_max']:.0f} bpm</b>" if has_hr
                    else "tidak terukur", S["cell"])],
                ["Laju jantung median", Paragraph(
                    f"<b>{summ['hr_median']:.0f} bpm</b>" if has_hr else "-", S["cell"])],
                ["Kualitas sinyal", Paragraph(
                    f"<b>{n_valid / max(n_all, 1):.0%} layak dianalisis</b>", S["cell"])]]
        story.append(table(rows, [50, 120]))
        story.append(Paragraph(
            "AFib dinyatakan terdeteksi hanya bila muncul pada sedikitnya 2 jendela berturut-turut "
            "(status ritme stabil).", S["small"]))

    # --- Log perubahan status ritme ---
    ev = summ.get("events", [])
    block = [Paragraph("Log Perubahan Status Ritme", S["h"])]
    if ev:
        rows = [["Waktu", "Ritme", "Laju Jantung"]]
        rows += [[mmss(e["t"]), NAMES.get(e.get("label"), e["label_name"]),
                  f"{e['hr']:.0f} bpm" if e.get("hr") else "-"] for e in ev[:40]]
        block.append(table(rows, [35, 90, 45], align_right=(2,)))
        if len(ev) > 40:
            block.append(Paragraph(f"... dan {len(ev) - 40} perubahan lainnya.", S["small"]))
    else:
        block.append(Paragraph("Tidak ada perubahan status ritme.", S["body"]))
    story.append(KeepTogether(block))

    if n_valid:
        # --- Proporsi per kategori ---
        story.append(Paragraph("Hasil per Kategori", S["h"]))
        rows = [["Ritme", "Proporsi"]]
        for k in sorted(ORDER, key=lambda k: -counts[k]):
            name = NAMES[k] + ("*" if k == "AFIB" and af_unconfirmed else "")
            rows.append([name, Paragraph(f"<b>{pct(counts[k] / n_valid)}</b>",
                                         ParagraphStyle("r", parent=S["cell"], alignment=2))])
        story.append(table(rows, [125, 45], align_right=(1,)))
        if af_unconfirmed:
            story.append(Paragraph("* Jendela tunggal yang tidak terkonfirmasi sebagai status "
                                   "ritme stabil.", S["small"]))

        # --- Linimasa ---
        story.append(KeepTogether([Paragraph("Linimasa Sesi", S["h"]),
                                   fig_to_image(timeline_figure(windows), 170)]))

    # --- Contoh sinyal + penjelasan ---
    if examples:
        story.append(Paragraph("Contoh Sinyal dan Penjelasan Model (Grad-CAM)", S["h"]))
        story.append(Paragraph(
            "Untuk setiap ritme yang terdeteksi, ditampilkan jendela dengan tingkat keyakinan "
            "tertinggi. Warna merah menunjukkan bagian sinyal yang paling memengaruhi keputusan "
            "model; segitiga biru menandai puncak R yang terdeteksi; angka ungu di antara dua "
            "puncak R berurutan adalah interval RR (ms) pada pasangan denyut tersebut.", S["small"]))
        for k in ORDER:
            ex = examples.get(k)
            if not ex:
                continue
            note = " - <i>tidak terkonfirmasi</i>" if k == "AFIB" and af_unconfirmed else ""
            head = (f"<b>{NAMES[k]}</b>{note} - pada {mmss(ex['t_start'])}-{mmss(ex['t_end'])} "
                    f"(menit:detik), keyakinan model {ex['conf']:.0%}")
            story.append(KeepTogether([
                Spacer(1, 4), Paragraph(head, S["body"]),
                fig_to_image(example_figure(ex), 170)]))

    # --- Catatan ---
    story.append(KeepTogether([Paragraph("Catatan", S["h"]), Paragraph(
        "Hasil ini merupakan <b>hasil pemantauan otomatis</b>, bukan diagnosis medis. Jika "
        "terdapat hasil yang mengkhawatirkan atau keluhan kesehatan, lakukan pemeriksaan lebih "
        "lanjut dengan tenaga kesehatan.", S["body"])]))

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(20 * mm, 10 * mm, f"{ses['id']}  |  dibuat {generated:%d-%m-%Y %H:%M}")
        canvas.drawRightString(190 * mm, 10 * mm, f"Halaman {doc.page}")
        canvas.setStrokeColor(LINE)
        canvas.line(20 * mm, 13 * mm, 190 * mm, 13 * mm)
        canvas.restoreState()

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=20 * mm,
                            rightMargin=20 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                            title=f"Laporan sesi {ses['id']}", author="Sistem klasifikasi aritmia",
                            subject="Laporan pemantauan ritme jantung")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return pdf_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("session_json")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    print("Laporan tersimpan:", generate_report(a.session_json, a.out))
