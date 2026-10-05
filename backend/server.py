"""
Backend real-time klasifikasi ritme.

Jalankan dari folder aritmia_ritme/:
  python backend/server.py --ckpt results/rhythm_cnn_best.pt
Lalu buka http://localhost:8000 untuk halaman uji.

REST:
  POST /api/session/start   {"source": "replay", "file": "rekaman.txt", "speed": 1}
                            {"source": "polar", "address": null}
  POST /api/session/stop
  GET  /api/status
WebSocket:
  /ws  -> pesan JSON bertipe "ecg", "rhythm", "status"
"""
import argparse
import asyncio
import hashlib
import json
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # modul backend
sys.path.insert(0, str(HERE.parent))   # preprocessing, model, gradcam, polar_io

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from processor import RhythmProcessor  # noqa: E402
from report import generate_report  # noqa: E402
from sources import PolarH10Device, PolarH10Source, ReplaySource  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
log = logging.getLogger("server")

CONFIG = {"ckpt": "results/rhythm_cnn_best.pt", "sessions_dir": "sessions", "reports_dir": "reports",
          "step_sec": 2.0, "polarity": "auto",
          "models_glob": ["results*/*.pt", "models/*.pt"]}   # checkpoint yang bisa dipilih


# ---------------------------------------------------------------------------
# WebSocket broadcast
# ---------------------------------------------------------------------------
class Hub:
    def __init__(self):
        self.clients: set[WebSocket] = set()

    async def send(self, msg: dict):
        data = json.dumps(msg)
        dead = []
        for ws in self.clients:
            try:
                await ws.send_text(data)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


hub = Hub()

# Status sistem untuk dashboard: pemuatan model & koneksi Polar H10
#   model : loading | ready | error
#   polar : idle | scanning | connecting | connected | error
system = {"model": "loading", "model_detail": None, "polar": "idle", "polar_detail": None}


async def set_system(**kw):
    system.update(kw)
    await hub.send({"type": "system", **system})


async def set_polar(state: str, detail: str | None = None):
    await set_system(polar=state, polar_detail=detail)


polar = PolarH10Device(on_state=set_polar)
polar_task: asyncio.Task | None = None   # pemindaian/koneksi yang dipicu dari dashboard


# ---------------------------------------------------------------------------
# Sesi
# ---------------------------------------------------------------------------
class Session:
    def __init__(self):
        self.task: asyncio.Task | None = None
        self.info: dict = {}
        self.results: list[dict] = []
        self.examples: dict = {}
        self.processor: RhythmProcessor | None = None
        self.model_task: asyncio.Task | None = None

    @property
    def running(self):
        return self.task is not None and not self.task.done()

    def load_model(self, force: bool = False) -> asyncio.Task:
        """Muat model di thread terpisah supaya server tetap responsif."""
        if force or self.model_task is None or (self.model_task.done() and self.processor is None):
            self.processor = None
            self.model_task = asyncio.create_task(self._load_model())
        return self.model_task

    async def _load_model(self):
        ckpt = CONFIG["ckpt"]
        await set_system(model="loading", model_detail=ckpt)
        try:
            self.processor = await asyncio.to_thread(
                RhythmProcessor, ckpt, CONFIG["step_sec"], polarity=CONFIG["polarity"])
        except Exception as e:  # noqa: BLE001
            log.exception("Gagal memuat model")
            await set_system(model="error", model_detail=f"{ckpt}: {e}")
            return
        log.info("Model dimuat: %s", ckpt)
        await set_system(model="ready", model_detail=ckpt)

    async def start(self, source, meta: dict):
        self.processor.reset()
        self.results = []
        self.examples = {}
        self.info = {"id": datetime.now().strftime("S-%Y%m%d-%H%M%S"),
                     "started_at": datetime.now().isoformat(timespec="seconds"),
                     "source": source.name, **meta}
        self.task = asyncio.create_task(self._run(source))
        await hub.send({"type": "status", "state": "running", **self.info})

    async def _run(self, source):
        error = None
        try:
            async for chunk in source.stream():
                await hub.send(self.processor.push(chunk))
                if self.processor.due():
                    msg = await asyncio.to_thread(self.processor.analyze)
                    self._record(msg)
                    await hub.send(msg)
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            error = str(e)
            log.exception("Sesi berhenti karena error")
        finally:
            path = self._save(error)
            report = None
            if path:
                try:
                    report = await asyncio.to_thread(generate_report, str(path),
                                                     CONFIG["reports_dir"])
                    log.info("Laporan PDF: %s", report)
                except Exception:  # noqa: BLE001
                    log.exception("Gagal membuat laporan PDF")
            await hub.send({"type": "status", "state": "stopped", "error": error,
                            "saved": str(path) if path else None,
                            "report": f"/api/sessions/{self.info['id']}/report" if report else None,
                            "report_sha256": sha256_file(Path(report)) if report else None,
                            **self.info})

    def _record(self, msg: dict):
        # Contoh terbaik per ritme (confidence tertinggi) untuk ditampilkan di laporan
        lab = msg.get("label")
        if lab and (lab not in self.examples or msg["conf"] > self.examples[lab]["conf"]):
            self.examples[lab] = {k: msg[k] for k in (
                "t_start", "t_end", "label", "label_name", "conf", "hr", "rr_cv", "rr_ms",
                "window", "cam", "r_peaks")}
        # Simpan ringkas (tanpa array sinyal) untuk laporan PDF nanti
        self.results.append({k: msg[k] for k in (
            "t_start", "t_end", "label", "label_name", "label_stable", "label_stable_name",
            "conf", "uncertain", "hr", "rr_cv", "rr_ms", "sqi", "sqi_reasons", "sqi_metrics",
            "probs")})

    def _save(self, error):
        if not self.results:
            return None
        out = Path(CONFIG["sessions_dir"])
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.info['id']}.json"
        summary = summarize(self.results)
        with open(path, "w") as f:
            json.dump({"session": {**self.info, "ended_at":
                                   datetime.now().isoformat(timespec="seconds"),
                                   "error": error},
                       "model": self.processor.model_info,
                       "summary": summary, "windows": self.results,
                       "examples": self.examples}, f)
        log.info("Sesi tersimpan: %s", path)
        return path

    async def stop(self):
        if self.running:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)


def summarize(results: list[dict]) -> dict:
    valid = [r for r in results if r["label"]]
    counts: dict[str, int] = {}
    for r in valid:
        counts[r["label_name"]] = counts.get(r["label_name"], 0) + 1
    hrs = [r["hr"] for r in valid if r["hr"]]
    # Kejadian = perubahan label berturut-turut (untuk log di laporan)
    # Kejadian = perubahan label STABIL (bukan label per jendela yang bisa berkedip)
    events, prev = [], None
    for r in valid:
        if r["label_stable"] and r["label_stable"] != prev:
            events.append({"t": r["t_end"], "label": r["label_stable"],
                           "label_name": r["label_stable_name"], "hr": r["hr"]})
            prev = r["label_stable"]
    return {"n_windows": len(results), "n_bad_signal": len(results) - len(valid),
            "counts": counts,
            "hr_min": min(hrs) if hrs else None, "hr_max": max(hrs) if hrs else None,
            "hr_median": sorted(hrs)[len(hrs) // 2] if hrs else None,
            "events": events}


def sha256_file(path: Path) -> str | None:
    """Sidik jari laporan PDF; dipakai dashboard untuk verifikasi keaslian file."""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


session = Session()

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app):
    session.load_model()   # muat model di latar belakang begitu server menyala
    yield
    await session.stop()
    await polar.disconnect()   # lepaskan strap supaya bisa dipakai perangkat lain


app = FastAPI(title="Backend Klasifikasi Aritmia", lifespan=lifespan)


class StartReq(BaseModel):
    source: str = "replay"
    file: str | None = None
    speed: float = 1.0
    address: str | None = None
    patient_id: str | None = None   # pseudonim, JANGAN nama asli


@app.post("/api/session/start")
async def start_session(req: StartReq):
    if session.running:
        raise HTTPException(409, "Sesi masih berjalan, hentikan dulu.")
    if session.processor is None:
        task = session.load_model()   # coba muat ulang bila sebelumnya gagal
        if not task.done():
            raise HTTPException(503, "Model masih dimuat, tunggu sebentar.")
        raise HTTPException(503, f"Model gagal dimuat: {system['model_detail']}")
    if req.source == "replay":
        if not req.file or not Path(req.file).exists():
            raise HTTPException(400, f"File tidak ditemukan: {req.file}")
        src = ReplaySource(req.file, speed=req.speed)
        meta = {"file": req.file, "speed": req.speed}
    elif req.source == "polar":
        src = PolarH10Source(polar, req.address)
        meta = {"address": req.address}
    else:
        raise HTTPException(400, "source harus 'replay' atau 'polar'")
    meta["patient_id"] = req.patient_id or "ANONIM"
    await session.start(src, meta)
    return {"ok": True, **session.info}


@app.post("/api/session/stop")
async def stop_session():
    await session.stop()
    return {"ok": True}


class PolarReq(BaseModel):
    address: str | None = None


@app.post("/api/polar/connect")
async def polar_connect(req: PolarReq):
    """Mulai pemindaian & koneksi di latar belakang; kemajuan dikirim lewat WebSocket."""
    global polar_task
    if not polar.connected and (polar_task is None or polar_task.done()):
        polar_task = asyncio.create_task(polar.connect(req.address))
        # Error sudah dilaporkan lewat status "error"; cegah warning "never retrieved"
        polar_task.add_done_callback(lambda t: t.cancelled() or t.exception())
    return {"ok": True, "state": polar.state}


@app.post("/api/polar/disconnect")
async def polar_disconnect():
    if session.running and session.info.get("source") == "polar":
        raise HTTPException(409, "Hentikan sesi dulu sebelum memutus Polar H10.")
    if polar_task is not None and not polar_task.done():
        polar_task.cancel()   # batalkan pemindaian yang sedang berjalan
        await asyncio.gather(polar_task, return_exceptions=True)
    await polar.disconnect()
    return {"ok": True}


_model_meta: dict = {}   # cache metadata checkpoint per (path, mtime)


def checkpoint_meta(path: Path) -> dict:
    key = (str(path), path.stat().st_mtime)
    if key not in _model_meta:
        import torch
        ck = torch.load(path, map_location="cpu")
        in_ch = ck.get("in_channels") or ck["model_state"]["features.0.0.weight"].shape[1]
        _model_meta[key] = {"epoch": ck.get("epoch"), "val_macro_f1": ck.get("val_macro_f1"),
                            "classes": ck.get("classes"), "in_channels": int(in_ch)}
    return _model_meta[key]


def available_models() -> list[dict]:
    paths = sorted({p for g in CONFIG["models_glob"] for p in Path(".").glob(g)})
    if Path(CONFIG["ckpt"]).exists():
        paths = sorted(set(paths) | {Path(CONFIG["ckpt"])})
    active = Path(CONFIG["ckpt"]).resolve()
    out = []
    for p in paths:
        item = {"path": str(p), "active": p.resolve() == active,
                "size_kb": round(p.stat().st_size / 1024),
                "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="minutes")}
        try:
            item.update(checkpoint_meta(p))
        except Exception as e:  # noqa: BLE001
            item["error"] = str(e)
        out.append(item)
    return out


@app.get("/api/models")
async def list_models():
    return await asyncio.to_thread(available_models)


class ModelReq(BaseModel):
    path: str


@app.post("/api/model")
async def select_model(req: ModelReq):
    if session.running:
        raise HTTPException(409, "Hentikan sesi dulu sebelum mengganti model.")
    if system["model"] == "loading":
        raise HTTPException(409, "Model sedang dimuat, tunggu sebentar.")
    # Hanya checkpoint dari daftar yang boleh dipilih (bukan path bebas dari klien)
    allowed = {m["path"] for m in await asyncio.to_thread(available_models)}
    if req.path not in allowed:
        raise HTTPException(400, f"Checkpoint tidak dikenal: {req.path}")
    CONFIG["ckpt"] = req.path
    session.load_model(force=True)
    return {"ok": True}


@app.get("/api/status")
async def status():
    return {"running": session.running, "n_windows": len(session.results), **session.info,
            "system": system}


@app.get("/api/sessions")
async def list_sessions():
    """Daftar sesi tersimpan (terbaru di atas) untuk halaman riwayat."""
    out = []
    for p in sorted(Path(CONFIG["sessions_dir"]).glob("S-*.json"), reverse=True):
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        sid = d["session"]["id"]
        pdf = Path(CONFIG["reports_dir"]) / f"{sid}.pdf"
        out.append({**d["session"], "summary": {k: v for k, v in d["summary"].items()
                                                if k != "events"},
                    "has_report": pdf.exists(), "report_sha256": sha256_file(pdf)})
    return out


@app.get("/api/sessions/{sid}/report")
async def get_report(sid: str, download: bool = False):
    """PDF laporan; default inline (ditampilkan di dashboard), ?download=1 untuk mengunduh."""
    pdf = Path(CONFIG["reports_dir"]) / f"{Path(sid).name}.pdf"
    if not pdf.exists():
        src = Path(CONFIG["sessions_dir"]) / f"{Path(sid).name}.json"
        if not src.exists():
            raise HTTPException(404, "Sesi tidak ditemukan")
        pdf = await asyncio.to_thread(generate_report, str(src), CONFIG["reports_dir"])
    return FileResponse(pdf, media_type="application/pdf", filename=pdf.name,
                        content_disposition_type="attachment" if download else "inline")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    hub.clients.add(ws)
    await ws.send_text(json.dumps({"type": "system", **system}))
    try:
        while True:
            await ws.receive_text()   # klien tidak perlu mengirim apa-apa
    except WebSocketDisconnect:
        hub.clients.discard(ws)


@app.get("/")
async def index():
    return FileResponse(HERE / "static" / "index.html")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CONFIG["ckpt"])
    ap.add_argument("--sessions_dir", default=CONFIG["sessions_dir"])
    ap.add_argument("--reports_dir", default=CONFIG["reports_dir"])
    ap.add_argument("--step", type=float, default=CONFIG["step_sec"])
    ap.add_argument("--polarity", choices=["auto", "normal", "invert"], default="auto")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    CONFIG.update(ckpt=a.ckpt, sessions_dir=a.sessions_dir, reports_dir=a.reports_dir, step_sec=a.step, polarity=a.polarity)
    log.info("Model: %s", a.ckpt)
    uvicorn.run(app, host=a.host, port=a.port)
