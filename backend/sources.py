"""
Sumber data EKG. Semua sumber menghasilkan potongan (chunk) numpy float64
dalam satuan mikrovolt pada 130 Hz, lewat async generator `stream()`.

  - PolarH10Source : langsung dari strap via Bluetooth Low Energy (bleak)
  - ReplaySource   : memutar ulang file rekaman secara real-time (mode demo)
"""
import asyncio
import logging

import numpy as np

from polar_io import load_polar_130
from preprocessing import TARGET_FS

log = logging.getLogger("sources")


# ---------------------------------------------------------------------------
# Replay dari file
# ---------------------------------------------------------------------------
class ReplaySource:
    name = "replay"

    def __init__(self, path: str, speed: float = 1.0, chunk: int = 73, loop: bool = True):
        self.path, self.speed, self.chunk, self.loop = path, speed, chunk, loop
        self.ecg = load_polar_130(path)
        log.info("Replay %s: %.1f detik", path, len(self.ecg) / TARGET_FS)

    async def stream(self):
        # chunk 73 sampel ~ ukuran paket notifikasi H10 (~0,56 s)
        while True:
            for i in range(0, len(self.ecg), self.chunk):
                yield self.ecg[i:i + self.chunk]
                await asyncio.sleep(self.chunk / TARGET_FS / self.speed)
            if not self.loop:
                return


# ---------------------------------------------------------------------------
# Polar H10 via BLE (protokol PMD)
# ---------------------------------------------------------------------------
PMD_CONTROL = "FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_DATA = "FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8"
# 0x02 = mulai pengukuran, 0x00 = ECG,
# 0x00 0x01 0x82 0x00 = sample rate 130 Hz, 0x01 0x01 0x0E 0x00 = resolusi 14 bit
ECG_START = bytearray([0x02, 0x00, 0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x0E, 0x00])
ECG_STOP = bytearray([0x03, 0x00])


def parse_pmd_ecg(data: bytearray) -> np.ndarray | None:
    """
    Format frame ECG H10:
      byte 0      : tipe pengukuran (0x00 = ECG)
      byte 1-8    : timestamp sensor (uint64, ns)
      byte 9      : tipe frame (0x00 = tidak terkompresi)
      byte 10...  : sampel int24 little-endian (signed), satuan uV
    """
    if len(data) < 10 or data[0] != 0x00 or data[9] != 0x00:
        return None
    payload = data[10:]
    n = len(payload) // 3
    samples = [int.from_bytes(payload[3 * i:3 * i + 3], "little", signed=True) for i in range(n)]
    return np.asarray(samples, dtype=np.float64)


class PolarH10Device:
    """
    Koneksi BLE ke Polar H10 yang bertahan di antara sesi: bisa dihubungkan dulu dari
    dashboard, lalu dipakai berulang kali. Stream EKG hanya aktif selama sesi berjalan.
    """

    def __init__(self, scan_timeout: float = 15.0, on_state=None):
        self.scan_timeout = scan_timeout
        self.on_state = on_state   # async callback(state, detail) untuk status di dashboard
        self.state, self.label = "idle", None
        self.client = None
        self._queue: asyncio.Queue | None = None
        self._lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self.client is not None and self.client.is_connected

    async def _state(self, state: str, detail: str | None = None):
        self.state = state
        if self.on_state:
            await self.on_state(state, detail)

    async def _find_device(self, address: str | None):
        from bleak import BleakScanner
        if address:
            dev = await BleakScanner.find_device_by_address(address, timeout=self.scan_timeout)
        else:
            dev = await BleakScanner.find_device_by_filter(
                lambda d, ad: bool(d.name) and "Polar H10" in d.name, timeout=self.scan_timeout)
        if dev is None:
            raise RuntimeError("Polar H10 tidak ditemukan. Pastikan strap dipakai (elektroda "
                               "basah) dan tidak sedang terhubung ke HP/aplikasi lain.")
        log.info("Ditemukan %s (%s)", dev.name, dev.address)
        return dev

    async def connect(self, address: str | None = None):
        from bleak import BleakClient
        async with self._lock:
            if self.connected:
                return
            client = None
            try:
                await self._state("scanning")
                dev = await self._find_device(address)
                self.label = f"{dev.name} ({dev.address})"
                await self._state("connecting", self.label)
                client = BleakClient(dev, disconnected_callback=self._on_disconnect)
                self.client = client
                await client.connect()
                await client.start_notify(PMD_CONTROL, self._on_control)
            except BaseException as e:
                self.client = None
                if client is not None and client.is_connected:
                    await client.disconnect()
                if isinstance(e, asyncio.CancelledError):
                    await self._state("idle")
                else:
                    await self._state("error", str(e))
                raise
            await self._state("connected", self.label)

    async def disconnect(self):
        client, self.client = self.client, None
        if client is not None and client.is_connected:
            try:
                await client.disconnect()
            except Exception:  # noqa: BLE001
                log.exception("Gagal memutus Polar H10")
        await self._state("idle")

    @staticmethod
    def _on_control(_, data: bytearray):
        # Respons: [0xF0, op_code, tipe, status, ...], status 0 = sukses
        if len(data) >= 4 and data[0] == 0xF0 and data[3] != 0:
            log.error("Polar menolak perintah (op %d, status %d)", data[1], data[3])

    def _on_disconnect(self, client):
        log.warning("Polar H10 terputus")
        if self._queue is not None:
            self._queue.put_nowait(None)
        if client is self.client:   # bukan diputus sendiri lewat disconnect()
            self.client = None
            asyncio.ensure_future(self._state("error", "Koneksi Polar H10 terputus"))

    async def stream(self):
        if not self.connected:
            raise RuntimeError("Polar H10 belum terhubung")
        client = self.client
        self._queue = queue = asyncio.Queue()

        def on_data(_, data: bytearray):
            s = parse_pmd_ecg(data)
            if s is not None and len(s):
                queue.put_nowait(s)

        try:
            await client.write_gatt_char(PMD_CONTROL, ECG_START, response=True)
            await client.start_notify(PMD_DATA, on_data)
            log.info("Streaming EKG dimulai")
            while True:
                chunk = await queue.get()
                if chunk is None:
                    raise RuntimeError("Koneksi Polar H10 terputus")
                yield chunk
        finally:
            self._queue = None
            if client.is_connected:
                try:
                    await client.stop_notify(PMD_DATA)
                    await client.write_gatt_char(PMD_CONTROL, ECG_STOP, response=True)
                except Exception:  # noqa: BLE001
                    pass


class PolarH10Source:
    """Sumber sesi dari Polar H10; menghubungkan dulu bila belum terhubung."""
    name = "polar"

    def __init__(self, device: PolarH10Device, address: str | None = None):
        self.device, self.address = device, address

    async def stream(self):
        await self.device.connect(self.address)
        async for chunk in self.device.stream():
            yield chunk
