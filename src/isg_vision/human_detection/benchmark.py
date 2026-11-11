"""Tekrarlanabilir CPU/MPS performans karşılaştırması (yalnızca hız/gecikme; doğruluk ölçümü DEĞİLDİR).

Yöntem
------
* Aynı video, aynı model, aynı eşikler, aynı ``imgsz``; yalnızca cihaz değişir.
* **Model yükleme**, **ısınma** (boş kare) ve **burn-in** (gerçek video karelerinde ölçüme alınmayan ilk ``burn_in`` kare)
  sürekli performanstan AYRI kaydedilir.
* Ölçülen ``passes`` geçişi cihazlar arasında **dönüşümlü sırayla** yapılır (``cpu,mps`` / ``mps,cpu`` / ...): ısınma/ısıl
  sürüklenme gibi sıra etkileri tek cihaza yazılmasın diye. Her geçiş ayrı raporlanır; geçişler arası fark, ölçümün gürültüsünü gösterir.
* **Yakalama hızı** (capture): aynı videonun modelsiz, çizimsiz, beklemesiz okunma hızı (``run_preview``); kameranın değil, dosya
  çözmenin hızıdır. Canlı kamera için ``run_person_detection.py --no-display`` ayrı çalıştırılır.
* **Uçtan uca FPS**: okuma + çıkarım döngüsü (pencere ve çizim yok) = ``run_detection`` özeti. Pencere/çizim dâhil DEĞİLDİR.
* Süreler ``time.perf_counter`` (çıkarım) ve ``time.monotonic`` (döngü) ile ölçülür; MPS'te çıkarım bitişinden önce senkronizasyon yapılır.

Kontrol EDİLEMEYEN etkenler: güç/ısıl durum, arka plandaki süreçler, macOS enerji modu. Bu yüzden sonuçlar bu makine ve bu oturum için
gözlemdir; evrensel performans iddiası değildir.

``make_synthetic_video`` ile üretilen video SENTETİKTİR (Ultralytics örnek görüntülerinden montaj). Yalnızca gecikme/FPS karşılaştırması
içindir; gerçek bir çalışma alanını temsil etmez ve doğruluk ölçümünde KULLANILMAZ.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from isg_vision.camera.capture import FrameSource
from isg_vision.camera.config import SourceConfig
from isg_vision.camera.preview import run_preview

from .config import DetectionConfig, resolve_device
from .errors import DetectionError, DeviceUnavailableError
from .person_detector import PersonDetector
from .pipeline import DetectionRunSummary, LatencyStats, run_detection

SYNTHETIC_DISCLAIMER = (
    "SENTETİK montaj (Ultralytics örnek görüntüleri); yalnızca hız/gecikme karşılaştırması içindir, doğruluk ölçümünde kullanılamaz."
)


@dataclass(frozen=True)
class CapturePass:
    frames: int
    elapsed_s: float
    fps: float


@dataclass
class DeviceBenchmark:
    """Tek cihazın sonuçları."""

    requested: str
    device: str = ""
    skipped_reason: str | None = None  # cihaz kullanılamadıysa neden (ör. MPS yok); bu durumda diğer alanlar boştur
    load_seconds: float | None = None
    warmup_seconds: float | None = None
    burn_in_frames: int = 0
    burn_in_seconds: float | None = None
    passes: list[DetectionRunSummary] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    video: str
    video_size: tuple[int, int] | None
    video_frames: int | None
    video_reported_fps: float | None
    frames_per_pass: int
    config: dict[str, Any]
    capture: list[CapturePass]
    devices: list[DeviceBenchmark]
    pass_order: list[list[str]]
    environment: dict[str, Any] = field(default_factory=dict)


def make_synthetic_video(path: Path, frames: int = 300, size: tuple[int, int] = (1280, 720), fps: float = 30.0) -> Path:
    """Ultralytics paketinin örnek görüntülerinden (indirme yok) deterministik, çok kişili SENTETİK bir video üretir.

    Sol tarafa ``bus.jpg`` (yayalar), sağ tarafa yatay kayan ``zidane.jpg`` yerleştirilir. Bkz. ``SYNTHETIC_DISCLAIMER``.
    """
    import importlib.util

    spec = importlib.util.find_spec("ultralytics")
    if spec is None or spec.origin is None:
        raise DetectionError("ultralytics kurulu değil; örnek görüntüler bulunamadı.")
    assets = Path(spec.origin).parent / "assets"
    bus, zidane = cv2.imread(str(assets / "bus.jpg")), cv2.imread(str(assets / "zidane.jpg"))
    if bus is None or zidane is None:
        raise DetectionError(f"Örnek görüntüler okunamadı: {assets}")
    width, height = size
    bus = cv2.resize(bus, (int(bus.shape[1] * height / bus.shape[0]), height))
    zidane = cv2.resize(zidane, (width // 2, height // 2))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise DetectionError(f"Video yazıcısı açılamadı: {path}")
    try:
        for i in range(frames):
            canvas = np.full((height, width, 3), 90, np.uint8)
            canvas[:, : bus.shape[1]] = bus
            shift = (i // 2) % (width // 4)  # sağdaki sahne yavaşça sola kayar
            x0 = width // 2 + shift
            visible = min(zidane.shape[1], width - x0)
            canvas[height // 4 : height // 4 + zidane.shape[0], x0 : x0 + visible] = zidane[:, :visible]
            writer.write(canvas)
    finally:
        writer.release()
    return path


def measure_capture(video: Path, passes: int, max_frames: int | None = None) -> list[CapturePass]:
    """Videonun modelsiz, çizimsiz, beklemesiz okunma hızı (``passes`` kez)."""
    results: list[CapturePass] = []
    for _ in range(passes):
        with FrameSource(SourceConfig.video(video)) as source:
            outcome = run_preview(source, None, max_frames=max_frames)
        results.append(CapturePass(outcome.frames_processed, outcome.elapsed_s, outcome.measured_fps))
    return results


def run_benchmark(
    video: Path,
    devices: Sequence[str],
    detection_config: DetectionConfig,
    *,
    passes: int = 3,
    burn_in: int = 10,
    max_frames: int | None = None,
    detector_factory: Callable[[DetectionConfig], Any] = PersonDetector,
    device_check: Callable[[str], Any] = resolve_device,
) -> BenchmarkReport:
    """Cihazları aynı video üzerinde karşılaştırır. Kullanılamayan cihaz atlanır ve nedeni kaydedilir (sessizce CPU'ya düşülmez)."""
    if isinstance(passes, bool) or not isinstance(passes, int) or passes < 1:
        raise ValueError(f"passes pozitif bir tam sayı olmalıdır: {passes!r}")
    if isinstance(burn_in, bool) or not isinstance(burn_in, int) or burn_in < 0:
        raise ValueError(f"burn_in negatif olmayan bir tam sayı olmalıdır: {burn_in!r}")
    if not devices:
        raise ValueError("En az bir cihaz verilmelidir.")

    with FrameSource(SourceConfig.video(video)) as probe:
        size, total, reported_fps = probe.actual_resolution, probe.frame_count, probe.source_fps
    frames_per_pass = min(total, max_frames) if (total and max_frames) else (max_frames or total or 0)

    capture = measure_capture(video, passes, max_frames)

    entries: list[DeviceBenchmark] = []
    detectors: dict[str, Any] = {}
    for requested in devices:
        entry = DeviceBenchmark(requested=requested)
        entries.append(entry)
        try:
            entry.device = device_check(requested).device
            cfg = DetectionConfig(model_path=detection_config.model_path, conf=detection_config.conf, iou=detection_config.iou,
                                  imgsz=detection_config.imgsz, device=requested, max_det=detection_config.max_det)
            detector = detector_factory(cfg)
            detector.load()
            entry.load_seconds = detector.load_seconds
            entry.warmup_seconds = detector.warmup(size or (640, 480), runs=2)
            if burn_in:
                start = time.monotonic()
                with FrameSource(SourceConfig.video(video)) as source:
                    burned = run_detection(source, detector, None, max_frames=burn_in)
                entry.burn_in_frames, entry.burn_in_seconds = burned.frames_processed, time.monotonic() - start
            detectors[requested] = detector
        except DeviceUnavailableError as exc:  # ör. MPS yok: yalnızca bu cihaz atlanır, neden kaydedilir.
            entry.skipped_reason = str(exc)   # Model/ağırlık hataları atlanmaz: benchmark durur (gerçek hata).

    active = [e.requested for e in entries if e.skipped_reason is None]
    order: list[list[str]] = []
    try:
        for p in range(passes):
            rotated = active[p % len(active):] + active[: p % len(active)] if active else []
            order.append(rotated)
            for requested in rotated:
                with FrameSource(SourceConfig.video(video)) as source:
                    summary = run_detection(source, detectors[requested], None, max_frames=max_frames)
                next(e for e in entries if e.requested == requested).passes.append(summary)
    finally:
        for detector in detectors.values():
            detector.close()

    return BenchmarkReport(
        video=str(video), video_size=size, video_frames=total, video_reported_fps=reported_fps, frames_per_pass=frames_per_pass,
        config={"model": detection_config.model_name, "conf": detection_config.conf, "iou": detection_config.iou,
                "imgsz": detection_config.imgsz, "max_det": detection_config.max_det, "passes": passes, "burn_in": burn_in},
        capture=capture, devices=entries, pass_order=order,
    )


# ---------------------------------------------------------------------- çıktı
def _stats(stats: LatencyStats | None) -> dict[str, float] | None:
    return None if stats is None else asdict(stats)


def report_to_dict(report: BenchmarkReport) -> dict[str, Any]:
    """JSON'a yazılabilir sözlük (görüntü içermez; yalnızca sayısal ölçümler)."""
    return {
        "video": report.video, "video_size": report.video_size, "video_frames": report.video_frames,
        "video_reported_fps": report.video_reported_fps, "frames_per_pass": report.frames_per_pass,
        "config": report.config, "environment": report.environment, "pass_order": report.pass_order,
        "capture_only": [asdict(c) for c in report.capture],
        "devices": [
            {
                "requested": d.requested, "device": d.device, "skipped_reason": d.skipped_reason,
                "load_seconds": d.load_seconds, "warmup_seconds": d.warmup_seconds,
                "burn_in_frames": d.burn_in_frames, "burn_in_seconds": d.burn_in_seconds,
                "passes": [
                    {
                        "frames": s.frames_processed, "elapsed_s": s.elapsed_s, "end_to_end_fps": s.end_to_end_fps,
                        "inference_ms": _stats(s.inference), "read_ms": _stats(s.read), "frame_to_result_ms": _stats(s.frame_to_result),
                        "persons_mean": s.persons_mean, "persons_max": s.persons_max, "dropped_boxes": s.dropped_boxes,
                    }
                    for s in d.passes
                ],
            }
            for d in report.devices
        ],
    }


def report_to_markdown(report: BenchmarkReport) -> str:
    """İnsan tarafından okunur özet tablo (Markdown)."""
    lines = [
        f"Video: `{Path(report.video).name}` | {report.video_size[0]}x{report.video_size[1]} | {report.frames_per_pass} kare/geçiş"
        if report.video_size else f"Video: `{Path(report.video).name}`",
        f"Model: {report.config['model']} | conf {report.config['conf']} | iou {report.config['iou']} | imgsz {report.config['imgsz']} | "
        f"{report.config['passes']} geçiş (dönüşümlü sıra) | burn-in {report.config['burn_in']} kare",
        "",
        "| Cihaz | Yükleme (sn) | Isınma (sn) | Geçiş | Uçtan uca FPS | Çıkarım medyan (ms) | Çıkarım ort (ms) | Çıkarım p95 (ms) | Çıkarım maks (ms) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for d in report.devices:
        if d.skipped_reason:
            lines.append(f"| {d.requested} | — | — | — | — | — | — | — | **atlandı:** {d.skipped_reason} |")
            continue
        for index, s in enumerate(d.passes, 1):
            inf = s.inference
            assert inf is not None
            first = index == 1
            lines.append(
                f"| {d.device} | {d.load_seconds:.2f} | {d.warmup_seconds:.2f} | {index} | {s.end_to_end_fps:.1f} | {inf.median_ms:.1f} | "
                f"{inf.mean_ms:.1f} | {inf.p95_ms:.1f} | {inf.max_ms:.1f} |" if first else
                f"| {d.device} | | | {index} | {s.end_to_end_fps:.1f} | {inf.median_ms:.1f} | {inf.mean_ms:.1f} | {inf.p95_ms:.1f} | {inf.max_ms:.1f} |"
            )
    cap = ", ".join(f"{c.fps:.1f}" for c in report.capture)
    lines += ["", f"Yakalama (modelsiz video okuma) FPS, geçişler: {cap}"]
    return "\n".join(lines)
