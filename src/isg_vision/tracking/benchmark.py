"""Takip (ByteTrack) ek yükünün tekrarlanabilir ölçümü: yalnızca-tespit ↔ tespit+takip, CPU ↔ MPS (doğruluk ölçümü DEĞİLDİR).

Yöntem (4A ``human_detection.benchmark`` ile aynı ilkeler)
-----------------------------------------------------------
* Aynı video, aynı YOLO modeli, aynı çözünürlük ve aynı eşikler; yalnızca cihaz ve mod değişir.
* **Model yükleme**, **takipçi yükleme**, **ısınma** (boş kare) ve **burn-in** (ölçüme alınmayan ilk ``burn_in`` kare) sürekli
  performanstan AYRI kaydedilir.
* Her ölçülen geçişte üç mod koşulur ve **sıra geçişler arasında döndürülür** (ısınma/ısıl sürüklenme tek moda yazılmasın diye):
  ``detect_only`` (``run_detection``: 4A döngüsü), ``track`` (``run_tracking``, pencere yok) ve ``track_draw`` (``run_tracking`` + hiçbir
  şey göstermeyen boş pencere: çizim maliyeti ölçülür, ``imshow`` maliyeti ÖLÇÜLEMEZ — o yalnızca gerçek pencerede, kullanıcının makinesinde).
* **Takip ek yükü** = ``track`` modunun kare süresi (1000/FPS) − ``detect_only`` modunun kare süresi, aynı oturumda ölçülen geçişlerin
  ortancası; ayrıca doğrudan ölçülen ``tracking`` süresiyle (``TrackingResult.tracking_ms``) karşılaştırılır.
* Süreler ``time.perf_counter`` (çıkarım, takip) ve ``time.monotonic`` (döngü) ile ölçülür; MPS'te çıkarım bitişinden önce senkronizasyon yapılır.
  ByteTrack her zaman CPU'da (NumPy) çalışır; MPS senkronizasyonu takip süresine karışmaz.
* ``measure_tracker_scaling``: YOLO OLMADAN, betiklenmiş sahte tespitlerle ByteTrack güncelleme süresinin kişi sayısıyla ölçeklenmesi.

Kontrol EDİLEMEYEN etkenler: güç/ısıl durum, arka plandaki süreçler, macOS enerji modu. Sonuçlar bu makine ve bu oturum için gözlemdir;
evrensel performans iddiası değildir. Sentetik video (``make_synthetic_video``) yalnızca hız/entegrasyon içindir; takip DOĞRULUĞU çıkarılamaz.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from isg_vision.camera.capture import FrameSource
from isg_vision.camera.config import SourceConfig
from isg_vision.human_detection.config import DetectionConfig, resolve_device
from isg_vision.human_detection.errors import DeviceUnavailableError
from isg_vision.human_detection.pipeline import LatencyStats, run_detection
from isg_vision.human_detection.person_detector import PersonDetector
from isg_vision.human_detection.results import DetectionResult, PersonDetection

from .bytetrack_adapter import PersonTracker
from .config import TrackingConfig
from .pipeline import TrackingRunSummary, run_tracking

MODES = ("detect_only", "track", "track_draw")
SCALING_COUNTS = (1, 5, 10, 20, 50, 100)


class _NullWindow:
    """Hiçbir şey göstermeyen pencere: ``run_tracking`` çizimi yapar ama ``imshow`` maliyeti yoktur."""

    closed_by_user = False

    def show(self, image: Any, delay_ms: int = 1) -> int:
        return -1


@dataclass
class ModeRun:
    """Tek bir geçişteki tek bir modun sonucu (süreler ms; örnek yoksa ``None``)."""

    mode: str
    frames: int
    elapsed_s: float
    end_to_end_fps: float
    read: LatencyStats | None
    inference: LatencyStats | None
    tracking: LatencyStats | None = None
    detect_plus_track: LatencyStats | None = None
    frame_to_result: LatencyStats | None = None
    draw: LatencyStats | None = None
    detections_mean: float = 0.0
    distinct_tracks: int | None = None
    active_max: int | None = None


@dataclass
class DeviceTrackingBenchmark:
    requested: str
    device: str = ""
    skipped_reason: str | None = None
    model_load_seconds: float | None = None
    tracker_load_seconds: float | None = None
    warmup_seconds: float | None = None
    burn_in_frames: int = 0
    burn_in_seconds: float | None = None
    runs: list[list[ModeRun]] = field(default_factory=list)  # geçiş -> modlar (o geçişte koşulan sırayla)


@dataclass
class ScalingRow:
    persons: int
    frames: int
    tracking: LatencyStats | None
    active_last: int


@dataclass
class TrackingBenchmarkReport:
    video: str
    video_size: tuple[int, int] | None
    video_frames: int | None
    frames_per_pass: int
    config: dict[str, Any]
    devices: list[DeviceTrackingBenchmark]
    pass_order: list[list[str]]
    mode_order: list[list[str]]
    scaling: list[ScalingRow] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)


def _from_detection(summary: Any) -> ModeRun:
    return ModeRun("detect_only", summary.frames_processed, summary.elapsed_s, summary.end_to_end_fps, summary.read, summary.inference,
                   frame_to_result=summary.frame_to_result, detections_mean=summary.persons_mean)


def _from_tracking(mode: str, summary: TrackingRunSummary) -> ModeRun:
    return ModeRun(mode, summary.frames_processed, summary.elapsed_s, summary.end_to_end_fps, summary.read, summary.inference, summary.tracking,
                   summary.detect_plus_track, summary.frame_to_result, summary.draw, summary.detections_mean, summary.distinct_tracks, summary.active_max)


def run_tracking_benchmark(
    video: Path,
    devices: Sequence[str],
    detection_config: DetectionConfig,
    tracking_config: TrackingConfig,
    *,
    passes: int = 3,
    burn_in: int = 10,
    max_frames: int | None = None,
    detector_factory: Callable[[DetectionConfig], Any] = PersonDetector,
    tracker_factory: Callable[[TrackingConfig], Any] = PersonTracker,
    device_check: Callable[[str], Any] = resolve_device,
) -> TrackingBenchmarkReport:
    """Cihazları ve modları aynı video üzerinde karşılaştırır. Kullanılamayan cihaz atlanır ve nedeni kaydedilir (sessizce CPU'ya düşülmez)."""
    if isinstance(passes, bool) or not isinstance(passes, int) or passes < 1:
        raise ValueError(f"passes pozitif bir tam sayı olmalıdır: {passes!r}")
    if isinstance(burn_in, bool) or not isinstance(burn_in, int) or burn_in < 0:
        raise ValueError(f"burn_in negatif olmayan bir tam sayı olmalıdır: {burn_in!r}")
    if not devices:
        raise ValueError("En az bir cihaz verilmelidir.")

    with FrameSource(SourceConfig.video(video)) as probe:
        size, total = probe.actual_resolution, probe.frame_count
    frames_per_pass = min(total, max_frames) if (total and max_frames) else (max_frames or total or 0)

    entries: list[DeviceTrackingBenchmark] = []
    detectors: dict[str, Any] = {}
    trackers: dict[str, Any] = {}
    for requested in devices:
        entry = DeviceTrackingBenchmark(requested=requested)
        entries.append(entry)
        try:
            entry.device = device_check(requested).device
            cfg = DetectionConfig(model_path=detection_config.model_path, conf=detection_config.conf, iou=detection_config.iou,
                                  imgsz=detection_config.imgsz, device=requested, max_det=detection_config.max_det)
            detector = detector_factory(cfg)
            detector.load()
            entry.model_load_seconds = detector.load_seconds
            tracker = tracker_factory(tracking_config)
            start = time.perf_counter()
            tracker.load()
            entry.tracker_load_seconds = time.perf_counter() - start
            entry.warmup_seconds = detector.warmup(size or (640, 480), runs=2)
            if burn_in:
                start = time.monotonic()
                with FrameSource(SourceConfig.video(video)) as source:
                    burned = run_tracking(source, detector, tracker, None, max_frames=burn_in)
                entry.burn_in_frames, entry.burn_in_seconds = burned.frames_processed, time.monotonic() - start
            detectors[requested], trackers[requested] = detector, tracker
        except DeviceUnavailableError as exc:  # ör. MPS yok: yalnızca bu cihaz atlanır, neden kaydedilir.
            entry.skipped_reason = str(exc)    # Model/ağırlık/bağımlılık hataları atlanmaz: benchmark durur (gerçek hata).

    active = [e.requested for e in entries if e.skipped_reason is None]
    device_order: list[list[str]] = []
    mode_order: list[list[str]] = []
    try:
        for p in range(passes):
            rotated_devices = active[p % len(active):] + active[: p % len(active)] if active else []
            rotated_modes = list(MODES[p % len(MODES):] + MODES[: p % len(MODES)])
            device_order.append(rotated_devices)
            mode_order.append(rotated_modes)
            for requested in rotated_devices:
                runs: list[ModeRun] = []
                for mode in rotated_modes:
                    with FrameSource(SourceConfig.video(video)) as source:
                        if mode == "detect_only":
                            runs.append(_from_detection(run_detection(source, detectors[requested], None, max_frames=max_frames)))
                        else:
                            window = _NullWindow() if mode == "track_draw" else None
                            runs.append(_from_tracking(mode, run_tracking(source, detectors[requested], trackers[requested], window, max_frames=max_frames)))
                next(e for e in entries if e.requested == requested).runs.append(runs)
    finally:
        for detector in detectors.values():
            detector.close()
        for tracker in trackers.values():
            tracker.close()

    return TrackingBenchmarkReport(
        video=str(video), video_size=size, video_frames=total, frames_per_pass=frames_per_pass,
        config={"model": detection_config.model_name, "conf": detection_config.conf, "iou": detection_config.iou, "imgsz": detection_config.imgsz,
                "max_det": detection_config.max_det, "passes": passes, "burn_in": burn_in, "tracking": tracking_config.bytetrack_args()},
        devices=entries, pass_order=device_order, mode_order=mode_order,
    )


# ---------------------------------------------------------------------- ByteTrack ölçeklenmesi (YOLO yok)
def scripted_detections(persons: int, frame_index: int, size: tuple[int, int] = (1280, 720)) -> DetectionResult:
    """Izgara düzeninde, kare başına birkaç piksel kayan ``persons`` sahte insan (örtüşme yok). Yalnızca takip hız ölçümü içindir."""
    width, height = size
    cols = max(1, min(persons, width // 60))
    rows = (persons + cols - 1) // cols
    cell_w, cell_h = width // cols, height // max(rows, 1)
    bw, bh = min(80, max(8, int(cell_w * 0.5))), min(200, max(16, int(cell_h * 0.7)))  # gerçekçi insan boyutu üst sınırı
    drift = abs((frame_index % 40) - 20) * 1.0  # üçgen dalga: ani sıçrama yok (iz kopmaz)
    people = []
    for k in range(persons):
        r, c = divmod(k, cols)
        x1 = c * cell_w + cell_w * 0.1 + drift * 0.5
        y1 = r * cell_h + cell_h * 0.1 + drift * 0.3
        people.append(PersonDetection(0, "person", 0.9 - 0.001 * k, (x1, y1, x1 + bw, y1 + bh), frame_index=frame_index, timestamp=frame_index / 30.0))
    return DetectionResult(tuple(people), size, frame_index, frame_index / 30.0, 0.0, "cpu")


def measure_tracker_scaling(
    counts: Sequence[int] = SCALING_COUNTS,
    frames: int = 300,
    tracking_config: TrackingConfig | None = None,
    *,
    burn_in: int = 20,
    tracker_factory: Callable[[TrackingConfig], Any] = PersonTracker,
) -> list[ScalingRow]:
    """ByteTrack güncelleme süresinin kişi sayısıyla ölçeklenmesi (YOLO yok; CPU). İlk ``burn_in`` kare ölçüme alınmaz."""
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise ValueError(f"frames pozitif bir tam sayı olmalıdır: {frames!r}")
    rows: list[ScalingRow] = []
    for persons in counts:
        if isinstance(persons, bool) or not isinstance(persons, int) or persons < 0:
            raise ValueError(f"kişi sayısı negatif olmayan tam sayı olmalıdır: {persons!r}")
        tracker = tracker_factory(tracking_config or TrackingConfig())
        tracker.load()
        samples: list[float] = []
        last_active = 0
        for i in range(burn_in + frames):
            result = tracker.update(scripted_detections(persons, i))
            last_active = result.active_count
            if i >= burn_in:
                samples.append(result.tracking_ms)
        tracker.close()
        rows.append(ScalingRow(persons, frames, LatencyStats.from_values(samples), last_active))
    return rows


# ---------------------------------------------------------------------- özetler
def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def overhead_summary(entry: DeviceTrackingBenchmark) -> dict[str, float | None]:
    """Cihaz için geçişler boyunca ortanca değerler: FPS'ler, kare süreleri ve takip ek yükü (ms/kare)."""

    def pick(mode: str, attr: str) -> list[float]:
        out = []
        for runs in entry.runs:
            for run in runs:
                if run.mode == mode:
                    value = getattr(run, attr)
                    value = value.median_ms if isinstance(value, LatencyStats) else value
                    if value is not None:
                        out.append(float(value))
        return out

    fps_detect, fps_track, fps_draw = (pick(m, "end_to_end_fps") for m in MODES)
    frame_detect = [1000.0 / f for f in fps_detect if f > 0]
    frame_track = [1000.0 / f for f in fps_track if f > 0]
    frame_draw = [1000.0 / f for f in fps_draw if f > 0]
    med = _median
    detect_ms, track_ms, draw_ms = med(frame_detect), med(frame_track), med(frame_draw)
    return {
        "fps_detect_only": med(fps_detect), "fps_track": med(fps_track), "fps_track_draw": med(fps_draw),
        "frame_ms_detect_only": detect_ms, "frame_ms_track": track_ms, "frame_ms_track_draw": draw_ms,
        "overhead_track_ms_per_frame": (track_ms - detect_ms) if (track_ms is not None and detect_ms is not None) else None,
        "overhead_draw_ms_per_frame": (draw_ms - track_ms) if (draw_ms is not None and track_ms is not None) else None,
        "inference_median_ms_detect_only": med(pick("detect_only", "inference")),
        "inference_median_ms_track": med(pick("track", "inference")),
        "tracking_median_ms": med(pick("track", "tracking")),
        "detect_plus_track_median_ms": med(pick("track", "detect_plus_track")),
        "frame_to_result_median_ms_detect_only": med(pick("detect_only", "frame_to_result")),
        "frame_to_result_median_ms_track": med(pick("track", "frame_to_result")),
        "draw_median_ms": med(pick("track_draw", "draw")),
        "read_median_ms": med(pick("track", "read")),
    }


def _stats(stats: LatencyStats | None) -> dict[str, float] | None:
    return None if stats is None else asdict(stats)


def report_to_dict(report: TrackingBenchmarkReport) -> dict[str, Any]:
    """JSON'a yazılabilir sözlük (görüntü içermez; yalnızca sayısal ölçümler)."""
    return {
        "video": report.video, "video_size": report.video_size, "video_frames": report.video_frames, "frames_per_pass": report.frames_per_pass,
        "config": report.config, "environment": report.environment, "pass_order": report.pass_order, "mode_order": report.mode_order,
        "devices": [
            {
                "requested": d.requested, "device": d.device, "skipped_reason": d.skipped_reason,
                "model_load_seconds": d.model_load_seconds, "tracker_load_seconds": d.tracker_load_seconds, "warmup_seconds": d.warmup_seconds,
                "burn_in_frames": d.burn_in_frames, "burn_in_seconds": d.burn_in_seconds,
                "summary": overhead_summary(d) if d.runs else None,
                "passes": [
                    [
                        {
                            "mode": r.mode, "frames": r.frames, "elapsed_s": r.elapsed_s, "end_to_end_fps": r.end_to_end_fps,
                            "read_ms": _stats(r.read), "inference_ms": _stats(r.inference), "tracking_ms": _stats(r.tracking),
                            "detect_plus_track_ms": _stats(r.detect_plus_track), "frame_to_result_ms": _stats(r.frame_to_result),
                            "draw_ms": _stats(r.draw), "detections_mean": r.detections_mean, "distinct_tracks": r.distinct_tracks,
                            "active_max": r.active_max,
                        }
                        for r in runs
                    ]
                    for runs in d.runs
                ],
            }
            for d in report.devices
        ],
        "tracker_scaling": [{"persons": s.persons, "frames": s.frames, "tracking_ms": _stats(s.tracking), "active_last": s.active_last} for s in report.scaling],
    }


def _fmt(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def report_to_markdown(report: TrackingBenchmarkReport) -> str:
    """İnsan tarafından okunur özet tablolar (Markdown). Cihaz yoksa (yalnızca ölçeklenme raporu) yalnızca ölçeklenme tablosu yazılır."""
    lines: list[str] = []
    if report.devices:
        lines += [
            (f"Video: `{Path(report.video).name}` | {report.video_size[0]}x{report.video_size[1]} | {report.frames_per_pass} kare/geçiş"
             if report.video_size else f"Video: `{Path(report.video).name}`"),
            f"Model: {report.config['model']} | conf {report.config['conf']} | iou {report.config['iou']} | imgsz {report.config['imgsz']} | "
            f"{report.config['passes']} geçiş (cihaz ve mod sırası döndürülür) | burn-in {report.config['burn_in']} kare",
            "",
            "| Cihaz | Model yükleme (sn) | Takipçi yükleme (sn) | Isınma (sn) | FPS yalnızca-tespit | FPS tespit+takip | FPS tespit+takip+çizim | "
            "Takip ek yükü (ms/kare) | Çıkarım medyan (ms) | ByteTrack medyan (ms) | Çizim medyan (ms) | Kare→sonuç medyan, takip (ms) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    for d in report.devices:
        if d.skipped_reason:
            lines.append(f"| {d.requested} | — | — | — | — | — | — | — | — | — | — | **atlandı:** {d.skipped_reason} |")
            continue
        s = overhead_summary(d)
        lines.append(
            f"| {d.device} | {_fmt(d.model_load_seconds, 2)} | {_fmt(d.tracker_load_seconds, 2)} | {_fmt(d.warmup_seconds, 2)} | {_fmt(s['fps_detect_only'])} | "
            f"{_fmt(s['fps_track'])} | {_fmt(s['fps_track_draw'])} | {_fmt(s['overhead_track_ms_per_frame'], 2)} | {_fmt(s['inference_median_ms_track'])} | "
            f"{_fmt(s['tracking_median_ms'], 2)} | {_fmt(s['draw_median_ms'], 2)} | {_fmt(s['frame_to_result_median_ms_track'])} |"
        )
    if report.scaling:
        lines += ["" if lines else "ByteTrack ölçeklenmesi (YOLO yok, CPU, betiklenmiş tespitler):"]
        if report.devices:
            lines += ["ByteTrack ölçeklenmesi (YOLO yok, CPU, betiklenmiş tespitler):"]
        lines += ["", "| Kişi sayısı | ByteTrack medyan (ms) | p95 (ms) | maks (ms) | aktif iz (son kare) |", "|---|---|---|---|---|"]
        for row in report.scaling:
            st = row.tracking
            lines.append(f"| {row.persons} | {_fmt(st.median_ms if st else None, 3)} | {_fmt(st.p95_ms if st else None, 3)} | {_fmt(st.max_ms if st else None, 3)} | {row.active_last} |")
    return "\n".join(lines)
