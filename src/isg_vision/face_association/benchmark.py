"""Birleşik hattın (YOLO11n + ByteTrack + YuNet + ilişkilendirme) performans ve OpenCV iş parçacığı ölçümü (``scripts/benchmark_face_person_association.py``).

**Aşama 5B'ye aittir; 4A/4B/5A performans sonuçları burada kullanılmaz ve 5B sonucu gibi sunulmaz. Hiçbir ölçüm gerçek kamera ölçümü DEĞİLDİR:** girdi, bilinen
bir SENTETİK montaj videosudur (``scenes``); kayıtlı videodan ölçülen FPS canlı kamera FPS'i gibi sunulmamalıdır (kameranın kare hızı, pozlama ve sürücü tamponlaması
yoktur).

Neden alt süreç? ``ultralytics`` içe aktarılınca süreç genelinde ``cv2.setNumThreads(0)`` çağırır ve OpenCV iş parçacığı durumu süreç boyunca kalır. Her koşulu
temiz ölçmek için her varyant YENİ bir alt süreçte (``--worker``) çalıştırılır; üst süreç yalnızca sonuçları toplar.

İki işçi modu:

* ``probe`` — yalnızca YuNet (ve isteğe bağlı YOLO) süreleri, AYNI süreçte üç aşamada: (1) hiçbir şey içe aktarılmadan önce, (2) ``ultralytics`` içe aktarıldıktan sonra,
  (3) isteğe bağlı ``cv2.setNumThreads(N)`` sonrası. Her aşamada ``cv2.getNumThreads()`` kaydedilir (bu OpenCV yapısında ``setNumThreads(1)`` sonrası da 10 gösterebilir;
  bu yüzden güvenilir gösterge değil, kayıt amaçlıdır).
* ``pipeline`` — kayıtlı videoda tam hat (``run_face_association``), ``--device`` ve ``--opencv-threads`` ile; kare başına süreler, FPS ve bir "imza" (her karedeki
  Track ID/eşleşme çiftleri) döner. Farklı iş parçacığı ayarlarında imzalar AYNI ise ayar davranışı değiştirmemiştir (yan etki denetimi).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SYNTHETIC_DISCLAIMER = (
    "Ölçümler kayıtlı SENTETİK montaj videosuyla yapılmıştır (aynı üç kişi kırpıntısı kaydırılır); gerçek kamera, gerçek sahne veya doğruluk ölçümü değildir. "
    "Kayıtlı videodan ölçülen FPS canlı kamera FPS'i olarak sunulmamalıdır."
)
DEFAULT_FRAMES = 240
PROBE_REPEATS = 60
PROBE_DISCARD = 10


# ---------------------------------------------------------------------- video üretimi
def make_benchmark_video(path: Path, frames: int = DEFAULT_FRAMES, scenario: str = "three_people") -> Path:
    """``scenes`` senaryosunu (varsayılan: üç kişi yan yana) ``frames`` kareye uzatıp mp4 olarak yazar. Bkz. ``SYNTHETIC_DISCLAIMER``."""
    from . import scenes  # noqa: PLC0415 (cv2/numpy; ultralytics içe aktarmaz)

    actors = scenes.load_actors()
    base = scenes.SCENARIOS[scenario](max(frames, 2))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, scenes.CANVAS)
    if not writer.isOpened():
        raise RuntimeError(f"Video yazıcısı açılamadı: {path}")
    try:
        for placements in base[:frames]:
            writer.write(scenes.render_frame(actors, placements).image)
    finally:
        writer.release()
    return path


# ---------------------------------------------------------------------- işçi: probe
def _summ(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {"median_ms": statistics.median(ordered), "mean_ms": statistics.fmean(ordered), "p95_ms": ordered[max(0, int(0.95 * len(ordered)) - 1)],
            "min_ms": ordered[0], "max_ms": ordered[-1], "n": len(ordered)}


def _time_yunet(detector: Any, image: np.ndarray) -> dict[str, float]:
    times = [detector.detect(image).inference_ms for _ in range(PROBE_REPEATS)]
    return _summ(times[PROBE_DISCARD:])


def _time_yolo(detector: Any, image: np.ndarray) -> dict[str, float]:
    times = [detector.detect(image).inference_ms for _ in range(PROBE_REPEATS)]
    return _summ(times[PROBE_DISCARD:])


def run_probe_worker(*, set_threads: int | None, device: str, with_yolo: bool, image_path: Path | None = None) -> dict[str, Any]:
    """Tek süreçte üç aşamalı OpenCV iş parçacığı sondası (bkz. modül açıklaması). ``ultralytics`` bu fonksiyonun İÇİNDE, ikinci aşamada içe aktarılır."""
    os.environ.setdefault("YOLO_OFFLINE", "1")
    from isg_vision.face_detection.benchmark import find_sample_images  # noqa: PLC0415
    from isg_vision.face_detection.detector import FaceDetector  # noqa: PLC0415

    path = image_path or find_sample_images()["zidane"]  # ultralytics İÇE AKTARILMADAN bulunur
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Görüntü okunamadı: {path}")
    face = FaceDetector()
    face.load()
    face.warmup((image.shape[1], image.shape[0]), runs=3)
    stages: list[dict[str, Any]] = []

    def record(name: str, **extra: Any) -> None:
        stages.append({"stage": name, "cv2_threads": int(cv2.getNumThreads()), "yunet": _time_yunet(face, image), **extra})

    record("1_before_ultralytics")
    from isg_vision.human_detection.config import DetectionConfig  # noqa: PLC0415
    from isg_vision.human_detection.person_detector import PersonDetector  # noqa: PLC0415

    yolo = PersonDetector(DetectionConfig(device=device))
    yolo.load()  # ultralytics burada içe aktarılır
    yolo.warmup((image.shape[1], image.shape[0]), runs=3)
    record("2_after_ultralytics", yolo=_time_yolo(yolo, image) if with_yolo else None, yolo_device=yolo.device)
    if set_threads is not None:
        cv2.setNumThreads(set_threads)
        record(f"3_after_setNumThreads({set_threads})", yolo=_time_yolo(yolo, image) if with_yolo else None, yolo_device=yolo.device)
    return {"mode": "probe", "image": path.name, "image_size": [image.shape[1], image.shape[0]], "stages": stages, "set_threads": set_threads,
            "environment": environment_info()}


# ---------------------------------------------------------------------- işçi: pipeline
def result_signature(results: Sequence[Any]) -> str:
    """Her karedeki ``(track_id, eşleşen yüz sırası, durum)`` çiftlerinin özeti (SHA-256 kısaltması); davranış değişimini tespit etmek içindir."""
    digest = hashlib.sha256()
    for r in results:
        digest.update(repr((r.frame_index, tuple((p.track_id, p.face_index, p.status) for p in r.persons))).encode())
    return digest.hexdigest()[:16]


def run_pipeline_worker(*, video: Path, device: str, opencv_threads: str, max_frames: int | None) -> dict[str, Any]:
    """Kayıtlı videoda tam hat (penceresiz). ``cli.run`` ile AYNI sıra: modeller yüklenir -> iş parçacığı ayarı -> kaynak -> ısınma -> döngü."""
    os.environ.setdefault("YOLO_OFFLINE", "1")
    from isg_vision.camera.capture import FrameSource  # noqa: PLC0415
    from isg_vision.camera.config import SourceConfig  # noqa: PLC0415
    from isg_vision.face_detection.detector import FaceDetector  # noqa: PLC0415
    from isg_vision.human_detection.config import DetectionConfig  # noqa: PLC0415
    from isg_vision.human_detection.person_detector import PersonDetector  # noqa: PLC0415
    from isg_vision.tracking.bytetrack_adapter import PersonTracker  # noqa: PLC0415

    from .associator import FacePersonAssociator  # noqa: PLC0415
    from .pipeline import run_face_association  # noqa: PLC0415
    from .threads import apply_opencv_threads  # noqa: PLC0415

    before_import = int(cv2.getNumThreads())
    detector = PersonDetector(DetectionConfig(device=device))
    tracker = PersonTracker()
    face = FaceDetector()
    associator = FacePersonAssociator()
    detector.load()
    tracker.load()
    face.load()
    after_import = int(cv2.getNumThreads())
    setting = apply_opencv_threads(opencv_threads)
    results: list[Any] = []
    with FrameSource(SourceConfig.video(video)) as source:
        size = source.actual_resolution or (1280, 720)
        detector.warmup(size, runs=2)
        face.warmup(size, runs=2)
        summary = run_face_association(source, detector, tracker, face, associator, None, max_frames=max_frames,
                                       on_result=lambda p, d, t, f, a: results.append(a))
    stats = {name: (None if getattr(summary, name) is None else {"median_ms": getattr(summary, name).median_ms, "mean_ms": getattr(summary, name).mean_ms,
                                                                   "p95_ms": getattr(summary, name).p95_ms, "n": getattr(summary, name).count})
             for name in ("read", "yolo", "tracking", "face", "association", "compute", "frame_to_result")}
    return {
        "mode": "pipeline", "video": video.name, "device": detector.device, "opencv_threads_request": opencv_threads, "cv2_threads_before_imports": before_import,
        "cv2_threads_after_model_load": after_import, "cv2_threads_after_setting": setting.after, "set_value": setting.value, "frames": summary.frames_processed,
        "elapsed_s": summary.elapsed_s, "end_to_end_fps": summary.end_to_end_fps, "latency": stats, "persons_total": summary.persons_total,
        "faces_total": summary.faces_total, "matched_total": summary.matched_total, "trusted_matched_total": summary.trusted_matched_total,
        "invalid_frames": summary.invalid_frames, "distinct_tracks": summary.distinct_tracks, "break_counts": summary.break_counts,
        "signature": result_signature(results), "environment": environment_info(),
    }


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(), "cv2": cv2.__version__,
                            "cpu_count": os.cpu_count()}
    try:
        info["chip"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        info["chip"] = None
    return info


# ---------------------------------------------------------------------- üst süreç: alt süreçleri çalıştırıp toplar
Runner = Callable[[list[str]], dict[str, Any]]


def subprocess_runner(script: Path, *, timeout: float = 900.0) -> Runner:
    """``script --worker ...`` komutunu YENİ bir alt süreçte çalıştırıp son JSON satırını döndüren çalıştırıcı."""

    def run(args: list[str]) -> dict[str, Any]:
        proc = subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=timeout, env=dict(os.environ, YOLO_OFFLINE="1"))
        if proc.returncode != 0:
            raise RuntimeError(f"alt süreç başarısız (kod {proc.returncode}): {' '.join(args)}\n{proc.stderr[-2000:]}")
        for line in reversed(proc.stdout.strip().splitlines()):
            if line.startswith("{"):
                return json.loads(line)
        raise RuntimeError(f"alt süreç JSON üretmedi: {' '.join(args)}\n{proc.stdout[-1000:]}")

    return run


@dataclass(frozen=True)
class Variant:
    device: str
    threads: str  # keep | restore | N


def run_matrix(
    runner: Runner,
    video: Path,
    *,
    devices: Sequence[str] = ("mps", "cpu"),
    threads: Sequence[str] = ("keep", "restore"),
    repeats: int = 3,
    max_frames: int | None = None,
    probe_thread_values: Sequence[int] = (0, 1, 2, 4, 8, -1),
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Varyant matrisini (cihaz x iş parçacığı ayarı x tekrar) ve iş parçacığı sondalarını çalıştırır. Varyantlar tekrarlar boyunca EŞİT olarak dönüşümlü sıralanır."""
    if repeats < 1:
        raise ValueError(f"repeats pozitif olmalıdır: {repeats}")
    runs: list[dict[str, Any]] = []
    variants = [Variant(d, t) for d in devices for t in threads]
    for repeat in range(repeats):
        order = variants[repeat % len(variants) :] + variants[: repeat % len(variants)]  # sıra etkisini dengele
        for v in order:
            args = ["--worker", "pipeline", "--video", str(video), "--device", v.device, "--opencv-threads", v.threads]
            if max_frames is not None:
                args += ["--max-frames", str(max_frames)]
            result = runner(args)
            result["repeat"] = repeat + 1
            runs.append(result)
    probes = []
    for value in probe_thread_values:
        probes.append(runner(["--worker", "probe", "--set-threads", str(value), "--device", devices[0] if devices else "cpu"]))
    return {"generated_at": clock(), "synthetic_disclaimer": SYNTHETIC_DISCLAIMER, "video": video.name, "repeats": repeats, "variants": [v.__dict__ for v in variants],
            "pipeline_runs": runs, "probes": probes, "summary": summarize(runs)}


def summarize(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Varyant başına medyanlar (tekrarlar arası) ve imza tutarlılığı."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in runs:
        groups.setdefault((run["device"], run["opencv_threads_request"]), []).append(run)
    out = []
    for (device, threads), items in groups.items():
        def med(path: tuple[str, ...]) -> float | None:
            values = []
            for it in items:
                node: Any = it
                for key in path:
                    node = None if node is None else node.get(key)
                if node is not None:
                    values.append(float(node))
            return statistics.median(values) if values else None

        out.append({
            "device": device, "opencv_threads": threads, "runs": len(items), "end_to_end_fps_median": med(("end_to_end_fps",)),
            "yolo_ms_median": med(("latency", "yolo", "median_ms")), "tracking_ms_median": med(("latency", "tracking", "median_ms")),
            "face_ms_median": med(("latency", "face", "median_ms")), "association_ms_median": med(("latency", "association", "median_ms")),
            "compute_ms_median": med(("latency", "compute", "median_ms")), "signatures": sorted({it["signature"] for it in items}),
            "cv2_threads_after_setting": sorted({it["cv2_threads_after_setting"] for it in items}),
        })
    out.sort(key=lambda row: (row["device"], row["opencv_threads"]))
    return out


def summary_to_markdown(summary: Sequence[dict[str, Any]]) -> str:
    def fmt(value: float | None, digits: int = 1) -> str:
        return "—" if value is None else f"{value:.{digits}f}"

    lines = ["| YOLO cihazı | --opencv-threads | tekrar | FPS (uçtan uca, medyan) | YOLO ms | ByteTrack ms | YuNet ms | İlişkilendirme ms | Toplam hesaplama ms | imza tutarlı |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for row in summary:
        lines.append(f"| {row['device']} | {row['opencv_threads']} | {row['runs']} | {fmt(row['end_to_end_fps_median'])} | {fmt(row['yolo_ms_median'])} | "
                     f"{fmt(row['tracking_ms_median'], 2)} | {fmt(row['face_ms_median'])} | {fmt(row['association_ms_median'], 3)} | {fmt(row['compute_ms_median'])} | "
                     f"{'evet' if len(row['signatures']) == 1 else 'HAYIR'} |")
    return "\n".join(lines)
