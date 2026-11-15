"""Yüz tespiti performans ölçümü (``scripts/benchmark_face_detection.py`` bu modülü kullanır).

**Aşama 5A'ya aittir; Aşama 4B/4A performans sonuçları burada kullanılmaz ve 5A sonucu gibi sunulmaz.**

Ölçülenler (birbirine karıştırılmaz): model yükleme süresi, ısınma süresi, çıkarım süresi (``FaceDetectionResult.inference_ms``),
farklı yüz sayılarında ve çözünürlüklerde çıkarım süresi, ``detect_width`` etkisi, tepe bellek (``ru_maxrss``), pencerenin ek
maliyeti (yalnızca gerçek pencereyle ve yalnızca istenirse). Uçtan uca FPS, gerçek kare akışı döngüsüyle (``pipeline``) ayrıca ölçülür.
Ölçülmeyen hiçbir değer tahmin edilip tabloya yazılmaz.

**Sentetik girdi uyarısı:** farklı yüz sayıları için kullanılan görüntüler, Ultralytics paketinin örnek fotoğraflarından kırpılan iki
yüzün ızgaraya yerleştirilmesiyle üretilen MONTAJLARDIR. Hız ve entegrasyon ölçümü içindir; gerçek sahne/doğruluk kanıtı DEĞİLDİR
(aynı iki yüz tekrarlanır, ışık/poz çeşitliliği yoktur). Rapordaki "tespit edilen yüz" sayıları, montajın beklenen sayısıyla
karşılaştırılmak içindir.

İpucu (ölçüldü): ``ultralytics`` içe aktarılınca OpenCV süreç genelinde tek iş parçacığına düşer ve YuNet ~2,8× yavaşlar
(``docs/face_detection.md``). Bu yüzden örnek görüntü yolları ``ultralytics`` İÇE AKTARILMADAN (``importlib.util.find_spec``) bulunur.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import resource
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from isg_vision.human_detection.pipeline import LatencyStats

from .config import FaceDetectionConfig
from .detector import FaceDetector
from .errors import FaceDetectionError

SYNTHETIC_DISCLAIMER = (
    "Yüz sayısı/çözünürlük ölçümleri SENTETİK montajlarla yapılmıştır (aynı iki yüz tekrarlanır); hız ve entegrasyon içindir, "
    "gerçek sahne veya doğruluk kanıtı değildir."
)
PATCH_MARGIN = 0.4  # yüz kutusunun her kenara oranla genişletilmesi (bağlam için)
MONTAGE_BACKGROUND = 114


@dataclass(frozen=True)
class Measurement:
    """Bir yapılandırmanın çıkarım süresi ölçümü.

    Attributes:
        label: Ölçümün adı (ör. ``faces=4``).
        stats: Çıkarım süresi özeti (ms) — ``FaceDetectionResult.inference_ms`` değerlerinden.
        detected_min / detected_max: Tekrarlar boyunca bulunan en az/en çok yüz (aynı görüntüde tutarlı olmalıdır).
        expected_faces: Montajın beklenen yüz sayısı (bilinmiyorsa ``None``).
        image_size / inference_size: Karenin ve modelin çalıştığı girdinin boyutu.
    """

    label: str
    stats: LatencyStats
    detected_min: int
    detected_max: int
    expected_faces: int | None
    image_size: tuple[int, int]
    inference_size: tuple[int, int] | None

    @property
    def fps_inference_only(self) -> float:
        """Yalnızca çıkarım süresinden türetilen FPS (``1000 / medyan``); uçtan uca FPS DEĞİLDİR."""
        return 1000.0 / self.stats.median_ms if self.stats.median_ms > 0 else float("inf")


def find_sample_images() -> dict[str, Path]:
    """Ultralytics paketinin örnek görüntülerini ``ultralytics`` İÇE AKTARMADAN bulur (indirme yok)."""
    spec = importlib.util.find_spec("ultralytics")
    if spec is None or spec.origin is None:
        raise FaceDetectionError("ultralytics kurulu değil; paketle gelen örnek görüntüler bulunamadı.")
    assets = Path(spec.origin).parent / "assets"
    found = {name: assets / f"{name}.jpg" for name in ("zidane", "bus") if (assets / f"{name}.jpg").is_file()}
    if not found:
        raise FaceDetectionError(f"Örnek görüntü bulunamadı: {assets}")
    return found


def extract_face_patches(detector: Any, image: np.ndarray, *, min_conf: float = 0.8, margin: float = PATCH_MARGIN) -> list[np.ndarray]:
    """Görüntüdeki net yüzleri (``min_conf`` ve üstü) bağlamıyla birlikte kırpar; montaj üretmek içindir."""
    result = detector.detect(image)
    patches: list[np.ndarray] = []
    height, width = image.shape[:2]
    for face in result.faces:
        if face.confidence < min_conf:
            continue
        x1, y1, x2, y2 = face.bbox_xyxy
        mx, my = (x2 - x1) * margin, (y2 - y1) * margin
        cx1, cy1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
        cx2, cy2 = min(width, int(x2 + mx)), min(height, int(y2 + my))
        if cx2 - cx1 >= 8 and cy2 - cy1 >= 8:
            patches.append(image[cy1:cy2, cx1:cx2].copy())
    return patches


def make_face_montage(
    patches: Sequence[np.ndarray],
    n_faces: int,
    face_px: int,
    canvas_size: tuple[int, int] = (1280, 720),
) -> np.ndarray:
    """``n_faces`` yüzü (kırpıntılar sırayla tekrarlanır) ızgaraya yerleştirip düz gri bir tuval üretir.

    Her kırpıntı, içindeki yüz kutusunun yüksekliği yaklaşık ``face_px`` olacak biçimde ölçeklenir. Yüzler üst üste binmez; tuvale
    sığmıyorsa ``ValueError`` verilir (sessizce sayı düşürülmez). ``n_faces=0`` düz tuval döndürür.
    """
    if n_faces < 0 or face_px < 8:
        raise ValueError(f"n_faces >= 0 ve face_px >= 8 olmalıdır: {n_faces}, {face_px}")
    width, height = canvas_size
    canvas = np.full((height, width, 3), MONTAGE_BACKGROUND, np.uint8)
    if n_faces == 0:
        return canvas
    if not patches:
        raise ValueError("Montaj için en az bir yüz kırpıntısı gerekir.")
    cell = int(round(face_px * (1 + 2 * PATCH_MARGIN)))  # kırpıntının hedef yüksekliği
    cols, rows = max(1, width // cell), max(1, height // cell)
    if n_faces > cols * rows:
        raise ValueError(f"{n_faces} yüz {width}x{height} tuvale {face_px} px yüzle sığmıyor (en çok {cols * rows}).")
    grid_cols = min(cols, n_faces)
    grid_rows = -(-n_faces // grid_cols)
    x_off = (width - grid_cols * cell) // 2
    y_off = (height - grid_rows * cell) // 2
    for i in range(n_faces):
        patch = patches[i % len(patches)]
        scale = cell / patch.shape[0]
        w = max(1, min(cell, int(round(patch.shape[1] * scale))))
        resized = cv2.resize(patch, (w, cell), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        x = x_off + (i % grid_cols) * cell
        y = y_off + (i // grid_cols) * cell
        canvas[y : y + cell, x : x + w] = resized
    return canvas


def measure_inference(
    detector: Any,
    frame: np.ndarray,
    *,
    repeats: int,
    warmup: int = 5,
    label: str = "",
    expected_faces: int | None = None,
) -> Measurement:
    """Aynı kare üzerinde ``repeats`` kez ``detect`` çağırır; ısınma çağrıları ölçüme girmez."""
    if repeats < 1 or warmup < 0:
        raise ValueError(f"repeats >= 1 ve warmup >= 0 olmalıdır: {repeats}, {warmup}")
    for _ in range(warmup):
        detector.detect(frame)
    times: list[float] = []
    counts: list[int] = []
    last = None
    for _ in range(repeats):
        last = detector.detect(frame)
        times.append(last.inference_ms)
        counts.append(last.face_count)
    stats = LatencyStats.from_values(times)
    assert stats is not None and last is not None  # noqa: S101 (repeats >= 1)
    return Measurement(label, stats, min(counts), max(counts), expected_faces, last.image_size, last.inference_size)


def measure_model_load(config: FaceDetectionConfig, loads: int, *, detector_factory: Callable[[FaceDetectionConfig], Any] = FaceDetector,
                       warmup_size: tuple[int, int] = (1280, 720)) -> dict[str, Any]:
    """Modeli ``loads`` kez yeni bir ``FaceDetector`` ile yükler (bütünlük denetimi dâhil) ve ısınma süresini ölçer."""
    load_s: list[float] = []
    warm_s: list[float] = []
    for _ in range(loads):
        detector = detector_factory(config)
        detector.load()
        load_s.append(detector.load_seconds or 0.0)
        warm_s.append(detector.warmup(warmup_size))
        detector.close()
    return {
        "loads": loads,
        "load_seconds": load_s,
        "load_ms_median": sorted(load_s)[len(load_s) // 2] * 1000.0,
        "warmup_seconds": warm_s,
        "note": "yükleme süresi SHA-256 bütünlük denetimini içerir; ilk yükleme işletim sistemi dosya önbelleğine bağlıdır",
    }


def measure_gui_cost(window: Any, frame: np.ndarray, runs: int, *, clock: Callable[[], float] = time.perf_counter) -> dict[str, float]:
    """Verilen (açık) pencerenin ``show`` süresini (imshow + waitKey(1)) ölçer; yalnızca gerçek pencereyle anlamlıdır."""
    times = []
    for _ in range(runs):
        start = clock()
        window.show(frame, 1)
        times.append((clock() - start) * 1000.0)
    stats = LatencyStats.from_values(times)
    assert stats is not None  # noqa: S101
    return {"runs": runs, "mean_ms": stats.mean_ms, "median_ms": stats.median_ms, "p95_ms": stats.p95_ms, "min_ms": stats.min_ms, "max_ms": stats.max_ms}


def peak_rss_mb() -> float:
    """Sürecin şimdiye dek ulaştığı tepe bellek (MiB; ``ru_maxrss``). Monoton artar; geri düşmez."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024.0 * 1024.0) if sys.platform == "darwin" else usage / 1024.0  # macOS bayt, Linux KiB verir


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "opencv": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "ultralytics_imported": "ultralytics" in sys.modules,
        "torch_imported": "torch" in sys.modules,
    }
    return info


def measurement_to_dict(m: Measurement) -> dict[str, Any]:
    return {
        "label": m.label,
        "image_size": list(m.image_size),
        "inference_size": list(m.inference_size) if m.inference_size else None,
        "expected_faces": m.expected_faces,
        "detected_min": m.detected_min,
        "detected_max": m.detected_max,
        "n": m.stats.count,
        "mean_ms": m.stats.mean_ms,
        "median_ms": m.stats.median_ms,
        "p95_ms": m.stats.p95_ms,
        "min_ms": m.stats.min_ms,
        "max_ms": m.stats.max_ms,
        "fps_inference_only": m.fps_inference_only,
    }


def measurements_to_markdown(title: str, rows: Sequence[Measurement]) -> str:
    lines = [f"### {title}", "", "| Ölçüm | Kare | Model girdisi | Beklenen yüz | Bulunan yüz | Medyan (ms) | p95 (ms) | Çıkarım-yalnız FPS |",
             "|---|---|---|---|---|---|---|---|"]
    for m in rows:
        found = f"{m.detected_min}" if m.detected_min == m.detected_max else f"{m.detected_min}-{m.detected_max}"
        size = f"{m.inference_size[0]}x{m.inference_size[1]}" if m.inference_size else "-"
        expected = "-" if m.expected_faces is None else str(m.expected_faces)
        lines.append(f"| {m.label} | {m.image_size[0]}x{m.image_size[1]} | {size} | {expected} | {found} | {m.stats.median_ms:.2f} | {m.stats.p95_ms:.2f} | {m.fps_inference_only:.1f} |")
    return "\n".join(lines) + "\n"


def write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_benchmark(
    config: FaceDetectionConfig,
    *,
    repeats: int = 100,
    face_counts: Sequence[int] = (0, 1, 2, 4, 8, 16),
    face_px: int = 96,
    sizes: Sequence[tuple[int, int]] = ((640, 360), (960, 540), (1280, 720), (1920, 1080)),
    detect_widths: Sequence[int] = (960, 640),
    loads: int = 5,
    gui: bool = False,
    gui_runs: int = 200,
    detector_factory: Callable[[FaceDetectionConfig], Any] = FaceDetector,
    window_factory: Callable[[], Any] | None = None,
    sample_images: dict[str, np.ndarray] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Tüm ölçümleri çalıştırır ve JSON'a yazılabilir bir sözlük döndürür (ölçülmeyen alanlar ``None`` kalır).

    Args:
        gui: ``True`` ve ``window_factory`` verilmişse çizim ve pencere ``show`` maliyeti de ölçülür (gerçek grafik oturumu gerekir).
        sample_images: Test için hazır görüntüler ``{"zidane": BGR, "bus": BGR}``; ``None`` ise Ultralytics örnekleri okunur.
    """
    from dataclasses import replace

    from .visualization import FaceOverlayInfo, draw_face_detections

    report: dict[str, Any] = {"environment": environment_info(), "disclaimer": SYNTHETIC_DISCLAIMER, "config": {
        "model": config.model_name, "conf": config.conf, "nms": config.nms, "max_faces": config.max_faces, "detect_width": config.detect_width}}
    rss: dict[str, float] = {"after_imports_mb": peak_rss_mb()}

    log(f"[BİLGİ] Model yükleme ölçümü ({loads} yükleme)...")
    report["model_load"] = measure_model_load(config, loads, detector_factory=detector_factory)

    detector = detector_factory(config)
    detector.load()
    report["warmup_seconds_1280x720"] = detector.warmup((1280, 720))
    rss["after_load_and_warmup_mb"] = peak_rss_mb()

    if sample_images is None:
        sample_images = {}
        for name, path in find_sample_images().items():
            image = cv2.imread(str(path))
            if image is None:
                raise FaceDetectionError(f"Örnek görüntü okunamadı: {path}")
            sample_images[name] = image
    patches: list[np.ndarray] = []
    for name in ("zidane", "bus"):
        if name in sample_images:
            patches.extend(extract_face_patches(detector, sample_images[name]))
    report["montage_patches"] = len(patches)
    log(f"[BİLGİ] Montaj için {len(patches)} net yüz kırpıntısı bulundu.")

    natural = [measure_inference(detector, image, repeats=repeats, label=f"{name} (doğal fotoğraf)") for name, image in sample_images.items()]
    report["natural_images"] = [measurement_to_dict(m) for m in natural]

    by_count: list[Measurement] = []
    for n in face_counts:
        frame = make_face_montage(patches, n, face_px)
        by_count.append(measure_inference(detector, frame, repeats=repeats, label=f"{n} yüz, ~{face_px} px", expected_faces=n))
    report["by_face_count"] = {"face_px": face_px, "canvas": [1280, 720], "rows": [measurement_to_dict(m) for m in by_count]}

    base = make_face_montage(patches, 4, face_px)
    by_size: list[Measurement] = []
    for (w, h) in sizes:
        frame = base if (w, h) == (1280, 720) else cv2.resize(base, (w, h), interpolation=cv2.INTER_AREA if w < 1280 else cv2.INTER_CUBIC)
        by_size.append(measure_inference(detector, frame, repeats=repeats, label=f"{w}x{h} (4 yüz montajı yeniden boyutlu)", expected_faces=4))
    report["by_resolution"] = [measurement_to_dict(m) for m in by_size]

    by_width: list[Measurement] = []
    big = cv2.resize(base, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    for frame_label, frame in (("1280x720", base), ("1920x1080", big)):
        by_width.append(measure_inference(detector, frame, repeats=repeats, label=f"{frame_label}, tespit genişliği: kare", expected_faces=4))
        for dw in detect_widths:
            if dw >= frame.shape[1]:
                continue
            narrow = detector_factory(replace(config, detect_width=dw))
            narrow.load()
            by_width.append(measure_inference(narrow, frame, repeats=repeats, label=f"{frame_label}, tespit genişliği: {dw}", expected_faces=4))
            narrow.close()
    report["by_detect_width"] = [measurement_to_dict(m) for m in by_width]

    report["gui"] = None
    if gui and window_factory is not None:
        result = detector.detect(base)
        info = FaceOverlayInfo("VIDEO bench", 30.0, result.inference_ms, (1280, 720), "cpu")
        draw_times = []
        for _ in range(gui_runs):
            start = time.perf_counter()
            image = draw_face_detections(base, result, info)
            draw_times.append((time.perf_counter() - start) * 1000.0)
        draw_stats = LatencyStats.from_values(draw_times)
        window = window_factory()
        window.open()
        try:
            show = measure_gui_cost(window, image, gui_runs)
        finally:
            window.close()
        report["gui"] = {"frame": [1280, 720], "faces": result.face_count,
                         "draw_median_ms": draw_stats.median_ms if draw_stats else None, "show": show,
                         "note": "show = imshow + waitKey(1) (gerçek pencere); çizim kopya üzerinde cv2 ile yapılır"}
    elif gui:
        log("[UYARI] --gui istendi ama pencere oluşturucu verilmedi; GUI maliyeti ölçülmedi.")

    detector.close()
    rss["end_mb"] = peak_rss_mb()
    report["peak_rss_mb"] = rss
    report["peak_rss_note"] = "ru_maxrss (tepe, monoton); 'after_imports' süreç başlangıcı + içe aktarmaları içerir; fark = model ve ölçüm arabellekleri"
    report["tables_markdown"] = "\n".join([
        measurements_to_markdown("Doğal fotoğraflar (1280x720 / 810x1080)", natural),
        measurements_to_markdown(f"Yüz sayısına göre (1280x720 montaj, ~{face_px} px yüz)", by_count),
        measurements_to_markdown("Çözünürlüğe göre (4 yüzlü montaj, tuval yeniden boyutlandırıldı)", by_size),
        measurements_to_markdown("Tespit genişliği (--detect-width) etkisi", by_width),
    ])
    return report
