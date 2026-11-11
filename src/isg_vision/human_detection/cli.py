"""Komut satırı arayüzü (``scripts/run_person_detection.py`` bu modülün ``main()`` fonksiyonunu çağırır).

Çıkış kodları (``run_camera.py`` ile aynı): 0 = normal bitiş (Q/Esc, pencere kapatma, video sonu, ``--max-frames``),
1 = kamera/video/pencere/model hatası, 2 = hatalı argüman, 130 = ``Ctrl+C``. Her koşulda kamera, pencere ve model serbest bırakılır.

Sıra bilinçli olarak şudur: (1) model yüklenir (başarısızsa kamera HİÇ açılmaz), (2) kaynak açılır, (3) model gerçek kare
boyutunda ısıtılır, (4) pencere açılır, (5) döngü. Görüntü kaydedilmez, hiçbir ağ işlemi yapılmaz (model indirme ayrı betiktir).

Bu modül import edildiğinde hiçbir kamera açılmaz, model yüklenmez ve torch/ultralytics import edilmez.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

from isg_vision.camera.capture import FrameSource
from isg_vision.camera.cli import EXIT_ERROR, EXIT_INTERRUPTED, EXIT_OK, EXIT_USAGE, config_from_args
from isg_vision.camera.config import BACKENDS, DEFAULT_HEIGHT, DEFAULT_WIDTH, SourceConfig
from isg_vision.camera.display import PreviewWindow
from isg_vision.camera.errors import DisplayError, SourceConfigError, SourceError
from isg_vision.camera.frame import SOURCE_KIND_CAMERA

from .config import (
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_IOU,
    DEFAULT_MAX_DET,
    DEFAULT_MODEL_PATH,
    DEVICES,
    DetectionConfig,
)
from .errors import DetectionConfigError, DetectionError
from .person_detector import PersonDetector
from .pipeline import (
    REASON_END_OF_STREAM,
    REASON_INTERRUPTED,
    REASON_MAX_FRAMES,
    REASON_USER_QUIT,
    REASON_WINDOW_CLOSED,
    DetectionRunSummary,
    LatencyStats,
    run_detection,
)
from .visualization import WINDOW_NAME

__all__ = ["EXIT_ERROR", "EXIT_INTERRUPTED", "EXIT_OK", "EXIT_USAGE", "build_parser", "main", "run"]

_REASON_TEXT = {
    REASON_USER_QUIT: "kullanıcı çıkışı (Q/Esc)",
    REASON_WINDOW_CLOSED: "pencere kapatıldı",
    REASON_END_OF_STREAM: "video sona erdi",
    REASON_MAX_FRAMES: "--max-frames sınırına ulaşıldı",
    REASON_INTERRUPTED: "Ctrl+C ile durduruldu",
}

_EXAMPLES = """\
örnekler:
  python scripts/run_person_detection.py --source camera
  python scripts/run_person_detection.py --source camera --camera-index 0
  python scripts/run_person_detection.py --source video --video-path /yol/video.mp4
  python scripts/run_person_detection.py --source camera --device mps       # Apple GPU (yoksa hata verir)
  python scripts/run_person_detection.py --source camera --device cpu       # CPU karşılaştırması
  python scripts/run_person_detection.py --source camera --conf 0.35        # güven eşiği
  python scripts/run_person_detection.py --source camera --no-display --max-frames 100   # pencere açmadan 100 kare

pencerede çıkış: Q veya Esc tuşu ya da pencereyi kapatma; terminalde Ctrl+C.
Çıkış kodları: 0 normal, 1 kamera/video/pencere/model hatası, 2 hatalı argüman, 130 Ctrl+C.
Ağırlık dosyası yoksa önce:  python scripts/download_person_model.py   (bu betik ağ kullanan tek adımdır)
"""


def _integer(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"tam sayı olmalıdır: {text!r}") from None


def _positive_integer(text: str) -> int:
    value = _integer(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"1 veya daha büyük olmalıdır: {text!r}")
    return value


def _unit_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"sayı olmalıdır: {text!r}") from None
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError(f"0'dan büyük ve 1'e eşit veya küçük olmalıdır: {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_person_detection.py",
        description=(
            "Kamera veya video dosyasındaki tüm insanları önceden eğitilmiş YOLO ile tespit eder; kutu, güven skoru, kişi sayısı, "
            "ölçülen FPS ve çıkarım süresini canlı gösterir. Aşama 4A: yalnızca tespit; insanlara takip numarası veya kimlik "
            "atanmaz, görüntü kaydedilmez, ağ kullanılmaz."
        ),
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_argument_group("kaynak (run_camera.py ile aynı seçenekler)")
    src.add_argument("--source", choices=("camera", "video"), default="camera", help="Kare kaynağı: camera (varsayılan) veya video")
    src.add_argument("--camera-index", type=_integer, default=None, metavar="N", help="Kamera indeksi (varsayılan: 0). Yalnızca --source camera ile.")
    src.add_argument("--width", type=_integer, default=None, metavar="PX",
                     help=f"İstenen kamera genişliği (varsayılan: {DEFAULT_WIDTH}); --height ile birlikte.")
    src.add_argument("--height", type=_integer, default=None, metavar="PX",
                     help=f"İstenen kamera yüksekliği (varsayılan: {DEFAULT_HEIGHT}); --width ile birlikte.")
    src.add_argument("--video-path", default=None, metavar="DOSYA", help="Video dosyası yolu (MP4/MOV/AVI). --source video için zorunludur.")
    src.add_argument("--backend", choices=BACKENDS, default="auto", help="Kamera backend'i (varsayılan: auto = macOS'ta AVFoundation).")

    det = parser.add_argument_group("model")
    det.add_argument("--model", default=None, metavar="DOSYA.pt",
                     help="YOLO ağırlık dosyası (varsayılan: models/human_detection/yolo11n.pt). 'person' sınıfı içermelidir.")
    det.add_argument("--conf", type=_unit_float, default=DEFAULT_CONF, metavar="F",
                     help=f"Güven eşiği, 0 < F <= 1 (varsayılan: {DEFAULT_CONF}). Düşürmek daha çok (yanlış pozitif dâhil), "
                          "yükseltmek daha az tespit verir. Güven skoru doğruluk değildir.")
    det.add_argument("--iou", type=_unit_float, default=DEFAULT_IOU, metavar="F", help=f"NMS IoU eşiği, 0 < F <= 1 (varsayılan: {DEFAULT_IOU}).")
    det.add_argument("--imgsz", type=_integer, default=DEFAULT_IMGSZ, metavar="PX",
                     help=f"Model girdi boyutu, 32'nin katı (varsayılan: {DEFAULT_IMGSZ}). Küçük = hızlı ama uzak insanları kaçırabilir.")
    det.add_argument("--max-det", type=_positive_integer, default=DEFAULT_MAX_DET, metavar="N",
                     help=f"Karede en fazla tespit (varsayılan: {DEFAULT_MAX_DET}).")
    det.add_argument("--device", choices=DEVICES, default="auto",
                     help="Çıkarım cihazı: auto (MPS varsa MPS, yoksa uyarıyla CPU), cpu veya mps (kullanılamıyorsa hata verir, sessizce CPU'ya düşmez).")

    run = parser.add_argument_group("çalışma")
    run.add_argument("--no-display", action="store_true", help="Pencere açma; kareleri işleyip özet yazar (video için bekleme yapılmaz).")
    run.add_argument("--max-frames", type=_positive_integer, default=None, metavar="N", help="N kare işlendikten sonra dur.")
    return parser


def detection_config_from_args(args: argparse.Namespace) -> DetectionConfig:
    """Argümanları doğrulayıp ``DetectionConfig`` üretir (``DetectionConfigError`` fırlatabilir)."""
    return DetectionConfig(
        model_path=DEFAULT_MODEL_PATH if args.model is None else args.model, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device, max_det=args.max_det
    )


def run(
    source_config: SourceConfig,
    detection_config: DetectionConfig,
    *,
    display: bool = True,
    max_frames: int | None = None,
    capture_factory: Callable[..., Any] | None = None,
    window_factory: Callable[[], Any] | None = None,
    detector_factory: Callable[[DetectionConfig], Any] = PersonDetector,
) -> int:
    """Modeli yükler, kaynağı açar, tespit döngüsünü çalıştırır ve her koşulda kaynakları serbest bırakır. Çıkış kodunu döndürür."""
    detector = detector_factory(detection_config)
    source = FrameSource(source_config, capture_factory=capture_factory)
    window: Any | None = None
    try:
        detector.load()  # başarısızsa kamera hiç açılmaz
        _print_model(detector, detection_config)
        source.open()
        _print_source(source, display)
        warmup_s = detector.warmup(source.actual_resolution or (640, 480))
        print(f"[BİLGİ] Model ısıtıldı: {warmup_s:.2f} sn (sürekli performanstan ayrı; ilk karelerin gecikmesini azaltır).")
        if display:
            window = (window_factory or partial(PreviewWindow, WINDOW_NAME))()
            window.open()
            if getattr(window, "close_button_enabled", None) is False:
                print("[UYARI] Pencerenin kırmızı kapatma düğmesi etkinleştirilemedi; pencereyi Q/Esc tuşuyla veya terminalde Ctrl+C ile kapatın.")
        summary = run_detection(source, detector, window, max_frames=max_frames)
    except KeyboardInterrupt:  # model yüklenirken, kaynak açılırken veya pencere oluşturulurken Ctrl+C
        print("\n[BİLGİ] Ctrl+C ile durduruldu.")
        return EXIT_INTERRUPTED
    except (SourceError, DisplayError, DetectionError) as exc:
        print(f"[HATA] {exc}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        if window is not None:
            window.close()
        source.close()
        detector.close()

    _print_summary(summary, source, detector, display)
    return EXIT_INTERRUPTED if summary.reason == REASON_INTERRUPTED else EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    *,
    capture_factory: Callable[..., Any] | None = None,
    window_factory: Callable[[], Any] | None = None,
    detector_factory: Callable[[DetectionConfig], Any] = PersonDetector,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        source_config = config_from_args(args)
        detection_config = detection_config_from_args(args)
    except (SourceConfigError, DetectionConfigError) as exc:
        parser.error(str(exc))  # kullanım bilgisini yazar ve 2 koduyla çıkar
    return run(
        source_config,
        detection_config,
        display=not args.no_display,
        max_frames=args.max_frames,
        capture_factory=capture_factory,
        window_factory=window_factory,
        detector_factory=detector_factory,
    )


# ---------------------------------------------------------------------- çıktı yardımcıları
def _fmt_size(size: tuple[int, int] | None) -> str:
    return f"{size[0]}x{size[1]}" if size else "bilinmiyor"


def _fmt_fps(fps: float | None) -> str:
    return f"{fps:.1f}" if fps else "bilinmiyor"


def _fmt_stats(stats: LatencyStats | None) -> str:
    if stats is None:
        return "ölçülmedi"
    return (f"ort {stats.mean_ms:.1f} | medyan {stats.median_ms:.1f} | p95 {stats.p95_ms:.1f} | "
            f"min {stats.min_ms:.1f} | maks {stats.max_ms:.1f} ms (n={stats.count})")


def _print_model(detector: Any, config: DetectionConfig) -> None:
    load_s = getattr(detector, "load_seconds", None)
    print(f"[BİLGİ] Model: {config.model_name} | cihaz: {detector.device} (istenen: {config.device}) | conf: {config.conf} | "
          f"iou: {config.iou} | imgsz: {config.imgsz}" + (f" | yükleme: {load_s:.2f} sn" if load_s is not None else ""))
    warning = getattr(detector, "device_warning", None)
    if warning:
        print(f"[UYARI] {warning}")


def _print_source(source: FrameSource, display: bool) -> None:
    print(f"[BİLGİ] Kaynak açıldı: {source.label} | çözünürlük: {_fmt_size(source.actual_resolution)} | "
          f"kaynak FPS (bildirilen, yalnızca bilgi): {_fmt_fps(source.source_fps)}")
    mismatch = source.resolution_mismatch
    if mismatch is not None:
        requested, actual = mismatch
        print(f"[UYARI] İstenen çözünürlük ({_fmt_size(requested)}) uygulanamadı; kaynak {_fmt_size(actual)} kullanıyor.")
    if display:
        print("[BİLGİ] Çıkmak için pencerede Q veya Esc tuşuna basın, pencerenin kapatma düğmesine tıklayın (ya da terminalde Ctrl+C).")


def _print_summary(summary: DetectionRunSummary, source: FrameSource, detector: Any, display: bool) -> None:
    print(f"[BİLGİ] Bitiş nedeni: {_REASON_TEXT.get(summary.reason, summary.reason)}")
    print(f"[BİLGİ] İşlenen kare: {summary.frames_processed} | süre: {summary.elapsed_s:.1f} sn | "
          f"uçtan uca FPS (ölçülen): {summary.end_to_end_fps:.1f} | gerçek çözünürlük: {_fmt_size(source.actual_resolution)} | "
          f"cihaz: {detector.device}")
    print(f"[BİLGİ] Çıkarım süresi: {_fmt_stats(summary.inference)}")
    print(f"[BİLGİ] Kare okuma süresi: {_fmt_stats(summary.read)}")
    print(f"[BİLGİ] Kare -> sonuç gecikmesi: {_fmt_stats(summary.frame_to_result)}")
    if display and summary.frame_to_display is not None:
        print(f"[BİLGİ] Kare -> görüntü gecikmesi: {_fmt_stats(summary.frame_to_display)}")
    print(f"[BİLGİ] Kişi sayısı: kare başına ortalama {summary.persons_mean:.2f} | en çok {summary.persons_max} | "
          f"en az bir kişi olan kare: {summary.frames_with_persons}/{summary.frames_processed}")
    if summary.dropped_boxes:
        print(f"[UYARI] {summary.dropped_boxes} geçersiz kutu atıldı.")
    if not display:
        print("[BİLGİ] Pencere kapalıydı; FPS çizim/gösterme maliyetini içermez.")
    if source.kind == SOURCE_KIND_CAMERA and source.source_fps is not None:
        print("[BİLGİ] Not: kameranın bildirdiği FPS performans ölçüsü değildir; ölçülen değerler yukarıdadır.")
    if source.premature_end:
        print(f"[UYARI] Video beklenenden erken bitti: {source.frames_read} kare okundu, dosya {source.frame_count} kare "
              "bildiriyor. Dosya kesik veya bozuk olabilir.", file=sys.stderr)
