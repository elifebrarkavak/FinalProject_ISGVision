"""Komut satırı arayüzü (``scripts/run_face_detection.py`` bu modülün ``main()`` fonksiyonunu çağırır).

Kaynak seçenekleri ``run_camera.py`` / ``run_person_detection.py`` ile aynıdır (aynı ``camera.cli.config_from_args`` kullanılır);
model seçenekleri yüz tespitine özgüdür. Cihaz seçeneği YOKTUR: model yalnızca CPU'da çalışır (``config`` açıklaması).

Çıkış kodları (diğer betiklerle aynı): 0 = normal bitiş (Q/Esc, pencere kapatma, video sonu, ``--max-frames``), 1 = kamera/video/
pencere/model hatası, 2 = hatalı argüman, 130 = ``Ctrl+C``. Her koşulda kamera, pencere ve model serbest bırakılır.

Sıra bilinçli olarak şudur: (1) model yüklenir (başarısızsa kamera HİÇ açılmaz), (2) kaynak açılır, (3) model gerçek kare boyutunda
ısıtılır, (4) pencere açılır, (5) döngü. Görüntü kaydedilmez, hiçbir ağ işlemi yapılmaz (model indirme ayrı betiktir).

Bu modül import edildiğinde hiçbir kamera açılmaz ve model yüklenmez.
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
from isg_vision.human_detection.cli import _fmt_fps, _fmt_size, _fmt_stats

from .config import (
    DEFAULT_CONF,
    DEFAULT_MAX_FACES,
    DEFAULT_MODEL_PATH,
    DEFAULT_NMS,
    MAX_DETECT_WIDTH,
    MIN_DETECT_WIDTH,
    FaceDetectionConfig,
)
from .detector import FaceDetector
from .errors import FaceConfigError, FaceDetectionError
from .pipeline import (
    REASON_END_OF_STREAM,
    REASON_INTERRUPTED,
    REASON_MAX_FRAMES,
    REASON_USER_QUIT,
    REASON_WINDOW_CLOSED,
    FaceRunSummary,
    run_face_detection,
)
from .visualization import WINDOW_NAME

__all__ = ["EXIT_ERROR", "EXIT_INTERRUPTED", "EXIT_OK", "EXIT_USAGE", "build_parser", "face_config_from_args", "main", "run"]

_REASON_TEXT = {
    REASON_USER_QUIT: "kullanıcı çıkışı (Q/Esc)",
    REASON_WINDOW_CLOSED: "pencere kapatıldı",
    REASON_END_OF_STREAM: "video sona erdi",
    REASON_MAX_FRAMES: "--max-frames sınırına ulaşıldı",
    REASON_INTERRUPTED: "Ctrl+C ile durduruldu",
}

_EXAMPLES = """\
örnekler:
  python scripts/run_face_detection.py --source camera
  python scripts/run_face_detection.py --source camera --camera-index 0
  python scripts/run_face_detection.py --source video --video-path /yol/video.mp4
  python scripts/run_face_detection.py --source camera --conf 0.7           # güven eşiği
  python scripts/run_face_detection.py --source camera --detect-width 960   # daha hızlı; küçük/uzak yüzler kaçabilir
  python scripts/run_face_detection.py --source camera --no-display --max-frames 100   # pencere açmadan 100 kare

pencerede çıkış: Q veya Esc tuşu ya da pencereyi kapatma; terminalde Ctrl+C.
Çıkış kodları: 0 normal, 1 kamera/video/pencere/model hatası, 2 hatalı argüman, 130 Ctrl+C.
Model dosyası yoksa önce:  python scripts/download_face_model.py   (bu betik ağ kullanan tek adımdır)
Önemli: yüz kutusu bir kimlik veya canlı insan kanıtı DEĞİLDİR; poster/fotoğraf/ekrandaki yüz de tespit edilebilir.
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
        prog="run_face_detection.py",
        description=(
            "Kamera veya video dosyasındaki tüm yüzleri önceden eğitilmiş YuNet (OpenCV) ile tespit eder; kutu, güven skoru, toplam yüz "
            "sayısı, ölçülen FPS ve çıkarım süresini canlı gösterir. Aşama 5A: yalnızca yüz TESPİTİ; yüzler kişiye/Track ID'ye "
            "bağlanmaz, yüz tanıma/çalışan kimliği/biyometrik veri yoktur, görüntü kaydedilmez, ağ kullanılmaz. Yüz tespiti canlı insan "
            "doğrulaması değildir."
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

    det = parser.add_argument_group("model (yalnızca CPU; cihaz seçeneği yoktur)")
    det.add_argument("--model", default=None, metavar="DOSYA.onnx",
                     help="YuNet ONNX dosyası (varsayılan: models/face_detection/face_detection_yunet_2026may.onnx). Resmî dosya yüklenirken SHA-256 ile doğrulanır.")
    det.add_argument("--conf", type=_unit_float, default=DEFAULT_CONF, metavar="F",
                     help=f"Güven eşiği, 0 < F <= 1 (varsayılan: {DEFAULT_CONF}). Düşürmek daha çok (yanlış pozitif dâhil), "
                          "yükseltmek daha az tespit verir. Güven skoru doğruluk değildir.")
    det.add_argument("--nms", type=_unit_float, default=DEFAULT_NMS, metavar="F", help=f"NMS IoU eşiği, 0 < F <= 1 (varsayılan: {DEFAULT_NMS}).")
    det.add_argument("--max-faces", type=_positive_integer, default=DEFAULT_MAX_FACES, metavar="N",
                     help=f"Karede en fazla yüz (varsayılan: {DEFAULT_MAX_FACES}); fazlası sayılır ve raporlanır.")
    det.add_argument("--detect-width", type=_integer, default=None, metavar="PX",
                     help=f"Verilirse kare bu genişliğe küçültülüp tespit yapılır ({MIN_DETECT_WIDTH}-{MAX_DETECT_WIDTH}); kutular özgün kareye geri "
                          "çevrilir. Varsayılan: kare çözünürlüğünde. Küçültmek hızlandırır ama küçük/uzak yüzleri kaçırabilir.")

    run = parser.add_argument_group("çalışma")
    run.add_argument("--no-display", action="store_true", help="Pencere açma; kareleri işleyip özet yazar (video için bekleme yapılmaz).")
    run.add_argument("--max-frames", type=_positive_integer, default=None, metavar="N", help="N kare işlendikten sonra dur.")
    return parser


def face_config_from_args(args: argparse.Namespace) -> FaceDetectionConfig:
    """Argümanları doğrulayıp ``FaceDetectionConfig`` üretir (``FaceConfigError`` fırlatabilir)."""
    return FaceDetectionConfig(
        model_path=DEFAULT_MODEL_PATH if args.model is None else args.model,
        conf=args.conf,
        nms=args.nms,
        max_faces=args.max_faces,
        detect_width=args.detect_width,
    )


def run(
    source_config: SourceConfig,
    face_config: FaceDetectionConfig,
    *,
    display: bool = True,
    max_frames: int | None = None,
    capture_factory: Callable[..., Any] | None = None,
    window_factory: Callable[[], Any] | None = None,
    detector_factory: Callable[[FaceDetectionConfig], Any] = FaceDetector,
) -> int:
    """Modeli yükler, kaynağı açar, tespit döngüsünü çalıştırır ve her koşulda kaynakları serbest bırakır. Çıkış kodunu döndürür."""
    detector = detector_factory(face_config)
    source = FrameSource(source_config, capture_factory=capture_factory)
    window: Any | None = None
    try:
        detector.load()  # başarısızsa kamera hiç açılmaz
        _print_model(detector, face_config)
        source.open()
        _print_source(source, display)
        warmup_s = detector.warmup(source.actual_resolution or (640, 480))
        print(f"[BİLGİ] Model ısıtıldı: {warmup_s:.2f} sn (sürekli performanstan ayrı; ilk karelerin gecikmesini azaltır).")
        if display:
            window = (window_factory or partial(PreviewWindow, WINDOW_NAME))()
            window.open()
            if getattr(window, "close_button_enabled", None) is False:
                print("[UYARI] Pencerenin kırmızı kapatma düğmesi etkinleştirilemedi; pencereyi Q/Esc tuşuyla veya terminalde Ctrl+C ile kapatın.")
        summary = run_face_detection(source, detector, window, max_frames=max_frames)
    except KeyboardInterrupt:  # model yüklenirken, kaynak açılırken veya pencere oluşturulurken Ctrl+C
        print("\n[BİLGİ] Ctrl+C ile durduruldu.")
        return EXIT_INTERRUPTED
    except (SourceError, DisplayError, FaceDetectionError) as exc:
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
    detector_factory: Callable[[FaceDetectionConfig], Any] = FaceDetector,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        source_config = config_from_args(args)
        face_config = face_config_from_args(args)
    except (SourceConfigError, FaceConfigError) as exc:
        parser.error(str(exc))  # kullanım bilgisini yazar ve 2 koduyla çıkar
    return run(
        source_config,
        face_config,
        display=not args.no_display,
        max_frames=args.max_frames,
        capture_factory=capture_factory,
        window_factory=window_factory,
        detector_factory=detector_factory,
    )


# ---------------------------------------------------------------------- çıktı yardımcıları
def _print_model(detector: Any, config: FaceDetectionConfig) -> None:
    load_s = getattr(detector, "load_seconds", None)
    width = f"{config.detect_width} px" if config.detect_width is not None else "kare çözünürlüğü"
    print(f"[BİLGİ] Model: {config.model_name} (YuNet, OpenCV DNN) | çalışma yeri: {detector.device} (cihaz seçeneği yok) | conf: {config.conf} | "
          f"nms: {config.nms} | tespit genişliği: {width}" + (f" | yükleme: {load_s:.3f} sn" if load_s is not None else ""))
    warning = getattr(detector, "device_warning", None)
    if warning:
        print(f"[UYARI] {warning}")
    print("[BİLGİ] Not: yüz tespiti bir kimlik veya canlı insan doğrulaması değildir; poster/fotoğraf/ekrandaki yüz de tespit edilebilir.")


def _print_source(source: FrameSource, display: bool) -> None:
    print(f"[BİLGİ] Kaynak açıldı: {source.label} | çözünürlük: {_fmt_size(source.actual_resolution)} | "
          f"kaynak FPS (bildirilen, yalnızca bilgi): {_fmt_fps(source.source_fps)}")
    mismatch = source.resolution_mismatch
    if mismatch is not None:
        requested, actual = mismatch
        print(f"[UYARI] İstenen çözünürlük ({_fmt_size(requested)}) uygulanamadı; kaynak {_fmt_size(actual)} kullanıyor.")
    if display:
        print("[BİLGİ] Çıkmak için pencerede Q veya Esc tuşuna basın, pencerenin kapatma düğmesine tıklayın (ya da terminalde Ctrl+C).")


def _print_summary(summary: FaceRunSummary, source: FrameSource, detector: Any, display: bool) -> None:
    print(f"[BİLGİ] Bitiş nedeni: {_REASON_TEXT.get(summary.reason, summary.reason)}")
    print(f"[BİLGİ] İşlenen kare: {summary.frames_processed} | süre: {summary.elapsed_s:.1f} sn | "
          f"uçtan uca FPS (ölçülen): {summary.end_to_end_fps:.1f} | gerçek çözünürlük: {_fmt_size(source.actual_resolution)} | "
          f"çalışma yeri: {detector.device}")
    print(f"[BİLGİ] Çıkarım süresi: {_fmt_stats(summary.inference)}")
    print(f"[BİLGİ] Kare okuma süresi: {_fmt_stats(summary.read)}")
    print(f"[BİLGİ] Kare -> sonuç gecikmesi: {_fmt_stats(summary.frame_to_result)}")
    if display and summary.frame_to_display is not None:
        print(f"[BİLGİ] Kare -> görüntü gecikmesi: {_fmt_stats(summary.frame_to_display)}")
    print(f"[BİLGİ] Toplam yüz tespiti: {summary.faces_total} (benzersiz kişi sayısı DEĞİLDİR; aynı yüz her karede yeniden sayılır) | "
          f"kare başına ortalama {summary.faces_mean:.2f} | en çok {summary.faces_max} | "
          f"en az bir yüz olan kare: {summary.frames_with_faces}/{summary.frames_processed}")
    if summary.dropped_boxes:
        print(f"[UYARI] {summary.dropped_boxes} geçersiz kutu atıldı.")
    if summary.truncated_faces:
        print(f"[UYARI] {summary.truncated_faces} yüz --max-faces sınırı nedeniyle kesildi (güveni en düşük olanlar).")
    if not display:
        print("[BİLGİ] Pencere kapalıydı; FPS çizim/gösterme maliyetini içermez.")
    if source.kind == SOURCE_KIND_CAMERA and source.source_fps is not None:
        print("[BİLGİ] Not: kameranın bildirdiği FPS performans ölçüsü değildir; ölçülen değerler yukarıdadır.")
    if source.premature_end:
        print(f"[UYARI] Video beklenenden erken bitti: {source.frames_read} kare okundu, dosya {source.frame_count} kare "
              "bildiriyor. Dosya kesik veya bozuk olabilir.", file=sys.stderr)
