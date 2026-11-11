"""Komut satırı arayüzü (``scripts/run_camera.py`` bu modülün ``main()`` fonksiyonunu çağırır).

Çıkış kodları: 0 = normal bitiş (Q/Esc, pencere kapatma, video sonu, ``--max-frames``), 1 = kamera/video/pencere hatası,
2 = hatalı argüman, 130 = ``Ctrl+C`` ile durdurma. Her koşulda kamera ve pencere serbest bırakılır.

Bu modül import edildiğinde hiçbir kamera açılmaz; kaynak yalnızca ``main()``/``run()`` çalışınca açılır.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from typing import Any

from .capture import FrameSource
from .config import BACKENDS, DEFAULT_HEIGHT, DEFAULT_WIDTH, SourceConfig
from .display import PreviewWindow
from .errors import DisplayError, SourceConfigError, SourceError
from .frame import SOURCE_KIND_CAMERA
from .preview import (
    REASON_END_OF_STREAM,
    REASON_INTERRUPTED,
    REASON_MAX_FRAMES,
    REASON_USER_QUIT,
    REASON_WINDOW_CLOSED,
    PreviewResult,
    run_preview,
)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

_REASON_TEXT = {
    REASON_USER_QUIT: "kullanıcı çıkışı (Q/Esc)",
    REASON_WINDOW_CLOSED: "pencere kapatıldı",
    REASON_END_OF_STREAM: "video sona erdi",
    REASON_MAX_FRAMES: "--max-frames sınırına ulaşıldı",
    REASON_INTERRUPTED: "Ctrl+C ile durduruldu",
}

_EXAMPLES = """\
örnekler:
  python scripts/run_camera.py --source camera
  python scripts/run_camera.py --source camera --camera-index 1
  python scripts/run_camera.py --source camera --width 640 --height 480
  python scripts/run_camera.py --source video --video-path /yol/video.mp4
  python scripts/run_camera.py --source camera --no-display --max-frames 100   # pencere açmadan 100 kare say

pencerede çıkış: Q veya Esc tuşu ya da pencereyi kapatma; terminalde Ctrl+C.
Çıkış kodları: 0 normal, 1 kamera/video/pencere hatası, 2 hatalı argüman, 130 Ctrl+C.
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_camera.py",
        description=(
            "Kamera veya video dosyasından kare okur ve canlı pencerede gösterir (kaynak, ölçülen FPS, çözünürlük, durum). "
            "Aşama 3: yalnızca görüntü altyapısı; hiçbir yapay zekâ modeli çalıştırılmaz ve görüntü kaydedilmez."
        ),
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", choices=("camera", "video"), default="camera",
                        help="Kare kaynağı: camera (varsayılan) veya video")
    parser.add_argument("--camera-index", type=_integer, default=None, metavar="N",
                        help="Kamera indeksi (varsayılan: 0). Yalnızca --source camera ile. "
                             "iPhone Süreklilik Kamerası da bir indeks alabilir.")
    parser.add_argument("--width", type=_integer, default=None, metavar="PX",
                        help=f"İstenen kamera genişliği (varsayılan: {DEFAULT_WIDTH}); --height ile birlikte. "
                             "Cihaz desteklemezse uygulanan çözünürlük farklı olabilir ve bildirilir.")
    parser.add_argument("--height", type=_integer, default=None, metavar="PX",
                        help=f"İstenen kamera yüksekliği (varsayılan: {DEFAULT_HEIGHT}); --width ile birlikte.")
    parser.add_argument("--video-path", default=None, metavar="DOSYA",
                        help="Video dosyası yolu (MP4/MOV/AVI). --source video için zorunludur.")
    parser.add_argument("--backend", choices=BACKENDS, default="auto",
                        help="Kamera backend'i (varsayılan: auto = macOS'ta AVFoundation). Yalnızca kamera için.")
    parser.add_argument("--no-display", action="store_true",
                        help="Pencere açma; kareleri okuyup sayar ve özet yazar (video için bekleme yapılmaz).")
    parser.add_argument("--max-frames", type=_positive_integer, default=None, metavar="N",
                        help="N kare işlendikten sonra dur.")
    return parser


def config_from_args(args: argparse.Namespace) -> SourceConfig:
    """Argümanları doğrulayıp ``SourceConfig`` üretir.

    Raises:
        SourceConfigError: Çelişkili veya geçersiz argümanlar.
    """
    if args.source == "video":
        if args.video_path is None:
            raise SourceConfigError("--source video için --video-path gerekir.")
        for flag, value in (("--camera-index", args.camera_index), ("--width", args.width), ("--height", args.height)):
            if value is not None:
                raise SourceConfigError(f"{flag} yalnızca --source camera ile kullanılabilir.")
        return SourceConfig.video(args.video_path)

    if args.video_path is not None:
        raise SourceConfigError("--video-path yalnızca --source video ile kullanılabilir.")
    if (args.width is None) != (args.height is None):
        raise SourceConfigError("--width ve --height birlikte verilmelidir.")
    width, height = (DEFAULT_WIDTH, DEFAULT_HEIGHT) if args.width is None else (args.width, args.height)
    index = 0 if args.camera_index is None else args.camera_index
    return SourceConfig.camera(index, width=width, height=height, backend=args.backend)


def run(
    config: SourceConfig,
    *,
    display: bool = True,
    max_frames: int | None = None,
    capture_factory: Callable[..., Any] | None = None,
    window_factory: Callable[[], Any] = PreviewWindow,
) -> int:
    """Kaynağı açar, önizlemeyi çalıştırır ve her koşulda kaynakları serbest bırakır. Çıkış kodunu döndürür."""
    source = FrameSource(config, capture_factory=capture_factory)
    window = window_factory() if display else None
    try:
        source.open()
        _print_opened(source, display)
        if window is not None:
            window.open()
            _warn_close_button(window)
        result = run_preview(source, window, max_frames=max_frames)
    except KeyboardInterrupt:  # kaynak açılırken veya pencere oluşturulurken Ctrl+C
        print("\n[BİLGİ] Ctrl+C ile durduruldu.")
        return EXIT_INTERRUPTED
    except (SourceError, DisplayError) as exc:
        print(f"[HATA] {exc}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        if window is not None:
            window.close()
        source.close()

    _print_summary(source, result, display)
    return EXIT_INTERRUPTED if result.reason == REASON_INTERRUPTED else EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    *,
    capture_factory: Callable[..., Any] | None = None,
    window_factory: Callable[[], Any] = PreviewWindow,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
    except SourceConfigError as exc:
        parser.error(str(exc))  # kullanım bilgisini yazar ve 2 koduyla çıkar
    return run(
        config,
        display=not args.no_display,
        max_frames=args.max_frames,
        capture_factory=capture_factory,
        window_factory=window_factory,
    )


# ---------------------------------------------------------------------- çıktı yardımcıları
def _fmt_size(size: tuple[int, int] | None) -> str:
    return f"{size[0]}x{size[1]}" if size else "bilinmiyor"


def _fmt_fps(fps: float | None) -> str:
    return f"{fps:.1f}" if fps else "bilinmiyor"


def _print_opened(source: FrameSource, display: bool) -> None:
    print(f"[BİLGİ] Kaynak açıldı: {source.label}")
    if source.kind == SOURCE_KIND_CAMERA:
        print(f"[BİLGİ] İstenen çözünürlük: {_fmt_size(source.requested_resolution) if source.requested_resolution else 'cihaz varsayılanı'}"
              f" | sürücünün bildirdiği: {_fmt_size(source.actual_resolution)} | kaynak FPS (bildirilen): {_fmt_fps(source.source_fps)}")
    else:
        total = source.frame_count
        print(f"[BİLGİ] Video çözünürlüğü: {_fmt_size(source.actual_resolution)} | kaynak FPS: {_fmt_fps(source.source_fps)}"
              f" | toplam kare: {total if total else 'bilinmiyor'}")
    _warn_resolution(source)
    if display:
        print("[BİLGİ] Çıkmak için pencerede Q veya Esc tuşuna basın, pencerenin kapatma düğmesine tıklayın (ya da terminalde Ctrl+C).")


def _warn_close_button(window: Any) -> None:
    """macOS'ta kapatma düğmesi etkinleştirilemediyse kullanıcıyı uyarır (``None`` = konu geçerli değil)."""
    if getattr(window, "close_button_enabled", None) is False:
        print("[UYARI] Pencerenin kırmızı kapatma düğmesi etkinleştirilemedi; pencereyi Q/Esc tuşuyla veya terminalde Ctrl+C ile kapatın.")


def _warn_resolution(source: FrameSource) -> None:
    mismatch = source.resolution_mismatch
    if mismatch is not None:
        requested, actual = mismatch
        print(f"[UYARI] İstenen çözünürlük ({_fmt_size(requested)}) uygulanamadı; kaynak {_fmt_size(actual)} kullanıyor.")


def _print_summary(source: FrameSource, result: PreviewResult, display: bool) -> None:
    print(f"[BİLGİ] Bitiş nedeni: {_REASON_TEXT.get(result.reason, result.reason)}")
    print(f"[BİLGİ] İşlenen kare: {result.frames_processed} | süre: {result.elapsed_s:.1f} sn | "
          f"ölçülen ortalama FPS: {result.measured_fps:.1f} | kaynak FPS (bildirilen): {_fmt_fps(source.source_fps)} | "
          f"gerçek çözünürlük: {_fmt_size(source.actual_resolution)}")
    if not display:
        print("[BİLGİ] Pencere kapalıydı; ölçülen FPS yalnızca kare okuma hızını gösterir (bekleme/oynatma yok).")
    _warn_resolution(source)
    if source.premature_end:
        print(f"[UYARI] Video beklenenden erken bitti: {source.frames_read} kare okundu, dosya {source.frame_count} kare "
              "bildiriyor. Dosya kesik veya bozuk olabilir.", file=sys.stderr)
