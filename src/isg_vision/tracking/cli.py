"""Komut satırı arayüzü (``scripts/run_person_tracking.py`` bu modülün ``main()`` fonksiyonunu çağırır).

Kaynak ve model seçenekleri ``run_person_detection.py`` ile BİREBİR aynıdır (4A'nın ``build_parser``'ı yeniden kullanılır); yalnızca
takip (ByteTrack) eşikleri eklenir. 4A betiği değişmemiştir.

Çıkış kodları (``run_camera.py`` / ``run_person_detection.py`` ile aynı): 0 = normal bitiş (Q/Esc, pencere kapatma, video sonu,
``--max-frames``), 1 = kamera/video/pencere/model/takip hatası, 2 = hatalı argüman, 130 = ``Ctrl+C``. Her koşulda kamera, pencere,
model ve takipçi serbest bırakılır.

Sıra bilinçli olarak şudur: (1) YOLO modeli yüklenir, (2) takipçi (bağımlılık denetimiyle) başlatılır — ikisi de başarısızsa kamera HİÇ
açılmaz, (3) kaynak açılır, (4) model gerçek kare boyutunda ısıtılır, (5) pencere açılır, (6) döngü. Görüntü kaydedilmez, hiçbir ağ işlemi
yapılmaz, hiçbir paket kurulmaz (``lap`` yoksa açıklayıcı hata verilir).

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
from isg_vision.camera.config import SourceConfig
from isg_vision.camera.display import PreviewWindow
from isg_vision.camera.errors import DisplayError, SourceConfigError, SourceError
from isg_vision.camera.frame import SOURCE_KIND_CAMERA
from isg_vision.human_detection.cli import _fmt_fps, _fmt_size, _fmt_stats, build_parser as build_detection_parser, detection_config_from_args
from isg_vision.human_detection.config import DetectionConfig
from isg_vision.human_detection.errors import DetectionConfigError, DetectionError
from isg_vision.human_detection.person_detector import PersonDetector

from .bytetrack_adapter import PersonTracker
from .config import (
    DEFAULT_MATCH_THRESH,
    DEFAULT_NEW_TRACK_THRESH,
    DEFAULT_TRACK_BUFFER,
    DEFAULT_TRACK_HIGH_THRESH,
    DEFAULT_TRACK_LOW_THRESH,
    MAX_TRACK_BUFFER,
    TrackingConfig,
    analyze_thresholds,
)
from .errors import TrackingConfigError, TrackingError
from .pipeline import (
    REASON_END_OF_STREAM,
    REASON_INTERRUPTED,
    REASON_MAX_FRAMES,
    REASON_USER_QUIT,
    REASON_WINDOW_CLOSED,
    TrackingRunSummary,
    run_tracking,
)
from .visualization import WINDOW_NAME

__all__ = ["EXIT_ERROR", "EXIT_INTERRUPTED", "EXIT_OK", "EXIT_USAGE", "build_parser", "main", "run"]

MAX_TABLE_ROWS = 20

_REASON_TEXT = {
    REASON_USER_QUIT: "kullanıcı çıkışı (Q/Esc)",
    REASON_WINDOW_CLOSED: "pencere kapatıldı",
    REASON_END_OF_STREAM: "video sona erdi",
    REASON_MAX_FRAMES: "--max-frames sınırına ulaşıldı",
    REASON_INTERRUPTED: "Ctrl+C ile durduruldu",
}

_EXAMPLES = """\
örnekler:
  python scripts/run_person_tracking.py --source camera --camera-index 0 --device mps
  python scripts/run_person_tracking.py --source camera --camera-index 0 --device cpu
  python scripts/run_person_tracking.py --source video --video-path /yol/video.mp4 --device mps
  python scripts/run_person_tracking.py --source camera --device mps --no-display --max-frames 300   # pencere açmadan 300 kare
  python scripts/run_person_tracking.py --source camera --device mps --conf 0.25                      # YOLO güven eşiği

  # yanlış pozitif denemesi (varsayılan DEĞİLDİR; etkisi ölçülmeden önerilmez): yeni iz için daha yüksek güven iste
  python scripts/run_person_tracking.py --source camera --new-track-thresh 0.6
  # ByteTrack düşük güvenli kurtarma aşamasını aç (YOLO conf, --track-high-thresh'in altına inmeli)
  python scripts/run_person_tracking.py --source camera --conf 0.10 --track-high-thresh 0.35 --new-track-thresh 0.35

pencerede çıkış: Q veya Esc tuşu ya da pencereyi kapatma; terminalde Ctrl+C.
Track ID geçici bir takip numarasıdır; çalışan kimliği DEĞİLDİR (oturum kapsamlıdır, her oturumda 1'den başlar).
Çıkış kodları: 0 normal, 1 kamera/video/pencere/model/takip hatası, 2 hatalı argüman, 130 Ctrl+C.
Ağırlık dosyası yoksa önce:  python scripts/download_person_model.py   (ağ kullanan tek adım)
ByteTrack 'lap' paketini ister; kurulu değilse hiçbir şey otomatik kurulmaz, kurulum komutu hata mesajında yazılıdır.
"""


def _unit_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"sayı olmalıdır: {text!r}") from None
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError(f"0'dan büyük ve 1'e eşit veya küçük olmalıdır: {text!r}")
    return value


def _buffer_frames(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"tam sayı olmalıdır: {text!r}") from None
    if not 1 <= value <= MAX_TRACK_BUFFER:
        raise argparse.ArgumentTypeError(f"1 ile {MAX_TRACK_BUFFER} arasında olmalıdır (kare sayısı): {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = build_detection_parser()  # kaynak + model + çalışma seçenekleri 4A ile aynı
    parser.prog = "run_person_tracking.py"
    parser.description = (
        "Kamera veya video dosyasındaki tüm insanları YOLO ile tespit edip ByteTrack ile kareler arasında takip eder; her aktif kişiye "
        "GEÇİCİ bir Track ID atar ve kutu + ID, tespit/aktif/kayıp iz sayıları, ölçülen FPS, çıkarım ve takip süresini canlı gösterir. "
        "Aşama 4B: YOLO her karede yalnızca bir kez çalışır. Track ID bir çalışan kimliği değildir; yüz/kimlik tanıma, KKD analizi yoktur; "
        "görüntü kaydedilmez, ağ kullanılmaz, paket kurulmaz."
    )
    parser.epilog = _EXAMPLES
    trk = parser.add_argument_group(
        "takip (ByteTrack eşikleri; varsayılanlar Ultralytics 8.4.210 bytetrack.yaml ile aynı, evrensel optimum değil)"
    )
    trk.add_argument("--track-high-thresh", type=_unit_float, default=DEFAULT_TRACK_HIGH_THRESH, metavar="F",
                     help=f"Birinci eşleştirmeye girecek tespitlerin alt güven sınırı (varsayılan: {DEFAULT_TRACK_HIGH_THRESH}). YOLO --conf bunun "
                          "altındaysa düşük güvenli kurtarma aşaması açılır.")
    trk.add_argument("--track-low-thresh", type=_unit_float, default=DEFAULT_TRACK_LOW_THRESH, metavar="F",
                     help=f"Düşük güvenli (kurtarma) aşaması için alt sınır, --track-high-thresh'ten küçük olmalı (varsayılan: {DEFAULT_TRACK_LOW_THRESH}).")
    trk.add_argument("--new-track-thresh", type=_unit_float, default=DEFAULT_NEW_TRACK_THRESH, metavar="F",
                     help=f"Yeni iz başlatmak için en düşük güven, --track-high-thresh'ten küçük olamaz (varsayılan: {DEFAULT_NEW_TRACK_THRESH}). "
                          "Yükseltmek yanlış pozitiflerden iz doğmasını azaltır ama gerçek kişinin iz almasını da geciktirebilir.")
    trk.add_argument("--track-buffer", type=_buffer_frames, default=DEFAULT_TRACK_BUFFER, metavar="KARE",
                     help=f"Kaybolan izin tutulacağı kare sayısı, 1..{MAX_TRACK_BUFFER} (varsayılan: {DEFAULT_TRACK_BUFFER}); süre değil KARE sayısıdır "
                          "(30 FPS'te 30 kare ≈ 1 sn).")
    trk.add_argument("--match-thresh", type=_unit_float, default=DEFAULT_MATCH_THRESH, metavar="F",
                     help=f"Eşleştirme maliyet eşiği (1-IoU), büyük = daha gevşek eşleştirme (varsayılan: {DEFAULT_MATCH_THRESH}).")
    trk.add_argument("--no-fuse-score", action="store_true", help="Eşleştirme maliyetini tespit skoruyla birleştirme (varsayılan: birleştirir).")
    return parser


def tracking_config_from_args(args: argparse.Namespace) -> TrackingConfig:
    """Argümanları doğrulayıp ``TrackingConfig`` üretir (``TrackingConfigError`` fırlatabilir)."""
    return TrackingConfig(
        track_high_thresh=args.track_high_thresh,
        track_low_thresh=args.track_low_thresh,
        new_track_thresh=args.new_track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        fuse_score=not args.no_fuse_score,
    )


def run(
    source_config: SourceConfig,
    detection_config: DetectionConfig,
    tracking_config: TrackingConfig,
    *,
    display: bool = True,
    max_frames: int | None = None,
    capture_factory: Callable[..., Any] | None = None,
    window_factory: Callable[[], Any] | None = None,
    detector_factory: Callable[[DetectionConfig], Any] = PersonDetector,
    tracker_factory: Callable[[TrackingConfig], Any] = PersonTracker,
) -> int:
    """Modeli ve takipçiyi yükler, kaynağı açar, takip döngüsünü çalıştırır ve her koşulda kaynakları serbest bırakır. Çıkış kodunu döndürür."""
    detector = detector_factory(detection_config)
    tracker = tracker_factory(tracking_config)
    source = FrameSource(source_config, capture_factory=capture_factory)
    window: Any | None = None
    try:
        detector.load()  # başarısızsa kamera hiç açılmaz
        _print_model(detector, detection_config)
        tracker.load()  # lap eksikse burada, kamera açılmadan hata verir
        _print_tracker(tracker, detection_config, tracking_config)
        source.open()
        _print_source(source, display)
        warmup_s = detector.warmup(source.actual_resolution or (640, 480))
        print(f"[BİLGİ] Model ısıtıldı: {warmup_s:.2f} sn (sürekli performanstan ayrı; ilk karelerin gecikmesini azaltır).")
        if display:
            window = (window_factory or partial(PreviewWindow, WINDOW_NAME))()
            window.open()
            if getattr(window, "close_button_enabled", None) is False:
                print("[UYARI] Pencerenin kırmızı kapatma düğmesi etkinleştirilemedi; pencereyi Q/Esc tuşuyla veya terminalde Ctrl+C ile kapatın.")
        summary = run_tracking(source, detector, tracker, window, max_frames=max_frames)
    except KeyboardInterrupt:  # model/takipçi yüklenirken, kaynak açılırken veya pencere oluşturulurken Ctrl+C
        print("\n[BİLGİ] Ctrl+C ile durduruldu.")
        return EXIT_INTERRUPTED
    except (SourceError, DisplayError, DetectionError, TrackingError) as exc:
        print(f"[HATA] {exc}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        if window is not None:
            window.close()
        source.close()
        tracker.close()
        detector.close()

    _print_summary(summary, source, detector, display)
    return EXIT_INTERRUPTED if summary.reason == REASON_INTERRUPTED else EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    *,
    capture_factory: Callable[..., Any] | None = None,
    window_factory: Callable[[], Any] | None = None,
    detector_factory: Callable[[DetectionConfig], Any] = PersonDetector,
    tracker_factory: Callable[[TrackingConfig], Any] = PersonTracker,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        source_config = config_from_args(args)
        detection_config = detection_config_from_args(args)
        tracking_config = tracking_config_from_args(args)
    except (SourceConfigError, DetectionConfigError, TrackingConfigError) as exc:
        parser.error(str(exc))  # kullanım bilgisini yazar ve 2 koduyla çıkar
    return run(
        source_config,
        detection_config,
        tracking_config,
        display=not args.no_display,
        max_frames=args.max_frames,
        capture_factory=capture_factory,
        window_factory=window_factory,
        detector_factory=detector_factory,
        tracker_factory=tracker_factory,
    )


# ---------------------------------------------------------------------- çıktı yardımcıları
def _print_model(detector: Any, config: DetectionConfig) -> None:
    load_s = getattr(detector, "load_seconds", None)
    print(f"[BİLGİ] Model: {config.model_name} | cihaz: {detector.device} (istenen: {config.device}) | conf: {config.conf} | "
          f"iou: {config.iou} | imgsz: {config.imgsz}" + (f" | yükleme: {load_s:.2f} sn" if load_s is not None else ""))
    warning = getattr(detector, "device_warning", None)
    if warning:
        print(f"[UYARI] {warning}")


def _print_tracker(tracker: Any, detection_config: DetectionConfig, config: TrackingConfig) -> None:
    info = getattr(tracker, "backend_info", {}) or {}
    versions = info.get("versions") or {}
    version_text = ", ".join(f"{k} {v}" for k, v in versions.items())
    print(f"[BİLGİ] Takip: ByteTrack (Ultralytics) | high {config.track_high_thresh} | low {config.track_low_thresh} | yeni-iz {config.new_track_thresh} | "
          f"tampon {config.track_buffer} kare | eşleştirme {config.match_thresh} | skor birleştirme {'açık' if config.fuse_score else 'kapalı'}"
          + (f" | {version_text}" if version_text else ""))
    if info.get("version_note"):
        print(f"[UYARI] {info['version_note']}")
    for note in analyze_thresholds(detection_config.conf, config).notes:
        print(f"[BİLGİ] Eşik notu: {note}")
    print("[BİLGİ] Track ID geçici ve oturum kapsamlıdır; çalışan kimliği DEĞİLDİR. YOLO her karede yalnızca bir kez çalışır.")


def _print_source(source: FrameSource, display: bool) -> None:
    print(f"[BİLGİ] Kaynak açıldı: {source.label} | çözünürlük: {_fmt_size(source.actual_resolution)} | "
          f"kaynak FPS (bildirilen, yalnızca bilgi): {_fmt_fps(source.source_fps)}")
    mismatch = source.resolution_mismatch
    if mismatch is not None:
        requested, actual = mismatch
        print(f"[UYARI] İstenen çözünürlük ({_fmt_size(requested)}) uygulanamadı; kaynak {_fmt_size(actual)} kullanıyor.")
    if display:
        print("[BİLGİ] Çıkmak için pencerede Q veya Esc tuşuna basın, pencerenin kapatma düğmesine tıklayın (ya da terminalde Ctrl+C).")


def _print_summary(summary: TrackingRunSummary, source: FrameSource, detector: Any, display: bool) -> None:
    print(f"[BİLGİ] Bitiş nedeni: {_REASON_TEXT.get(summary.reason, summary.reason)}")
    print(f"[BİLGİ] İşlenen kare: {summary.frames_processed} | süre: {summary.elapsed_s:.1f} sn | "
          f"uçtan uca FPS (ölçülen): {summary.end_to_end_fps:.1f} | gerçek çözünürlük: {_fmt_size(source.actual_resolution)} | "
          f"YOLO cihazı: {detector.device} (ByteTrack: CPU)")
    print(f"[BİLGİ] Kare okuma süresi: {_fmt_stats(summary.read)}")
    print(f"[BİLGİ] YOLO çıkarım süresi: {_fmt_stats(summary.inference)}")
    print(f"[BİLGİ] ByteTrack takip süresi: {_fmt_stats(summary.tracking)}")
    print(f"[BİLGİ] YOLO + ByteTrack (çıkarım+takip): {_fmt_stats(summary.detect_plus_track)}")
    print(f"[BİLGİ] Kare -> takip sonucu gecikmesi: {_fmt_stats(summary.frame_to_result)}")
    if display:
        if summary.draw is not None:
            print(f"[BİLGİ] Çizim süresi: {_fmt_stats(summary.draw)}")
        if summary.show is not None:
            print(f"[BİLGİ] Pencereye gösterme süresi (imshow+waitKey): {_fmt_stats(summary.show)}")
        if summary.frame_to_display is not None:
            print(f"[BİLGİ] Kare -> görüntü gecikmesi: {_fmt_stats(summary.frame_to_display)}")
    print(f"[BİLGİ] Tespit: kare başına ortalama {summary.detections_mean:.2f} | en çok {summary.detections_max} | "
          f"aktif iz en çok: {summary.active_max} | kayıp (tamponda) iz en çok: {summary.lost_max} | "
          f"en az bir aktif iz olan kare: {summary.frames_with_tracks}/{summary.frames_processed}")
    print(f"[BİLGİ] Görülen farklı Track ID: {summary.distinct_tracks} (gerçek kişi sayısından belirgin büyükse izler parçalanıyor "
          "veya yanlış pozitifler iz alıyor olabilir; tek başına doğruluk ölçüsü DEĞİLDİR)")
    _print_track_table(summary)
    if summary.dropped_boxes:
        print(f"[UYARI] {summary.dropped_boxes} geçersiz tespit kutusu atıldı.")
    if summary.dropped_tracks:
        print(f"[UYARI] {summary.dropped_tracks} geçersiz iz kutusu atıldı.")
    if summary.reset_events:
        print(f"[UYARI] Oturum sırasında takipçi {summary.reset_events} kez otomatik sıfırlandı (kaynak/çözünürlük değişimi).")
    if not display:
        print("[BİLGİ] Pencere kapalıydı; FPS çizim/gösterme maliyetini içermez.")
    if source.kind == SOURCE_KIND_CAMERA and source.source_fps is not None:
        print("[BİLGİ] Not: kameranın bildirdiği FPS performans ölçüsü değildir; ölçülen değerler yukarıdadır.")
    if source.premature_end:
        print(f"[UYARI] Video beklenenden erken bitti: {source.frames_read} kare okundu, dosya {source.frame_count} kare "
              "bildiriyor. Dosya kesik veya bozuk olabilir.", file=sys.stderr)


def _print_track_table(summary: TrackingRunSummary) -> None:
    if not summary.tracks:
        print("[BİLGİ] Hiç aktif iz oluşmadı.")
        return
    print("[BİLGİ] İz özeti (Track ID | ilk-son kare | aktif kare | süre | ort/en yüksek YOLO güveni) — kısa süren izler yanlış pozitif olabilir:")
    for span in summary.tracks[:MAX_TABLE_ROWS]:
        duration = f"{span.duration_s:.2f} sn" if span.duration_s is not None else "—"
        print(f"           ID {span.track_id:>3} (oturum {span.session_id}) | kare {span.first_frame}-{span.last_frame} | {span.frames_active} kare | "
              f"{duration} | {span.confidence_mean:.2f} / {span.confidence_max:.2f}")
    if len(summary.tracks) > MAX_TABLE_ROWS:
        print(f"           ... ve {len(summary.tracks) - MAX_TABLE_ROWS} iz daha")
    if summary.spans_truncated:
        print("[UYARI] İz sayısı sınırı aşıldı; özet tüm izleri içermiyor.")
