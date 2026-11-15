"""Komut satırı arayüzü (``scripts/run_face_person_association.py`` bu modülün ``main()`` fonksiyonunu çağırır).

Kaynak, YOLO ve ByteTrack seçenekleri ``run_person_tracking.py`` ile BİREBİR aynıdır (4B'nin ``build_parser``'ı yeniden kullanılır); yüz modeli
(``--face-*``), ilişkilendirme (``--min-containment`` vb.) ve ``--opencv-threads`` eklenir. 4A/4B/5A betikleri değişmemiştir.

Çıkış kodları (diğer betiklerle aynı): 0 = normal bitiş (Q/Esc, pencere kapatma, video sonu, ``--max-frames``), 1 = kamera/video/pencere/model/takip/yüz
hatası, 2 = hatalı argüman, 130 = ``Ctrl+C``. Her koşulda kamera, pencere, modeller ve takipçi serbest bırakılır.

Sıra bilinçli olarak şudur: (1) YOLO yüklenir, (2) takipçi başlatılır (``lap`` denetimi), (3) YuNet yüklenir — hiçbiri başarısızsa kamera HİÇ açılmaz,
(4) ``--opencv-threads`` uygulanır (``ultralytics`` içe aktarıldıktan SONRA), (5) kaynak açılır, (6) modeller gerçek kare boyutunda ısıtılır, (7) pencere
açılır, (8) döngü. Görüntü kaydedilmez, ağ kullanılmaz, paket kurulmaz, biyometrik veri yazılmaz.

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
from isg_vision.face_detection.config import (
    DEFAULT_CONF as DEFAULT_FACE_CONF,
    DEFAULT_MAX_FACES,
    DEFAULT_MODEL_PATH as DEFAULT_FACE_MODEL_PATH,
    DEFAULT_NMS as DEFAULT_FACE_NMS,
    MAX_DETECT_WIDTH,
    MIN_DETECT_WIDTH,
    FaceDetectionConfig,
)
from isg_vision.face_detection.detector import FaceDetector
from isg_vision.face_detection.errors import FaceConfigError, FaceDetectionError
from isg_vision.human_detection.cli import _fmt_fps, _fmt_size, _fmt_stats, detection_config_from_args
from isg_vision.human_detection.config import DetectionConfig
from isg_vision.human_detection.errors import DetectionConfigError, DetectionError
from isg_vision.human_detection.person_detector import PersonDetector
from isg_vision.tracking.bytetrack_adapter import PersonTracker
from isg_vision.tracking.cli import build_parser as build_tracking_parser, tracking_config_from_args
from isg_vision.tracking.config import TrackingConfig
from isg_vision.tracking.errors import TrackingConfigError, TrackingError

from .associator import FacePersonAssociator
from .config import (
    DEFAULT_AMBIGUITY_MARGIN,
    DEFAULT_MIN_CONTAINMENT,
    DEFAULT_MIN_SCORE,
    DEFAULT_OVERLAP_GUARD,
    AssociationConfig,
)
from .errors import AssociationConfigError, FaceAssociationError
from .pipeline import (
    REASON_END_OF_STREAM,
    REASON_INTERRUPTED,
    REASON_MAX_FRAMES,
    REASON_USER_QUIT,
    REASON_WINDOW_CLOSED,
    FaceAssociationRunSummary,
    run_face_association,
)
from .threads import MODE_KEEP, MODE_RESTORE, ThreadSetting, apply_opencv_threads, threads_argument
from .visualization import WINDOW_NAME

__all__ = ["EXIT_ERROR", "EXIT_INTERRUPTED", "EXIT_OK", "EXIT_USAGE", "association_config_from_args", "build_parser", "face_config_from_args", "main", "run"]

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
  python scripts/run_face_person_association.py --source camera --camera-index 0 --device mps
  python scripts/run_face_person_association.py --source video --video-path /yol/video.mp4 --device mps
  python scripts/run_face_person_association.py --source camera --device mps --no-display --max-frames 300   # pencere açmadan 300 kare
  python scripts/run_face_person_association.py --source camera --device mps --opencv-threads restore       # YuNet yavaşlamasını deney (ölçüm: docs)
  python scripts/run_face_person_association.py --source camera --face-conf 0.7 --ambiguity-margin 0.2     # daha sıkı eşleştirme

pencerede çıkış: Q veya Esc tuşu ya da pencereyi kapatma; terminalde Ctrl+C.
Track ID geçici bir takip numarasıdır; çalışan kimliği DEĞİLDİR. Yüz–kişi bağı yalnızca GEOMETRİKTİR (kimlik/canlılık değildir); belirsizde eşleştirilmez.
Çıkış kodları: 0 normal, 1 kamera/video/pencere/model/takip/yüz hatası, 2 hatalı argüman, 130 Ctrl+C.
Ağırlıklar yoksa önce:  python scripts/download_person_model.py  ve  python scripts/download_face_model.py   (ağ kullanan tek adımlar)
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
    parser = build_tracking_parser()  # kaynak + YOLO + ByteTrack + çalışma seçenekleri 4B ile aynı
    parser.prog = "run_face_person_association.py"
    parser.description = (
        "Kamera veya video dosyasında YOLO11n ile insanları tespit eder, ByteTrack ile geçici Track ID verir, YuNet ile yüzleri tespit eder ve her yüzü, "
        "GEOMETRİK kanıt güvenilirse, ilgili Track ID'ye bağlar; belirsizde eşleştirmez. Aşama 5B: her model kare başına yalnızca bir kez çalışır. "
        "Track ID bir çalışan kimliği DEĞİLDİR; yüz tanıma/çalışan kimliği/biyometrik veri/KKD analizi yoktur; yüz–kişi bağı canlı insan doğrulaması "
        "değildir (poster/ekrandaki yüz de bağlanabilir); görüntü kaydedilmez, ağ kullanılmaz, paket kurulmaz."
    )
    parser.epilog = _EXAMPLES
    face = parser.add_argument_group("yüz modeli (YuNet; yalnızca CPU; --model YOLO ağırlığıdır)")
    face.add_argument("--face-model", default=None, metavar="DOSYA.onnx",
                      help="YuNet ONNX dosyası (varsayılan: models/face_detection/face_detection_yunet_2026may.onnx). Resmî dosya SHA-256 ile doğrulanır.")
    face.add_argument("--face-conf", type=_unit_float, default=DEFAULT_FACE_CONF, metavar="F",
                      help=f"Yüz güven eşiği, 0 < F <= 1 (varsayılan: {DEFAULT_FACE_CONF}). Güven skoru kimlik doğruluğu değildir.")
    face.add_argument("--face-nms", type=_unit_float, default=DEFAULT_FACE_NMS, metavar="F", help=f"Yüz NMS IoU eşiği (varsayılan: {DEFAULT_FACE_NMS}).")
    face.add_argument("--face-max-faces", type=_positive_integer, default=DEFAULT_MAX_FACES, metavar="N", help=f"Karede en fazla yüz (varsayılan: {DEFAULT_MAX_FACES}).")
    face.add_argument("--face-detect-width", type=_integer, default=None, metavar="PX",
                      help=f"Verilirse kare bu genişliğe küçültülüp yüz tespiti yapılır ({MIN_DETECT_WIDTH}-{MAX_DETECT_WIDTH}); küçük/uzak yüzler kaçabilir.")

    assoc = parser.add_argument_group("yüz–kişi ilişkilendirme (varsayılanlar gerekçeli başlangıç değerleridir; docs/face_association.md)")
    assoc.add_argument("--min-containment", type=_unit_float, default=DEFAULT_MIN_CONTAINMENT, metavar="F",
                       help=f"Yüzün kişi kutusu içinde kalması gereken en düşük oran (varsayılan: {DEFAULT_MIN_CONTAINMENT}).")
    assoc.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE, metavar="F",
                       help=f"Eşleşme için en düşük geometrik skor, 0..1 (varsayılan: {DEFAULT_MIN_SCORE}). Altı 'belirsiz' sayılır.")
    assoc.add_argument("--ambiguity-margin", type=_unit_float, default=DEFAULT_AMBIGUITY_MARGIN, metavar="F",
                       help=f"İlk iki aday arasındaki en az skor farkı (varsayılan: {DEFAULT_AMBIGUITY_MARGIN}). Daha küçük fark = belirsiz = eşleştirme yok. "
                            "Büyütmek daha çok 'belirsiz', küçültmek daha çok (ve daha riskli) eşleşme demektir.")
    assoc.add_argument("--overlap-guard", type=_unit_float, default=DEFAULT_OVERLAP_GUARD, metavar="F",
                       help=f"İki kişi kutusunun ortak alanı küçük kutunun bu oranını aşarsa 'örtüşme' (iz güvensiz sayılır; varsayılan: {DEFAULT_OVERLAP_GUARD}).")

    runtime = parser.add_argument_group("OpenCV iş parçacığı (ölçüm ve gerekçe: docs/face_association.md)")
    runtime.add_argument("--opencv-threads", type=threads_argument, default=MODE_KEEP, metavar="keep|restore|N",
                         help="ultralytics içe aktarılınca OpenCV tek iş parçacığına düşer ve YuNet ≈ 2,7x yavaşlar. keep (varsayılan): hiçbir şey yapma; "
                              "restore: cv2.setNumThreads(-1) ile OpenCV varsayılanına dön; N: cv2.setNumThreads(N). Süreç genelindedir; yalnızca bu betiği etkiler.")
    return parser


def face_config_from_args(args: argparse.Namespace) -> FaceDetectionConfig:
    """Argümanları doğrulayıp ``FaceDetectionConfig`` üretir (``FaceConfigError`` fırlatabilir)."""
    return FaceDetectionConfig(
        model_path=DEFAULT_FACE_MODEL_PATH if args.face_model is None else args.face_model,
        conf=args.face_conf,
        nms=args.face_nms,
        max_faces=args.face_max_faces,
        detect_width=args.face_detect_width,
    )


def association_config_from_args(args: argparse.Namespace) -> AssociationConfig:
    """Argümanları doğrulayıp ``AssociationConfig`` üretir (``AssociationConfigError`` fırlatabilir)."""
    return AssociationConfig(
        min_containment=args.min_containment, min_score=args.min_score, ambiguity_margin=args.ambiguity_margin, overlap_guard=args.overlap_guard
    )


def run(
    source_config: SourceConfig,
    detection_config: DetectionConfig,
    tracking_config: TrackingConfig,
    face_config: FaceDetectionConfig,
    association_config: AssociationConfig,
    *,
    display: bool = True,
    max_frames: int | None = None,
    opencv_threads: str = MODE_KEEP,
    capture_factory: Callable[..., Any] | None = None,
    window_factory: Callable[[], Any] | None = None,
    detector_factory: Callable[[DetectionConfig], Any] = PersonDetector,
    tracker_factory: Callable[[TrackingConfig], Any] = PersonTracker,
    face_detector_factory: Callable[[FaceDetectionConfig], Any] = FaceDetector,
    associator_factory: Callable[[AssociationConfig], Any] = FacePersonAssociator,
    thread_applier: Callable[[str], ThreadSetting] = apply_opencv_threads,
) -> int:
    """Modelleri ve takipçiyi yükler, kaynağı açar, döngüyü çalıştırır ve her koşulda kaynakları serbest bırakır. Çıkış kodunu döndürür."""
    detector = detector_factory(detection_config)
    tracker = tracker_factory(tracking_config)
    face_detector = face_detector_factory(face_config)
    associator = associator_factory(association_config)
    source = FrameSource(source_config, capture_factory=capture_factory)
    window: Any | None = None
    try:
        detector.load()  # başarısızsa kamera hiç açılmaz
        _print_model(detector, detection_config)
        tracker.load()  # lap eksikse burada, kamera açılmadan hata verir
        _print_tracker(tracker, tracking_config)
        face_detector.load()
        _print_face_model(face_detector, face_config)
        setting = thread_applier(opencv_threads)  # ultralytics YUKARIDA içe aktarıldı; ayar ondan SONRA uygulanmalıdır
        _print_threads(setting)
        print(f"[BİLGİ] İlişkilendirme: min içerme {association_config.min_containment} | min skor {association_config.min_score} | belirsizlik payı "
              f"{association_config.ambiguity_margin} | örtüşme eşiği {association_config.overlap_guard}. Her model kare başına yalnızca bir kez çalışır.")
        source.open()
        _print_source(source, display)
        size = source.actual_resolution or (640, 480)
        warmup_s = detector.warmup(size) + face_detector.warmup(size)
        print(f"[BİLGİ] Modeller ısıtıldı: {warmup_s:.2f} sn (sürekli performanstan ayrı; ilk karelerin gecikmesini azaltır).")
        if display:
            window = (window_factory or partial(PreviewWindow, WINDOW_NAME))()
            window.open()
            if getattr(window, "close_button_enabled", None) is False:
                print("[UYARI] Pencerenin kırmızı kapatma düğmesi etkinleştirilemedi; pencereyi Q/Esc tuşuyla veya terminalde Ctrl+C ile kapatın.")
        summary = run_face_association(source, detector, tracker, face_detector, associator, window, max_frames=max_frames)
    except KeyboardInterrupt:  # modeller yüklenirken, kaynak açılırken veya pencere oluşturulurken Ctrl+C
        print("\n[BİLGİ] Ctrl+C ile durduruldu.")
        return EXIT_INTERRUPTED
    except (SourceError, DisplayError, DetectionError, TrackingError, FaceDetectionError, FaceAssociationError) as exc:
        print(f"[HATA] {exc}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        if window is not None:
            window.close()
        source.close()
        tracker.close()
        face_detector.close()
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
    face_detector_factory: Callable[[FaceDetectionConfig], Any] = FaceDetector,
    associator_factory: Callable[[AssociationConfig], Any] = FacePersonAssociator,
    thread_applier: Callable[[str], ThreadSetting] = apply_opencv_threads,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        source_config = config_from_args(args)
        detection_config = detection_config_from_args(args)
        tracking_config = tracking_config_from_args(args)
        face_config = face_config_from_args(args)
        association_config = association_config_from_args(args)
    except (SourceConfigError, DetectionConfigError, TrackingConfigError, FaceConfigError, AssociationConfigError) as exc:
        parser.error(str(exc))  # kullanım bilgisini yazar ve 2 koduyla çıkar
    return run(
        source_config, detection_config, tracking_config, face_config, association_config,
        display=not args.no_display, max_frames=args.max_frames, opencv_threads=args.opencv_threads,
        capture_factory=capture_factory, window_factory=window_factory, detector_factory=detector_factory,
        tracker_factory=tracker_factory, face_detector_factory=face_detector_factory, associator_factory=associator_factory,
        thread_applier=thread_applier,
    )


# ---------------------------------------------------------------------- çıktı yardımcıları
def _print_model(detector: Any, config: DetectionConfig) -> None:
    load_s = getattr(detector, "load_seconds", None)
    print(f"[BİLGİ] YOLO: {config.model_name} | cihaz: {detector.device} (istenen: {config.device}) | conf: {config.conf} | "
          f"iou: {config.iou} | imgsz: {config.imgsz}" + (f" | yükleme: {load_s:.2f} sn" if load_s is not None else ""))
    warning = getattr(detector, "device_warning", None)
    if warning:
        print(f"[UYARI] {warning}")


def _print_tracker(tracker: Any, config: TrackingConfig) -> None:
    info = getattr(tracker, "backend_info", {}) or {}
    versions = info.get("versions") or {}
    version_text = ", ".join(f"{k} {v}" for k, v in versions.items())
    print(f"[BİLGİ] Takip: ByteTrack (Ultralytics) | high {config.track_high_thresh} | low {config.track_low_thresh} | yeni-iz {config.new_track_thresh} | "
          f"tampon {config.track_buffer} kare | eşleştirme {config.match_thresh}" + (f" | {version_text}" if version_text else ""))
    if info.get("version_note"):
        print(f"[UYARI] {info['version_note']}")
    print("[BİLGİ] Track ID geçici ve oturum kapsamlıdır; çalışan kimliği DEĞİLDİR.")


def _print_face_model(detector: Any, config: FaceDetectionConfig) -> None:
    load_s = getattr(detector, "load_seconds", None)
    width = f"{config.detect_width} px" if config.detect_width is not None else "kare çözünürlüğü"
    print(f"[BİLGİ] Yüz: {config.model_name} (YuNet, OpenCV DNN) | çalışma yeri: {detector.device} | conf: {config.conf} | nms: {config.nms} | "
          f"tespit genişliği: {width}" + (f" | yükleme: {load_s:.3f} sn" if load_s is not None else ""))
    print("[BİLGİ] Not: yüz tespiti/eşleşmesi kimlik veya canlı insan doğrulaması değildir; poster/fotoğraf/ekrandaki yüz de tespit edilebilir ve bağlanabilir.")


def _print_threads(setting: ThreadSetting) -> None:
    if setting.changed:
        print(f"[BİLGİ] OpenCV iş parçacığı: istek '{setting.requested}' -> cv2.setNumThreads({setting.value}); getNumThreads önce {setting.before}, sonra {setting.after}.")
    elif setting.before == 1:
        print(f"[UYARI] OpenCV iş parçacığı: getNumThreads()=1 (ultralytics ayarladı); YuNet bu durumda ≈ 2,7x yavaş ölçüldü (1280x720: ≈ 30 ms; docs/face_association.md). "
              f"Değiştirmedim (varsayılan '{MODE_KEEP}'). Denemek için: --opencv-threads {MODE_RESTORE}")
    else:
        print(f"[BİLGİ] OpenCV iş parçacığı: getNumThreads()={setting.before} (değiştirilmedi).")


def _print_source(source: FrameSource, display: bool) -> None:
    print(f"[BİLGİ] Kaynak açıldı: {source.label} | çözünürlük: {_fmt_size(source.actual_resolution)} | "
          f"kaynak FPS (bildirilen, yalnızca bilgi): {_fmt_fps(source.source_fps)}")
    mismatch = source.resolution_mismatch
    if mismatch is not None:
        requested, actual = mismatch
        print(f"[UYARI] İstenen çözünürlük ({_fmt_size(requested)}) uygulanamadı; kaynak {_fmt_size(actual)} kullanıyor.")
    if display:
        print("[BİLGİ] Çıkmak için pencerede Q veya Esc tuşuna basın, pencerenin kapatma düğmesine tıklayın (ya da terminalde Ctrl+C).")


def _print_summary(summary: FaceAssociationRunSummary, source: FrameSource, detector: Any, display: bool) -> None:
    s = summary
    print(f"[BİLGİ] Bitiş nedeni: {_REASON_TEXT.get(s.reason, s.reason)}")
    print(f"[BİLGİ] İşlenen kare: {s.frames_processed} | süre: {s.elapsed_s:.1f} sn | uçtan uca FPS (ölçülen): {s.end_to_end_fps:.1f} | "
          f"gerçek çözünürlük: {_fmt_size(source.actual_resolution)} | YOLO cihazı: {detector.device} (ByteTrack/YuNet/ilişkilendirme: CPU)")
    print(f"[BİLGİ] Kare okuma süresi: {_fmt_stats(s.read)}")
    print(f"[BİLGİ] YOLO çıkarım süresi: {_fmt_stats(s.yolo)}")
    print(f"[BİLGİ] ByteTrack takip süresi: {_fmt_stats(s.tracking)}")
    print(f"[BİLGİ] YuNet yüz çıkarım süresi: {_fmt_stats(s.face)}")
    print(f"[BİLGİ] İlişkilendirme süresi: {_fmt_stats(s.association)}")
    print(f"[BİLGİ] Toplam hesaplama (YOLO+ByteTrack+YuNet+ilişkilendirme): {_fmt_stats(s.compute)}")
    print(f"[BİLGİ] Kare -> ilişkilendirme sonucu gecikmesi: {_fmt_stats(s.frame_to_result)}")
    if display:
        if s.draw is not None:
            print(f"[BİLGİ] Çizim süresi: {_fmt_stats(s.draw)}")
        if s.show is not None:
            print(f"[BİLGİ] Pencereye gösterme süresi (imshow+waitKey): {_fmt_stats(s.show)}")
        if s.frame_to_display is not None:
            print(f"[BİLGİ] Kare -> görüntü gecikmesi: {_fmt_stats(s.frame_to_display)}")
    print(f"[BİLGİ] Kişi: kare başına ortalama {s.persons_mean:.2f} | en çok {s.persons_max} | yüz: kare başına ortalama {s.faces_mean:.2f} | en çok {s.faces_max} "
          "(yüz sayısı benzersiz kişi sayısı DEĞİLDİR)")
    print(f"[BİLGİ] Eşleşme (kare başına toplamların toplamı): eşleşen {s.matched_total} (güvenilir süreklilikli {s.trusted_matched_total}) | "
          f"yüzü bulunamayan kişi {s.no_face_total} | belirsiz kişi {s.ambiguous_persons_total} / belirsiz yüz {s.ambiguous_faces_total} | "
          f"eşleşmeyen yüz {s.unmatched_faces_total} | en az bir eşleşmeli kare {s.frames_with_match}/{s.frames_processed}")
    kinds = ", ".join(f"{k}: {v}" for k, v in sorted(s.break_counts.items())) or "yok"
    print(f"[BİLGİ] Süreklilik kesintileri (ID switch/devralma/yeni iz/oturum/örtüşme sonu): {kinds} | görülen farklı Track ID: {s.distinct_tracks} "
          "(gerçek kişi sayısından belirgin büyükse izler parçalanıyor olabilir; tek başına doğruluk ölçüsü DEĞİLDİR)")
    if s.possibly_static_tracks:
        print(f"[BİLGİ] Hareketsiz bayrağı alan iz: {s.possibly_static_tracks} (poster/yorgan benzeri kalıcı yanlış pozitif ADAYI veya oturan kişi; yargı değildir)")
    _print_track_table(s)
    if s.invalid_frames:
        print(f"[UYARI] {s.invalid_frames} karede yüz ve takip sonuçları tutarsızdı; o karelerde hiçbir eşleştirme yapılmadı.")
    if s.dropped_person_boxes:
        print(f"[UYARI] {s.dropped_person_boxes} geçersiz kişi tespit/iz kutusu atıldı.")
    if s.dropped_face_boxes:
        print(f"[UYARI] {s.dropped_face_boxes} geçersiz yüz kutusu atıldı.")
    if s.truncated_faces:
        print(f"[UYARI] {s.truncated_faces} yüz --face-max-faces sınırı nedeniyle kesildi.")
    if s.reset_events:
        print(f"[UYARI] Oturum sırasında takipçi {s.reset_events} kez otomatik sıfırlandı (kaynak/çözünürlük değişimi).")
    if not display:
        print("[BİLGİ] Pencere kapalıydı; FPS çizim/gösterme maliyetini içermez.")
    if source.kind == SOURCE_KIND_CAMERA and source.source_fps is not None:
        print("[BİLGİ] Not: kameranın bildirdiği FPS performans ölçüsü değildir; ölçülen değerler yukarıdadır.")
    if source.premature_end:
        print(f"[UYARI] Video beklenenden erken bitti: {source.frames_read} kare okundu, dosya {source.frame_count} kare "
              "bildiriyor. Dosya kesik veya bozuk olabilir.", file=sys.stderr)


def _print_track_table(summary: FaceAssociationRunSummary) -> None:
    if not summary.tracks:
        print("[BİLGİ] Hiç aktif iz oluşmadı.")
        return
    print("[BİLGİ] İz özeti (Track ID | ilk-son kare | aktif kare | eşleşen kare | güvenilir eşleşen | süreklilik dönemi sayısı):")
    for span in summary.tracks[:MAX_TABLE_ROWS]:
        print(f"           ID {span.track_id:>3} (oturum {span.session_id}) | kare {span.first_frame}-{span.last_frame} | {span.frames_active} | "
              f"{span.frames_matched} | {span.frames_trusted_matched} | epoch {span.epochs}")
    if len(summary.tracks) > MAX_TABLE_ROWS:
        print(f"           ... ve {len(summary.tracks) - MAX_TABLE_ROWS} iz daha")
    if summary.spans_truncated:
        print("[UYARI] İz sayısı sınırı aşıldı; özet tüm izleri içermiyor.")
