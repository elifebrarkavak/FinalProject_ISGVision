"""Kare akışı -> YOLO insan tespiti -> ByteTrack -> YuNet yüz tespiti -> yüz–kişi ilişkilendirme -> görselleştirme döngüsü.

Akış (HER KARE İÇİN; her model kare başına TAM BİR KEZ çalışır)::

    FrameSource.frames() -> FramePacket
        -> PersonDetector.detect   (YOLO11n)            -> DetectionResult
        -> PersonTracker.update    (ByteTrack adaptörü) -> TrackingResult        # YOLO'yu İKİNCİ kez çağırmaz
        -> FaceDetector.detect     (YuNet)              -> FaceDetectionResult   # aynı karenin aynı pikselleri
        -> FacePersonAssociator.associate               -> FaceAssociationResult
        -> draw_face_associations -> pencere

Döngü kamera okuma, model çağrısı, takip algoritması, eşleştirme ve çizim kodunu BİLMEZ; yalnızca bu arayüzleri birleştirir. MacBook kamerası,
video dosyası (ve ileride IP/RTSP) aynı kodla çalışır. Kamera/pencere altyapısı (``FrameSource``, ``PreviewWindow``, ``FpsMeter``, ``REASON_*``,
çıkış tuşları) Aşama 3/4'ten yeniden kullanılır, yeniden yazılmamıştır.

Neden ``run_tracking(..., on_result=...)`` kancası burada KULLANILMADI (kaynak koddan doğrulandı)
-------------------------------------------------------------------------------------------------
``run_tracking`` kancasını ``(paket, tespit, takip)`` ile çağırır ve ``None`` bekler; yüz tespiti ve ilişkilendirme kancadan çalıştırılabilir
(``tests/test_face_association_pipeline.py`` bunu, bu döngüyle AYNI sonuçları verdiğini doğrulayarak gösterir) ancak ekrana çizim
``draw_person_tracks``'e sabit bağlıdır ve özet yüz/ilişkilendirme sürelerini taşımaz. Bu yüzden, 4B'nin ``run_detection`` -> ``run_tracking``
için yaptığı gibi, aynı yapıda kardeş bir döngü yazıldı; 4A/4B/5A döngüleri DEĞİŞMEMİŞTİR.

Ölçümlerin tanımı (``tracking.pipeline`` ile aynı; birbirine karıştırılmamalıdır)
---------------------------------------------------------------------------------
``read``: kare okuma. ``yolo``: ``DetectionResult.inference_ms``. ``tracking``: ``TrackingResult.tracking_ms``. ``face``:
``FaceDetectionResult.inference_ms``. ``association``: ``FaceAssociationResult.association_ms``. ``compute``: aynı karedeki dört sürenin toplamı
(okuma/çizim/pencere hariç). ``frame_to_result``: ``FramePacket.timestamp`` ile ilişkilendirmenin bittiği an arası. ``frame_to_display``: aynı
başlangıçtan ``window.show`` dönüşüne kadar (video+pencere modunda ``None``). ``end_to_end_fps``: ``(N-1)/süre``; kaynağın bildirdiği FPS değildir.
YuNet CPU'da, YOLO ``device``'ta, ByteTrack ve ilişkilendirme CPU'dadır; ``ultralytics`` aynı süreçteyse OpenCV iş parçacığı etkisi için
``docs/face_association.md``.

Oturum: döngü başında ``tracker.reset()`` ve ``associator.reset()`` çağrılır; önceki kaynaktan/oturumdan iz ve süreklilik geçmişi taşınmaz.
Zamanlar ``time.monotonic`` ile ölçülür; ``clock``, ``FrameSource``'un damga için kullandığı saatle AYNI olmalıdır.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from isg_vision.camera.capture import FrameSource
from isg_vision.camera.display import is_exit_key
from isg_vision.camera.fps import FpsMeter
from isg_vision.camera.frame import SOURCE_KIND_CAMERA, SOURCE_KIND_VIDEO, FramePacket
from isg_vision.camera.preview import (
    DEFAULT_VIDEO_FPS,
    REASON_END_OF_STREAM,
    REASON_INTERRUPTED,
    REASON_MAX_FRAMES,
    REASON_USER_QUIT,
    REASON_WINDOW_CLOSED,
)
from isg_vision.face_detection.results import FaceDetectionResult
from isg_vision.human_detection.pipeline import RECENT_INFERENCE_WINDOW, LatencyStats
from isg_vision.human_detection.results import DetectionResult
from isg_vision.tracking.results import TrackingResult

from .results import FaceAssociationResult
from .visualization import AssociationOverlayInfo, draw_face_associations

__all__ = [
    "REASON_END_OF_STREAM",
    "REASON_INTERRUPTED",
    "REASON_MAX_FRAMES",
    "REASON_USER_QUIT",
    "REASON_WINDOW_CLOSED",
    "FaceAssociationRunSummary",
    "LatencyStats",
    "TrackAssociationSpan",
    "run_face_association",
]

MAX_TRACK_SPANS = 5000  # çok uzun oturumlarda bellek büyümesini sınırlar (aşılırsa ``spans_truncated``)


class DetectorLike(Protocol):
    def detect(self, image: Any, *, frame_index: int | None = None, timestamp: float | None = None) -> DetectionResult: ...


class TrackerLike(Protocol):
    def update(self, result: DetectionResult, *, source: str | None = None) -> TrackingResult: ...

    def reset(self) -> None: ...


class FaceDetectorLike(Protocol):
    def detect(self, image: Any, *, frame_index: int | None = None, timestamp: float | None = None) -> FaceDetectionResult: ...


class AssociatorLike(Protocol):
    def associate(self, tracking: TrackingResult, faces: FaceDetectionResult) -> FaceAssociationResult: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class TrackAssociationSpan:
    """Bir izin oturum boyunca yüz ilişkilendirme özeti.

    Attributes:
        session_id / track_id: Anahtar ``(session_id, track_id)`` (kimlik DEĞİL).
        first_frame / last_frame: İzin ilk ve son aktif olduğu kare.
        frames_active: Aktif kare sayısı. ``frames_matched``: yüz eşleşmesi alınan kare. ``frames_trusted_matched``: eşleşme + güvenilir süreklilik.
        epochs: İzin geçtiği süreklilik dönemi sayısı (1'den büyükse süreklilik en az bir kez bozuldu: ID switch/devralma şüphesi, uzun kayıp, örtüşme sonrası).
    """

    session_id: int
    track_id: int
    first_frame: int
    last_frame: int
    frames_active: int
    frames_matched: int
    frames_trusted_matched: int
    epochs: int


@dataclass(frozen=True)
class FaceAssociationRunSummary:
    """Bir yüz–kişi ilişkilendirme oturumunun özeti.

    Sayılar birbirine KARIŞTIRILMAMALIDIR: ``persons_total``/``faces_total`` kare başına sayıların toplamıdır (benzersiz kişi/yüz sayısı DEĞİL);
    ``distinct_tracks`` görülen farklı ``(session_id, track_id)``'dir (gerçek kişi sayısından büyükse izler parçalanıyor olabilir).

    Attributes:
        reason: Bitiş nedeni (``REASON_*``). frames_processed / elapsed_s / end_to_end_fps: bkz. modül açıklaması.
        persons_total / persons_max, faces_total / faces_max: Kare başına toplam ve en yüksek sayılar.
        matched_total / trusted_matched_total: Eşleşen çift toplamı; ikincisi süreklilik bakımından da güvenilir olanlar.
        ambiguous_persons_total / ambiguous_faces_total / no_face_total / unmatched_faces_total: Durum toplamları (kare başına toplamların toplamı).
        frames_with_match: En az bir eşleşme alınan kare. invalid_frames: Girdi tutarsızlığı nedeniyle hiç eşleştirme yapılmayan kare.
        break_counts: Süreklilik kesintisi sayıları (``BREAK_*`` -> sayı). possibly_static_tracks: Hareketsiz bayrağı alan farklı iz sayısı.
        distinct_tracks / reset_events: Farklı iz sayısı; takipçinin otomatik sıfırlandığı kare sayısı.
        dropped_person_boxes / dropped_face_boxes / truncated_faces: Geçersiz olduğu için atılan kutular / ``max_faces`` ile kesilen yüzler.
        tracks / spans_truncated: İz bazlı özet (en çok ``MAX_TRACK_SPANS``).
        read / yolo / tracking / face / association / compute / frame_to_result / frame_to_display / draw / show: ``LatencyStats`` (örnek yoksa ``None``).
    """

    reason: str
    frames_processed: int
    elapsed_s: float
    end_to_end_fps: float
    persons_total: int
    persons_max: int
    faces_total: int
    faces_max: int
    matched_total: int
    trusted_matched_total: int
    ambiguous_persons_total: int
    ambiguous_faces_total: int
    no_face_total: int
    unmatched_faces_total: int
    frames_with_match: int
    invalid_frames: int
    break_counts: dict[str, int]
    possibly_static_tracks: int
    distinct_tracks: int
    reset_events: int
    dropped_person_boxes: int
    dropped_face_boxes: int
    truncated_faces: int
    tracks: tuple[TrackAssociationSpan, ...]
    spans_truncated: bool
    read: LatencyStats | None
    yolo: LatencyStats | None
    tracking: LatencyStats | None
    face: LatencyStats | None
    association: LatencyStats | None
    compute: LatencyStats | None
    frame_to_result: LatencyStats | None
    frame_to_display: LatencyStats | None
    draw: LatencyStats | None
    show: LatencyStats | None

    @property
    def persons_mean(self) -> float:
        return self.persons_total / self.frames_processed if self.frames_processed else 0.0

    @property
    def faces_mean(self) -> float:
        return self.faces_total / self.frames_processed if self.frames_processed else 0.0


class _SpanBook:
    """İz bazlı eşleşme/epoch özetini biriktirir."""

    def __init__(self) -> None:
        self._data: dict[tuple[int, int], list[int]] = {}
        self.truncated = False

    def add(self, result: FaceAssociationResult) -> None:
        for person in result.persons:
            entry = self._data.get(person.key)
            if entry is None:
                if len(self._data) >= MAX_TRACK_SPANS:
                    self.truncated = True
                    continue
                entry = self._data[person.key] = [result.frame_index or 0, result.frame_index or 0, 0, 0, 0, 1]
            entry[1] = result.frame_index or entry[1]
            entry[2] += 1
            entry[3] += 1 if person.is_matched else 0
            entry[4] += 1 if person.trusted_match else 0
            if person.continuity is not None:
                entry[5] = max(entry[5], person.continuity.epoch)

    def spans(self) -> tuple[TrackAssociationSpan, ...]:
        out = [TrackAssociationSpan(s, t, *values) for (s, t), values in self._data.items()]
        out.sort(key=lambda sp: (sp.session_id, sp.first_frame, sp.track_id))
        return tuple(out)


def run_face_association(
    source: FrameSource,
    detector: DetectorLike,
    tracker: TrackerLike,
    face_detector: FaceDetectorLike,
    associator: AssociatorLike,
    window: Any | None = None,
    *,
    max_frames: int | None = None,
    meter: FpsMeter | None = None,
    clock: Callable[[], float] = time.monotonic,
    on_result: Callable[[FramePacket, DetectionResult, TrackingResult, FaceDetectionResult, FaceAssociationResult], None] | None = None,
) -> FaceAssociationRunSummary:
    """Açık bir kaynaktan kare okuyup her karede YOLO + ByteTrack + YuNet + ilişkilendirme yapar; ``window`` verilmişse çizip gösterir.

    Kaynağın, pencerenin, modellerin ve takipçinin açılıp kapatılması çağıranın sorumluluğundadır. Q/Esc, pencerenin kapatılması, video sonu,
    ``max_frames`` veya ``Ctrl+C`` ile biter. Kaynak/görüntü/tespit/takip/yüz hataları çağırana yayılır.

    Args:
        on_result: Her kare için ``(paket, tespit, takip, yüz sonucu, ilişkilendirme)`` ile çağrılır (ör. kayıt veya Aşama 6/7 için).
    """
    if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1):
        raise ValueError(f"max_frames pozitif bir tam sayı olmalıdır: {max_frames!r}")

    tracker.reset()  # yeni oturum: önceki kaynaktan/oturumdan iz taşınmaz
    associator.reset()  # ... ve süreklilik geçmişi de
    meter = meter or FpsMeter(clock=clock)
    interval: float | None = None
    if window is not None and source.kind == SOURCE_KIND_VIDEO:
        interval = 1.0 / (source.source_fps or DEFAULT_VIDEO_FPS)
    device = getattr(detector, "device", None)

    keys = ("read", "yolo", "tracking", "face", "association", "compute", "to_result", "to_display", "draw", "show")
    samples: dict[str, list[float]] = {k: [] for k in keys}
    recent: dict[str, deque[float]] = {k: deque(maxlen=RECENT_INFERENCE_WINDOW) for k in ("yolo", "tracking", "face", "association")}
    counts: Counter[str] = Counter()
    breaks: Counter[str] = Counter()
    book = _SpanBook()
    seen_keys: set[tuple[int, int]] = set()
    static_keys: set[tuple[int, int]] = set()
    reason = REASON_END_OF_STREAM
    next_deadline: float | None = None
    persons_max = faces_max = 0
    frames = source.frames()

    try:
        while True:
            before_read = clock()
            try:
                packet = next(frames)
            except StopIteration:
                reason = REASON_END_OF_STREAM
                break
            after_read = clock()
            samples["read"].append((after_read - before_read) * 1000.0)
            meter.tick(after_read)

            detection = detector.detect(packet.frame, frame_index=packet.frame_index, timestamp=packet.timestamp)  # YOLO: kare başına TEK çağrı
            tracking = tracker.update(detection, source=packet.source)  # takipçi YOLO'yu çağırmaz
            faces = face_detector.detect(packet.frame, frame_index=packet.frame_index, timestamp=packet.timestamp)  # YuNet: kare başına TEK çağrı
            association = associator.associate(tracking, faces)
            after_assoc = clock()

            for key, value in (("yolo", detection.inference_ms), ("tracking", tracking.tracking_ms), ("face", faces.inference_ms),
                               ("association", association.association_ms)):
                samples[key].append(value)
                recent[key].append(value)
            samples["compute"].append(detection.inference_ms + tracking.tracking_ms + faces.inference_ms + association.association_ms)
            samples["to_result"].append(max(0.0, (after_assoc - packet.timestamp) * 1000.0))

            counts["persons"] += association.person_count
            counts["faces"] += association.face_count
            persons_max = max(persons_max, association.person_count)
            faces_max = max(faces_max, association.face_count)
            counts["matched"] += association.matched_count
            counts["trusted_matched"] += association.trusted_match_count
            counts["ambiguous_persons"] += association.ambiguous_person_count
            counts["ambiguous_faces"] += association.ambiguous_face_count
            counts["no_face"] += association.no_face_count
            counts["unmatched_faces"] += association.unmatched_face_count
            counts["frames_with_match"] += 1 if association.matched_count else 0
            counts["invalid_frames"] += 0 if association.input_valid else 1
            counts["resets"] += 1 if tracking.reset_reason else 0
            counts["dropped_person_boxes"] += detection.dropped_boxes + tracking.dropped_tracks
            counts["dropped_face_boxes"] += faces.dropped_boxes
            counts["truncated_faces"] += faces.truncated_faces
            for _, why in association.breaks:
                breaks[why] += 1
            for person in association.persons:
                seen_keys.add(person.key)
                if person.continuity is not None and person.continuity.possibly_static:
                    static_keys.add(person.key)
            book.add(association)
            if on_result is not None:
                on_result(packet, detection, tracking, faces, association)

            if window is not None:
                info = _overlay_info(source, packet, meter.fps, {k: sum(v) / len(v) for k, v in recent.items()}, device)
                before_draw = clock()
                image = draw_face_associations(packet.frame, association, info)
                after_draw = clock()
                samples["draw"].append((after_draw - before_draw) * 1000.0)
                delay_ms = 1
                if interval is not None:
                    now = clock()
                    next_deadline = now + interval if next_deadline is None else max(next_deadline + interval, now)
                    delay_ms = max(1, int((next_deadline - clock()) * 1000))
                key = window.show(image, delay_ms)
                after_show = clock()
                if interval is None:
                    samples["show"].append((after_show - after_draw) * 1000.0)
                    samples["to_display"].append(max(0.0, (after_show - packet.timestamp) * 1000.0))
                counts["processed"] += 1
                if is_exit_key(key):
                    reason = REASON_USER_QUIT
                    break
                if window.closed_by_user:
                    reason = REASON_WINDOW_CLOSED
                    break
            else:
                counts["processed"] += 1
            if max_frames is not None and counts["processed"] >= max_frames:
                reason = REASON_MAX_FRAMES
                break
    except KeyboardInterrupt:
        reason = REASON_INTERRUPTED

    return FaceAssociationRunSummary(
        reason=reason,
        frames_processed=counts["processed"],
        elapsed_s=meter.elapsed,
        end_to_end_fps=meter.average_fps,
        persons_total=counts["persons"],
        persons_max=persons_max,
        faces_total=counts["faces"],
        faces_max=faces_max,
        matched_total=counts["matched"],
        trusted_matched_total=counts["trusted_matched"],
        ambiguous_persons_total=counts["ambiguous_persons"],
        ambiguous_faces_total=counts["ambiguous_faces"],
        no_face_total=counts["no_face"],
        unmatched_faces_total=counts["unmatched_faces"],
        frames_with_match=counts["frames_with_match"],
        invalid_frames=counts["invalid_frames"],
        break_counts=dict(breaks),
        possibly_static_tracks=len(static_keys),
        distinct_tracks=len(seen_keys),
        reset_events=counts["resets"],
        dropped_person_boxes=counts["dropped_person_boxes"],
        dropped_face_boxes=counts["dropped_face_boxes"],
        truncated_faces=counts["truncated_faces"],
        tracks=book.spans(),
        spans_truncated=book.truncated,
        read=LatencyStats.from_values(samples["read"]),
        yolo=LatencyStats.from_values(samples["yolo"]),
        tracking=LatencyStats.from_values(samples["tracking"]),
        face=LatencyStats.from_values(samples["face"]),
        association=LatencyStats.from_values(samples["association"]),
        compute=LatencyStats.from_values(samples["compute"]),
        frame_to_result=LatencyStats.from_values(samples["to_result"]),
        frame_to_display=LatencyStats.from_values(samples["to_display"]),
        draw=LatencyStats.from_values(samples["draw"]),
        show=LatencyStats.from_values(samples["show"]),
    )


def _overlay_info(source: FrameSource, packet: FramePacket, fps: float, recent_ms: dict[str, float], device: str | None) -> AssociationOverlayInfo:
    if source.kind == SOURCE_KIND_CAMERA:
        source_text = f"CAMERA {source.config.camera_index}"
    else:
        source_text = f"VIDEO {Path(source.config.video_path).name}"  # type: ignore[arg-type]
    note = None
    mismatch = source.resolution_mismatch
    if mismatch is not None:
        (rw, rh), (aw, ah) = mismatch
        note = f"requested {rw}x{rh}, got {aw}x{ah}"
    return AssociationOverlayInfo(
        source_text=source_text,
        fps=fps,
        yolo_ms=recent_ms["yolo"],
        tracking_ms=recent_ms["tracking"],
        face_ms=recent_ms["face"],
        association_ms=recent_ms["association"],
        resolution=(packet.width, packet.height),
        device=device,
        note=note,
    )
