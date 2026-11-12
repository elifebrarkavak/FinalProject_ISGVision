"""Kare akışı -> YOLO insan tespiti -> ByteTrack takibi -> görselleştirme döngüsü ve performans ölçümü.

Akış: ``FrameSource.frames()`` -> ``FramePacket`` -> ``PersonDetector.detect`` (YOLO, kare başına TAM BİR KEZ) -> ``DetectionResult`` ->
``PersonTracker.update`` -> ``TrackingResult`` -> ``draw_person_tracks`` -> pencere. Döngü kamera okuma, model çağrısı, takip algoritması
ve çizim kodunu BİLMEZ; yalnızca bu arayüzleri birleştirir. MacBook kamerası, video dosyası (ve ileride IP/RTSP kaynağı) aynı kodla çalışır.

Neden ``run_detection(..., on_result=...)`` kancası KULLANILMADI (kaynak koddan doğrulandı)
-----------------------------------------------------------------------------------------
Kanca ``(paket, sonuç)`` ile çağrılır ve ``None`` döndürür; çizim ise ``draw_person_detections``'a sabit bağlıdır ve özet takip süresini
taşımaz. Yani takip, kancayla çalıştırılabilirdi ama sonucu ekrana çizdiremez ve tracking süresini ölçüme katamazdı. 4A kodunu
değiştirmeden bunu sağlamak için aynı yapıdaki kardeş döngü burada yazıldı (aynı ``FpsMeter``, ``LatencyStats``, ``REASON_*``, çıkış
tuşu ve pencere mantığı yeniden kullanılır). 4A betiği ve ``run_detection`` değişmemiştir.

Ölçümlerin tanımı (birbirine karıştırılmamalıdır)
-------------------------------------------------
* ``read``: kare okuma (``FrameSource``). ``inference``: YOLO çıkarımı (``DetectionResult.inference_ms``, MPS'te senkronize).
  ``tracking``: ``PersonTracker.update`` (dönüştürme + ByteTrack + sonuç oluşturma; ``TrackingResult.tracking_ms``).
  ``detect_plus_track``: aynı karedeki ``inference + tracking`` toplamı (çizim/okuma/pencere hariç).
* ``frame_to_result``: ``FramePacket.timestamp`` (okumanın tamamlandığı an) ile takip sonucunun hazır olduğu an arası (ms).
  ``frame_to_display``: aynı başlangıçtan ``window.show`` dönüşüne kadar. Kapsam dışı: sensör pozlaması, sürücü/OS tamponlaması.
  Video + pencere modunda gösterme süresi bilinçli oynatma beklemesini içerdiği için ``None`` bırakılır (``draw`` ayrıca ölçülür).
* ``draw``: yalnızca çizim (``draw_person_tracks``) süresi; ``show``: ``window.show`` (imshow + waitKey) süresi.
* ``end_to_end_fps``: döngünün gerçekte işlediği kare hızı, ilk ve son kare arası ``(N-1)/süre``. Kaynağın bildirdiği FPS değildir.

Takip oturumu: döngü başında ``tracker.reset()`` çağrılır; böylece aynı takipçi nesnesi farklı bir kaynakta/oturumda yeniden kullanılsa
bile önceki izler taşınmaz. Her karede ``tracker.update`` tam bir kez (tespitsiz kareler dahil) çağrılır.

Zamanlar ``time.monotonic`` ile ölçülür; ``clock``, ``FrameSource``'un damga için kullandığı saatle AYNI olmalıdır.
"""

from __future__ import annotations

import time
from collections import deque
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
from isg_vision.human_detection.pipeline import RECENT_INFERENCE_WINDOW, LatencyStats
from isg_vision.human_detection.results import DetectionResult

from .results import TrackingResult
from .visualization import TrackingOverlayInfo, draw_person_tracks

__all__ = [
    "REASON_END_OF_STREAM",
    "REASON_INTERRUPTED",
    "REASON_MAX_FRAMES",
    "REASON_USER_QUIT",
    "REASON_WINDOW_CLOSED",
    "LatencyStats",
    "TrackSpan",
    "TrackingRunSummary",
    "run_tracking",
]

MAX_TRACK_SPANS = 5000  # çok uzun oturumlarda bellek büyümesini sınırlar (aşılırsa ``spans_truncated``)


class DetectorLike(Protocol):
    def detect(self, image: Any, *, frame_index: int | None = None, timestamp: float | None = None) -> DetectionResult: ...


class TrackerLike(Protocol):
    def update(self, result: DetectionResult, *, source: str | None = None) -> TrackingResult: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class TrackSpan:
    """Bir izin oturum boyunca görüntülendiği aralığın özeti (yanlış pozitif / ID kesintisi incelemesi için).

    Attributes:
        session_id / track_id: Benzersiz anahtar ``(session_id, track_id)``.
        first_frame / last_frame: İzin ilk ve son GÖRÜNTÜLENDİĞİ kare numarası.
        frames_active: İzin kaç karede aktif (görüntülenen) olduğu. ``last_frame - first_frame + 1``'den küçükse arada kayıp kareler vardır.
        duration_s: İlk ve son aktif kare arası süre (``FramePacket.timestamp`` farkı; biliniyorsa), aksi halde ``None``.
        confidence_mean / confidence_max: Aktif karelerdeki YOLO güven skorları (takip doğruluğu DEĞİLDİR).
    """

    session_id: int
    track_id: int
    first_frame: int
    last_frame: int
    frames_active: int
    duration_s: float | None
    confidence_mean: float
    confidence_max: float


@dataclass(frozen=True)
class TrackingRunSummary:
    """Bir takip oturumunun özeti.

    Attributes:
        reason: Bitiş nedeni (``REASON_*``).
        frames_processed: İşlenen kare sayısı.
        elapsed_s: İlk ve son kare arası süre (sn).
        end_to_end_fps: Uçtan uca ortalama FPS (bkz. modül açıklaması).
        detections_total / detections_max: Tüm karelerdeki toplam YOLO tespiti ve tek karedeki en yüksek sayı.
        active_max: Tek karedeki en yüksek aktif iz sayısı. ``frames_with_tracks``: en az bir aktif iz olan kare sayısı.
        lost_max: Tek karedeki en yüksek kayıp (tamponda bekleyen) iz sayısı.
        distinct_tracks: Görülen farklı ``(session_id, track_id)`` sayısı (gerçek kişi sayısından çok büyükse izler parçalanıyor olabilir).
        dropped_boxes / dropped_tracks: Geçersiz olduğu için atılan tespit kutusu / iz sayısı.
        reset_events: Döngü içinde takipçinin otomatik sıfırlandığı kare sayısı (normal bir oturumda 0).
        tracks: İz özetleri (ilk görüldükleri sıraya göre; en çok ``MAX_TRACK_SPANS``); ``spans_truncated`` kesildiyse ``True``.
        read / inference / tracking / detect_plus_track / frame_to_result / frame_to_display / draw / show: ``LatencyStats`` (örnek yoksa ``None``).
    """

    reason: str
    frames_processed: int
    elapsed_s: float
    end_to_end_fps: float
    detections_total: int
    detections_max: int
    active_max: int
    frames_with_tracks: int
    lost_max: int
    distinct_tracks: int
    dropped_boxes: int
    dropped_tracks: int
    reset_events: int
    tracks: tuple[TrackSpan, ...]
    spans_truncated: bool
    read: LatencyStats | None
    inference: LatencyStats | None
    tracking: LatencyStats | None
    detect_plus_track: LatencyStats | None
    frame_to_result: LatencyStats | None
    frame_to_display: LatencyStats | None
    draw: LatencyStats | None
    show: LatencyStats | None

    @property
    def detections_mean(self) -> float:
        return self.detections_total / self.frames_processed if self.frames_processed else 0.0


class _SpanBook:
    """Aktif izlerin ilk/son kare, sayı ve güven özetini biriktirir."""

    def __init__(self) -> None:
        self._data: dict[tuple[int, int], list[Any]] = {}
        self.truncated = False

    def add(self, result: TrackingResult) -> None:
        for track in result.tracks:
            entry = self._data.get(track.key)
            if entry is None:
                if len(self._data) >= MAX_TRACK_SPANS:
                    self.truncated = True
                    continue
                self._data[track.key] = [track.frame_index, track.frame_index, 1, track.confidence, track.confidence, track.timestamp, track.timestamp]
            else:
                entry[1] = track.frame_index
                entry[2] += 1
                entry[3] += track.confidence
                entry[4] = max(entry[4], track.confidence)
                entry[6] = track.timestamp

    def spans(self) -> tuple[TrackSpan, ...]:
        out = []
        for (session, track_id), (first, last, n, conf_sum, conf_max, t0, t1) in self._data.items():
            duration = (t1 - t0) if (t0 is not None and t1 is not None) else None
            out.append(TrackSpan(session, track_id, first, last, n, duration, conf_sum / n, conf_max))
        out.sort(key=lambda s: (s.session_id, s.first_frame, s.track_id))
        return tuple(out)


def run_tracking(
    source: FrameSource,
    detector: DetectorLike,
    tracker: TrackerLike,
    window: Any | None = None,
    *,
    max_frames: int | None = None,
    meter: FpsMeter | None = None,
    clock: Callable[[], float] = time.monotonic,
    on_result: Callable[[FramePacket, DetectionResult, TrackingResult], None] | None = None,
) -> TrackingRunSummary:
    """Açık bir kaynaktan kare okuyup her karede (1) YOLO ile insan tespiti, (2) ByteTrack ile takip yapar; ``window`` verilmişse çizip gösterir.

    Kaynağın, pencerenin, modelin ve takipçinin açılıp kapatılması çağıranın sorumluluğundadır (her koşulda serbest bırakılabilsin diye).
    Q/Esc, pencerenin kapatılması, video sonu, ``max_frames`` veya ``Ctrl+C`` ile biter. Kaynak/görüntü/tespit/takip hataları
    (``SourceError``, ``DisplayError``, ``DetectionError``, ``TrackingError``) çağırana yayılır.

    Args:
        on_result: Her kare için ``(paket, tespit sonucu, takip sonucu)`` ile çağrılır (ör. Aşama 5 yüz-kişi ilişkilendirmesi veya kayıt için).
    """
    if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1):
        raise ValueError(f"max_frames pozitif bir tam sayı olmalıdır: {max_frames!r}")

    tracker.reset()  # yeni oturum: önceki kaynaktan/oturumdan iz taşınmaz
    meter = meter or FpsMeter(clock=clock)
    interval: float | None = None
    if window is not None and source.kind == SOURCE_KIND_VIDEO:
        interval = 1.0 / (source.source_fps or DEFAULT_VIDEO_FPS)
    device = getattr(detector, "device", None)

    samples: dict[str, list[float]] = {k: [] for k in ("read", "inference", "tracking", "both", "to_result", "to_display", "draw", "show")}
    recent_inf: deque[float] = deque(maxlen=RECENT_INFERENCE_WINDOW)
    recent_trk: deque[float] = deque(maxlen=RECENT_INFERENCE_WINDOW)
    book = _SpanBook()
    processed = det_total = det_max = active_max = frames_with_tracks = lost_max = dropped_boxes = dropped_tracks = resets = 0
    seen_keys: set[tuple[int, int]] = set()
    reason = REASON_END_OF_STREAM
    next_deadline: float | None = None
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
            after_track = clock()

            samples["inference"].append(detection.inference_ms)
            samples["tracking"].append(tracking.tracking_ms)
            samples["both"].append(detection.inference_ms + tracking.tracking_ms)
            samples["to_result"].append(max(0.0, (after_track - packet.timestamp) * 1000.0))
            recent_inf.append(detection.inference_ms)
            recent_trk.append(tracking.tracking_ms)
            det_total += detection.person_count
            det_max = max(det_max, detection.person_count)
            active_max = max(active_max, tracking.active_count)
            frames_with_tracks += 1 if tracking.active_count else 0
            lost_max = max(lost_max, tracking.lost_count)
            dropped_boxes += detection.dropped_boxes
            dropped_tracks += tracking.dropped_tracks
            resets += 1 if tracking.reset_reason else 0
            seen_keys.update(t.key for t in tracking.tracks)
            book.add(tracking)
            if on_result is not None:
                on_result(packet, detection, tracking)

            if window is not None:
                info = _overlay_info(source, packet, meter.fps, sum(recent_inf) / len(recent_inf), sum(recent_trk) / len(recent_trk), device)
                before_draw = clock()
                image = draw_person_tracks(packet.frame, tracking, info)
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
                processed += 1
                if is_exit_key(key):
                    reason = REASON_USER_QUIT
                    break
                if window.closed_by_user:
                    reason = REASON_WINDOW_CLOSED
                    break
            else:
                processed += 1
            if max_frames is not None and processed >= max_frames:
                reason = REASON_MAX_FRAMES
                break
    except KeyboardInterrupt:
        reason = REASON_INTERRUPTED

    return TrackingRunSummary(
        reason=reason,
        frames_processed=processed,
        elapsed_s=meter.elapsed,
        end_to_end_fps=meter.average_fps,
        detections_total=det_total,
        detections_max=det_max,
        active_max=active_max,
        frames_with_tracks=frames_with_tracks,
        lost_max=lost_max,
        distinct_tracks=len(seen_keys),
        dropped_boxes=dropped_boxes,
        dropped_tracks=dropped_tracks,
        reset_events=resets,
        tracks=book.spans(),
        spans_truncated=book.truncated,
        read=LatencyStats.from_values(samples["read"]),
        inference=LatencyStats.from_values(samples["inference"]),
        tracking=LatencyStats.from_values(samples["tracking"]),
        detect_plus_track=LatencyStats.from_values(samples["both"]),
        frame_to_result=LatencyStats.from_values(samples["to_result"]),
        frame_to_display=LatencyStats.from_values(samples["to_display"]),
        draw=LatencyStats.from_values(samples["draw"]),
        show=LatencyStats.from_values(samples["show"]),
    )


def _overlay_info(
    source: FrameSource, packet: FramePacket, fps: float, inference_ms: float, tracking_ms: float, device: str | None
) -> TrackingOverlayInfo:
    if source.kind == SOURCE_KIND_CAMERA:
        source_text = f"CAMERA {source.config.camera_index}"
    else:
        source_text = f"VIDEO {Path(source.config.video_path).name}"  # type: ignore[arg-type]
    note = None
    mismatch = source.resolution_mismatch
    if mismatch is not None:
        (rw, rh), (aw, ah) = mismatch
        note = f"requested {rw}x{rh}, got {aw}x{ah}"
    return TrackingOverlayInfo(
        source_text=source_text,
        fps=fps,
        inference_ms=inference_ms,
        tracking_ms=tracking_ms,
        resolution=(packet.width, packet.height),
        device=device,
        note=note,
    )
