"""Kare akışı -> yüz tespiti -> görselleştirme döngüsü ve performans ölçümü.

Akış: ``FrameSource.frames()`` -> ``FramePacket`` -> ``FaceDetector.detect`` -> ``draw_face_detections`` -> pencere. Döngü kamera okuma,
model çağrısı ve çizim kodunu BİLMEZ; yalnızca bu üç arayüzü birleştirir. Kamera/video kaynağı ve pencere, Aşama 3/4'te yazılmış kodla
(``camera.capture.FrameSource``, ``camera.display.PreviewWindow``) aynıdır; burada yeniden yazılmamıştır.

Ölçümlerin tanımı ``human_detection.pipeline`` ile AYNIDIR (``LatencyStats`` da oradan yeniden kullanılır):

* **Çıkarım gecikmesi** (``inference``, ms): yalnızca ``FaceDetector.detect`` içindeki model çağrısı (``FaceDetectionResult.inference_ms``).
* **Uçtan uca FPS** (``end_to_end_fps``): döngünün gerçekte işlediği kare hızı (okuma + çıkarım + çizim + gösterme dâhil); ilk ve son
  kare arası ``(N-1)/süre``. Kaynağın bildirdiği FPS değildir.
* **Kare -> sonuç gecikmesi** (``frame_to_result``, ms): ``FramePacket.timestamp`` ile tespitin bitişi arası. **Kare -> görüntü**
  (``frame_to_display``): aynı başlangıçtan ``window.show`` dönüşüne kadar. Video + pencere modunda gösterme süresi, bilinçli oynatma
  beklemesini içerdiği için ``None`` bırakılır.

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

from .results import FaceDetectionResult
from .visualization import FaceOverlayInfo, draw_face_detections

__all__ = [
    "REASON_END_OF_STREAM",
    "REASON_INTERRUPTED",
    "REASON_MAX_FRAMES",
    "REASON_USER_QUIT",
    "REASON_WINDOW_CLOSED",
    "FaceRunSummary",
    "LatencyStats",
    "run_face_detection",
]


class FaceDetectorLike(Protocol):
    def detect(self, image: Any, *, frame_index: int | None = None, timestamp: float | None = None) -> FaceDetectionResult: ...


@dataclass(frozen=True)
class FaceRunSummary:
    """Bir yüz tespiti oturumunun özeti.

    Attributes:
        reason: Bitiş nedeni (``REASON_*``).
        frames_processed: İşlenen kare sayısı.
        elapsed_s: İlk ve son kare arası süre (sn).
        end_to_end_fps: Uçtan uca ortalama FPS (bkz. modül açıklaması).
        faces_total / faces_max: Tüm karelerdeki toplam yüz tespiti ve tek karedeki en yüksek yüz sayısı. ``faces_total``
            **benzersiz kişi sayısı DEĞİLDİR** (aynı yüz her karede yeniden sayılır).
        frames_with_faces: En az bir yüz bulunan kare sayısı.
        dropped_boxes / truncated_faces: Geçersiz olduğu için atılan kutular / ``max_faces`` aşıldığı için kesilen yüzler (toplam).
        inference / read / frame_to_result / frame_to_display: ``LatencyStats`` (örnek yoksa ``None``).
    """

    reason: str
    frames_processed: int
    elapsed_s: float
    end_to_end_fps: float
    faces_total: int
    faces_max: int
    frames_with_faces: int
    dropped_boxes: int
    truncated_faces: int
    inference: LatencyStats | None
    read: LatencyStats | None
    frame_to_result: LatencyStats | None
    frame_to_display: LatencyStats | None

    @property
    def faces_mean(self) -> float:
        return self.faces_total / self.frames_processed if self.frames_processed else 0.0


def run_face_detection(
    source: FrameSource,
    detector: FaceDetectorLike,
    window: Any | None = None,
    *,
    max_frames: int | None = None,
    meter: FpsMeter | None = None,
    clock: Callable[[], float] = time.monotonic,
    on_result: Callable[[FramePacket, FaceDetectionResult], None] | None = None,
) -> FaceRunSummary:
    """Açık bir kaynaktan kare okuyup her karede yüz tespiti yapar; ``window`` verilmişse sonucu çizip gösterir.

    Kaynağın ve pencerenin açılıp kapatılması ve modelin yüklenmesi çağıranın sorumluluğundadır (her koşulda serbest bırakılabilsin
    diye). Q/Esc, pencerenin kapatılması, video sonu, ``max_frames`` veya ``Ctrl+C`` ile biter. Kaynak/görüntü/tespit hataları
    (``SourceError``, ``DisplayError``, ``FaceDetectionError``) çağırana yayılır.

    Args:
        on_result: Her kare için ``(paket, sonuç)`` ile çağrılır (ör. Aşama 5B'nin kare başına ``TrackingResult`` ile aynı kareden
            ilişkilendirme yapması için kanca). Aşama 5A'da yüz–Track ID eşleştirmesi YOKTUR.
    """
    if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1):
        raise ValueError(f"max_frames pozitif bir tam sayı olmalıdır: {max_frames!r}")

    meter = meter or FpsMeter(clock=clock)
    interval: float | None = None
    if window is not None and source.kind == SOURCE_KIND_VIDEO:
        interval = 1.0 / (source.source_fps or DEFAULT_VIDEO_FPS)
    device = getattr(detector, "device", None)

    read_ms: list[float] = []
    inference_ms: list[float] = []
    to_result_ms: list[float] = []
    to_display_ms: list[float] = []
    recent: deque[float] = deque(maxlen=RECENT_INFERENCE_WINDOW)
    processed = faces_total = faces_max = frames_with_faces = dropped = truncated = 0
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
            read_ms.append((after_read - before_read) * 1000.0)
            meter.tick(after_read)

            result = detector.detect(packet.frame, frame_index=packet.frame_index, timestamp=packet.timestamp)
            after_detect = clock()
            inference_ms.append(result.inference_ms)
            recent.append(result.inference_ms)
            to_result_ms.append(max(0.0, (after_detect - packet.timestamp) * 1000.0))
            faces_total += result.face_count
            faces_max = max(faces_max, result.face_count)
            frames_with_faces += 1 if result.face_count else 0
            dropped += result.dropped_boxes
            truncated += result.truncated_faces
            if on_result is not None:
                on_result(packet, result)

            if window is not None:
                info = _overlay_info(source, packet, meter.fps, sum(recent) / len(recent), device)
                image = draw_face_detections(packet.frame, result, info)
                delay_ms = 1
                if interval is not None:
                    now = clock()
                    next_deadline = now + interval if next_deadline is None else max(next_deadline + interval, now)
                    delay_ms = max(1, int((next_deadline - clock()) * 1000))
                key = window.show(image, delay_ms)
                if interval is None:
                    to_display_ms.append(max(0.0, (clock() - packet.timestamp) * 1000.0))
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

    return FaceRunSummary(
        reason=reason,
        frames_processed=processed,
        elapsed_s=meter.elapsed,
        end_to_end_fps=meter.average_fps,
        faces_total=faces_total,
        faces_max=faces_max,
        frames_with_faces=frames_with_faces,
        dropped_boxes=dropped,
        truncated_faces=truncated,
        inference=LatencyStats.from_values(inference_ms),
        read=LatencyStats.from_values(read_ms),
        frame_to_result=LatencyStats.from_values(to_result_ms),
        frame_to_display=LatencyStats.from_values(to_display_ms),
    )


def _overlay_info(source: FrameSource, packet: FramePacket, fps: float, inference_ms: float, device: str | None) -> FaceOverlayInfo:
    if source.kind == SOURCE_KIND_CAMERA:
        source_text = f"CAMERA {source.config.camera_index}"
    else:
        source_text = f"VIDEO {Path(source.config.video_path).name}"  # type: ignore[arg-type]
    note = None
    mismatch = source.resolution_mismatch
    if mismatch is not None:
        (rw, rh), (aw, ah) = mismatch
        note = f"requested {rw}x{rh}, got {aw}x{ah}"
    return FaceOverlayInfo(
        source_text=source_text,
        fps=fps,
        inference_ms=inference_ms,
        resolution=(packet.width, packet.height),
        device=device,
        note=note,
    )
