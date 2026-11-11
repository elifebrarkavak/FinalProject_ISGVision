"""Kare akışını pencerede gösteren (veya penceresiz sayan) önizleme döngüsü.

Döngü yalnızca ``FrameSource`` ile ``PreviewWindow``'u birleştirir; hiçbir yapay zekâ modeli çalıştırmaz.
Kaynak ve pencerenin açılıp kapatılması çağıranın (``with`` blokları) sorumluluğundadır; böylece her koşulda
kaynak serbest bırakılır. Gelecekteki modüller aynı ``FrameSource.frames()`` akışını doğrudan kullanır.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .capture import FrameSource
from .display import OverlayInfo, PreviewWindow, draw_overlay, is_exit_key
from .fps import FpsMeter
from .frame import SOURCE_KIND_CAMERA, SOURCE_KIND_VIDEO, FramePacket

REASON_USER_QUIT = "user_quit"
REASON_WINDOW_CLOSED = "window_closed"
REASON_END_OF_STREAM = "end_of_stream"
REASON_MAX_FRAMES = "max_frames"
REASON_INTERRUPTED = "interrupted"

DEFAULT_VIDEO_FPS = 30.0  # video dosyası kare hızını bildirmiyorsa oynatma hızı


@dataclass(frozen=True)
class PreviewResult:
    """Önizleme oturumunun özeti.

    Attributes:
        reason: Döngünün bitiş nedeni (``REASON_*`` sabitlerinden biri).
        frames_processed: İşlenen (pencereli modda gösterilen) kare sayısı.
        elapsed_s: İlk ve son işlenen kare arasındaki süre (saniye).
        measured_fps: Oturum boyunca ölçülen ortalama FPS (kaynağın bildirdiği FPS değildir).
    """

    reason: str
    frames_processed: int
    elapsed_s: float
    measured_fps: float


def run_preview(
    source: FrameSource,
    window: PreviewWindow | None = None,
    *,
    max_frames: int | None = None,
    meter: FpsMeter | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> PreviewResult:
    """Açık bir kaynaktan kare okur; ``window`` verilmişse overlay ile gösterir.

    * Kamera: kare hızını kamera belirler (``waitKey(1)``).
    * Video + pencere: kaynağın bildirdiği FPS'e uygun hızda oynatılır (kare atlanmaz; işlem yavaşsa
      oynatma yavaşlar, ölçülen FPS bunu yansıtır).
    * Pencere yoksa (``--no-display``) bekleme yapılmaz; kareler olabildiğince hızlı okunur.

    Q/Esc tuşu, pencerenin kapatılması, video sonu, ``max_frames`` veya ``Ctrl+C`` ile biter.
    Kaynak hataları (``SourceError``, ``DisplayError``) çağırana yayılır.
    """
    if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1):
        raise ValueError(f"max_frames pozitif bir tam sayı olmalıdır: {max_frames!r}")

    meter = meter or FpsMeter(clock=clock)
    interval: float | None = None
    if window is not None and source.kind == SOURCE_KIND_VIDEO:
        interval = 1.0 / (source.source_fps or DEFAULT_VIDEO_FPS)

    reason = REASON_END_OF_STREAM
    processed = 0
    next_deadline: float | None = None
    try:
        for packet in source.frames():
            meter.tick()
            if window is not None:
                image = draw_overlay(packet.frame, _overlay_info(source, packet, meter.fps))
                delay_ms = 1
                if interval is not None:
                    now = clock()
                    next_deadline = now + interval if next_deadline is None else max(next_deadline + interval, now)
                    delay_ms = max(1, int((next_deadline - clock()) * 1000))
                key = window.show(image, delay_ms)
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

    return PreviewResult(reason, processed, meter.elapsed, meter.average_fps)


def _overlay_info(source: FrameSource, packet: FramePacket, fps: float) -> OverlayInfo:
    if source.kind == SOURCE_KIND_CAMERA:
        source_text = f"CAMERA {source.config.camera_index}"
        status = "LIVE"
    else:
        source_text = f"VIDEO {Path(source.config.video_path).name}"  # type: ignore[arg-type]
        total = source.frame_count
        status = f"PLAYING {packet.frame_index + 1}/{total}" if total else f"PLAYING {packet.frame_index + 1}"

    note = None
    mismatch = source.resolution_mismatch
    if mismatch is not None:
        (rw, rh), (aw, ah) = mismatch
        note = f"requested {rw}x{rh}, got {aw}x{ah}"
    return OverlayInfo(
        source_text=source_text,
        fps=fps,
        resolution=(packet.width, packet.height),
        source_fps=source.source_fps,
        status=status,
        note=note,
    )
