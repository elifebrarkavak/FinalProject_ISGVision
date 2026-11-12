"""Aktif izleri (kutu + ``ID n``) ve durum bilgisini görüntüye çizen katman.

* Çizim **girdi görüntüsünün kopyası** üzerinde yapılır; girdi değiştirilmez.
* **Yalnızca aktif izler çizilir** (bu karede eşleşmiş ve onaylanmış). Kaybolmuş/tamponda bekleyen ve henüz onaylanmamış izler
  ÇİZİLMEZ: görünmeyen birini görünüyormuş gibi göstermek yanıltıcıdır. Sayılar ayrı gösterilir:
  ``DETECTIONS`` (YOLO kutuları) ≠ ``ACTIVE TRACKS`` (çizilen izler) ≠ ``LOST TRACKS`` (tamponda bekleyen, görünmez).
* **Renk yalnızca görsel ayrım içindir**: aynı ``track_id`` aynı rengi alır, ancak renk bir kimlik veya doğruluk anlamına gelmez
  (12 renklik palet; ``track_id`` 13 ve üzeri renkleri tekrar eder; iki farklı kişi aynı renkte olabilir).
* ``ID n`` geçici bir takip numarasıdır, çalışan kimliği DEĞİLDİR (alt şeritte de belirtilir).
* Metinler yalnızca ASCII'dir (OpenCV yazı tipleri ``ç ş ğ ı ö ü`` çizemez); ``camera.display.ascii_text`` kullanılır.
* Pencere yönetimi burada DEĞİLDİR; pencere ve macOS kapatma düzeltmesi ``camera.display.PreviewWindow``'dadır.

Bu modül import edildiğinde hiçbir pencere açılmaz.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from isg_vision.camera.display import ascii_text

from .results import TrackedPerson, TrackingResult

WINDOW_NAME = "ISG Vision - Person Tracking"
EXIT_HINT = "Q / Esc / Close to exit"
ID_NOTE = "ID = temporary track number (not an identity)"

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_GREEN = (80, 255, 80)  # BGR
_AMBER = (0, 200, 255)
_GRAY = (190, 190, 190)

# BGR; parlak ve birbirinden ayrışan renkler (siyah etiket yazısı hepsinde okunur).
TRACK_PALETTE: tuple[tuple[int, int, int], ...] = (
    (80, 255, 80), (255, 200, 60), (60, 200, 255), (200, 120, 255), (60, 255, 255), (255, 120, 120),
    (120, 255, 200), (60, 140, 255), (255, 80, 200), (180, 255, 60), (255, 160, 60), (160, 160, 255),
)


def color_for_track(track_id: int) -> tuple[int, int, int]:
    """``track_id``'ye göre sabit (BGR) renk. Yalnızca görsel ayrım içindir; kimlik/doğruluk anlamı taşımaz."""
    return TRACK_PALETTE[(int(track_id) - 1) % len(TRACK_PALETTE)]


@dataclass(frozen=True)
class TrackingOverlayInfo:
    """Üst şeritte gösterilecek bilgiler (üç sayı ``TrackingResult``'tan gelir).

    Attributes:
        source_text: Örn. ``"CAMERA 0"`` veya ``"VIDEO klip.mp4"``.
        fps: Ölçülen uçtan uca FPS (kayan pencere ortalaması). Kaynağın bildirdiği FPS değildir.
        inference_ms: Gösterilecek YOLO çıkarım süresi (ms); ``None`` ise ``--`` yazılır.
        tracking_ms: Gösterilecek takip (ByteTrack) süresi (ms); ``None`` ise ``--`` yazılır.
        resolution: Gerçek kare boyutu ``(genişlik, yükseklik)``.
        device: YOLO çıkarım cihazı (``cpu``/``mps``); ByteTrack her zaman CPU'dadır.
        note: İsteğe bağlı uyarı satırı.
    """

    source_text: str
    fps: float
    inference_ms: float | None
    tracking_ms: float | None
    resolution: tuple[int, int] | None
    device: str | None = None
    note: str | None = None


def draw_person_tracks(image: np.ndarray, result: TrackingResult, info: TrackingOverlayInfo) -> np.ndarray:
    """Görüntünün **kopyasına** aktif izleri ve durum şeridini çizer; girdiyi değiştirmez."""
    out = image.copy()
    height, width = out.shape[:2]
    scale = max(0.35, min(1.0, min(width, height) / 720.0 * 0.6 + 0.1))
    thickness = 1 if scale < 0.8 else 2
    (_, text_h), baseline = cv2.getTextSize("Ag", _FONT, scale, thickness)
    pad = max(4, int(10 * scale))
    line_h = text_h + baseline + pad // 2

    resolution = f"{info.resolution[0]}x{info.resolution[1]}" if info.resolution else "N/A"
    inference = f"{info.inference_ms:.0f} ms" if info.inference_ms is not None else "-- ms"
    tracking = f"{info.tracking_ms:.1f} ms" if info.tracking_ms is not None else "-- ms"
    left = [
        f"DETECTIONS: {result.detection_count}",
        f"ACTIVE TRACKS: {result.active_count}",
        f"LOST TRACKS: {result.lost_count}",
        f"SOURCE: {ascii_text(info.source_text)}",
        f"RESOLUTION: {resolution}",
    ]
    right = [f"FPS: {info.fps:.1f}", f"INFERENCE: {inference}", f"TRACKING: {tracking}"]
    if info.device:
        right.append(f"DEVICE: {ascii_text(info.device).upper()}")
    note = f"NOTE: {ascii_text(info.note)}" if info.note else None

    rows = max(len(left) + (1 if note else 0), len(right))
    band_h = min(height, pad * 2 + rows * line_h)
    foot_h = min(height, text_h + baseline + pad * 2)
    out[:band_h] = cv2.convertScaleAbs(out[:band_h], alpha=0.4)  # okunabilirlik için şeritleri koyulaştır
    out[height - foot_h :] = cv2.convertScaleAbs(out[height - foot_h :], alpha=0.4)

    _draw_tracks(out, result.tracks, scale, band_h)

    y = pad + text_h
    for index, text in enumerate(left):
        color = _GREEN if index == 1 else (_GRAY if index == 2 else _WHITE)  # ACTIVE TRACKS vurgulu; LOST soluk (görünmeyen izler)
        cv2.putText(out, text, (pad, y), _FONT, scale, color, thickness, cv2.LINE_AA)
        y += line_h
    if note:
        cv2.putText(out, note, (pad, y), _FONT, scale, _AMBER, thickness, cv2.LINE_AA)

    y = pad + text_h
    for index, text in enumerate(right):
        (text_w, _), _ = cv2.getTextSize(text, _FONT, scale, thickness)
        cv2.putText(out, text, (max(pad, width - text_w - pad), y), _FONT, scale, _GREEN if index == 0 else _WHITE, thickness, cv2.LINE_AA)
        y += line_h

    foot_y = height - pad - baseline
    cv2.putText(out, EXIT_HINT, (pad, foot_y), _FONT, scale, _WHITE, thickness, cv2.LINE_AA)
    (hint_w, _), _ = cv2.getTextSize(EXIT_HINT, _FONT, scale, thickness)
    (note_w, _), _ = cv2.getTextSize(ID_NOTE, _FONT, scale * 0.8, 1)
    if pad * 3 + hint_w + note_w <= width:  # yer varsa kimlik olmadığı notu sağ altta
        cv2.putText(out, ID_NOTE, (width - note_w - pad, foot_y), _FONT, scale * 0.8, _GRAY, 1, cv2.LINE_AA)
    return out


def _draw_tracks(out: np.ndarray, tracks: tuple[TrackedPerson, ...], scale: float, band_h: int) -> None:
    height, width = out.shape[:2]
    box_thickness = max(2, int(round(min(width, height) / 360)))
    label_scale = max(0.35, scale * 0.8)
    label_thickness = 1 if label_scale < 0.7 else 2
    for track in tracks:
        color = color_for_track(track.track_id)
        x1, y1, x2, y2 = (int(round(v)) for v in track.bbox_xyxy)
        x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
        y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thickness, cv2.LINE_AA)

        label = f"ID {track.track_id} {track.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, _FONT, label_scale, label_thickness)
        box_h = text_h + baseline + 4
        top = y1 - box_h
        if top < band_h:  # üstte yer yoksa (kutu görüntünün tepesinde) etiketi kutunun içine al
            top = min(y1, height - box_h)
        left = max(0, min(x1, width - text_w - 6))
        cv2.rectangle(out, (left, top), (left + text_w + 6, top + box_h), color, cv2.FILLED)
        cv2.putText(out, label, (left + 3, top + text_h + 1), _FONT, label_scale, _BLACK, label_thickness, cv2.LINE_AA)
