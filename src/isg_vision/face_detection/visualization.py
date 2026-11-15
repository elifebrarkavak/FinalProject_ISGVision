"""Yüz kutularını ve durum bilgisini görüntüye çizen katman.

* Çizim **girdi görüntüsünün kopyası** üzerinde yapılır; girdi değiştirilmez (aynı kareyi kullanan diğer modüller etkilenmez).
* Tüm yüz kutuları aynı renkte çizilir (kimlik/takip yok; farklı renkler kimlik izlenimi verirdi) ve insan kutularının yeşilinden
  ayrı bir renktedir.
* Alt şeritte **her zaman** bir uyarı yazar: ``FACE BOX = detection only (not identity, not liveness)``. Yüz kutusu bir kişi
  kimliği veya canlı insan kanıtı değildir.
* Kadraj kenarına değen (kısmen görünüyor olabilecek) yüzlerin etiketine ``EDGE`` eklenir.
* Metinler yalnızca ASCII'dir (OpenCV yazı tipleri ``ç ş ğ ı ö ü`` çizemez); ``camera.display.ascii_text`` kullanılır.
* Pencere yönetimi burada DEĞİLDİR; pencere ve macOS kapatma düzeltmesi ``camera.display.PreviewWindow``'dadır.

Bu modül import edildiğinde hiçbir pencere açılmaz.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from isg_vision.camera.display import ascii_text

from .results import FaceDetectionResult

WINDOW_NAME = "ISG Vision - Face Detection"
EXIT_HINT = "Q / Esc / Close to exit"
NOTICE = "FACE BOX = detection only (not identity, not liveness)"

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_CYAN = (255, 220, 0)  # BGR: yüz kutuları (insan kutuları yeşildir)
_AMBER = (0, 200, 255)


@dataclass(frozen=True)
class FaceOverlayInfo:
    """Üst şeritte gösterilecek bilgiler (yüz sayısı ``FaceDetectionResult``'tan gelir).

    Attributes:
        source_text: Örn. ``"CAMERA 0"`` veya ``"VIDEO klip.mp4"``.
        fps: Ölçülen uçtan uca FPS (kayan pencere ortalaması). Kaynağın bildirdiği FPS değildir.
        inference_ms: Gösterilecek çıkarım süresi (ms); ``None`` ise ``--`` yazılır.
        resolution: Gerçek kare boyutu ``(genişlik, yükseklik)``.
        device: Çıkarımın yapıldığı yer (``cpu``); bilinmiyorsa ``None``.
        note: İsteğe bağlı uyarı satırı.
    """

    source_text: str
    fps: float
    inference_ms: float | None
    resolution: tuple[int, int] | None
    device: str | None = None
    note: str | None = None


def draw_face_detections(image: np.ndarray, result: FaceDetectionResult, info: FaceOverlayInfo) -> np.ndarray:
    """Görüntünün **kopyasına** yüz kutularını, güven skorlarını ve durum şeridini çizer; girdiyi değiştirmez."""
    out = image.copy()
    height, width = out.shape[:2]
    scale = max(0.35, min(1.0, min(width, height) / 720.0 * 0.6 + 0.1))
    thickness = 1 if scale < 0.8 else 2
    (_, text_h), baseline = cv2.getTextSize("Ag", _FONT, scale, thickness)
    pad = max(4, int(10 * scale))
    line_h = text_h + baseline + pad // 2

    resolution = f"{info.resolution[0]}x{info.resolution[1]}" if info.resolution else "N/A"
    inference = f"{info.inference_ms:.0f} ms" if info.inference_ms is not None else "-- ms"
    left = [f"FACES: {result.face_count}", f"SOURCE: {ascii_text(info.source_text)}", f"RESOLUTION: {resolution}"]
    right = [f"FPS: {info.fps:.1f}", f"INFERENCE: {inference}"]
    if info.device:
        right.append(f"DEVICE: {ascii_text(info.device).upper()}")
    note = f"NOTE: {ascii_text(info.note)}" if info.note else None

    rows = max(len(left) + (1 if note else 0), len(right))
    band_h = min(height, pad * 2 + rows * line_h)
    foot_h = min(height, pad * 2 + 2 * line_h - pad // 2)
    out[:band_h] = cv2.convertScaleAbs(out[:band_h], alpha=0.4)  # okunabilirlik için şeritleri koyulaştır
    out[height - foot_h :] = cv2.convertScaleAbs(out[height - foot_h :], alpha=0.4)

    _draw_boxes(out, result, scale, band_h)

    y = pad + text_h
    for index, text in enumerate(left):
        cv2.putText(out, text, (pad, y), _FONT, scale, _CYAN if index == 0 else _WHITE, thickness, cv2.LINE_AA)
        y += line_h
    if note:
        cv2.putText(out, note, (pad, y), _FONT, scale, _AMBER, thickness, cv2.LINE_AA)

    y = pad + text_h
    for index, text in enumerate(right):
        (text_w, _), _ = cv2.getTextSize(text, _FONT, scale, thickness)
        cv2.putText(out, text, (max(pad, width - text_w - pad), y), _FONT, scale, _CYAN if index == 0 else _WHITE, thickness, cv2.LINE_AA)
        y += line_h

    # Alt şerit: çıkış ipucu + her zaman görünen "kimlik/canlılık değildir" uyarısı (küçük görüntülerde yazı ölçeği kendiliğinden küçülür)
    notice_scale = scale
    (notice_w, _), _ = cv2.getTextSize(NOTICE, _FONT, notice_scale, thickness)
    while notice_w > width - 2 * pad and notice_scale > 0.25:
        notice_scale -= 0.05
        (notice_w, _), _ = cv2.getTextSize(NOTICE, _FONT, notice_scale, thickness)
    cv2.putText(out, EXIT_HINT, (pad, height - pad - line_h - baseline + pad // 2), _FONT, scale, _WHITE, thickness, cv2.LINE_AA)
    cv2.putText(out, NOTICE, (pad, height - pad - baseline), _FONT, notice_scale, _AMBER, thickness, cv2.LINE_AA)
    return out


def _draw_boxes(out: np.ndarray, result: FaceDetectionResult, scale: float, band_h: int) -> None:
    height, width = out.shape[:2]
    box_thickness = max(2, int(round(min(width, height) / 360)))
    label_scale = max(0.35, scale * 0.8)
    label_thickness = 1 if label_scale < 0.7 else 2
    for face in result.faces:
        x1, y1, x2, y2 = (int(round(v)) for v in face.bbox_xyxy)
        x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
        y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
        cv2.rectangle(out, (x1, y1), (x2, y2), _CYAN, box_thickness, cv2.LINE_AA)

        label = f"FACE {face.confidence:.2f}" + (" EDGE" if face.at_border else "")
        (text_w, text_h), baseline = cv2.getTextSize(label, _FONT, label_scale, label_thickness)
        box_h = text_h + baseline + 4
        top = y1 - box_h
        if top < band_h:  # üstte yer yoksa (kutu görüntünün tepesinde) etiketi kutunun içine al
            top = min(y1, height - box_h)
        left = max(0, min(x1, width - text_w - 6))
        cv2.rectangle(out, (left, top), (left + text_w + 6, top + box_h), _CYAN, cv2.FILLED)
        cv2.putText(out, label, (left + 3, top + text_h + 1), _FONT, label_scale, _BLACK, label_thickness, cv2.LINE_AA)
