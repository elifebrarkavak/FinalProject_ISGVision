"""Yüz–kişi ilişkilendirmesini (kişi kutusu + Track ID, yüz kutusu + bağlı ID, sayılar, süreler) görüntüye çizen katman.

* Çizim **girdi görüntüsünün kopyası** üzerinde yapılır; girdi değiştirilmez.
* **Kişi kutusu** tam dikdörtgen + ``ID n`` etiketidir (renk ``tracking.visualization.color_for_track``; renk yalnızca görsel ayrımdır, kimlik değildir).
  Durum etikete yazılır: ``FACE`` (eşleşti), ``NO FACE`` (yüz bulunamadı), ``FACE ?`` (belirsiz), ``INVALID`` (tutarsız girdi). Yıldız (``*``) =
  eşleşme var ama izin sürekliliği henüz güvenilir değil (yeni iz, örtüşme, şüpheli ID değişimi sonrası). ``STATIC?`` = kutu uzun süredir
  hiç değişmiyor (poster/yorgan benzeri kalıcı yanlış pozitif ADAYI; yargı değil).
* **Yüz kutusu** köşe ayraçlarıyla çizilir (kişi kutusundan görsel olarak ayrışır). EŞLEŞEN yüz, bağlı olduğu kişinin RENGİYLE ve ``FACE>ID n``
  etiketiyle çizilir. Eşleşmeyen yüz (amber, ``FACE unmatched``) veya belirsiz yüz (macenta, ``FACE ambiguous``) HİÇBİR Track ID yazılmadan çizilir:
  eşleşmeyen/belirsiz yüze yanlış ID yazılmaz.
* Alt şeritte **her zaman** şu uyarı yazar: ``LINK = geometry only (not identity, not liveness)``.
* Metinler yalnızca ASCII'dir (OpenCV yazı tipleri ``ç ş ğ ı ö ü`` çizemez); ``camera.display.ascii_text`` kullanılır.
* Pencere yönetimi burada DEĞİLDİR (``camera.display.PreviewWindow``).

Bu modül import edildiğinde hiçbir pencere açılmaz.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from isg_vision.camera.display import ascii_text
from isg_vision.tracking.visualization import color_for_track

from .results import (
    STATUS_AMBIGUOUS,
    STATUS_INVALID,
    STATUS_MATCHED,
    STATUS_NO_FACE,
    STATUS_UNMATCHED_FACE,
    AssociatedFace,
    AssociatedPerson,
    FaceAssociationResult,
)

WINDOW_NAME = "ISG Vision - Face-Person Association"
EXIT_HINT = "Q / Esc / Close to exit"
NOTICE = "LINK = geometry only (not identity, not liveness)"
LEGEND = "* = continuity not trusted yet"

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_GREEN = (80, 255, 80)
_AMBER = (0, 200, 255)  # BGR: eşleşmeyen yüz / uyarı
_MAGENTA = (255, 0, 255)  # belirsiz
_GRAY = (170, 170, 170)


@dataclass(frozen=True)
class AssociationOverlayInfo:
    """Üst şeritte gösterilecek bilgiler (sayılar ``FaceAssociationResult``'tan gelir).

    Attributes:
        source_text: Örn. ``"CAMERA 0"`` veya ``"VIDEO klip.mp4"``.
        fps: Ölçülen uçtan uca FPS (kayan pencere ortalaması). Kaynağın bildirdiği FPS değildir.
        yolo_ms / tracking_ms / face_ms / association_ms: Ayrı ayrı süreler (ms); ``None`` ise ``--`` yazılır.
        resolution: Gerçek kare boyutu ``(genişlik, yükseklik)``.
        device: YOLO çıkarım cihazı (``cpu``/``mps``); ByteTrack, YuNet ve ilişkilendirme CPU'dadır.
        note: İsteğe bağlı uyarı satırı.
    """

    source_text: str
    fps: float
    yolo_ms: float | None
    tracking_ms: float | None
    face_ms: float | None
    association_ms: float | None
    resolution: tuple[int, int] | None
    device: str | None = None
    note: str | None = None


def person_label(person: AssociatedPerson) -> str:
    """Kişi kutusunun etiketi (ASCII)."""
    base = f"ID {person.track_id}"
    if person.status == STATUS_MATCHED:
        text = f"{base} FACE" + ("" if person.trusted_match else "*")
    elif person.status == STATUS_NO_FACE:
        text = f"{base} NO FACE"
    elif person.status == STATUS_AMBIGUOUS:
        text = f"{base} FACE ?"
    else:
        text = f"{base} INVALID"
    if person.continuity is not None and person.continuity.possibly_static:
        text += " STATIC?"
    return text


def face_label(face: AssociatedFace) -> str:
    """Yüz kutusunun etiketi (ASCII). Yalnızca ``matched`` yüzde Track ID yazılır."""
    edge = " EDGE" if face.at_border else ""
    if face.status == STATUS_MATCHED:
        return f"FACE>ID {face.track_id} {face.face_confidence:.2f}{edge}"
    if face.status == STATUS_UNMATCHED_FACE:
        return f"FACE unmatched {face.face_confidence:.2f}{edge}"
    if face.status == STATUS_AMBIGUOUS:
        return f"FACE ambiguous {face.face_confidence:.2f}{edge}"
    return f"FACE ? {face.face_confidence:.2f}{edge}"


def draw_face_associations(image: np.ndarray, result: FaceAssociationResult, info: AssociationOverlayInfo) -> np.ndarray:
    """Görüntünün **kopyasına** kişi/yüz kutularını, ilişkilendirmeyi ve durum şeridini çizer; girdiyi değiştirmez."""
    out = image.copy()
    height, width = out.shape[:2]
    scale = max(0.35, min(1.0, min(width, height) / 720.0 * 0.6 + 0.1))
    thickness = 1 if scale < 0.8 else 2
    (_, text_h), baseline = cv2.getTextSize("Ag", _FONT, scale, thickness)
    pad = max(4, int(10 * scale))
    line_h = text_h + baseline + pad // 2

    resolution = f"{info.resolution[0]}x{info.resolution[1]}" if info.resolution else "N/A"
    device = f" ({ascii_text(info.device).upper()})" if info.device else ""
    left = [
        f"PERSONS: {result.person_count}   FACES: {result.face_count}",
        f"MATCHED: {result.matched_count} (trusted {result.trusted_match_count})",
        f"NO FACE: {result.no_face_count}  AMBIGUOUS: {result.ambiguous_person_count}  UNMATCHED FACES: {result.unmatched_face_count}",
        f"SOURCE: {ascii_text(info.source_text)}  {resolution}",
    ]
    right = [
        f"FPS: {info.fps:.1f}",
        f"YOLO{device}: {_ms(info.yolo_ms)}",
        f"BYTETRACK: {_ms(info.tracking_ms, 1)}",
        f"YUNET: {_ms(info.face_ms)}",
        f"ASSOC: {_ms(info.association_ms, 2)}",
    ]
    note = f"NOTE: {ascii_text(info.note)}" if info.note else None
    if not result.input_valid:
        note = "INPUT MISMATCH - NO ASSOCIATION MADE"
    rows = max(len(left) + (1 if note else 0), len(right))
    band_h = min(height, pad * 2 + rows * line_h)
    foot_h = min(height, pad * 2 + 2 * line_h - pad // 2)
    out[:band_h] = cv2.convertScaleAbs(out[:band_h], alpha=0.4)  # okunabilirlik için şeritleri koyulaştır
    out[height - foot_h :] = cv2.convertScaleAbs(out[height - foot_h :], alpha=0.4)

    _draw_persons(out, result.persons, scale, band_h)
    _draw_faces(out, result.faces, scale, band_h)

    y = pad + text_h
    for index, text in enumerate(left):
        color = _GREEN if index == 1 and result.matched_count else _WHITE
        cv2.putText(out, text, (pad, y), _FONT, scale, color, thickness, cv2.LINE_AA)
        y += line_h
    if note:
        cv2.putText(out, note, (pad, y), _FONT, scale, _AMBER, thickness, cv2.LINE_AA)

    y = pad + text_h
    for index, text in enumerate(right):
        (text_w, _), _ = cv2.getTextSize(text, _FONT, scale, thickness)
        cv2.putText(out, text, (max(pad, width - text_w - pad), y), _FONT, scale, _GREEN if index == 0 else _WHITE, thickness, cv2.LINE_AA)
        y += line_h

    foot_scale = scale
    (notice_w, _), _ = cv2.getTextSize(NOTICE, _FONT, foot_scale, thickness)
    while notice_w > width - 2 * pad and foot_scale > 0.25:
        foot_scale -= 0.05
        (notice_w, _), _ = cv2.getTextSize(NOTICE, _FONT, foot_scale, thickness)
    cv2.putText(out, EXIT_HINT, (pad, height - pad - line_h - baseline + pad // 2), _FONT, scale, _WHITE, thickness, cv2.LINE_AA)
    (hint_w, _), _ = cv2.getTextSize(EXIT_HINT, _FONT, scale, thickness)
    (legend_w, _), _ = cv2.getTextSize(LEGEND, _FONT, scale * 0.8, 1)
    if pad * 3 + hint_w + legend_w <= width:
        cv2.putText(out, LEGEND, (width - legend_w - pad, height - pad - line_h - baseline + pad // 2), _FONT, scale * 0.8, _GRAY, 1, cv2.LINE_AA)
    cv2.putText(out, NOTICE, (pad, height - pad - baseline), _FONT, foot_scale, _AMBER, thickness, cv2.LINE_AA)
    return out


def _ms(value: float | None, digits: int = 0) -> str:
    return f"{value:.{digits}f} ms" if value is not None else "-- ms"


def _clamp_box(box: tuple[float, ...], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    return max(0, min(x1, width - 1)), max(0, min(y1, height - 1)), max(0, min(x2, width - 1)), max(0, min(y2, height - 1))


def _draw_label(out: np.ndarray, text: str, x: int, y_top: int, color: tuple[int, int, int], scale: float, thickness: int,
                min_top: int, *, above: bool = True, anchor_y: int | None = None) -> None:
    """Dolgulu etiket; ``above`` ise ``y_top``'un üstüne, değilse altına koyar. Görüntü dışına taşmaz, üst şeridin altında kalır."""
    height, width = out.shape[:2]
    (text_w, text_h), baseline = cv2.getTextSize(text, _FONT, scale, thickness)
    box_h = text_h + baseline + 4
    top = y_top - box_h if above else y_top
    if top < min_top:
        top = min(y_top if above else y_top, height - box_h)
        top = max(top, min_top)
    if top + box_h > height:
        top = max(min_top, (anchor_y if anchor_y is not None else y_top) - box_h)
    left = max(0, min(x, width - text_w - 6))
    cv2.rectangle(out, (left, top), (left + text_w + 6, top + box_h), color, cv2.FILLED)
    cv2.putText(out, text, (left + 3, top + text_h + 1), _FONT, scale, _BLACK, thickness, cv2.LINE_AA)


def _draw_persons(out: np.ndarray, persons: tuple[AssociatedPerson, ...], scale: float, band_h: int) -> None:
    height, width = out.shape[:2]
    box_thickness = max(2, int(round(min(width, height) / 360)))
    label_scale = max(0.35, scale * 0.8)
    label_thickness = 1 if label_scale < 0.7 else 2
    for person in persons:
        color = _GRAY if person.status == STATUS_INVALID else color_for_track(person.track_id)
        x1, y1, x2, y2 = _clamp_box(person.person_bbox, width, height)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thickness, cv2.LINE_AA)
        _draw_label(out, person_label(person), x1, y1, color, label_scale, label_thickness, band_h, above=True)


def _draw_faces(out: np.ndarray, faces: tuple[AssociatedFace, ...], scale: float, band_h: int) -> None:
    height, width = out.shape[:2]
    box_thickness = max(2, int(round(min(width, height) / 360)))
    label_scale = max(0.35, scale * 0.8)
    label_thickness = 1 if label_scale < 0.7 else 2
    for face in faces:
        if face.status == STATUS_MATCHED and face.track_id is not None:
            color = color_for_track(face.track_id)
        elif face.status == STATUS_UNMATCHED_FACE:
            color = _AMBER
        elif face.status == STATUS_AMBIGUOUS:
            color = _MAGENTA
        else:
            color = _GRAY
        x1, y1, x2, y2 = _clamp_box(face.face_bbox, width, height)
        _draw_corners(out, (x1, y1, x2, y2), color, box_thickness)
        _draw_label(out, face_label(face), x1, y2, color, label_scale, label_thickness, band_h, above=False, anchor_y=y1)


def _draw_corners(out: np.ndarray, box: tuple[int, int, int, int], color: tuple[int, int, int], thickness: int) -> None:
    """Dikdörtgenin dört köşesine L-biçimli ayraçlar çizer (yüz kutusu, tam dikdörtgen olan kişi kutusundan ayrışsın diye)."""
    x1, y1, x2, y2 = box
    length = max(4, min(x2 - x1, y2 - y1) // 3)
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(out, (cx, cy), (cx + dx * length, cy), color, thickness, cv2.LINE_AA)
        cv2.line(out, (cx, cy), (cx, cy + dy * length), color, thickness, cv2.LINE_AA)
