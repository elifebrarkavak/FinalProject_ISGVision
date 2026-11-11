"""OpenCV ile canlı görüntü penceresi ve durum bilgisi katmanı (overlay).

Bu modül yalnızca **durum bilgisini** (kaynak, FPS, çözünürlük, durum) gösterir. İnsan kutusu, isim, baret/yelek
gibi yapay zekâ çıktıları bu aşamada gösterilmez.

Metinler bilinçli olarak yalnızca ASCII'dir: OpenCV'nin ``putText`` yazı tipleri ``ç ş ğ ı ö ü`` gibi
karakterleri çizemez ve ``?`` olarak gösterir.

Modül import edildiğinde pencere açılmaz; pencere ``PreviewWindow.open()`` veya ilk ``show()`` ile oluşturulur.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import cv2
import numpy as np

from .errors import DisplayError

WINDOW_NAME = "ISG Vision - Camera"
EXIT_KEYS = (27, ord("q"), ord("Q"))  # Esc, q, Q
EXIT_HINT = "Press Q or ESC to exit"

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_GREEN = (80, 255, 80)
_AMBER = (0, 200, 255)  # BGR


def ascii_text(text: str) -> str:
    """Metni OpenCV'nin çizebileceği ASCII karakterlere indirger (``ş -> s``, ``ı -> i``; çizilemeyen -> ``?``)."""
    text = text.replace("ı", "i").replace("İ", "I")
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.encode("ascii", "replace").decode("ascii")


def is_exit_key(key: int) -> bool:
    """``cv2.waitKey`` sonucunun çıkış tuşu (Q, q veya Esc) olup olmadığını söyler. ``-1`` = tuş yok."""
    return key >= 0 and (key & 0xFF) in EXIT_KEYS


@dataclass(frozen=True)
class OverlayInfo:
    """Overlay'de gösterilecek bilgiler.

    Attributes:
        source_text: Örn. ``"CAMERA 0"`` veya ``"VIDEO klip.mp4"`` (ASCII'ye çevrilir).
        fps: Ölçülen FPS (uygulamanın gerçekten işlediği kare hızı).
        resolution: Gerçek kare boyutu (genişlik, yükseklik).
        source_fps: Kaynağın bildirdiği FPS; bilinmiyorsa ``None``.
        status: Örn. ``"LIVE"`` veya ``"PLAYING 12/300"``.
        note: İsteğe bağlı uyarı satırı (örn. istenen çözünürlük uygulanamadı).
    """

    source_text: str
    fps: float
    resolution: tuple[int, int] | None
    source_fps: float | None
    status: str
    note: str | None = None


def draw_overlay(image: np.ndarray, info: OverlayInfo) -> np.ndarray:
    """Görüntünün **kopyası** üzerine durum bilgisini çizer; girdi görüntüsünü değiştirmez."""
    out = image.copy()
    height, width = out.shape[:2]
    scale = max(0.35, min(1.0, min(width, height) / 720.0 * 0.6 + 0.1))
    thickness = 1 if scale < 0.8 else 2
    (_, text_h), baseline = cv2.getTextSize("Ag", _FONT, scale, thickness)
    pad = max(4, int(10 * scale))
    line_h = text_h + baseline + pad // 2

    resolution = f"{info.resolution[0]}x{info.resolution[1]}" if info.resolution else "N/A"
    source_fps = f"{info.source_fps:.1f}" if info.source_fps else "N/A"
    left = [
        f"SOURCE: {ascii_text(info.source_text)}",
        f"RESOLUTION: {resolution}",
        f"STATUS: {ascii_text(info.status)}",
    ]
    right = [f"FPS: {info.fps:.1f}", f"SOURCE FPS: {source_fps}"]
    note = f"NOTE: {ascii_text(info.note)}" if info.note else None

    rows = len(left) + (1 if note else 0)
    band_h = min(height, pad * 2 + rows * line_h)
    out[:band_h] = cv2.convertScaleAbs(out[:band_h], alpha=0.4)  # üst şerit: okunabilirlik için koyulaştır

    y = pad + text_h
    for text in left:
        cv2.putText(out, text, (pad, y), _FONT, scale, _WHITE, thickness, cv2.LINE_AA)
        y += line_h
    if note:
        cv2.putText(out, note, (pad, y), _FONT, scale, _AMBER, thickness, cv2.LINE_AA)

    y = pad + text_h
    for text, color in zip(right, (_GREEN, _WHITE)):
        (text_w, _), _ = cv2.getTextSize(text, _FONT, scale, thickness)
        cv2.putText(out, text, (max(pad, width - text_w - pad), y), _FONT, scale, color, thickness, cv2.LINE_AA)
        y += line_h

    foot_h = min(height, text_h + baseline + pad * 2)
    out[height - foot_h :] = cv2.convertScaleAbs(out[height - foot_h :], alpha=0.4)
    cv2.putText(out, EXIT_HINT, (pad, height - pad - baseline), _FONT, scale, _WHITE, thickness, cv2.LINE_AA)
    return out


class PreviewWindow:
    """Tek bir OpenCV penceresini yönetir; ``with`` bloğunda kullanıldığında kapanışta pencere temizlenir."""

    def __init__(self, name: str = WINDOW_NAME) -> None:
        self.name = name
        self._created = False
        self._shown = False

    def open(self) -> None:
        """Pencereyi oluşturur.

        Raises:
            DisplayError: OpenCV pencere oluşturamadı (ör. grafik oturumu yok, GUI desteği olmayan OpenCV).
        """
        if self._created:
            return
        try:
            cv2.namedWindow(self.name, cv2.WINDOW_AUTOSIZE)
        except cv2.error as exc:
            raise DisplayError(
                f"OpenCV görüntü penceresi oluşturulamadı: {exc}. Grafik oturumu olan bir terminalde "
                "(Terminal.app, VS Code) çalıştırın veya penceresiz çalışmak için --no-display kullanın."
            ) from exc
        self._created = True

    def show(self, image: np.ndarray, delay_ms: int = 1) -> int:
        """Görüntüyü gösterir, en fazla ``delay_ms`` kadar tuş bekler ve tuş kodunu döndürür (tuş yoksa ``-1``)."""
        self.open()
        try:
            cv2.imshow(self.name, image)
            key = int(cv2.waitKey(max(1, int(delay_ms))))
        except cv2.error as exc:
            raise DisplayError(f"Görüntü pencerede gösterilemedi: {exc}") from exc
        self._shown = True
        return key

    @property
    def closed_by_user(self) -> bool:
        """Kullanıcı pencereyi kapatma düğmesiyle kapattıysa ``True`` (ilk kare gösterilmeden önce hep ``False``)."""
        if not self._created or not self._shown:
            return False
        try:
            return cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def close(self) -> None:
        """Pencereyi yok eder. Birden fazla kez çağrılması güvenlidir."""
        if not self._created:
            return
        self._created = False
        self._shown = False
        try:
            cv2.destroyWindow(self.name)
            for _ in range(3):  # macOS'ta pencerenin gerçekten kapanması için olay döngüsünü işlet
                cv2.waitKey(1)
        except cv2.error:
            pass

    def __enter__(self) -> PreviewWindow:
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
