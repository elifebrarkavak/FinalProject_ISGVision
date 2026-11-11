"""Kamera/video modülünün dışarıya verdiği tek veri yapısı: ``FramePacket``.

Gelecekteki modüller (insan tespiti, takip, yüz tespiti, KKD) kareyi bu yapı üzerinden alır; kamera
okuma kodunu bilmeleri veya değiştirmeleri gerekmez.

Not: ``frame`` bir NumPy dizisidir ve kopyalanmadan paylaşılır. Üzerine çizim yapacak (kutu, yazı...)
modüller önce ``frame.copy()`` almalıdır; böylece aynı kareyi kullanan diğer modüller etkilenmez.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SOURCE_KIND_CAMERA = "camera"
SOURCE_KIND_VIDEO = "video"
SOURCE_KINDS = (SOURCE_KIND_CAMERA, SOURCE_KIND_VIDEO)


@dataclass(frozen=True)
class FramePacket:
    """Tek bir görüntü karesi ve metadata'sı.

    Attributes:
        frame: OpenCV BGR görüntüsü (``uint8``, şekil ``(yükseklik, genişlik, 3)``).
        frame_index: Kaynaktan başarıyla okunan karenin 0'dan başlayan sıra numarası.
            Başarısız okumalar numara tüketmez; sıra her zaman ardışıktır.
        timestamp: Karenin okunduğu an, ``time.monotonic()`` saniyesi (süre/gecikme/FPS hesabı için;
            saat ayarından etkilenmez, yalnızca farkları anlamlıdır).
        wall_time: Karenin okunduğu an, Unix zamanı (``time.time()``; olay kaydında tarih/saat için).
        source: Kaynak etiketi, ör. ``"camera:0"`` veya ``"video:/yol/klip.mp4"``.
        source_kind: ``"camera"`` veya ``"video"``.
    """

    frame: np.ndarray
    frame_index: int
    timestamp: float
    wall_time: float
    source: str
    source_kind: str

    def __post_init__(self) -> None:
        if not isinstance(self.frame, np.ndarray) or self.frame.ndim != 3 or self.frame.shape[2] != 3:
            raise ValueError("frame, (yükseklik, genişlik, 3) şeklinde bir BGR NumPy dizisi olmalıdır")
        if self.frame.shape[0] < 1 or self.frame.shape[1] < 1:
            raise ValueError("frame boş olamaz")
        if self.frame_index < 0:
            raise ValueError("frame_index negatif olamaz")
        if self.source_kind not in SOURCE_KINDS:
            raise ValueError(f"source_kind {SOURCE_KINDS} değerlerinden biri olmalıdır: {self.source_kind!r}")

    @property
    def width(self) -> int:
        """Gerçek kare genişliği (piksel)."""
        return int(self.frame.shape[1])

    @property
    def height(self) -> int:
        """Gerçek kare yüksekliği (piksel)."""
        return int(self.frame.shape[0])
