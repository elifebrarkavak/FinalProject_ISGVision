"""İnsan tespiti sonuç veri yapıları.

Koordinat kuralı (tüm modül için tek tanım): ``bbox_xyxy = (x1, y1, x2, y2)``; **görüntünün piksel koordinatları**,
başlangıç sol-üst köşe, ``x`` sağa ve ``y`` aşağı artar, ``x1 < x2`` ve ``y1 < y2``. Kutular görüntü sınırlarına
kırpılmıştır (``0 <= x <= genişlik``, ``0 <= y <= yükseklik``). Değerler ``float`` pikseldir (alt piksel kesirli olabilir).

Bu aşamada sonuçlarda ``track_id``, çalışan adı veya yüz kimliği YOKTUR; bunlar sonraki aşamalarda (takip, yüz tanıma)
ayrı yapılarla bu kutulara ilişkilendirilecektir.

Bu modül yalnızca standart kütüphane kullanır (torch/OpenCV/NumPy import etmez).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

BBox = tuple[float, float, float, float]
MIN_BOX_SIDE_PX = 1.0  # kırpmadan sonra kenarı bundan küçük kutular geçersiz sayılır


@dataclass(frozen=True)
class PersonDetection:
    """Tek bir insan tespiti.

    Attributes:
        class_id: Modelin sınıf kimliği (modelin ``names`` haritasından doğrulanan ``person`` kimliği).
        class_name: Sınıf adı (``"person"``).
        confidence: Modelin güven skoru, ``0..1``. Doğruluk olasılığı değildir; kalibre edilmemiş bir skordur.
        bbox_xyxy: ``(x1, y1, x2, y2)`` piksel koordinatları (görüntü sınırlarına kırpılmış).
        frame_index: Tespitin yapıldığı karenin kaynak sıra numarası (biliniyorsa).
        timestamp: Karenin ``FramePacket.timestamp`` değeri (monotonic saniye; biliniyorsa).
    """

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: BBox
    frame_index: int | None = None
    timestamp: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int) or self.class_id < 0:
            raise ValueError(f"class_id negatif olmayan bir tam sayı olmalıdır: {self.class_id!r}")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence 0..1 aralığında sonlu bir sayı olmalıdır: {self.confidence!r}")
        box = tuple(float(v) for v in self.bbox_xyxy)
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise ValueError(f"bbox_xyxy dört sonlu sayıdan oluşmalıdır: {self.bbox_xyxy!r}")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"bbox_xyxy için x1<x2 ve y1<y2 olmalıdır: {self.bbox_xyxy!r}")
        object.__setattr__(self, "bbox_xyxy", box)
        if self.frame_index is not None and self.frame_index < 0:
            raise ValueError("frame_index negatif olamaz")

    @property
    def x1(self) -> float:
        return self.bbox_xyxy[0]

    @property
    def y1(self) -> float:
        return self.bbox_xyxy[1]

    @property
    def x2(self) -> float:
        return self.bbox_xyxy[2]

    @property
    def y2(self) -> float:
        return self.bbox_xyxy[3]

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True)
class DetectionResult:
    """Tek bir karenin tespit sonucu (``detections`` güven skoruna göre azalan sıradadır).

    Attributes:
        detections: Bu karedeki insanlar; hiç insan yoksa boş demet.
        image_size: Tespit yapılan görüntünün gerçek boyutu ``(genişlik, yükseklik)``.
        frame_index: Karenin kaynak sıra numarası (biliniyorsa).
        timestamp: Karenin ``FramePacket.timestamp`` değeri (biliniyorsa).
        inference_ms: Bu kare için model çağrısının duvar saati süresi (ms): ön işleme + ileri geçiş + NMS + sonuçların
            CPU'ya alınması. Kamera okuma, çizim ve görüntüleme dâhil DEĞİLDİR.
        device: Çıkarımın gerçekte çalıştığı cihaz (``cpu`` veya ``mps``).
        dropped_boxes: Geçersiz (sonlu olmayan, görüntü dışı/dejenere) olduğu için atılan kutu sayısı.
    """

    detections: tuple[PersonDetection, ...]
    image_size: tuple[int, int]
    frame_index: int | None = None
    timestamp: float | None = None
    inference_ms: float = 0.0
    device: str = "cpu"
    dropped_boxes: int = 0

    @property
    def person_count(self) -> int:
        """Karedeki toplam insan sayısı."""
        return len(self.detections)

    @property
    def is_empty(self) -> bool:
        return not self.detections

    def __len__(self) -> int:
        return len(self.detections)

    def __iter__(self) -> Iterator[PersonDetection]:
        return iter(self.detections)


def clip_box_xyxy(box: Sequence[float], width: int, height: int) -> BBox | None:
    """Kutuyu görüntü sınırlarına kırpar; geçersizse ``None`` döndürür.

    ``None`` döndürülen durumlar: dört sayıdan oluşmuyor, NaN/sonsuz değer içeriyor, ``x1 >= x2`` veya ``y1 >= y2``,
    tamamen görüntü dışında ya da kırpmadan sonra bir kenarı ``MIN_BOX_SIDE_PX``'ten küçük.
    Kısmen görüntü dışına taşan kutular (görüntü kenarındaki insanlar) **atılmaz**, kırpılır.
    """
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    if x2 - x1 < MIN_BOX_SIDE_PX or y2 - y1 < MIN_BOX_SIDE_PX:
        return None
    return (x1, y1, x2, y2)


def build_detections(
    boxes: Iterable[Sequence[float]],
    confidences: Iterable[float],
    class_ids: Iterable[int],
    class_names: dict[int, str],
    *,
    person_class_id: int,
    image_size: tuple[int, int],
    frame_index: int | None = None,
    timestamp: float | None = None,
) -> tuple[tuple[PersonDetection, ...], int]:
    """Ham model çıktısından yalnızca ``person`` tespitlerini üretir.

    Returns:
        ``(tespitler, atılan_geçersiz_kutu_sayısı)``. ``person`` dışındaki sınıflar sessizce elenir (bunlar "geçersiz"
        sayılmaz). Tespitler güvene göre azalan sıradadır (eşitlikte model sırası korunur).
    """
    width, height = image_size
    kept: list[PersonDetection] = []
    dropped = 0
    for box, conf, cls in zip(boxes, confidences, class_ids):
        if int(cls) != person_class_id:
            continue
        clipped = clip_box_xyxy(box, width, height)
        conf = float(conf)
        if clipped is None or not math.isfinite(conf):
            dropped += 1
            continue
        kept.append(
            PersonDetection(
                class_id=person_class_id,
                class_name=class_names.get(person_class_id, "person"),
                confidence=min(1.0, max(0.0, conf)),
                bbox_xyxy=clipped,
                frame_index=frame_index,
                timestamp=timestamp,
            )
        )
    kept.sort(key=lambda d: -d.confidence)  # sort kararlıdır
    return tuple(kept), dropped
