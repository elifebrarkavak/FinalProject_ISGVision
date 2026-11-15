"""Yüz tespiti sonuç veri yapıları.

Koordinat kuralı ``human_detection.results`` ve ``tracking.results`` ile AYNIDIR (tek tanım): ``bbox_xyxy = (x1, y1, x2, y2)``;
görüntünün piksel koordinatları, başlangıç sol-üst köşe, ``x`` sağa ve ``y`` aşağı artar, ``x1 < x2`` ve ``y1 < y2``. Kutular
görüntü sınırlarına kırpılmıştır. Böylece bir yüz kutusu ile ``TrackedPerson.bbox_xyxy`` / ``PersonDetection.bbox_xyxy`` aynı
koordinat sisteminde karşılaştırılabilir (Aşama 5B'nin işi; burada eşleştirme YOKTUR).

**Bir yüz tespiti bir kimlik, yaşayan bir insan veya çalışan kanıtı DEĞİLDİR.** Bu sonuçlarda ``track_id``, çalışan adı, yüz
kimliği, gömme (embedding) veya biyometrik şablon bulunmaz. Poster, basılı fotoğraf veya ekrandaki yüz de yüksek güvenli bir yüz
tespiti üretebilir; yüz görünmediğinde kimlik hakkında hiçbir çıkarım yapılmaz.

Aşama 5B ile ilişkilendirme anahtarı: ``(frame_index, image_size)``. Aynı kare için ``FaceDetectionResult`` ve ``TrackingResult``
aynı ``frame_index`` ve ``image_size`` değerlerini taşır; 5B bunları karşılaştırarak aynı kareden olduklarını doğrulamalıdır.
``FaceDetection.face_index``, aynı karenin ``FaceDetectionResult.faces`` içindeki sırasıdır (``TrackedPerson.detection_index`` gibi).

Bu modül yalnızca standart kütüphane (+ ``human_detection.results``) kullanır; NumPy/OpenCV import etmez.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from isg_vision.human_detection.results import BBox, clip_box_xyxy

__all__ = ["BORDER_TOLERANCE_PX", "BBox", "FaceDetection", "FaceDetectionResult", "build_faces", "clip_box_xyxy"]

# Kırpılmış kutunun herhangi bir kenarı görüntü sınırına bu kadar pikselden yakınsa yüz "kenarda" sayılır (kadrajın kestiği/kestiği
# olabilecek, kısmen görünen yüz). Bu bir eşik/sezgidir; yüzün gerçekten kesildiğinin kanıtı değildir.
BORDER_TOLERANCE_PX = 2.0


@dataclass(frozen=True)
class FaceDetection:
    """Tek bir yüz tespiti.

    Attributes:
        confidence: Modelin güven skoru, ``0..1``. Doğruluk olasılığı değildir; kalibre edilmemiş bir skordur.
        bbox_xyxy: ``(x1, y1, x2, y2)`` piksel koordinatları (görüntü sınırlarına kırpılmış).
        frame_index: Tespitin yapıldığı karenin kaynak sıra numarası (biliniyorsa).
        timestamp: Karenin ``FramePacket.timestamp`` değeri (monotonic saniye; biliniyorsa).
        image_size: Tespit yapılan karenin gerçek boyutu ``(genişlik, yükseklik)`` (biliniyorsa); kutu bu görüntünün içindedir.
        face_index: Bu yüzün aynı karenin ``FaceDetectionResult.faces`` içindeki sırası (0'dan; güvene göre azalan).
        at_border: Kutu görüntü sınırına ``BORDER_TOLERANCE_PX`` pikselden yakın (yüz kadraj kenarında, kısmen görünüyor olabilir).
    """

    confidence: float
    bbox_xyxy: BBox
    frame_index: int | None = None
    timestamp: float | None = None
    image_size: tuple[int, int] | None = None
    face_index: int | None = None
    at_border: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise ValueError(f"confidence sayı olmalıdır: {self.confidence!r}")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence 0..1 aralığında sonlu bir sayı olmalıdır: {self.confidence!r}")
        box = tuple(float(v) for v in self.bbox_xyxy)
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise ValueError(f"bbox_xyxy dört sonlu sayıdan oluşmalıdır: {self.bbox_xyxy!r}")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"bbox_xyxy için x1<x2 ve y1<y2 olmalıdır: {self.bbox_xyxy!r}")
        object.__setattr__(self, "bbox_xyxy", box)
        if self.frame_index is not None and (isinstance(self.frame_index, bool) or self.frame_index < 0):
            raise ValueError(f"frame_index negatif olmayan bir tam sayı olmalıdır: {self.frame_index!r}")
        if self.face_index is not None and (isinstance(self.face_index, bool) or self.face_index < 0):
            raise ValueError(f"face_index negatif olmayan bir tam sayı olmalıdır: {self.face_index!r}")
        if self.image_size is not None:
            size = tuple(self.image_size)
            if len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in size):
                raise ValueError(f"image_size iki pozitif tam sayı (genişlik, yükseklik) olmalıdır: {self.image_size!r}")
            if box[0] < 0 or box[1] < 0 or box[2] > size[0] + 1e-6 or box[3] > size[1] + 1e-6:
                raise ValueError(f"bbox_xyxy image_size {size} içinde olmalıdır: {self.bbox_xyxy!r}")
            object.__setattr__(self, "image_size", size)

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
class FaceDetectionResult:
    """Tek bir karenin yüz tespiti sonucu (``faces`` güven skoruna göre azalan sıradadır).

    Attributes:
        faces: Bu karedeki yüzler; hiç yüz yoksa boş demet (hata değildir).
        image_size: Tespit yapılan görüntünün gerçek boyutu ``(genişlik, yükseklik)``; kutular bu boyuttadır.
        frame_index: Karenin kaynak sıra numarası (biliniyorsa).
        timestamp: Karenin ``FramePacket.timestamp`` değeri (biliniyorsa).
        inference_ms: Bu kare için model çağrısının duvar saati süresi (ms): (varsa) yeniden boyutlandırma + girdi boyutu ayarı +
            ileri geçiş + NMS + sonuçların NumPy'a alınması. Kamera okuma, çizim ve görüntüleme dâhil DEĞİLDİR.
        device: Çıkarımın gerçekte çalıştığı yer (``cpu``; bkz. ``config`` açıklaması).
        dropped_boxes: Geçersiz (sonlu olmayan, görüntü dışı/dejenere) olduğu için atılan kutu sayısı.
        truncated_faces: ``max_faces`` sınırını aştığı için (güveni en düşükler) atılan geçerli yüz sayısı.
        inference_size: Modelin gerçekte çalıştığı girdi boyutu ``(genişlik, yükseklik)``; ``detect_width`` kullanılmadıysa
            ``image_size`` ile aynıdır. ``None`` = bilinmiyor.
    """

    faces: tuple[FaceDetection, ...]
    image_size: tuple[int, int]
    frame_index: int | None = None
    timestamp: float | None = None
    inference_ms: float = 0.0
    device: str = "cpu"
    dropped_boxes: int = 0
    truncated_faces: int = 0
    inference_size: tuple[int, int] | None = None

    @property
    def face_count(self) -> int:
        """Karedeki toplam yüz sayısı."""
        return len(self.faces)

    @property
    def is_empty(self) -> bool:
        return not self.faces

    def __len__(self) -> int:
        return len(self.faces)

    def __iter__(self) -> Iterator[FaceDetection]:
        return iter(self.faces)


def build_faces(
    boxes: Iterable[Sequence[float]],
    scores: Iterable[float],
    *,
    image_size: tuple[int, int],
    frame_index: int | None = None,
    timestamp: float | None = None,
    max_faces: int | None = None,
) -> tuple[tuple[FaceDetection, ...], int, int]:
    """Ham model çıktısından doğrulanmış ``FaceDetection`` demeti üretir.

    Args:
        boxes: ``(x1, y1, x2, y2)`` kutuları (özgün kare piksel koordinatlarında; görüntü dışına taşabilir).
        scores: Her kutunun ham güven skoru.
        max_faces: Verilirse güvene göre en yüksek bu kadar yüz tutulur.

    Returns:
        ``(yüzler, atılan_geçersiz_kutu_sayısı, kesilen_geçerli_yüz_sayısı)``. Kısmen görüntü dışına taşan kutular (kadraj
        kenarındaki yüzler) **atılmaz**, kırpılır ve ``at_border`` ile işaretlenir. Yüzler güvene göre azalan sıradadır
        (eşitlikte model sırası korunur) ve ``face_index`` bu sırayı taşır.
    """
    width, height = image_size
    candidates: list[tuple[float, BBox, bool]] = []
    dropped = 0
    for box, score in zip(boxes, scores):
        clipped = clip_box_xyxy(box, width, height)
        try:
            conf = float(score)
        except (TypeError, ValueError):
            dropped += 1
            continue
        if clipped is None or not math.isfinite(conf):
            dropped += 1
            continue
        x1, y1, x2, y2 = clipped
        at_border = x1 <= BORDER_TOLERANCE_PX or y1 <= BORDER_TOLERANCE_PX or x2 >= width - BORDER_TOLERANCE_PX or y2 >= height - BORDER_TOLERANCE_PX
        candidates.append((min(1.0, max(0.0, conf)), clipped, at_border))

    candidates.sort(key=lambda item: -item[0])  # sort kararlıdır
    truncated = 0
    if max_faces is not None and len(candidates) > max_faces:
        truncated = len(candidates) - max_faces
        candidates = candidates[:max_faces]
    faces = tuple(
        FaceDetection(
            confidence=conf,
            bbox_xyxy=box,
            frame_index=frame_index,
            timestamp=timestamp,
            image_size=(width, height),
            face_index=index,
            at_border=at_border,
        )
        for index, (conf, box, at_border) in enumerate(candidates)
    )
    return faces, dropped, truncated
