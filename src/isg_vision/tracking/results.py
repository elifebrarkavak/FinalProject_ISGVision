"""İnsan takibi sonuç veri yapıları.

Koordinat kuralı ``human_detection.results`` ile aynıdır: ``bbox_xyxy = (x1, y1, x2, y2)``, görüntünün piksel koordinatları,
sol-üst köşe başlangıç, ``x1 < x2`` ve ``y1 < y2``, görüntü sınırlarına kırpılmış ``float`` değerler.

**Track ID bir kimlik DEĞİLDİR.** ``track_id``:

* geçicidir ve **oturum kapsamlıdır**: takipçi sıfırlandığında (yeni kaynak, yeni oturum) 1'den yeniden başlar; farklı oturumlarda
  aynı sayı farklı kişilerdir. Bu yüzden her sonuçta ``session_id`` de taşınır; oturumlar arası benzersiz anahtar
  ``(session_id, track_id)``'dir.
* bir çalışan adı, personel numarası veya biyometrik kimlik değildir; yalnızca "bu kutu bir önceki karedeki şu kutuyla aynı
  kişi gibi görünüyor" bilgisidir. Hatalı eşleştirme (ID switch) olabilir.
* bitişik sayılar olmak zorunda değildir: onaylanmadan silinen geçici izler de numara tüketir (1, 2, 5, 6 gibi boşluklar normaldir).

Bu modül yalnızca standart kütüphane (+ ``human_detection.results``) kullanır; ``torch``/``ultralytics``/NumPy import etmez.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from isg_vision.human_detection.results import BBox

__all__ = ["BBox", "LostTrack", "TrackedPerson", "TrackingResult"]


@dataclass(frozen=True)
class TrackedPerson:
    """Bu karede takip edilen (eşleşmiş ve onaylanmış) bir insan.

    Attributes:
        track_id: Oturum içinde geçici takip numarası (``>= 1``). Kimlik değildir.
        class_id / class_name: Tespitin sınıfı (``person``).
        confidence: Bu karede bu ize eşleşen YOLO tespitinin güven skoru, ``0..1``. **Takip doğruluğu değildir.**
        bbox_xyxy: Takipçinin (Kalman filtreli) kutu tahmini; ham YOLO kutusuyla birebir aynı OLMAZ (hareket başlarken
            ham kutunun biraz gerisinde kalabilir). Ham kutu ``detection_index`` ile ``DetectionResult.detections`` içinden alınır.
        frame_index: Kaynak kare sıra numarası.
        timestamp: ``FramePacket.timestamp`` (monotonic saniye; bilinmiyorsa ``None``).
        session_id: Takip oturumu (``>= 1``). Her sıfırlamada artar.
        detection_index: Bu iz için kullanılan tespitin, aynı karenin ``DetectionResult.detections`` içindeki sırası.
        age_frames: İzin oluşturulduğu andan bu kareye kadar geçen işlenen kare sayısı (kayıp olduğu kareler dahil).
    """

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: BBox
    frame_index: int
    timestamp: float | None
    session_id: int
    detection_index: int
    age_frames: int

    def __post_init__(self) -> None:
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int) or self.track_id < 1:
            raise ValueError(f"track_id 1 veya daha büyük bir tam sayı olmalıdır: {self.track_id!r}")
        if isinstance(self.session_id, bool) or not isinstance(self.session_id, int) or self.session_id < 1:
            raise ValueError(f"session_id 1 veya daha büyük bir tam sayı olmalıdır: {self.session_id!r}")
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int) or self.class_id < 0:
            raise ValueError(f"class_id negatif olmayan bir tam sayı olmalıdır: {self.class_id!r}")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence 0..1 aralığında sonlu bir sayı olmalıdır: {self.confidence!r}")
        box = tuple(float(v) for v in self.bbox_xyxy)
        if len(box) != 4 or not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"bbox_xyxy için dört sonlu sayı ve x1<x2, y1<y2 gerekir: {self.bbox_xyxy!r}")
        object.__setattr__(self, "bbox_xyxy", box)
        if self.frame_index < 0 or self.detection_index < 0 or self.age_frames < 1:
            raise ValueError("frame_index/detection_index negatif, age_frames 1'den küçük olamaz")

    @property
    def key(self) -> tuple[int, int]:
        """Oturumlar arası benzersiz anahtar ``(session_id, track_id)``."""
        return (self.session_id, self.track_id)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox_xyxy[0] + self.bbox_xyxy[2]) / 2.0, (self.bbox_xyxy[1] + self.bbox_xyxy[3]) / 2.0)


@dataclass(frozen=True)
class LostTrack:
    """Kısa süre önce kaybolmuş ama tampon süresi dolmadığı için bellekte tutulan iz.

    Ekranda/aktif sayıda GÖSTERİLMEZ: görünmeyen bir kişiyi görünüyormuş gibi göstermek yanıltıcıdır. Yalnızca tampon durumunu
    izlemek içindir (kişi geri dönerse aynı ``track_id`` ile devam edebilir).

    Attributes:
        track_id: Kayıp izin numarası.
        frames_since_seen: İzin en son eşleştiği kareden bu yana işlenen kare sayısı (``>= 1``).
    """

    track_id: int
    frames_since_seen: int


@dataclass(frozen=True)
class TrackingResult:
    """Tek bir karenin takip sonucu.

    Üç sayı birbirine KARIŞTIRILMAMALIDIR: ``detection_count`` (YOLO'nun bu karede verdiği kutular), ``active_count`` (bu karede
    eşleşmiş ve onaylanmış izler = görüntülenen kutular) ve ``lost_count`` (görünmeyen, tamponda bekleyen izler).

    Attributes:
        tracks: Aktif izler (``track_id``'ye göre artan sırada, benzersiz ``track_id``).
        lost_tracks: Tamponda bekleyen kayıp izler (aktif değildir).
        tentative_count: Henüz onaylanmamış yeni izler (ilk kez görüldü; bir sonraki karede eşleşirse onaylanır, görüntülenmez).
        detection_count: Bu karede takipçiye verilen tespit sayısı.
        image_size: ``(genişlik, yükseklik)``.
        frame_index / timestamp: Karenin kaynak sıra numarası ve zaman damgası.
        session_id: Takip oturumu (bkz. ``TrackedPerson``).
        tracking_ms: ``PersonTracker.update`` süresi (ms): dönüştürme + ByteTrack güncellemesi + sonuç oluşturma. YOLO çıkarımı,
            kare okuma ve çizim dahil DEĞİLDİR.
        dropped_tracks: Kutusu görüntü sınırlarına kırpılınca geçersiz kaldığı için sonuçtan çıkarılan iz sayısı.
        reset_reason: Bu kareden hemen önce takipçi otomatik sıfırlandıysa nedeni (``source_changed``, ``image_size_changed``);
            aksi halde ``None``. Sıfırlama varsa önceki karelerin ``track_id``'leri geçersizdir.
    """

    tracks: tuple[TrackedPerson, ...]
    lost_tracks: tuple[LostTrack, ...]
    tentative_count: int
    detection_count: int
    image_size: tuple[int, int]
    frame_index: int
    timestamp: float | None
    session_id: int
    tracking_ms: float = 0.0
    dropped_tracks: int = 0
    reset_reason: str | None = None

    @property
    def active_count(self) -> int:
        """Aktif (görüntülenen) iz sayısı."""
        return len(self.tracks)

    @property
    def lost_count(self) -> int:
        return len(self.lost_tracks)

    @property
    def is_empty(self) -> bool:
        return not self.tracks

    @property
    def track_ids(self) -> tuple[int, ...]:
        return tuple(t.track_id for t in self.tracks)

    def __len__(self) -> int:
        return len(self.tracks)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.tracks)
