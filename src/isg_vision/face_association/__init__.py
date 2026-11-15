"""Yüz–kişi ilişkilendirme ve çoklu insan takibi entegrasyonu (Aşama 5B).

Aynı kareden gelen insan takibi (``tracking``: YOLO11n + ByteTrack) ve yüz tespiti (``face_detection``: YuNet) sonuçlarını birleştirir;
her yüzü, GEOMETRİK kanıt güvenilirse, ilgili geçici Track ID'ye bağlar. Belirsizde eşleştirmez (güvenli varsayılan).

Bu paket yalnızca ilişkilendirir:

* gerçek çalışan kimliği, isim, personel numarası, yüz tanıma, embedding veya biyometrik veri YOKTUR (Aşama 6-7); ``track_id`` kimliğe
  dönüştürülmez; yüz tespit skoru kimlik doğruluğu olarak yorumlanmaz;
* yüz eşleşmesi canlı insan doğrulaması DEĞİLDİR: poster/ekrandaki yüz bir kişi kutusunun kafa bölgesine denk gelirse eşleşebilir;
* görüntü kaydetmez, ağ kullanmaz.

Ayrıntı: ``docs/face_association.md``. Bu dosya hafiftir: model açmaz, kamera açmaz, ağ kullanmaz, ``cv2``/``ultralytics`` import etmez.
"""

from .associator import AssociatorStats, FacePersonAssociator
from .config import AssociationConfig
from .continuity import ContinuityMonitor, TrackObservation
from .errors import AssociationConfigError, AssociationInputError, AssociationResultError, FaceAssociationError
from .results import (
    STATUS_AMBIGUOUS,
    STATUS_INVALID,
    STATUS_MATCHED,
    STATUS_NO_FACE,
    STATUS_UNMATCHED_FACE,
    AssociatedFace,
    AssociatedPerson,
    FaceAssociationResult,
    TrackContinuity,
)

__all__ = [
    "STATUS_AMBIGUOUS",
    "STATUS_INVALID",
    "STATUS_MATCHED",
    "STATUS_NO_FACE",
    "STATUS_UNMATCHED_FACE",
    "AssociatedFace",
    "AssociatedPerson",
    "AssociationConfig",
    "AssociationConfigError",
    "AssociationInputError",
    "AssociationResultError",
    "AssociatorStats",
    "ContinuityMonitor",
    "FaceAssociationError",
    "FaceAssociationResult",
    "FacePersonAssociator",
    "TrackContinuity",
    "TrackObservation",
]
