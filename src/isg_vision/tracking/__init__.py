"""Çoklu insan takibi (Aşama 4B): 4A tespit sonuçlarına Ultralytics ByteTrack ile geçici Track ID atar.

Akış: ``PersonDetector.detect`` -> ``DetectionResult`` -> ``PersonTracker.update`` -> ``TrackingResult``. YOLO her karede YALNIZCA
BİR KEZ çalışır (takipçi modeli çağırmaz). **Track ID geçici ve oturum kapsamlıdır; çalışan kimliği, personel numarası veya
biyometrik kimlik DEĞİLDİR.** Ayrıntılar: docs/person_tracking.md

Bu paket import edildiğinde ``torch``/``ultralytics`` yüklenmez, model açılmaz, kamera açılmaz, ağ bağlantısı kurulmaz, paket kurulmaz.
Görselleştirme (``visualization``), kare akışı döngüsü (``pipeline``) ve komut satırı (``cli``) OpenCV/kamera paketini kullanır;
bunlar bilinçli olarak burada içe aktarılmaz, gerektiğinde kendi modül yollarından import edilir.
"""

from .bytetrack_adapter import PersonTracker, TrackerStats, check_tracker_dependencies
from .config import ThresholdAnalysis, TrackingConfig, analyze_thresholds
from .errors import (
    FrameOrderError,
    InvalidTrackingInputError,
    TrackerBackendError,
    TrackerDependencyError,
    TrackingConfigError,
    TrackingError,
)
from .metrics import IdSwitchReport, count_id_switches
from .results import LostTrack, TrackedPerson, TrackingResult

__all__ = [
    "FrameOrderError",
    "IdSwitchReport",
    "InvalidTrackingInputError",
    "LostTrack",
    "PersonTracker",
    "ThresholdAnalysis",
    "TrackedPerson",
    "TrackerBackendError",
    "TrackerDependencyError",
    "TrackerStats",
    "TrackingConfig",
    "TrackingConfigError",
    "TrackingError",
    "TrackingResult",
    "analyze_thresholds",
    "check_tracker_dependencies",
    "count_id_switches",
]
