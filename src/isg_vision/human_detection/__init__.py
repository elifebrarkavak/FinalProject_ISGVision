"""Çoklu insan tespiti (Aşama 4A): önceden eğitilmiş YOLO ile karedeki tüm insanları tespit eder.

Bu paket import edildiğinde ``torch``/``ultralytics`` yüklenmez, model açılmaz, kamera açılmaz, ağ bağlantısı kurulmaz.
Takip (Track ID) bu pakette DEĞİLDİR; o Aşama 4B'dedir (``isg_vision.tracking``). Ayrıntılar: docs/person_detection.md

Görselleştirme (``visualization``), kare akışı döngüsü (``pipeline``) ve komut satırı (``cli``) OpenCV/kamera paketini
kullanır; bunlar bilinçli olarak burada içe aktarılmaz, gerektiğinde kendi modül yollarından import edilir.
"""

from .config import DEVICES, DetectionConfig, ResolvedDevice, resolve_device
from .errors import (
    DetectionConfigError,
    DetectionError,
    DeviceUnavailableError,
    InferenceError,
    InvalidFrameError,
    ModelLoadError,
    WeightsDownloadError,
)
from .person_detector import PersonDetector
from .results import DetectionResult, PersonDetection

__all__ = [
    "DEVICES",
    "DetectionConfig",
    "DetectionConfigError",
    "DetectionError",
    "DetectionResult",
    "DeviceUnavailableError",
    "InferenceError",
    "InvalidFrameError",
    "ModelLoadError",
    "PersonDetection",
    "PersonDetector",
    "ResolvedDevice",
    "WeightsDownloadError",
    "resolve_device",
]
