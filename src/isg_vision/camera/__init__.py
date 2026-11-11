"""Kamera/video kaynağı işlemleri (yapay zekâ modüllerinden bağımsız).

Bu paket import edildiğinde hiçbir kamera açılmaz ve hiçbir pencere oluşturulmaz.
Ayrıntılar: docs/camera_module.md
"""

from .capture import FrameSource
from .config import SourceConfig
from .errors import (
    CameraOpenError,
    FrameReadError,
    SourceConfigError,
    SourceDisconnectedError,
    SourceError,
    VideoFileError,
)
from .frame import FramePacket

__all__ = [
    "CameraOpenError",
    "FramePacket",
    "FrameReadError",
    "FrameSource",
    "SourceConfig",
    "SourceConfigError",
    "SourceDisconnectedError",
    "SourceError",
    "VideoFileError",
]
