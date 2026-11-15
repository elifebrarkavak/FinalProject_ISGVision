"""Çoklu yüz tespiti (Aşama 5A).

Seçilen model: YuNet (OpenCV ``cv2.FaceDetectorYN``, MIT lisanslı ONNX, yalnızca CPU). Bu paket yalnızca yüz TESPİTİ yapar:

* yüzleri bir kişiye veya Track ID'ye bağlamaz (Aşama 5B), yüz tanıma/çalışan kimliği/embedding/biyometrik şablon üretmez (Aşama 6-7);
* görüntü kaydetmez, ağ kullanmaz (model indirme ayrı ve açık bir betiktir: ``scripts/download_face_model.py``);
* yüz tespiti bir kimlik veya canlı insan doğrulaması değildir: poster/fotoğraf/ekrandaki yüz de tespit edilebilir.

Ayrıntı: ``docs/face_detection.md``. Bu dosya hafiftir: model açmaz, kamera açmaz, ağ kullanmaz.
"""

from .config import (
    DEFAULT_CONF,
    DEFAULT_MAX_FACES,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PATH,
    DEFAULT_NMS,
    FaceDetectionConfig,
)
from .detector import FaceDetector
from .errors import (
    FaceConfigError,
    FaceDetectionError,
    FaceInferenceError,
    FaceInvalidFrameError,
    FaceModelLoadError,
    FaceWeightsDownloadError,
)
from .results import FaceDetection, FaceDetectionResult

__all__ = [
    "DEFAULT_CONF",
    "DEFAULT_MAX_FACES",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_NMS",
    "FaceConfigError",
    "FaceDetection",
    "FaceDetectionConfig",
    "FaceDetectionError",
    "FaceDetectionResult",
    "FaceDetector",
    "FaceInferenceError",
    "FaceInvalidFrameError",
    "FaceModelLoadError",
    "FaceWeightsDownloadError",
]
