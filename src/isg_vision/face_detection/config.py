"""Yüz tespiti yapılandırması.

Bu modül ``cv2``/model import etmez, model yüklemez ve ağ bağlantısı kurmaz; yalnızca değerleri doğrular.

Cihaz seçimi YOKTUR: seçilen model (YuNet, OpenCV ``FaceDetectorYN``) bu ortamda (OpenCV 5.0) yalnızca CPU'da çalışır. OpenCV'nin
yeni DNN motoru ``setPreferableTarget`` isteğini yok sayar (OpenCL hedefi denendi, süre değişmedi) ve klasik motor bu modelle hata
verir (``docs/face_detection.md``). Var olmayan bir hızlandırma seçeneği sunmak yanıltıcı olurdu.

Varsayılan değerler evrensel optimum değil, kısıtlı ölçümlere (``docs/face_detection.md``) dayanan makul başlangıç noktalarıdır:
``conf=0.6`` (örnek görüntülerdeki net yüzler 0,89–0,94, bulanık arka plan yüzü 0,61, yüz içermeyen bölgelerde ≥ 0,4 yanlış pozitif
görülmedi), ``nms=0.3`` (OpenCV/OpenCV Zoo örneğinin varsayılanı).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from isg_vision.utils.paths import project_path

from .errors import FaceConfigError

DEFAULT_MODEL_NAME = "face_detection_yunet_2026may.onnx"
DEFAULT_MODEL_PATH: Path = project_path("models", "face_detection", DEFAULT_MODEL_NAME)
DEFAULT_CONF = 0.6
DEFAULT_NMS = 0.3
DEFAULT_MAX_FACES = 100
MAX_MAX_FACES = 1000
MIN_DETECT_WIDTH = 64
MAX_DETECT_WIDTH = 8192
TOP_K_BEFORE_NMS = 5000  # YuNet'in NMS öncesi tuttuğu en yüksek aday sayısı (OpenCV Zoo örneğiyle aynı); kullanıcıya açılmadı
MODEL_SUFFIXES = (".onnx",)
DEVICE = "cpu"  # tek desteklenen çalışma biçimi (yukarıdaki açıklamaya bakın)


@dataclass(frozen=True)
class FaceDetectionConfig:
    """Doğrulanmış yüz tespiti yapılandırması.

    Attributes:
        model_path: ONNX model dosyası (``.onnx``). Varsayılan: ``models/face_detection/face_detection_yunet_2026may.onnx``.
            Dosyanın varlığı yapılandırma sırasında değil, model yüklenirken (``FaceDetector.load``) denetlenir.
        conf: Güven eşiği, ``0 < conf <= 1``. Bunun altındaki tespitler atılır. Güven skoru modelin kendi kutusuna ne kadar
            emin olduğunun göstergesidir; **doğruluk (precision/recall) değildir** ve gerçek insan yüzü olduğunu kanıtlamaz
            (poster/fotoğraf/ekrandaki yüz de yüksek skor alır).
        nms: NMS IoU eşiği, ``0 < nms <= 1``. Düşük değer üst üste binen kutuları daha agresif eler.
        max_faces: Karede tutulacak en fazla yüz (güvene göre en yüksekler); fazlası sayılır (``truncated_faces``), sessizce atılmaz.
        detect_width: ``None`` ise model kare çözünürlüğünde (yeniden boyutlandırma yok) çalışır. Bir sayı verilirse, kare bu
            genişlikten genişse en-boy oranı korunarak o genişliğe KÜÇÜLTÜLÜR; kutular özgün kare koordinatlarına geri
            çevrilir. Daha hızlıdır ama küçük/uzak yüzler kaçabilir. Kareden dar kareler büyütülmez.
        verify_model_hash: ``True`` ise dosya adı resmî izin listesindeyse (``weights.OFFICIAL_MODELS``) yükleme sırasında
            SHA-256 doğrulanır; uyuşmazlıkta model yüklenmez (ONNX ayrıştırıcıya güvenilmeyen dosya verilmez). İzin
            listesinde olmayan özel modellerde denetim yapılamaz.
    """

    model_path: Path = DEFAULT_MODEL_PATH
    conf: float = DEFAULT_CONF
    nms: float = DEFAULT_NMS
    max_faces: int = DEFAULT_MAX_FACES
    detect_width: int | None = None
    verify_model_hash: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.model_path, bool) or not isinstance(self.model_path, (str, Path)) or str(self.model_path).strip() == "":
            raise FaceConfigError(f"Model yolu boş olamaz ve metin/yol olmalıdır: {self.model_path!r}")
        path = Path(self.model_path).expanduser()
        if path.suffix.lower() not in MODEL_SUFFIXES:
            raise FaceConfigError(f"Model dosyası {MODEL_SUFFIXES} uzantılarından biri olmalıdır (ONNX): {str(path)!r}")
        object.__setattr__(self, "model_path", path)

        _check_unit_interval("conf", self.conf)
        _check_unit_interval("nms", self.nms)
        if isinstance(self.max_faces, bool) or not isinstance(self.max_faces, int) or not 1 <= self.max_faces <= MAX_MAX_FACES:
            raise FaceConfigError(f"max_faces 1 ile {MAX_MAX_FACES} arasında bir tam sayı olmalıdır: {self.max_faces!r}")
        if self.detect_width is not None:
            if isinstance(self.detect_width, bool) or not isinstance(self.detect_width, int):
                raise FaceConfigError(f"detect_width tam sayı veya None olmalıdır: {self.detect_width!r}")
            if not MIN_DETECT_WIDTH <= self.detect_width <= MAX_DETECT_WIDTH:
                raise FaceConfigError(f"detect_width {MIN_DETECT_WIDTH} ile {MAX_DETECT_WIDTH} arasında olmalıdır: {self.detect_width}")
        if not isinstance(self.verify_model_hash, bool):
            raise FaceConfigError(f"verify_model_hash True/False olmalıdır: {self.verify_model_hash!r}")

    @property
    def model_name(self) -> str:
        """Model dosyasının adı (ör. ``face_detection_yunet_2026may.onnx``)."""
        return self.model_path.name


def _check_unit_interval(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise FaceConfigError(f"{name} sayı olmalıdır (0 < {name} <= 1): {value!r}")
    if not 0.0 < float(value) <= 1.0:
        raise FaceConfigError(f"{name} 0'dan büyük ve 1'e eşit veya küçük olmalıdır: {value!r}")
