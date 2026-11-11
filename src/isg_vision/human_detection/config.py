"""İnsan tespiti yapılandırması ve cihaz seçimi.

Bu modül ``torch``/``ultralytics`` import etmez, model yüklemez ve ağ bağlantısı kurmaz; yalnızca değerleri doğrular.
Torch yalnızca ``resolve_device`` çağrıldığında (ve ``mps_available`` verilmediyse) içeriden import edilir.

Varsayılan değerler Ultralytics'in varsayılanlarıdır (``conf=0.25``, ``iou=0.7``, ``imgsz=640``); evrensel optimum
değildirler, yalnızca makul bir başlangıç noktasıdır. Kamera yüksekliğine/sahneye göre ayarlanmaları beklenir.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from isg_vision.utils.paths import project_path

from .errors import DetectionConfigError, DeviceUnavailableError

DEVICES = ("auto", "cpu", "mps")
DEFAULT_MODEL_NAME = "yolo11n.pt"
DEFAULT_MODEL_PATH: Path = project_path("models", "human_detection", DEFAULT_MODEL_NAME)
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7
DEFAULT_IMGSZ = 640
DEFAULT_MAX_DET = 100
MIN_IMGSZ = 32
MAX_IMGSZ = 4096
IMGSZ_STRIDE = 32  # YOLO ağları 32'nin katı girdi boyutu ister; sessizce yuvarlamak yerine açıkça reddederiz
MAX_MAX_DET = 1000
MODEL_SUFFIXES = (".pt",)


@dataclass(frozen=True)
class DetectionConfig:
    """Doğrulanmış insan tespiti yapılandırması.

    Attributes:
        model_path: PyTorch ağırlık dosyası (``.pt``). Varsayılan: ``models/human_detection/yolo11n.pt``. Dosyanın
            varlığı yapılandırma sırasında değil, model yüklenirken (``PersonDetector.load``) denetlenir.
        conf: Güven eşiği, ``0 < conf <= 1``. Bu değerin altındaki tespitler atılır. Güven skoru, modelin kendi
            kutusuna ne kadar emin olduğunun göstergesidir; **doğruluk (precision/recall) değildir**.
        iou: NMS IoU eşiği, ``0 < iou <= 1``. Düşük değer üst üste binen kutuları daha agresif eler.
        imgsz: Model girdi boyutu (uzun kenar, piksel); 32'nin katı. Küçük değer hızlıdır ama küçük/uzak insanları
            kaçırabilir.
        device: ``"auto"`` (MPS varsa MPS, yoksa CPU — uyarıyla), ``"cpu"`` veya ``"mps"`` (kullanılamıyorsa hata).
        max_det: Karede tutulacak en fazla tespit sayısı.
    """

    model_path: Path = DEFAULT_MODEL_PATH
    conf: float = DEFAULT_CONF
    iou: float = DEFAULT_IOU
    imgsz: int = DEFAULT_IMGSZ
    device: str = "auto"
    max_det: int = DEFAULT_MAX_DET

    def __post_init__(self) -> None:
        if isinstance(self.model_path, bool) or not isinstance(self.model_path, (str, Path)) or str(self.model_path).strip() == "":
            raise DetectionConfigError(f"Model yolu boş olamaz ve metin/yol olmalıdır: {self.model_path!r}")
        path = Path(self.model_path).expanduser()
        if path.suffix.lower() not in MODEL_SUFFIXES:
            raise DetectionConfigError(
                f"Model dosyası {MODEL_SUFFIXES} uzantılarından biri olmalıdır (PyTorch ağırlığı): {str(path)!r}"
            )
        object.__setattr__(self, "model_path", path)

        _check_unit_interval("conf", self.conf)
        _check_unit_interval("iou", self.iou)
        _check_imgsz(self.imgsz)
        _check_max_det(self.max_det)
        if self.device not in DEVICES:
            raise DetectionConfigError(f"Geçersiz cihaz {self.device!r}; geçerli değerler: {DEVICES}")

    @property
    def model_name(self) -> str:
        """Ağırlık dosyasının adı (ör. ``yolo11n.pt``)."""
        return self.model_path.name


@dataclass(frozen=True)
class ResolvedDevice:
    """Cihaz seçiminin sonucu.

    Attributes:
        requested: Kullanıcının istediği değer (``auto``/``cpu``/``mps``).
        device: Gerçekte kullanılacak cihaz (``cpu`` veya ``mps``).
        warning: Geri dönüş olduysa kullanıcıya gösterilecek uyarı; yoksa ``None``.
    """

    requested: str
    device: str
    warning: str | None = None


def resolve_device(requested: str, *, mps_available: Callable[[], tuple[bool, str]] | None = None) -> ResolvedDevice:
    """İstenen cihazı gerçek cihaza çevirir.

    * ``cpu``  -> her zaman CPU.
    * ``mps``  -> MPS kullanılabilirse MPS; **kullanılamıyorsa** ``DeviceUnavailableError`` (sessizce CPU'ya düşülmez).
    * ``auto`` -> MPS kullanılabilirse MPS; değilse CPU ve bir uyarı.

    Args:
        mps_available: ``(kullanılabilir_mi, neden)`` döndüren fonksiyon (testlerde torch'suz çalışmak için).
    """
    if requested not in DEVICES:
        raise DetectionConfigError(f"Geçersiz cihaz {requested!r}; geçerli değerler: {DEVICES}")
    if requested == "cpu":
        return ResolvedDevice("cpu", "cpu")

    available, reason = (mps_available or torch_mps_status)()
    if available:
        return ResolvedDevice(requested, "mps")
    if requested == "mps":
        raise DeviceUnavailableError(
            f"MPS (Apple GPU) istendi ancak kullanılamıyor: {reason}. CPU ile çalışmak için --device cpu, "
            "uygun olanı otomatik seçmek için --device auto kullanın."
        )
    return ResolvedDevice(
        "auto", "cpu", f"MPS kullanılamıyor ({reason}); CPU'ya dönüldü. Çıkarım MPS'e göre daha yavaş olabilir."
    )


def torch_mps_status() -> tuple[bool, str]:
    """PyTorch'un MPS durumunu ``(kullanılabilir, neden)`` olarak döndürür. Torch burada (çağrı anında) import edilir."""
    try:
        import torch
    except ImportError:
        return False, "PyTorch kurulu değil"
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False, "bu PyTorch sürümünde MPS desteği yok"
    if not backend.is_built():
        return False, "bu PyTorch derlemesi MPS desteğiyle derlenmemiş"
    if not backend.is_available():
        return False, "macOS/donanım MPS'i bu süreçte sunmuyor (macOS sürümü veya sanal ortam olabilir)"
    return True, "kullanılabilir"


# ---------------------------------------------------------------------- doğrulama yardımcıları
def _check_unit_interval(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DetectionConfigError(f"{name} sayı olmalıdır (0 < {name} <= 1): {value!r}")
    if not 0.0 < float(value) <= 1.0:
        raise DetectionConfigError(f"{name} 0'dan büyük ve 1'e eşit veya küçük olmalıdır: {value!r}")


def _check_imgsz(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DetectionConfigError(f"imgsz tam sayı olmalıdır: {value!r}")
    if not MIN_IMGSZ <= value <= MAX_IMGSZ:
        raise DetectionConfigError(f"imgsz {MIN_IMGSZ} ile {MAX_IMGSZ} arasında olmalıdır: {value}")
    if value % IMGSZ_STRIDE != 0:
        raise DetectionConfigError(f"imgsz {IMGSZ_STRIDE}'nin katı olmalıdır (ör. 320, 480, 640, 960): {value}")


def _check_max_det(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_MAX_DET:
        raise DetectionConfigError(f"max_det 1 ile {MAX_MAX_DET} arasında bir tam sayı olmalıdır: {value!r}")
