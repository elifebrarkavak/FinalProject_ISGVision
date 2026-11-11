"""Kamera/video kaynağı yapılandırması ve doğrulaması.

Bu modül ``cv2`` import etmez ve hiçbir kaynak açmaz; yalnızca değerleri doğrular.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import SourceConfigError, VideoFileError
from .frame import SOURCE_KIND_CAMERA, SOURCE_KIND_VIDEO, SOURCE_KINDS

DEFAULT_CAMERA_INDEX = 0
MAX_CAMERA_INDEX = 99
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
MIN_DIMENSION = 16
MAX_DIMENSION = 7680
BACKENDS = ("auto", "avfoundation", "any")


@dataclass(frozen=True)
class SourceConfig:
    """Kare kaynağının (kamera veya video dosyası) doğrulanmış yapılandırması.

    Doğrudan oluşturmak yerine ``SourceConfig.camera(...)`` veya ``SourceConfig.video(...)`` kullanın.

    Attributes:
        kind: ``"camera"`` veya ``"video"``.
        camera_index: Kamera indeksi (yalnızca kamera için).
        video_path: Video dosyası yolu (yalnızca video için).
        width: İstenen kamera genişliği (piksel). ``None`` = cihazın varsayılanı. Gerçekte uygulanan değer
            cihaza bağlıdır ve ``FrameSource.actual_resolution`` ile okunur. Video için verilemez.
        height: İstenen kamera yüksekliği (``width`` ile birlikte verilmelidir).
        backend: ``"auto"`` (macOS'ta AVFoundation, diğerlerinde OpenCV varsayılanı),
            ``"avfoundation"`` veya ``"any"``. Video dosyalarında her zaman OpenCV varsayılanı kullanılır.
    """

    kind: str
    camera_index: int = DEFAULT_CAMERA_INDEX
    video_path: Path | None = None
    width: int | None = None
    height: int | None = None
    backend: str = "auto"

    def __post_init__(self) -> None:
        if self.kind not in SOURCE_KINDS:
            raise SourceConfigError(f"Kaynak türü {SOURCE_KINDS} değerlerinden biri olmalıdır: {self.kind!r}")
        if self.backend not in BACKENDS:
            raise SourceConfigError(f"Geçersiz backend {self.backend!r}; geçerli değerler: {BACKENDS}")

        if self.kind == SOURCE_KIND_CAMERA:
            if self.video_path is not None:
                raise SourceConfigError("Kamera kaynağı için video yolu verilemez.")
            _check_camera_index(self.camera_index)
            _check_resolution(self.width, self.height)
        else:
            if self.video_path is None or str(self.video_path).strip() == "":
                raise SourceConfigError("Video kaynağı için bir video dosyası yolu gerekir.")
            if self.width is not None or self.height is not None:
                raise SourceConfigError(
                    "Çözünürlük yalnızca kamera için ayarlanabilir; video dosyası kendi çözünürlüğünde okunur."
                )
            object.__setattr__(self, "video_path", Path(self.video_path).expanduser())

    # ------------------------------------------------------------------ oluşturucular
    @classmethod
    def camera(
        cls,
        index: int = DEFAULT_CAMERA_INDEX,
        *,
        width: int | None = DEFAULT_WIDTH,
        height: int | None = DEFAULT_HEIGHT,
        backend: str = "auto",
    ) -> SourceConfig:
        """Kamera kaynağı yapılandırması (varsayılan istek: 1280x720; ``width=height=None`` = cihaz varsayılanı)."""
        return cls(kind=SOURCE_KIND_CAMERA, camera_index=index, width=width, height=height, backend=backend)

    @classmethod
    def video(cls, path: str | Path) -> SourceConfig:
        """Video dosyası kaynağı yapılandırması (dosyanın varlığı ``FrameSource.open`` sırasında denetlenir)."""
        return cls(kind=SOURCE_KIND_VIDEO, video_path=path)

    # ------------------------------------------------------------------ etiketler
    @property
    def label(self) -> str:
        """FramePacket.source alanına yazılan etiket: ``camera:0`` veya ``video:/yol/klip.mp4``."""
        if self.kind == SOURCE_KIND_CAMERA:
            return f"camera:{self.camera_index}"
        return f"video:{self.video_path}"


def _check_camera_index(index: object) -> None:
    # bool, int'in alt sınıfıdır; True/False'ı indeks olarak kabul etmeyiz.
    if isinstance(index, bool) or not isinstance(index, int):
        raise SourceConfigError(f"Kamera indeksi tam sayı olmalıdır: {index!r}")
    if not 0 <= index <= MAX_CAMERA_INDEX:
        raise SourceConfigError(f"Kamera indeksi 0 ile {MAX_CAMERA_INDEX} arasında olmalıdır: {index}")


def _check_resolution(width: object, height: object) -> None:
    if width is None and height is None:
        return
    if width is None or height is None:
        raise SourceConfigError("Çözünürlük için genişlik ve yükseklik birlikte verilmelidir.")
    for name, value in (("genişlik", width), ("yükseklik", height)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise SourceConfigError(f"Çözünürlük {name} değeri tam sayı olmalıdır: {value!r}")
        if not MIN_DIMENSION <= value <= MAX_DIMENSION:
            raise SourceConfigError(
                f"Çözünürlük {name} değeri {MIN_DIMENSION} ile {MAX_DIMENSION} arasında olmalıdır: {value}"
            )


def validate_video_path(path: Path) -> Path:
    """Video dosyasının var olduğunu ve okunabilir bir dosya olduğunu denetler.

    Raises:
        VideoFileError: Dosya yoksa, klasörse, boşsa veya okunamıyorsa.
    """
    if not path.exists():
        raise VideoFileError(f"Video dosyası bulunamadı: {path}. Yolu ve dosya adını kontrol edin.")
    if not path.is_file():
        raise VideoFileError(f"Video yolu bir dosya değil (klasör olabilir): {path}")
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        raise VideoFileError(f"Video dosyası okunamıyor (izin sorunu olabilir): {path} ({exc})") from exc
    if size == 0:
        raise VideoFileError(f"Video dosyası boş (0 bayt): {path}")
    return path
