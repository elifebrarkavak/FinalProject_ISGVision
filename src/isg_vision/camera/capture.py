"""Kamera ve video dosyası için ortak kare okuma altyapısı.

``FrameSource``, ``cv2.VideoCapture`` üzerinde ince bir sarmalayıcıdır. Kamera ve video dosyası aynı arayüzle
kullanılır; kareler ``FramePacket`` olarak verilir. Bu modül hiçbir yapay zekâ modeline bağımlı değildir ve
**import edildiğinde kamera açmaz** — kaynak yalnızca ``open()`` çağrıldığında (veya ``with`` bloğuna girildiğinde)
açılır.

Kullanım::

    config = SourceConfig.camera(0)            # veya SourceConfig.video("klip.mp4")
    with FrameSource(config) as source:
        for packet in source.frames():
            ...                                # packet.frame, packet.frame_index, packet.timestamp
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable, Iterator
from typing import Any

import cv2

from .config import SourceConfig, validate_video_path
from .errors import (
    CameraOpenError,
    FrameReadError,
    SourceConfigError,
    SourceDisconnectedError,
    SourceError,
    VideoFileError,
)
from .frame import SOURCE_KIND_CAMERA, FramePacket

CAMERA_OPEN_HINT = (
    "Olası nedenler: (1) macOS kamera izni verilmemiş: Sistem Ayarları > Gizlilik ve Güvenlik > Kamera "
    "bölümünde bu programı çalıştıran uygulamaya (Terminal, VS Code, Claude vb.) izin verin ve uygulamayı "
    "yeniden başlatın; (2) kamera başka bir uygulama tarafından kullanılıyor (FaceTime, Zoom, Photo Booth...); "
    "(3) bu indekste kamera yok: farklı bir kamera indeksi deneyin (iPhone Süreklilik Kamerası da bir "
    "indeks alabilir)."
)


DEFAULT_MAX_READ_FAILURES = 30
DEFAULT_RETRY_DELAY_S = 0.05
# Video bildirilen kare sayısının bu oranından azıyla bittiyse "beklenenden erken bitti" sayılır.
# (OpenCV'nin kare sayısı bazı biçimlerde yaklaşıktır; küçük sapmalar uyarı üretmesin diye tolerans vardır.)
PREMATURE_END_RATIO = 0.98


def resolve_backend(name: str, platform: str | None = None) -> int:
    """Backend adını OpenCV sabitine çevirir (``auto`` -> macOS'ta AVFoundation, aksi halde varsayılan)."""
    platform = sys.platform if platform is None else platform
    if name == "avfoundation":
        return int(cv2.CAP_AVFOUNDATION)
    if name == "auto" and platform == "darwin":
        return int(cv2.CAP_AVFOUNDATION)
    return int(cv2.CAP_ANY)


class FrameSource:
    """Kamera veya video dosyasından sırayla ``FramePacket`` okur.

    Args:
        config: Doğrulanmış kaynak yapılandırması.
        capture_factory: ``cv2.VideoCapture`` yerine kullanılacak üretici (testlerde sahte kaynak vermek için).
            ``None`` ise çağrı anında ``cv2.VideoCapture`` kullanılır.
        clock: Zaman damgası için monotonic saat (testlerde sahte saat vermek için).
        max_read_failures: Kameradan art arda bu kadar okuma başarısız olursa hata fırlatılır (kamera ısınırken
            veya anlık takılmalarda ilk başarısız okumada pes edilmez; sonsuz döngü de oluşmaz).
        retry_delay_s: Başarısız okumalar arasındaki bekleme süresi (saniye).
        sleep: Bekleme fonksiyonu (testlerde gerçek bekleme yapmamak için).
    """

    def __init__(
        self,
        config: SourceConfig,
        *,
        capture_factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        max_read_failures: int = DEFAULT_MAX_READ_FAILURES,
        retry_delay_s: float = DEFAULT_RETRY_DELAY_S,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if isinstance(max_read_failures, bool) or not isinstance(max_read_failures, int) or max_read_failures < 1:
            raise SourceConfigError(f"max_read_failures pozitif bir tam sayı olmalıdır: {max_read_failures!r}")
        if retry_delay_s < 0:
            raise SourceConfigError(f"retry_delay_s negatif olamaz: {retry_delay_s!r}")
        self.config = config
        self._capture_factory = capture_factory
        self._clock = clock
        self._max_read_failures = max_read_failures
        self._retry_delay_s = retry_delay_s
        self._sleep = sleep if sleep is not None else time.sleep  # çağrı anında çözülür (test edilebilirlik)
        self._cap: Any | None = None
        self._next_index = 0
        self._ended = False
        self._reported_resolution: tuple[int, int] | None = None
        self._frame_size: tuple[int, int] | None = None
        self._source_fps: float | None = None
        self._frame_count: int | None = None

    # ------------------------------------------------------------------ durum
    @property
    def is_open(self) -> bool:
        return self._cap is not None

    @property
    def frames_read(self) -> int:
        """Şimdiye kadar başarıyla okunan kare sayısı."""
        return self._next_index

    @property
    def label(self) -> str:
        return self.config.label

    @property
    def kind(self) -> str:
        return self.config.kind

    @property
    def requested_resolution(self) -> tuple[int, int] | None:
        """İstenen (genişlik, yükseklik); istek yoksa (cihaz varsayılanı / video) ``None``."""
        if self.config.width is None or self.config.height is None:
            return None
        return (self.config.width, self.config.height)

    @property
    def actual_resolution(self) -> tuple[int, int] | None:
        """Gerçekten uygulanan (genişlik, yükseklik).

        İlk kare okunduktan sonra kareden ölçülen gerçek boyuttur; öncesinde sürücünün bildirdiği değerdir
        (bildirmezse ``None``). Kaynak kapatıldıktan sonra da okunabilir.
        """
        return self._frame_size or self._reported_resolution

    @property
    def resolution_mismatch(self) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """İstenen çözünürlük uygulanmadıysa ``(istenen, gerçek)``; aksi halde ``None``."""
        requested, actual = self.requested_resolution, self.actual_resolution
        if requested is not None and actual is not None and requested != actual:
            return requested, actual
        return None

    @property
    def source_fps(self) -> float | None:
        """Kaynağın (kamera sürücüsü veya video dosyası) bildirdiği kare hızı; bilinmiyorsa ``None``.

        Bu, uygulamanın gerçekten işlediği (ölçülen) FPS değildir; bkz. ``fps.FpsMeter``.
        """
        return self._source_fps

    @property
    def frame_count(self) -> int | None:
        """Video dosyasının bildirdiği toplam kare sayısı (kamera ve bilinmeyen durumlarda ``None``)."""
        return self._frame_count

    @property
    def premature_end(self) -> bool:
        """Video, bildirilen kare sayısından belirgin biçimde önce bittiyse ``True`` (kesik/bozuk dosya belirtisi).

        OpenCV, dosya sonu ile okuma hatasını ayırt edemez; bu nedenle bildirilen kare sayısıyla karşılaştırılır.
        Kamera veya kare sayısı bilinmeyen videolar için her zaman ``False``.
        """
        total = self._frame_count
        return self._ended and total is not None and self._next_index < total * PREMATURE_END_RATIO

    # ------------------------------------------------------------------ açma / kapatma
    def open(self) -> None:
        """Kaynağı açar. Başarısız olursa kaynak serbest bırakılır ve açıklayıcı bir ``SourceError`` fırlatılır."""
        if self._cap is not None:
            raise SourceError("Kaynak zaten açık; önce close() çağırın.")

        is_camera = self.config.kind == SOURCE_KIND_CAMERA
        if is_camera:
            args: tuple[Any, ...] = (self.config.camera_index, resolve_backend(self.config.backend))
        else:
            args = (str(validate_video_path(self.config.video_path)),)  # type: ignore[arg-type]

        factory = self._capture_factory or cv2.VideoCapture
        try:
            cap = factory(*args)
        except Exception as exc:  # cv2.error veya beklenmeyen sürücü hatası
            raise self._open_error(f"Kaynak açılırken hata oluştu: {exc}") from exc

        if not cap.isOpened():
            cap.release()
            raise self._open_error("Kaynak açılamadı.")

        self._cap = cap
        self._next_index = 0
        self._ended = False
        self._frame_size = None
        if is_camera and self.requested_resolution is not None:
            # İstek yalnızca bir talebdir; cihaz desteklemiyorsa başka bir çözünürlük seçebilir.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._reported_resolution = self._positive_size(cap)
        self._source_fps = _positive_float(cap.get(cv2.CAP_PROP_FPS))
        count = _positive_float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_camera else None
        self._frame_count = int(count) if count is not None else None

    def close(self) -> None:
        """Kaynağı serbest bırakır. Birden fazla kez çağrılması güvenlidir."""
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()

    def __enter__(self) -> FrameSource:
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ kare okuma
    def read(self) -> FramePacket | None:
        """Bir sonraki kareyi okur.

        Returns:
            Kare paketi; video dosyası bittiyse ``None`` (kamera akışı sonsuzdur, ``None`` dönmez).

        Raises:
            FrameReadError: Kamera açıldı ama hiç kare gelmedi (izin/başka uygulama/yanlış indeks).
            SourceDisconnectedError: Kare akışı başladıktan sonra kamera bağlantısı koptu.
            VideoFileError: Video açıldı ama hiç kare okunamadı (bozuk dosya / desteklenmeyen codec).
            SourceError: Kaynak açık değil.
        """
        cap = self._cap
        if cap is None:
            raise SourceError("Kaynak açık değil; önce open() çağırın veya 'with' bloğunu kullanın.")
        if self._ended:
            return None

        if self.config.kind == SOURCE_KIND_CAMERA:
            frame = self._read_camera_frame(cap)
        else:
            frame = self._read_video_frame(cap)
            if frame is None:
                return None
        timestamp = self._clock()
        wall_time = time.time()

        self._frame_size = (int(frame.shape[1]), int(frame.shape[0]))
        packet = FramePacket(
            frame=frame,
            frame_index=self._next_index,
            timestamp=timestamp,
            wall_time=wall_time,
            source=self.label,
            source_kind=self.config.kind,
        )
        self._next_index += 1
        return packet

    def _read_camera_frame(self, cap: Any) -> Any:
        """Kameradan kare alır; geçici başarısızlıklarda sınırlı sayıda yeniden dener (sonsuz döngü yok)."""
        failures = 0
        last_error: Exception | None = None
        while True:
            try:
                ok, frame = cap.read()
            except cv2.error as exc:
                ok, frame, last_error = False, None, exc
            if ok and frame is not None:
                return frame
            failures += 1
            if failures >= self._max_read_failures or not cap.isOpened():
                raise self._camera_read_failure(failures, last_error)
            self._sleep(self._retry_delay_s)

    def _read_video_frame(self, cap: Any) -> Any:
        """Videodan kare alır; dosya sonunda ``None`` döndürür."""
        try:
            ok, frame = cap.read()
        except cv2.error as exc:
            raise FrameReadError(f"Video okunurken hata oluştu ({self.config.video_path}): {exc}") from exc
        if ok and frame is not None:
            return frame
        self._ended = True
        if self._next_index == 0:
            raise VideoFileError(
                f"Video açıldı ancak hiç kare okunamadı: {self.config.video_path}. "
                "Dosya bozuk olabilir veya codec desteklenmiyor olabilir."
            )
        return None

    def _camera_read_failure(self, failures: int, last_error: Exception | None) -> FrameReadError:
        detail = f" Son hata: {last_error}." if last_error is not None else ""
        if self._next_index == 0:
            return FrameReadError(
                f"Kamera açıldı ancak hiç kare okunamadı ({self.label}; {failures} deneme).{detail} {CAMERA_OPEN_HINT}"
            )
        return SourceDisconnectedError(
            f"Kamera bağlantısı kesildi ({self.label}): {self._next_index} kare okunduktan sonra {failures} "
            f"ardışık okuma başarısız oldu.{detail} Kameranın/iPhone bağlantısının sürdüğünü ve kamerayı başka "
            "bir uygulamanın almadığını kontrol edin."
        )

    def frames(self) -> Iterator[FramePacket]:
        """Kaynak bitene kadar (video) veya hata oluşana / dışarıdan durdurulana kadar (kamera) kare verir."""
        while True:
            packet = self.read()
            if packet is None:
                return
            yield packet

    # ------------------------------------------------------------------ yardımcılar
    def _open_error(self, detail: str) -> SourceError:
        if self.config.kind == SOURCE_KIND_CAMERA:
            return CameraOpenError(f"Kamera {self.config.camera_index} açılamadı. {detail} {CAMERA_OPEN_HINT}")
        return VideoFileError(
            f"Video dosyası açılamadı: {self.config.video_path}. {detail} "
            "Dosya bozuk olabilir veya codec desteklenmiyor olabilir (MP4/MOV/AVI desteklenir)."
        )

    @staticmethod
    def _positive_size(cap: Any) -> tuple[int, int] | None:
        width = _positive_float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = _positive_float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width is None or height is None:
            return None
        return int(width), int(height)


def _positive_float(value: object) -> float | None:
    """Sürücünün bildirdiği değeri sayıya çevirir; 0, negatif, NaN veya sonsuz değerler "bilinmiyor" sayılır."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None
