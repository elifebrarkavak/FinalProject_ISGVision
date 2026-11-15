"""YuNet (OpenCV ``FaceDetectorYN``) ile çoklu yüz tespiti.

``FaceDetector`` yalnızca "görüntü içeri, yüz kutuları dışarı" işini yapar; kamera okumaz, görüntü çizmez, pencere açmaz. Kare
kaynağından bağımsızdır: ``detect`` bir BGR ``numpy`` görüntüsü alır (kamera, video, IP kamera fark etmez).

Önemli tasarım kararları
------------------------
* **Model uygulama başına bir kez yüklenir** (``load``); her karede yeniden yüklenmez. ``load`` tekrar çağrılırsa bir şey yapmaz.
* **Ağ bağlantısı kurulmaz ve indirme yapılmaz.** Model dosyası yoksa indirilmez, anlaşılır bir hata verilir (indirme ayrı ve açık bir
  adımdır: ``scripts/download_face_model.py``).
* **Bütünlük denetimi:** dosya adı resmî izin listesindeyse (``weights.OFFICIAL_MODELS``) yükleme sırasında boyutu ve SHA-256'sı
  doğrulanır; uyuşmazlıkta dosya OpenCV'nin ONNX ayrıştırıcısına verilmez. Denetim ~230 KB için milisaniyenin altındadır.
* **Yalnızca CPU.** OpenCV 5.0'ın yeni DNN motoru hesaplama hedefi isteğini yok sayar; bu yüzden cihaz seçeneği yoktur
  (``config`` açıklaması). ``device`` her zaman ``"cpu"``'dur.
* **Girdi doğrulaması bu modülde yapılır:** model yanlış tür/kanal sayısını hata vermeden kabul ettiği için (ölçüldü) geçersiz
  görüntüler burada ``FaceInvalidFrameError`` ile reddedilir.
* **Süre ölçümü** (``FaceDetectionResult.inference_ms``): ``time.perf_counter`` ile; (varsa) yeniden boyutlandırma + girdi boyutu
  ayarı + ``detect`` (ileri geçiş + NMS) + sonuçların NumPy'a alınması. Kamera okuma, çizim ve görüntüleme dâhil değildir.
  Sonuç kutularının doğrulanması/kırpılması (``build_faces``) ölçüme dâhil değildir (ihmal edilebilir).
* İlk çıkarım sonrakilerden yavaş olabilir; ``warmup`` bu maliyeti sürekli performanstan ayrı ölçer.

Bu modül import edildiğinde model açılmaz, kamera açılmaz, ağ kullanılmaz.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import DEVICE, TOP_K_BEFORE_NMS, FaceDetectionConfig
from .errors import FaceInferenceError, FaceInvalidFrameError, FaceModelLoadError
from .results import FaceDetectionResult, build_faces
from .weights import OFFICIAL_MODELS, sha256_file

DOWNLOAD_HINT = (
    "Resmî modeli indirmek için: .venv/bin/python scripts/download_face_model.py "
    "(yalnızca bu betik ağ kullanır; kaynak, lisans ve SHA-256 bilgisi yazdırılır)."
)
_YUNET_COLUMNS = 15  # x, y, w, h, 5 işaret noktası (10 değer), skor
_SCORE_COLUMN = 14

ModelLoader = Callable[[Path, FaceDetectionConfig], Any]


def load_yunet_model(path: Path, config: FaceDetectionConfig) -> Any:
    """Gerçek ``cv2.FaceDetectorYN`` nesnesini oluşturur (girdi boyutu ``detect`` sırasında kareye göre ayarlanır)."""
    return cv2.FaceDetectorYN.create(str(path), "", (320, 320), float(config.conf), float(config.nms), TOP_K_BEFORE_NMS)


class FaceDetector:
    """Önceden eğitilmiş YuNet modeliyle karedeki tüm yüzleri tespit eder.

    Args:
        config: Doğrulanmış yapılandırma (``None`` = varsayılanlar).
        model_loader: ``(model_yolu, yapılandırma) -> model`` fonksiyonu; testlerde sahte model vermek için. Dönen nesne
            ``setInputSize((genişlik, yükseklik))`` ve ``detect(görüntü) -> (retval, N×15 dizi | None)`` sağlamalıdır
            (sütunlar: ``x, y, w, h``, 10 işaret noktası değeri, skor).
        clock: Süre ölçümü için monotonic sayaç (varsayılan ``time.perf_counter``).
    """

    def __init__(
        self,
        config: FaceDetectionConfig | None = None,
        *,
        model_loader: ModelLoader = load_yunet_model,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config or FaceDetectionConfig()
        self._model_loader = model_loader
        self._clock = clock
        self._model: Any | None = None
        self._input_size: tuple[int, int] | None = None
        self._load_seconds: float | None = None
        self._warmup_seconds: float | None = None
        self.load_count = 0  # model gerçekten kaç kez yüklendi (testlerde "yalnızca bir kez" denetimi için)

    # ------------------------------------------------------------------ durum
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        """Çıkarımın çalıştığı yer; her zaman ``cpu`` (bkz. modül açıklaması)."""
        return DEVICE

    @property
    def device_warning(self) -> str | None:
        """Cihaz geri dönüş uyarısı yoktur (seçenek sunulmaz); arayüz uyumluluğu için ``None``."""
        return None

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def load_seconds(self) -> float | None:
        """Model yükleme süresi (sn; bütünlük denetimi dâhil); sürekli çıkarım performansından ayrı tutulur."""
        return self._load_seconds

    @property
    def warmup_seconds(self) -> float | None:
        """Isınma çıkarımlarının toplam süresi (sn); yapılmadıysa ``None``."""
        return self._warmup_seconds

    # ------------------------------------------------------------------ yükleme
    def load(self) -> None:
        """Modeli (yalnızca bir kez) yükler.

        Raises:
            FaceModelLoadError: Dosya yok, bütünlük denetimi başarısız, OpenCV dosyayı yükleyemedi veya model beklenen arayüzü
                (``setInputSize``/``detect``) sağlamıyor.
        """
        if self._model is not None:
            return
        path = self.config.model_path
        if not path.is_file():
            raise FaceModelLoadError(f"Yüz tespit modeli bulunamadı: {path}. {DOWNLOAD_HINT}")

        start = self._clock()
        self._verify_integrity(path)
        try:
            model = self._model_loader(path, self.config)
        except Exception as exc:  # bozuk dosya, uyumsuz OpenCV sürümü...
            raise FaceModelLoadError(
                f"Yüz tespit modeli yüklenemedi: {path} ({type(exc).__name__}: {exc}). Dosya bozuk veya yüklü OpenCV "
                f"sürümüyle uyumsuz olabilir. {DOWNLOAD_HINT}"
            ) from exc
        if not callable(getattr(model, "detect", None)) or not callable(getattr(model, "setInputSize", None)):
            raise FaceModelLoadError(f"Model nesnesi beklenen arayüzü (setInputSize/detect) sağlamıyor: {type(model).__name__}")
        self._model = model
        self._input_size = None
        self._load_seconds = self._clock() - start
        self.load_count += 1

    def warmup(self, image_size: tuple[int, int] = (640, 480), runs: int = 1) -> float:
        """Boş bir görüntüyle ısınma çıkarımı yapar ve toplam süreyi (sn) döndürür.

        Args:
            image_size: ``(genişlik, yükseklik)``; gerçek kare boyutunu verin (girdi şekline özel maliyet ilk gerçek karede
                yaşanmasın).
            runs: Isınma çıkarımı sayısı.
        """
        if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
            raise ValueError(f"runs pozitif bir tam sayı olmalıdır: {runs!r}")
        self.load()
        width, height = image_size
        blank = np.zeros((int(height), int(width), 3), dtype=np.uint8)
        start = self._clock()
        for _ in range(runs):
            self._predict(blank)
        self._warmup_seconds = self._clock() - start
        return self._warmup_seconds

    # ------------------------------------------------------------------ tespit
    def detect(self, image: np.ndarray, *, frame_index: int | None = None, timestamp: float | None = None) -> FaceDetectionResult:
        """Bir BGR görüntüdeki tüm yüzleri tespit eder. Girdi görüntüsü değiştirilmez.

        Args:
            image: ``uint8`` BGR görüntü, şekil ``(yükseklik, genişlik, 3)`` (OpenCV/``FramePacket.frame`` biçimi).
            frame_index: Sonuca yazılacak kare sıra numarası (isteğe bağlı).
            timestamp: Sonuca yazılacak kare zaman damgası (isteğe bağlı; ``FramePacket.timestamp``).

        Returns:
            Tespit sonucu; yüz yoksa ``faces`` boştur (hata değildir). Kutular **özgün karenin** piksel koordinatlarındadır.

        Raises:
            FaceInvalidFrameError: Görüntü ``None``/boş/yanlış biçimde.
            FaceInferenceError: Model çağrısı beklenmeyen bir hatayla başarısız oldu veya beklenmeyen çıktı verdi.
        """
        _validate_image(image)
        self.load()
        height, width = image.shape[:2]
        detect_width = self.config.detect_width

        start = self._clock()
        work = image
        run_w, run_h = width, height
        if detect_width is not None and width > detect_width:
            run_w = detect_width
            run_h = max(1, int(round(height * detect_width / width)))
            work = cv2.resize(image, (run_w, run_h), interpolation=cv2.INTER_AREA)
        rows = self._predict(work)
        inference_ms = (self._clock() - start) * 1000.0

        sx, sy = width / run_w, height / run_h
        boxes = [(r[0] * sx, r[1] * sy, (r[0] + r[2]) * sx, (r[1] + r[3]) * sy) for r in rows]
        scores = [r[_SCORE_COLUMN] for r in rows]
        faces, dropped, truncated = build_faces(
            boxes,
            scores,
            image_size=(width, height),
            frame_index=frame_index,
            timestamp=timestamp,
            max_faces=self.config.max_faces,
        )
        return FaceDetectionResult(
            faces=faces,
            image_size=(width, height),
            frame_index=frame_index,
            timestamp=timestamp,
            inference_ms=inference_ms,
            device=self.device,
            dropped_boxes=dropped,
            truncated_faces=truncated,
            inference_size=(run_w, run_h),
        )

    def close(self) -> None:
        """Model referansını bırakır (birden fazla kez çağrılması güvenlidir)."""
        self._model = None
        self._input_size = None

    def __enter__(self) -> FaceDetector:
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ iç işlevler
    def _verify_integrity(self, path: Path) -> None:
        if not self.config.verify_model_hash:
            return
        official = OFFICIAL_MODELS.get(path.name)
        if official is None:
            return  # özel model: sabitlenmiş özet yok
        try:
            size = path.stat().st_size
            digest = sha256_file(path)
        except OSError as exc:
            raise FaceModelLoadError(f"Model dosyası okunamadı: {path} ({exc}). {DOWNLOAD_HINT}") from exc
        if size != official.size_bytes or digest != official.sha256:
            raise FaceModelLoadError(
                f"Model bütünlük denetimi başarısız ({path.name}): beklenen {official.size_bytes} bayt / sha256 {official.sha256}, "
                f"bulunan {size} bayt / sha256 {digest}. Dosya bozuk veya değiştirilmiş; ayrıştırılmadı. Silip yeniden indirin: "
                f"{DOWNLOAD_HINT}"
            )

    def _predict(self, image: np.ndarray) -> list[np.ndarray]:
        """Model çağrısı; her yüz için ``[x, y, w, h, ..., skor]`` satırlarını (modelin girdi koordinatlarında) döndürür."""
        assert self._model is not None  # noqa: S101 (load() sonrası)
        height, width = image.shape[:2]
        try:
            if self._input_size != (width, height):
                self._model.setInputSize((width, height))
                self._input_size = (width, height)
            _, faces = self._model.detect(image)
        except Exception as exc:
            self._input_size = None  # girdi boyutu durumu belirsiz; sonraki karede yeniden ayarlanır
            raise FaceInferenceError(f"Yüz tespiti çıkarımı başarısız oldu ({type(exc).__name__}): {exc}") from exc
        if faces is None:
            return []
        try:
            array = np.asarray(faces, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise FaceInferenceError(f"Model çıktısı sayısal bir diziye çevrilemedi: {type(faces).__name__}") from exc
        if array.ndim != 2 or array.shape[1] < _YUNET_COLUMNS:
            raise FaceInferenceError(f"Beklenmeyen model çıktısı: şekil {array.shape} (N×{_YUNET_COLUMNS} bekleniyordu).")
        return list(array)


# ---------------------------------------------------------------------- yardımcılar
def _validate_image(image: object) -> None:
    if image is None:
        raise FaceInvalidFrameError("Görüntü None; yüz tespiti için bir BGR görüntü gerekir.")
    if not isinstance(image, np.ndarray):
        raise FaceInvalidFrameError(f"Görüntü bir NumPy dizisi olmalıdır: {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise FaceInvalidFrameError(f"Görüntü (yükseklik, genişlik, 3) şeklinde olmalıdır: {image.shape}")
    if image.shape[0] < 1 or image.shape[1] < 1 or image.size == 0:
        raise FaceInvalidFrameError(f"Görüntü boş olamaz: {image.shape}")
    if image.dtype != np.uint8:
        raise FaceInvalidFrameError(f"Görüntü uint8 olmalıdır: {image.dtype}")
