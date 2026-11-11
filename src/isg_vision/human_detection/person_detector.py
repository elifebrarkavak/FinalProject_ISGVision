"""YOLO ile insan (``person``) tespiti.

``PersonDetector`` yalnızca "görüntü içeri, insan kutuları dışarı" işini yapar; kamera okumaz, görüntü çizmez, pencere
açmaz. Kare kaynağından bağımsızdır: ``detect`` bir BGR ``numpy`` görüntüsü alır (kamera, video, IP kamera fark etmez).

Önemli tasarım kararları
------------------------
* **Model uygulama başına bir kez yüklenir** (``load``); her karede yeniden yüklenmez. ``load`` tekrar çağrılırsa bir şey yapmaz.
* **Ağ bağlantısı kurulmaz ve indirme yapılmaz.** Ultralytics ``import`` sırasında varsayılan olarak DNS yoklaması yapar
  (``is_online``); bunu önlemek için model yüklenmeden önce ``YOLO_OFFLINE=1`` ayarlanır. Ağırlık dosyası yoksa indirilmez,
  anlaşılır bir hata verilir (indirme ayrı ve açık bir adımdır: ``scripts/download_person_model.py``).
* **``person`` sınıf kimliği modelin metadata'sından (``model.names``) doğrulanır**; COCO'nun ``0`` değeri varsayılmaz.
  Yalnızca ``person`` işlenir: model çağrısına ``classes=[person_id]`` verilir ve çıktı ayrıca süzülür (iki katman).
* **Süre ölçümü** (``DetectionResult.inference_ms``): ``time.perf_counter`` (monotonic, yüksek çözünürlüklü) ile, model
  çağrısından önce başlar; sonuçlar CPU'ya alınıp ``PersonDetection``'a dönüştürüldükten sonra biter. MPS asenkron çalıştığı
  için bitişten önce ``torch.mps.synchronize()`` çağrılır. Kapsam: ön işleme + ileri geçiş + NMS + CPU'ya aktarım.
  Kare okuma, çizim ve görüntüleme dâhil değildir.
* İlk çıkarım (özellikle MPS'te) belirgin biçimde yavaştır; ``warmup`` bu maliyeti sürekli performanstan ayrı ölçer.

Bu modül import edildiğinde ``torch``/``ultralytics`` yüklenmez, model açılmaz, kamera açılmaz.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .config import DetectionConfig, ResolvedDevice, resolve_device
from .errors import InferenceError, InvalidFrameError, ModelLoadError
from .results import DetectionResult, build_detections

PERSON_CLASS_NAME = "person"
DOWNLOAD_HINT = (
    "Resmî ağırlığı indirmek için: .venv/bin/python scripts/download_person_model.py "
    "(yalnızca bu betik ağ kullanır; kaynak ve SHA-256 bilgisi yazdırılır)."
)

ModelLoader = Callable[[Path, str], Any]


def load_ultralytics_model(path: Path, device: str) -> Any:
    """Gerçek Ultralytics YOLO modelini yükler (``device`` çıkarım sırasında uygulanır).

    ``YOLO_OFFLINE=1`` ayarlanır: Ultralytics'in import sırasındaki DNS yoklaması ve olası otomatik indirme/kurulum
    denemeleri kapatılır. ``ultralytics`` burada (çağrı anında) import edilir.
    """
    os.environ.setdefault("YOLO_OFFLINE", "1")
    from ultralytics import YOLO  # noqa: PLC0415 (bilinçli gecikmeli import)

    return YOLO(str(path), task="detect")


def torch_synchronize(device: str) -> None:
    """Asenkron cihazlarda (MPS/CUDA) bekleyen işleri bitirir; CPU'da hiçbir şey yapmaz (torch import edilmez)."""
    if device == "cpu":
        return
    import torch  # noqa: PLC0415

    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


class PersonDetector:
    """Önceden eğitilmiş YOLO modeliyle karedeki tüm insanları tespit eder.

    Args:
        config: Doğrulanmış tespit yapılandırması (``None`` = varsayılanlar).
        model_loader: ``(ağırlık_yolu, cihaz) -> model`` fonksiyonu; testlerde sahte model vermek için. Dönen nesne
            ``names`` (sınıf kimliği -> ad) ve Ultralytics uyumlu ``predict(...)`` sağlamalıdır.
        device_resolver: ``config.device`` değerini gerçek cihaza çeviren fonksiyon (varsayılan ``resolve_device``).
        clock: Süre ölçümü için monotonic sayaç (varsayılan ``time.perf_counter``).
        synchronize: ``(cihaz) -> None``; asenkron cihazlarda ölçüm bitişinden önce çağrılır.
    """

    def __init__(
        self,
        config: DetectionConfig | None = None,
        *,
        model_loader: ModelLoader = load_ultralytics_model,
        device_resolver: Callable[[str], ResolvedDevice] = resolve_device,
        clock: Callable[[], float] = time.perf_counter,
        synchronize: Callable[[str], None] = torch_synchronize,
    ) -> None:
        self.config = config or DetectionConfig()
        self._model_loader = model_loader
        self._device_resolver = device_resolver
        self._clock = clock
        self._synchronize = synchronize
        self._model: Any | None = None
        self._resolved: ResolvedDevice | None = None
        self._person_id: int | None = None
        self._names: dict[int, str] = {}
        self._load_seconds: float | None = None
        self._warmup_seconds: float | None = None
        self.load_count = 0  # model gerçekten kaç kez yüklendi (testlerde "yalnızca bir kez" denetimi için)

    # ------------------------------------------------------------------ durum
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        """Çıkarımın çalıştığı cihaz (``cpu``/``mps``); ``load`` çağrılana kadar ``config.device`` çözülmemiştir."""
        return self._resolve().device

    @property
    def device_warning(self) -> str | None:
        """``auto`` seçilip MPS yoksa CPU'ya dönüş uyarısı; aksi halde ``None``."""
        return self._resolve().warning

    @property
    def person_class_id(self) -> int:
        """Modelin metadata'sından bulunan ``person`` sınıf kimliği (yüklenmeden önce hata verir)."""
        if self._person_id is None:
            raise ModelLoadError("Model henüz yüklenmedi; önce load() çağırın.")
        return self._person_id

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def load_seconds(self) -> float | None:
        """Model yükleme süresi (sn); sürekli çıkarım performansından ayrı tutulur. Yüklenmediyse ``None``."""
        return self._load_seconds

    @property
    def warmup_seconds(self) -> float | None:
        """Isınma çıkarımlarının toplam süresi (sn); yapılmadıysa ``None``."""
        return self._warmup_seconds

    # ------------------------------------------------------------------ yükleme
    def load(self) -> None:
        """Modeli (yalnızca bir kez) yükler ve ``person`` sınıf kimliğini doğrular.

        Raises:
            DeviceUnavailableError: ``mps`` açıkça istendi ama kullanılamıyor.
            ModelLoadError: Ağırlık dosyası yok/bozuk/uyumsuz veya modelde ``person`` sınıfı yok.
        """
        if self._model is not None:
            return
        resolved = self._resolve()  # cihaz hatası, dosya denetiminden ve yüklemeden önce ortaya çıkar
        path = self.config.model_path
        if not path.is_file():
            raise ModelLoadError(f"Model ağırlık dosyası bulunamadı: {path}. {DOWNLOAD_HINT}")

        start = self._clock()
        try:
            model = self._model_loader(path, resolved.device)
        except Exception as exc:  # bozuk dosya, uyumsuz sürüm, torch hatası...
            raise ModelLoadError(
                f"Model yüklenemedi: {path} ({type(exc).__name__}: {exc}). Dosya bozuk veya yüklü Ultralytics "
                "sürümüyle uyumsuz olabilir."
            ) from exc

        names = _names_of(model)
        person_id = _find_person_id(names)
        if person_id is None:
            raise ModelLoadError(
                f"Model sınıf haritasında '{PERSON_CLASS_NAME}' sınıfı bulunamadı ({path.name}; sınıflar: "
                f"{sorted(names.values())[:10]}...). İnsan tespiti için COCO gibi 'person' sınıfı olan bir model gerekir."
            )
        self._model, self._names, self._person_id = model, names, person_id
        self._load_seconds = self._clock() - start
        self.load_count += 1

    def warmup(self, image_size: tuple[int, int] = (640, 480), runs: int = 1) -> float:
        """Boş bir görüntüyle ısınma çıkarımı yapar ve toplam süreyi (sn) döndürür.

        Args:
            image_size: ``(genişlik, yükseklik)``; gerçek kare boyutunu verin (girdi şekline özel kernel/derleme
                maliyeti ilk gerçek karede yaşanmasın).
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
    def detect(self, image: np.ndarray, *, frame_index: int | None = None, timestamp: float | None = None) -> DetectionResult:
        """Bir BGR görüntüdeki tüm insanları tespit eder. Girdi görüntüsü değiştirilmez.

        Args:
            image: ``uint8`` BGR görüntü, şekil ``(yükseklik, genişlik, 3)`` (OpenCV/``FramePacket.frame`` biçimi).
            frame_index: Sonuca yazılacak kare sıra numarası (isteğe bağlı).
            timestamp: Sonuca yazılacak kare zaman damgası (isteğe bağlı; ``FramePacket.timestamp``).

        Returns:
            Tespit sonucu; insan yoksa ``detections`` boştur (hata değildir).

        Raises:
            InvalidFrameError: Görüntü ``None``/boş/yanlış biçimde.
            InferenceError: Model çağrısı beklenmeyen bir hatayla başarısız oldu.
        """
        _validate_image(image)
        self.load()
        height, width = image.shape[:2]

        start = self._clock()
        boxes, confidences, class_ids = self._predict(image)
        # Ölçüm bitişi: sonuçlar CPU'ya alınmış (_predict içinde) ve asenkron iş bitmiştir.
        inference_ms = (self._clock() - start) * 1000.0

        detections, dropped = build_detections(
            boxes,
            confidences,
            class_ids,
            self._names,
            person_class_id=self.person_class_id,
            image_size=(width, height),
            frame_index=frame_index,
            timestamp=timestamp,
        )
        return DetectionResult(
            detections=detections,
            image_size=(width, height),
            frame_index=frame_index,
            timestamp=timestamp,
            inference_ms=inference_ms,
            device=self.device,
            dropped_boxes=dropped,
        )

    def close(self) -> None:
        """Model referansını bırakır (birden fazla kez çağrılması güvenlidir)."""
        self._model = None

    def __enter__(self) -> PersonDetector:
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ iç işlevler
    def _resolve(self) -> ResolvedDevice:
        if self._resolved is None:
            self._resolved = self._device_resolver(self.config.device)
        return self._resolved

    def _predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Model çağrısı; ``(kutular_xyxy[N,4], güven[N], sınıf[N])`` NumPy dizilerini döndürür (CPU'da)."""
        assert self._model is not None  # noqa: S101 (load() sonrası)
        cfg = self.config
        device = self.device
        try:
            results = self._model.predict(
                source=image,
                conf=cfg.conf,
                iou=cfg.iou,
                imgsz=cfg.imgsz,
                device=device,
                classes=[self.person_class_id],
                max_det=cfg.max_det,
                verbose=False,
            )
            first = results[0] if len(results) else None
            boxes_obj = getattr(first, "boxes", None)
            if boxes_obj is None or len(boxes_obj) == 0:
                self._synchronize(device)
                return np.zeros((0, 4), dtype=np.float64), np.zeros(0), np.zeros(0, dtype=np.int64)
            xyxy = _to_numpy(boxes_obj.xyxy).reshape(-1, 4)
            conf = _to_numpy(boxes_obj.conf).reshape(-1)
            cls = _to_numpy(boxes_obj.cls).reshape(-1)
            self._synchronize(device)
        except Exception as exc:
            raise InferenceError(f"Model çıkarımı başarısız oldu ({type(exc).__name__}): {exc}") from exc
        return xyxy, conf, cls


# ---------------------------------------------------------------------- yardımcılar
def _validate_image(image: object) -> None:
    if image is None:
        raise InvalidFrameError("Görüntü None; tespit için bir BGR görüntü gerekir.")
    if not isinstance(image, np.ndarray):
        raise InvalidFrameError(f"Görüntü bir NumPy dizisi olmalıdır: {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidFrameError(f"Görüntü (yükseklik, genişlik, 3) şeklinde olmalıdır: {image.shape}")
    if image.shape[0] < 1 or image.shape[1] < 1 or image.size == 0:
        raise InvalidFrameError(f"Görüntü boş olamaz: {image.shape}")
    if image.dtype != np.uint8:
        raise InvalidFrameError(f"Görüntü uint8 olmalıdır: {image.dtype}")


def _to_numpy(value: Any) -> np.ndarray:
    """Torch tensörünü (herhangi bir cihazda), NumPy dizisini veya listeyi NumPy dizisine çevirir."""
    if hasattr(value, "detach"):  # torch.Tensor
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _names_of(model: Any) -> dict[int, str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    raise ModelLoadError("Model sınıf haritası (names) okunamadı; desteklenmeyen model biçimi.")


def _find_person_id(names: dict[int, str]) -> int | None:
    """Sınıf haritasında adı ``person`` olan sınıfın kimliği; yoksa veya belirsizse ``None``."""
    matches = [cid for cid, name in names.items() if name.strip().lower() == PERSON_CLASS_NAME]
    return matches[0] if len(matches) == 1 else None
