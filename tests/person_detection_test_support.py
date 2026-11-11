"""İnsan tespiti testleri için ortak yardımcılar (test modülü değildir; ``test_*.py`` desenine uymaz).

* ``FakeYoloModel``: Ultralytics ``YOLO`` nesnesini taklit eder (``names`` + ``predict``). Gerçek model yüklenmez,
  ağırlık indirilmez, internet gerekmez.
* ``FakeBoxes`` / ``FakeYoloResult``: Ultralytics ``Results``/``Boxes`` çıktısını taklit eder.
* ``FakeTensor``: ``detach().cpu().numpy()`` zincirini destekleyen torch benzeri sahte tensör.
* ``make_detector``: sahte modelle ``PersonDetector`` üretir (geçici, boş bir ``.pt`` dosyasıyla).

Yalnızca sahte modeller, GERÇEK YOLO çıkarımının çalıştığını kanıtlamaz; gerçek çıkarım ``test_person_detection_real_model.py``
dosyasında ayrıca doğrulanır.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from isg_vision.human_detection.config import DetectionConfig, ResolvedDevice
from isg_vision.human_detection.person_detector import PersonDetector

COCO_LIKE_NAMES = {0: "person", 1: "bicycle", 2: "car"}


class FakeTensor:
    """``torch.Tensor`` benzeri: ``detach().cpu().numpy()`` zinciri. Kaç kez ``cpu()`` çağrıldığını sayar."""

    def __init__(self, array: Any) -> None:
        self._array = np.asarray(array)
        self.cpu_calls = 0

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        self.cpu_calls += 1
        return self

    def numpy(self) -> np.ndarray:
        return self._array

    def __len__(self) -> int:
        return len(self._array)


class FakeBoxes:
    def __init__(self, xyxy: Any, conf: Any, cls: Any, *, as_tensor: bool = False) -> None:
        wrap: Callable[[Any], Any] = FakeTensor if as_tensor else np.asarray
        self.xyxy = wrap(np.asarray(xyxy, dtype=np.float64).reshape(-1, 4))
        self.conf = wrap(np.asarray(conf, dtype=np.float64).reshape(-1))
        self.cls = wrap(np.asarray(cls, dtype=np.float64).reshape(-1))
        self._n = len(np.asarray(conf).reshape(-1))

    def __len__(self) -> int:
        return self._n


class FakeYoloResult:
    def __init__(self, boxes: FakeBoxes | None) -> None:
        self.boxes = boxes


Detection = tuple[Sequence[float], float, int]  # (xyxy, güven, sınıf)


class FakeYoloModel:
    """``YOLO`` taklidi.

    Args:
        script: Her ``predict`` çağrısı için dönecek tespit listeleri (liste bitince son eleman tekrarlanır;
            ``None`` = ``boxes=None`` döndür).
        names: Sınıf haritası.
        honor_classes: ``True`` ise ``classes=`` filtresi uygulanır (gerçek Ultralytics gibi). ``False`` ise model
            filtreyi yok sayar; böylece ``PersonDetector``'ın kendi ikinci süzme katmanı sınanır.
        as_tensor: Çıktıları torch benzeri ``FakeTensor`` olarak ver.
        error: Verilirse ``predict`` bu hatayı fırlatır.
    """

    def __init__(
        self,
        script: Sequence[Sequence[Detection] | None] = ((),),
        names: dict[int, str] | list[str] | None = None,
        *,
        honor_classes: bool = False,
        as_tensor: bool = False,
        error: Exception | None = None,
    ) -> None:
        self.script = list(script)
        self.names = COCO_LIKE_NAMES if names is None else names
        self.honor_classes = honor_classes
        self.as_tensor = as_tensor
        self.error = error
        self.predict_calls: list[dict[str, Any]] = []

    def predict(self, source: Any = None, **kwargs: Any) -> list[FakeYoloResult]:
        self.predict_calls.append({"source": source, **kwargs})
        if self.error is not None:
            raise self.error
        step = self.script[min(len(self.predict_calls) - 1, len(self.script) - 1)]
        if step is None:
            return [FakeYoloResult(None)]
        rows = list(step)
        classes = kwargs.get("classes")
        if self.honor_classes and classes is not None:
            rows = [r for r in rows if r[2] in classes]
        if not rows:
            return [FakeYoloResult(FakeBoxes(np.zeros((0, 4)), [], [], as_tensor=self.as_tensor))]
        xyxy = [r[0] for r in rows]
        return [FakeYoloResult(FakeBoxes(xyxy, [r[1] for r in rows], [r[2] for r in rows], as_tensor=self.as_tensor))]


class LoaderSpy:
    """``model_loader`` taklidi: çağrı sayısını ve argümanlarını kaydeder."""

    def __init__(self, model: Any | Exception) -> None:
        self.model = model
        self.calls: list[tuple[Path, str]] = []

    def __call__(self, path: Path, device: str) -> Any:
        self.calls.append((path, device))
        if isinstance(self.model, Exception):
            raise self.model
        return self.model


class SequenceClock:
    """Her çağrıda sıradaki değeri döndüren sahte sayaç (süre hesabını deterministik sınamak için)."""

    def __init__(self, values: Sequence[float] | None = None, step: float = 0.010) -> None:
        self._values = list(values) if values is not None else None
        self._step = step
        self._now = 0.0

    def __call__(self) -> float:
        if self._values is not None:
            return self._values.pop(0)
        self._now += self._step
        return self._now


def fake_weights_file(directory: Path, name: str = "fake.pt") -> Path:
    """Boş bir sahte ``.pt`` dosyası (yalnızca 'dosya var' denetimi için; içerik hiç okunmaz)."""
    path = directory / name
    path.write_bytes(b"not-a-real-model")
    return path


def make_detector(
    model: Any,
    directory: Path,
    *,
    device: str = "cpu",
    conf: float = 0.25,
    clock: Callable[[], float] | None = None,
    synchronize: Callable[[str], None] | None = None,
    device_resolver: Callable[[str], ResolvedDevice] | None = None,
) -> tuple[PersonDetector, LoaderSpy]:
    spy = LoaderSpy(model)
    config = DetectionConfig(model_path=fake_weights_file(directory), conf=conf, device=device)
    kwargs: dict[str, Any] = {"model_loader": spy}
    if clock is not None:
        kwargs["clock"] = clock
    if synchronize is not None:
        kwargs["synchronize"] = synchronize
    if device_resolver is not None:
        kwargs["device_resolver"] = device_resolver
    return PersonDetector(config, **kwargs), spy


class TempDirMixin:
    """Her test için geçici bir dizin (test sonunda silinir)."""

    def setUp(self) -> None:  # noqa: N802
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:  # noqa: N802
        self._tmp.cleanup()


def blank_frame(width: int = 64, height: int = 48) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------- kare akışı döngüsü/CLI testleri için sahte tespitçi
class FakeDetector:
    """``PersonDetector`` yerine kullanılan sahte tespitçi (pipeline ve CLI testleri için; model/torch yoktur).

    Args:
        counts: Kare başına döndürülecek kişi sayıları (liste bitince son eleman tekrarlanır).
        clock: Verilirse her ``detect`` çağrısında saati ``inference_s`` kadar ilerletir (çıkarım süresini taklit eder).
        inference_s: Her çıkarımın süresi (saniye); sonucun ``inference_ms`` değeri de budur.
        load_error: ``load`` çağrısında fırlatılır.
        detect_errors: ``{çağrı_sırası(1'den): Exception}``.
    """

    def __init__(
        self,
        counts: Sequence[int] = (1,),
        *,
        clock: Any | None = None,
        inference_s: float = 0.040,
        device: str = "cpu",
        device_warning: str | None = None,
        load_error: BaseException | None = None,
        detect_errors: dict[int, BaseException] | None = None,
    ) -> None:
        self.counts = list(counts)
        self.clock = clock
        self.inference_s = inference_s
        self.device = device
        self.device_warning = device_warning
        self.load_error = load_error
        self.detect_errors = detect_errors or {}
        self.load_seconds: float | None = 0.01
        self.load_calls = 0
        self.warmup_calls: list[tuple[int, int]] = []
        self.close_calls = 0
        self.detect_calls: list[dict[str, Any]] = []
        self.config: Any | None = None

    def load(self) -> None:
        self.load_calls += 1
        if self.load_error is not None:
            raise self.load_error

    def warmup(self, image_size: tuple[int, int] = (640, 480), runs: int = 1) -> float:
        self.warmup_calls.append(tuple(image_size))
        return 0.01

    def close(self) -> None:
        self.close_calls += 1

    def detect(self, image: Any, *, frame_index: int | None = None, timestamp: float | None = None):  # type: ignore[no-untyped-def]
        from isg_vision.human_detection.results import DetectionResult, PersonDetection

        call = len(self.detect_calls) + 1
        self.detect_calls.append({"image": image, "frame_index": frame_index, "timestamp": timestamp})
        if call in self.detect_errors:
            raise self.detect_errors[call]
        if self.clock is not None:
            self.clock.advance(self.inference_s)
        height, width = image.shape[:2]
        n = self.counts[min(call - 1, len(self.counts) - 1)]
        people = tuple(
            PersonDetection(0, "person", 0.9 - 0.01 * i, (2.0 + 3 * i, 5.0, 2.0 + 3 * i + 2, height - 2.0),
                            frame_index=frame_index, timestamp=timestamp)
            for i in range(n)
            if 2.0 + 3 * i + 2 < width
        )
        return DetectionResult(people, (width, height), frame_index, timestamp, self.inference_s * 1000.0, self.device)
