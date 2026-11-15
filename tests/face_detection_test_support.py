"""Yüz tespiti testleri için ortak yardımcılar (test modülü değildir; ``test_*.py`` desenine uymaz).

* ``FakeYuNet``: ``cv2.FaceDetectorYN`` nesnesini taklit eder (``setInputSize`` + ``detect``). Gerçek model yüklenmez, dosya
  okunmaz, ağ kullanılmaz.
* ``yunet_rows``: modelin ``N×15`` çıktı biçimini üretir (``x, y, w, h``, 10 işaret noktası değeri, skor).
* ``LoaderSpy`` / ``make_detector``: sahte modelle ``FaceDetector`` üretir (geçici, boş bir ``.onnx`` dosyasıyla).
* ``FakeFaceDetector``: boru hattı ve CLI testleri için sahte tespitçi (model yok).

Yalnızca sahte modeller GERÇEK yüz tespitinin çalıştığını kanıtlamaz; gerçek çıkarım ``test_face_detection_real_model.py`` dosyasında
ayrıca doğrulanır.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isg_vision.face_detection.config import FaceDetectionConfig  # noqa: E402
from isg_vision.face_detection.detector import FaceDetector  # noqa: E402
from isg_vision.face_detection.results import FaceDetection, FaceDetectionResult  # noqa: E402

Row = tuple[float, float, float, float, float]  # (x, y, w, h, skor) — modelin girdi koordinatlarında


def yunet_rows(rows: Sequence[Row]) -> np.ndarray | None:
    """Gerçek ``FaceDetectorYN.detect`` çıktısı gibi ``N×15`` dizi; yüz yoksa ``None`` (gerçek model de ``None`` döndürür)."""
    if not rows:
        return None
    out = np.zeros((len(rows), 15), dtype=np.float32)
    for i, (x, y, w, h, score) in enumerate(rows):
        out[i, 0:4] = (x, y, w, h)
        out[i, 4:14] = np.arange(10)  # işaret noktaları: bu testlerde kullanılmaz
        out[i, 14] = score
    return out


class FakeYuNet:
    """``cv2.FaceDetectorYN`` taklidi.

    Args:
        script: Her ``detect`` çağrısı için dönecek satır listeleri (liste bitince son eleman tekrarlanır). Bir eleman ``None``
            ise model ``None`` döndürür; bir ``np.ndarray`` ise olduğu gibi döner (bozuk çıktıları sınamak için).
        error: Verilirse ``detect`` bu hatayı fırlatır.
    """

    def __init__(self, script: Sequence[Any] = ((),), *, error: Exception | None = None) -> None:
        self.script = list(script)
        self.error = error
        self.input_sizes: list[tuple[int, int]] = []
        self.detect_calls: list[np.ndarray] = []

    def setInputSize(self, size: tuple[int, int]) -> None:  # noqa: N802 (OpenCV adı)
        self.input_sizes.append(tuple(size))

    def detect(self, image: np.ndarray) -> tuple[int, Any]:
        self.detect_calls.append(image)
        if self.error is not None:
            raise self.error
        step = self.script[min(len(self.detect_calls) - 1, len(self.script) - 1)]
        if step is None or isinstance(step, np.ndarray):
            return 1, step
        return 1, yunet_rows(list(step))


class LoaderSpy:
    """``model_loader`` taklidi: çağrı sayısını ve argümanlarını kaydeder."""

    def __init__(self, model: Any | Exception) -> None:
        self.model = model
        self.calls: list[tuple[Path, FaceDetectionConfig]] = []

    def __call__(self, path: Path, config: FaceDetectionConfig) -> Any:
        self.calls.append((path, config))
        if isinstance(self.model, Exception):
            raise self.model
        return self.model


def fake_model_file(directory: Path, name: str = "fake.onnx", content: bytes = b"not-a-real-model") -> Path:
    """Boş/sahte bir ``.onnx`` dosyası (yalnızca 'dosya var' denetimi için; içerik hiç ayrıştırılmaz)."""
    path = directory / name
    path.write_bytes(content)
    return path


def make_detector(model: Any, directory: Path, *, clock: Any | None = None, **config_overrides: Any) -> tuple[FaceDetector, LoaderSpy]:
    spy = LoaderSpy(model)
    config = FaceDetectionConfig(model_path=fake_model_file(directory), **config_overrides)
    kwargs: dict[str, Any] = {"model_loader": spy}
    if clock is not None:
        kwargs["clock"] = clock
    return FaceDetector(config, **kwargs), spy


def blank_frame(width: int = 64, height: int = 48) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


class FakeFaceDetector:
    """``FaceDetector`` yerine kullanılan sahte tespitçi (boru hattı ve CLI testleri için; model yoktur).

    Args:
        counts: Kare başına döndürülecek yüz sayıları (liste bitince son eleman tekrarlanır).
        clock: Verilirse her ``detect`` çağrısında saati ``inference_s`` kadar ilerletir.
        load_error: ``load`` çağrısında fırlatılır.
        detect_errors: ``{çağrı_sırası(1'den): Exception}``.
    """

    def __init__(
        self,
        counts: Sequence[int] = (1,),
        *,
        clock: Any | None = None,
        inference_s: float = 0.010,
        load_error: BaseException | None = None,
        detect_errors: dict[int, BaseException] | None = None,
        dropped: int = 0,
        truncated: int = 0,
    ) -> None:
        self.counts = list(counts)
        self.clock = clock
        self.inference_s = inference_s
        self.device = "cpu"
        self.device_warning: str | None = None
        self.load_error = load_error
        self.detect_errors = detect_errors or {}
        self.dropped = dropped
        self.truncated = truncated
        self.load_seconds: float | None = 0.002
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

    def detect(self, image: Any, *, frame_index: int | None = None, timestamp: float | None = None) -> FaceDetectionResult:
        call = len(self.detect_calls) + 1
        self.detect_calls.append({"image": image, "frame_index": frame_index, "timestamp": timestamp})
        if call in self.detect_errors:
            raise self.detect_errors[call]
        if self.clock is not None:
            self.clock.advance(self.inference_s)
        height, width = image.shape[:2]
        n = self.counts[min(call - 1, len(self.counts) - 1)]
        faces = tuple(
            FaceDetection(0.9 - 0.01 * i, (2.0 + 10 * i, 5.0, 10.0 + 10 * i, 5.0 + 8), frame_index, timestamp, (width, height), i)
            for i in range(n)
            if 10.0 + 10 * i < width and 13.0 < height
        )
        return FaceDetectionResult(faces, (width, height), frame_index, timestamp, self.inference_s * 1000.0, self.device,
                                   self.dropped, self.truncated, (width, height))
