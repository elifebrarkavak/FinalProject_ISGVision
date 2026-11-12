"""Aşama 4B testleri için ortak yardımcılar: betiklenmiş (sahte) tespit dizileri, sahte backend ve kimlik eşleme.

DİKKAT: Burada üretilen tespitler SAHTEDİR (YOLO çalışmaz). Bunlarla yapılan testler takipçi mantığını ve (``bytetrack_real`` dosyasında)
gerçek ByteTrack algoritmasını sınar; **gerçek YOLO + gerçek kamera başarısı kanıtı DEĞİLDİR**.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isg_vision.human_detection.results import DetectionResult, PersonDetection  # noqa: E402
from isg_vision.tracking.bytetrack_adapter import BackendFrame, PersonTracker  # noqa: E402
from isg_vision.tracking.config import TrackingConfig  # noqa: E402
from isg_vision.tracking.metrics import IdSwitchReport, count_id_switches  # noqa: E402
from isg_vision.tracking.results import TrackingResult  # noqa: E402

W, H = 640, 480
PERSON_W, PERSON_H = 60.0, 170.0

Item = tuple[str, tuple[float, float, float, float], float]  # (gerçek kimlik adı, kutu xyxy, güven)


def box_at(cx: float, bottom: float = 420.0, w: float = PERSON_W, h: float = PERSON_H) -> tuple[float, float, float, float]:
    """Merkez x ve alt kenar y'ye göre insan boyutunda kutu (xyxy)."""
    return (cx - w / 2, bottom - h, cx + w / 2, bottom)


def make_result(
    items: Sequence[Item],
    frame_index: int | None,
    *,
    size: tuple[int, int] = (W, H),
    timestamp: float | None = None,
) -> tuple[DetectionResult, list[str]]:
    """Betiklenmiş tespitlerden ``DetectionResult`` ve tespit sırasıyla hizalı kimlik adı listesi üretir (sıra olduğu gibi korunur)."""
    detections = tuple(
        PersonDetection(0, "person", conf, box, frame_index=frame_index, timestamp=timestamp) for _, box, conf in items
    )
    return DetectionResult(detections, size, frame_index, timestamp, 5.0, "cpu"), [name for name, _, _ in items]


def assignment(result: TrackingResult, names: Sequence[str]) -> dict[str, int]:
    """Aktif izlerin ``{gerçek kimlik: track_id}`` eşlemesi (``detection_index`` ile; kutu eşleştirmesi tahmini YOK)."""
    return {names[t.detection_index]: t.track_id for t in result.tracks}


@dataclass
class Outcome:
    results: list[TrackingResult] = field(default_factory=list)
    assignments: list[dict[str, int]] = field(default_factory=list)

    @property
    def report(self) -> IdSwitchReport:
        return count_id_switches(self.assignments)

    def ids(self, identity: str) -> list[int | None]:
        return [a.get(identity) for a in self.assignments]

    def assert_unique_ids_every_frame(self, case: Any) -> None:
        for r in self.results:
            case.assertEqual(len(set(r.track_ids)), len(r.track_ids), f"aynı karede yinelenen ID: kare {r.frame_index} {r.track_ids}")


def run_frames(
    tracker: PersonTracker,
    frames: Sequence[Sequence[Item]],
    *,
    start_index: int = 0,
    source: str | None = None,
    size: tuple[int, int] = (W, H),
) -> Outcome:
    """Kare kare tespit listesini takipçiye verir; her karede sonuç ve kimlik eşlemesini kaydeder."""
    outcome = Outcome()
    for offset, items in enumerate(frames):
        result, names = make_result(items, start_index + offset, size=size, timestamp=(start_index + offset) / 30.0)
        tracking = tracker.update(result, source=source)
        outcome.results.append(tracking)
        outcome.assignments.append(assignment(tracking, names))
    return outcome


def make_tracker(**config_overrides: Any) -> PersonTracker:
    """Gerçek ByteTrack'li takipçi (``lap`` ve ``ultralytics`` kurulu olmalıdır)."""
    return PersonTracker(TrackingConfig(**config_overrides))


def lerp(a: float, b: float, i: int, n: int) -> float:
    """``i`` 0..n-1 için a'dan b'ye doğrusal ara değer."""
    return a + (b - a) * (i / (n - 1) if n > 1 else 0.0)


# ---------------------------------------------------------------------- sahte backend
class FakeBackend:
    """Sahte ByteTrack backend'i: betiklenmiş ``BackendFrame``'ler döndürür, çağrıları kaydeder (algoritma YOKTUR)."""

    def __init__(self, frames: Sequence[BackendFrame | Exception | Callable[[np.ndarray], BackendFrame]] = ()) -> None:
        self.frames = list(frames)
        self.calls: list[dict[str, Any]] = []
        self.reset_calls = 0
        self.versions = {"ultralytics": "fake", "lap": "fake"}
        self.version_note: str | None = None

    def update(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray, image_hw: tuple[int, int]) -> BackendFrame:
        self.calls.append({"xyxy": xyxy.copy(), "conf": conf.copy(), "cls": cls.copy(), "image_hw": image_hw})
        item = self.frames.pop(0) if self.frames else empty_backend_frame()
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(xyxy)
        return item

    def reset(self) -> None:
        self.reset_calls += 1


def empty_backend_frame() -> BackendFrame:
    return BackendFrame(np.zeros((0, 8), dtype=np.float32), {}, (), 0)


def backend_frame(
    rows: Sequence[Sequence[float]],
    *,
    ages: dict[int, int] | None = None,
    lost: Sequence[tuple[int, int]] = (),
    tentative: int = 0,
) -> BackendFrame:
    arr = np.asarray(rows, dtype=np.float32).reshape(-1, 8)
    return BackendFrame(arr, ages if ages is not None else {int(r[4]): 1 for r in arr}, tuple(lost), tentative)


def fake_tracker(backend: FakeBackend, **kwargs: Any) -> PersonTracker:
    return PersonTracker(backend_factory=lambda _config: backend, **kwargs)


class SequenceClock:
    """Her çağrıda sıradaki değeri döndüren sahte sayaç."""

    def __init__(self, values: Sequence[float]) -> None:
        self._values = list(values)

    def __call__(self) -> float:
        return self._values.pop(0)


class EchoBackend:
    """Sahte backend: her tespite ``track_id = tespit sırası + 1`` verir (algoritma YOKTUR; yalnızca pipeline/CLI testleri için)."""

    def __init__(self) -> None:
        self.calls = 0
        self.reset_calls = 0
        self.versions = {"ultralytics": "fake", "lap": "fake"}
        self.version_note: str | None = None

    def update(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray, image_hw: tuple[int, int]) -> BackendFrame:
        self.calls += 1
        rows = [[*xyxy[i].tolist(), i + 1, float(conf[i]), float(cls[i]), i] for i in range(len(conf))]
        return backend_frame(rows)

    def reset(self) -> None:
        self.reset_calls += 1


def echo_tracker(**kwargs: Any) -> PersonTracker:
    backend = EchoBackend()
    tracker = PersonTracker(backend_factory=lambda _config: backend, **kwargs)
    tracker.echo_backend = backend  # type: ignore[attr-defined]
    return tracker


class FakeTracker:
    """``PersonTracker`` yerine kullanılan sahte takipçi (pipeline/CLI testleri; ByteTrack/lap yoktur).

    Args:
        clock: Verilirse her ``update``'te saati ``track_s`` kadar ilerletir (takip süresini taklit eder).
        track_s: Her güncellemenin süresi (saniye); sonucun ``tracking_ms`` değeri de budur.
        load_error: ``load`` çağrısında fırlatılır.
        update_errors: ``{çağrı_sırası(1'den): Exception}``.
        lost: Her sonuca eklenecek kayıp iz sayısı.
    """

    def __init__(
        self,
        config: Any | None = None,
        *,
        clock: Any | None = None,
        track_s: float = 0.003,
        load_error: BaseException | None = None,
        update_errors: dict[int, BaseException] | None = None,
        lost: int = 0,
    ) -> None:
        self.config = config
        self.clock = clock
        self.track_s = track_s
        self.load_error = load_error
        self.update_errors = update_errors or {}
        self.lost = lost
        self.load_calls = self.close_calls = self.reset_calls = 0
        self.updates: list[dict[str, Any]] = []
        self.events: list[str] = []
        self.backend_info: dict[str, Any] = {"versions": {"ultralytics": "fake", "lap": "fake"}, "version_note": None}
        self._session = 1
        self._frames = 0

    def load(self) -> None:
        self.load_calls += 1
        self.events.append("load")
        if self.load_error is not None:
            raise self.load_error

    def reset(self) -> None:
        self.reset_calls += 1
        self.events.append("reset")
        if self._frames:
            self._session += 1
        self._frames = 0

    def close(self) -> None:
        self.close_calls += 1

    def update(self, result: DetectionResult, *, source: str | None = None) -> TrackingResult:
        from isg_vision.tracking.results import LostTrack, TrackedPerson

        call = len(self.updates) + 1
        self.updates.append({"result": result, "source": source})
        self.events.append("update")
        if call in self.update_errors:
            raise self.update_errors[call]
        if self.clock is not None:
            self.clock.advance(self.track_s)
        self._frames += 1
        tracks = tuple(
            TrackedPerson(i + 1, d.class_id, d.class_name, d.confidence, d.bbox_xyxy, result.frame_index or 0, result.timestamp,
                          self._session, i, self._frames)
            for i, d in enumerate(result.detections)
        )
        lost = tuple(LostTrack(100 + k, k + 1) for k in range(self.lost))
        return TrackingResult(tracks, lost, 0, len(result.detections), result.image_size, result.frame_index or 0, result.timestamp,
                              self._session, self.track_s * 1000.0)
