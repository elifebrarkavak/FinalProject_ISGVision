"""Aşama 5B testleri için ortak yardımcılar (test modülü değildir; ``test_*.py`` desenine uymaz).

DİKKAT: Burada üretilen takip/yüz sonuçları SAHTEDİR (YOLO/ByteTrack/YuNet çalışmaz). Bunlarla yapılan testler ilişkilendirme ve süreklilik MANTIĞINI
(geometri, belirsizlik, ID switch/devralma, oturum, hata yönetimi) sınar; gerçek model + gerçek ByteTrack + gerçek YuNet zinciri
``test_face_association_real_model.py``'dedir. Sahte testler gerçek kamera, doğruluk veya canlı performans kanıtı DEĞİLDİR.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from isg_vision.face_association.associator import FacePersonAssociator  # noqa: E402
from isg_vision.face_association.config import AssociationConfig  # noqa: E402
from isg_vision.face_detection.results import FaceDetection, FaceDetectionResult  # noqa: E402
from isg_vision.human_detection.results import DetectionResult, PersonDetection  # noqa: E402
from isg_vision.tracking.results import TrackedPerson, TrackingResult  # noqa: E402

W, H = 1280, 720
Box = tuple[float, float, float, float]


def person_box(cx: float, top: float = 100.0, w: float = 140.0, h: float = 420.0) -> Box:
    """Ayakta, tam boy kişi kutusu (merkez x, üst kenar)."""
    return (cx - w / 2, top, cx + w / 2, top + h)


def head_box(person: Box, size: float = 54.0, down: float = 0.06, dx: float = 0.0) -> Box:
    """Kişi kutusunun ÜST-ORTASINDA bir yüz kutusu (``down``: kutu yüksekliğine oranla üstten kayma, ``dx``: yatay kayma oranı)."""
    pw, ph = person[2] - person[0], person[3] - person[1]
    cx = (person[0] + person[2]) / 2 + dx * pw
    top = person[1] + down * ph
    return (cx - size / 2, top, cx + size / 2, top + size)


def tracked(track_id: int, box: Box, frame: int = 0, *, session: int = 1, conf: float = 0.9, age: int = 5) -> TrackedPerson:
    return TrackedPerson(track_id, 0, "person", conf, box, frame, frame / 30.0, session, 0, age)


def tracking(
    people: Sequence[tuple[int, Box]], frame: int = 0, *, size: tuple[int, int] = (W, H), session: int = 1,
    timestamp: float | None = ..., reset_reason: str | None = None,  # type: ignore[assignment]
) -> TrackingResult:
    ts = frame / 30.0 if timestamp is ... else timestamp
    tracks = tuple(TrackedPerson(tid, 0, "person", 0.9, box, frame, ts, session, i, 5) for i, (tid, box) in enumerate(people))
    return TrackingResult(tracks, (), 0, len(tracks), size, frame, ts, session, 0.4, 0, reset_reason)


def face(box: Box, conf: float = 0.92, frame: int | None = 0, *, size: tuple[int, int] | None = (W, H), index: int | None = None, at_border: bool = False,
         timestamp: float | None = None) -> FaceDetection:
    return FaceDetection(conf, box, frame, timestamp, size, index, at_border)


def faces(boxes: Sequence[Box], frame: int | None = 0, *, size: tuple[int, int] = (W, H), timestamp: float | None = ..., conf: float = 0.92,  # type: ignore[assignment]
          at_border: Sequence[bool] = ()) -> FaceDetectionResult:
    ts = (frame / 30.0 if frame is not None else None) if timestamp is ... else timestamp
    detections = tuple(
        FaceDetection(conf - 0.001 * i, box, frame, ts, size, i, bool(at_border[i]) if i < len(at_border) else False) for i, box in enumerate(boxes)
    )
    return FaceDetectionResult(detections, size, frame, ts, 11.0, "cpu", 0, 0, size)


def associate_once(people: Sequence[tuple[int, Box]], face_boxes: Sequence[Box], **config: Any) -> Any:
    """Tek karelik ilişkilendirme (varsayılan yapılandırma + ``config`` geçersiz kılmaları)."""
    associator = FacePersonAssociator(AssociationConfig(**config))
    return associator.associate(tracking(people), faces(face_boxes))


def person_by_id(result: Any, track_id: int) -> Any:
    found = result.person(track_id)
    assert found is not None, f"iz {track_id} sonuçta yok"
    return found


class ScriptedTracker:
    """Betiklenmiş izlerle ``PersonTracker`` taklidi: her ``update`` çağrısında ``frames[i]`` (``[(track_id, box), ...]``) döner (algoritma YOKTUR).

    ``session`` ve ``reset_reason`` kare başına verilebilir. Çağrı sayısını ve aldığı ``DetectionResult``'ları kaydeder.
    """

    def __init__(self, frames: Sequence[Sequence[tuple[int, Box]]], *, sessions: Sequence[int] | None = None, reset_reasons: dict[int, str] | None = None) -> None:
        self.frames = [list(f) for f in frames]
        self.sessions = list(sessions) if sessions is not None else None
        self.reset_reasons = reset_reasons or {}
        self.updates: list[dict[str, Any]] = []
        self.events: list[str] = []
        self.reset_calls = 0
        self.config = None
        self.backend_info: dict[str, Any] = {"versions": {"ultralytics": "fake", "lap": "fake"}, "version_note": None}

    def load(self) -> None:
        self.events.append("load")

    def reset(self) -> None:
        self.reset_calls += 1
        self.events.append("reset")

    def close(self) -> None:
        self.events.append("close")

    def update(self, result: DetectionResult, *, source: str | None = None) -> TrackingResult:
        call = len(self.updates)
        self.updates.append({"result": result, "source": source})
        self.events.append("update")
        people = self.frames[min(call, len(self.frames) - 1)]
        session = self.sessions[min(call, len(self.sessions) - 1)] if self.sessions else 1
        return tracking(people, result.frame_index or 0, size=result.image_size, session=session, timestamp=result.timestamp,
                        reset_reason=self.reset_reasons.get(call))


class SizedFakeDetector:
    """Her çağrıda boş ``DetectionResult`` veren sahte insan tespitçisi (çağrı sayısı sayılır; takipçi ``ScriptedTracker`` ise tespit içeriği önemsizdir)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.device = "cpu"
        self.device_warning: str | None = None
        self.load_seconds: float | None = 0.01
        self.config = None
        self.load_calls = self.close_calls = 0
        self.warmup_calls: list[tuple[int, int]] = []

    def load(self) -> None:
        self.load_calls += 1

    def close(self) -> None:
        self.close_calls += 1

    def warmup(self, image_size: tuple[int, int] = (640, 480), runs: int = 1) -> float:
        self.warmup_calls.append(tuple(image_size))
        return 0.01

    def detect(self, image: Any, *, frame_index: int | None = None, timestamp: float | None = None) -> DetectionResult:
        self.calls.append({"frame_index": frame_index, "timestamp": timestamp})
        height, width = image.shape[:2]
        return DetectionResult((), (width, height), frame_index, timestamp, 12.0, "cpu")


class ScriptedFaceDetector:
    """Betiklenmiş yüz kutularıyla ``FaceDetector`` taklidi: ``detect`` çağrısı ``i`` için ``frames[i]`` (yüz kutuları listesi) döner."""

    def __init__(self, frames: Sequence[Sequence[Box]]) -> None:
        self.frames = [list(f) for f in frames]
        self.calls: list[dict[str, Any]] = []
        self.device = "cpu"
        self.device_warning: str | None = None
        self.load_seconds: float | None = 0.002
        self.config = None
        self.load_calls = self.close_calls = 0
        self.warmup_calls: list[tuple[int, int]] = []

    def load(self) -> None:
        self.load_calls += 1

    def close(self) -> None:
        self.close_calls += 1

    def warmup(self, image_size: tuple[int, int] = (640, 480), runs: int = 1) -> float:
        self.warmup_calls.append(tuple(image_size))
        return 0.01

    def detect(self, image: Any, *, frame_index: int | None = None, timestamp: float | None = None) -> FaceDetectionResult:
        call = len(self.calls)
        self.calls.append({"frame_index": frame_index, "timestamp": timestamp})
        height, width = image.shape[:2]
        boxes = self.frames[min(call, len(self.frames) - 1)]
        return faces(boxes, frame_index, size=(width, height), timestamp=timestamp)


def blank(width: int = W, height: int = H) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)
