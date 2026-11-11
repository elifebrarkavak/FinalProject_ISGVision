"""Kaynak modüllerin import edilmesinin yan etkisiz olduğunu doğrular.

- Kamera açılmaz (cv2.VideoCapture çağrılırsa test başarısız olur).
- Ağ bağlantısı kurulmaz (model/veri indirilemez).
- data/ ve models/ altındaki hiçbir dosya oluşmaz, değişmez veya silinmez.
"""

from __future__ import annotations

import importlib
import socket
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODULES = (
    "isg_vision",
    "isg_vision.camera",
    "isg_vision.camera.errors",
    "isg_vision.camera.frame",
    "isg_vision.camera.config",
    "isg_vision.camera.capture",
    "isg_vision.camera.fps",
    "isg_vision.camera.macos_window",
    "isg_vision.camera.display",
    "isg_vision.camera.preview",
    "isg_vision.camera.cli",
    "isg_vision.human_detection",
    "isg_vision.human_detection.errors",
    "isg_vision.human_detection.config",
    "isg_vision.human_detection.results",
    "isg_vision.human_detection.person_detector",
    "isg_vision.human_detection.weights",
    "isg_vision.human_detection.visualization",
    "isg_vision.human_detection.pipeline",
    "isg_vision.human_detection.cli",
    "isg_vision.human_detection.benchmark",
    "isg_vision.human_detection.evaluation",
    "isg_vision.tracking",
    "isg_vision.face_detection",
    "isg_vision.face_recognition",
    "isg_vision.ppe_detection",
    "isg_vision.integration",
    "isg_vision.storage",
    "isg_vision.utils",
    "isg_vision.utils.paths",
    "isg_vision.utils.device",
)
WATCHED = ("data", "models")
# Aşama 4A: models/ altında YALNIZCA bu onaylı resmî ağırlık bulunabilir (SHA-256'sı weights.py'de sabitli; Git dışında).
APPROVED_WEIGHTS = frozenset({"models/human_detection/yolo11n.pt"})


def snapshot() -> dict[str, tuple[int, int]]:
    """data/ ve models/ altındaki tüm dosyaların (boyut, mtime) kaydı."""
    state: dict[str, tuple[int, int]] = {}
    for name in WATCHED:
        for path in (ROOT / name).rglob("*"):
            if path.is_file():
                st = path.stat()
                state[str(path.relative_to(ROOT))] = (st.st_size, st.st_mtime_ns)
    return state


class ImportSideEffectsTest(unittest.TestCase):
    def test_import_does_not_open_camera_or_network(self) -> None:
        import cv2

        before = snapshot()
        with mock.patch.object(
            cv2, "VideoCapture", side_effect=AssertionError("Kamera açıldı!")
        ), mock.patch.object(
            socket.socket, "connect", side_effect=AssertionError("Ağ bağlantısı kuruldu!")
        ):
            for name in MODULES:
                sys.modules.pop(name, None)
                importlib.import_module(name)
        self.assertEqual(snapshot(), before, "Import sırasında data/models değişti")

    def test_no_model_weights_created(self) -> None:
        weights = [
            p
            for suffix in ("*.pt", "*.onnx", "*.pth")
            for p in (ROOT / "models").rglob(suffix)
            if p.relative_to(ROOT).as_posix() not in APPROVED_WEIGHTS
        ]
        self.assertEqual(weights, [], f"Beklenmeyen model dosyaları: {weights}")


class DeviceSelectionTest(unittest.TestCase):
    def test_select_device_returns_valid_name(self) -> None:
        from isg_vision.utils.device import select_device

        self.assertIn(select_device("auto"), {"mps", "cuda", "cpu"})
        self.assertEqual(select_device("cpu"), "cpu")
        self.assertEqual(select_device("olmayan_cihaz"), "cpu")


if __name__ == "__main__":
    unittest.main()
