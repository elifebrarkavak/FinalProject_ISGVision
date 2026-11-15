"""Yüz tespiti yapılandırması testleri: varsayılanlar, doğrulama, hatalı değerler (model yüklenmez, ağ yok)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isg_vision.face_detection import config as cfg  # noqa: E402
from isg_vision.face_detection.config import FaceDetectionConfig  # noqa: E402
from isg_vision.face_detection.errors import FaceConfigError  # noqa: E402


class DefaultsTest(unittest.TestCase):
    def test_defaults(self) -> None:
        c = FaceDetectionConfig()
        self.assertEqual((c.conf, c.nms, c.max_faces, c.detect_width, c.verify_model_hash), (0.6, 0.3, 100, None, True))
        self.assertEqual(c.model_name, "face_detection_yunet_2026may.onnx")
        self.assertEqual(c.model_path, cfg.DEFAULT_MODEL_PATH)
        self.assertEqual(c.model_path.parent.name, "face_detection")
        self.assertEqual(c.model_path.parent.parent.name, "models")

    def test_only_cpu_is_supported_and_there_is_no_device_setting(self) -> None:
        self.assertEqual(cfg.DEVICE, "cpu")
        self.assertNotIn("device", FaceDetectionConfig.__dataclass_fields__)

    def test_config_is_frozen(self) -> None:
        with self.assertRaises(Exception):
            FaceDetectionConfig().conf = 0.1  # type: ignore[misc]

    def test_string_model_path_is_converted(self) -> None:
        self.assertIsInstance(FaceDetectionConfig(model_path="x/model.onnx").model_path, Path)

    def test_construction_does_not_touch_the_model_file(self) -> None:
        FaceDetectionConfig(model_path="/yok/boyle/bir/dosya.onnx")  # varlık denetimi yükleme sırasındadır


class ValidationTest(unittest.TestCase):
    def test_valid_boundaries_accepted(self) -> None:
        FaceDetectionConfig(conf=1.0, nms=1.0, max_faces=1)
        FaceDetectionConfig(conf=0.0001, nms=0.0001, max_faces=cfg.MAX_MAX_FACES)
        FaceDetectionConfig(detect_width=cfg.MIN_DETECT_WIDTH)
        FaceDetectionConfig(detect_width=cfg.MAX_DETECT_WIDTH)

    def test_invalid_conf_and_nms_rejected(self) -> None:
        for name in ("conf", "nms"):
            for bad in (0, 0.0, -0.1, 1.0001, 2, float("nan"), float("inf"), "0.5", None, True):
                with self.subTest(name=name, value=bad), self.assertRaises(FaceConfigError):
                    FaceDetectionConfig(**{name: bad})

    def test_invalid_max_faces_rejected(self) -> None:
        for bad in (0, -1, cfg.MAX_MAX_FACES + 1, 1.5, "10", None, True):
            with self.subTest(value=bad), self.assertRaises(FaceConfigError):
                FaceDetectionConfig(max_faces=bad)  # type: ignore[arg-type]

    def test_invalid_detect_width_rejected(self) -> None:
        for bad in (0, -5, cfg.MIN_DETECT_WIDTH - 1, cfg.MAX_DETECT_WIDTH + 1, 640.0, "640", True):
            with self.subTest(value=bad), self.assertRaises(FaceConfigError):
                FaceDetectionConfig(detect_width=bad)  # type: ignore[arg-type]

    def test_invalid_model_path_rejected(self) -> None:
        for bad in ("", "   ", None, 5, True, "model.pt", "model.onnx.txt", "model", "model.tflite"):
            with self.subTest(value=bad), self.assertRaises(FaceConfigError):
                FaceDetectionConfig(model_path=bad)  # type: ignore[arg-type]

    def test_verify_flag_must_be_bool(self) -> None:
        for bad in (1, 0, "yes", None):
            with self.subTest(value=bad), self.assertRaises(FaceConfigError):
                FaceDetectionConfig(verify_model_hash=bad)  # type: ignore[arg-type]

    def test_config_error_is_a_value_error(self) -> None:
        self.assertTrue(issubclass(FaceConfigError, ValueError))

    def test_messages_are_actionable(self) -> None:
        with self.assertRaises(FaceConfigError) as ctx:
            FaceDetectionConfig(conf=2)
        self.assertIn("conf", str(ctx.exception))
        with self.assertRaises(FaceConfigError) as ctx:
            FaceDetectionConfig(model_path="a.pt")
        self.assertIn(".onnx", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
