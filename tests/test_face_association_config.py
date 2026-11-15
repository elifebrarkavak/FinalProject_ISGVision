"""AssociationConfig doğrulaması ve OpenCV iş parçacığı ayarının ayrıştırılması [Mock/birim; model/kamera/ağ yok]."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isg_vision.face_association import threads  # noqa: E402
from isg_vision.face_association.config import AssociationConfig  # noqa: E402
from isg_vision.face_association.errors import AssociationConfigError  # noqa: E402


class ConfigTest(unittest.TestCase):
    def test_defaults_are_valid_and_floats(self) -> None:
        cfg = AssociationConfig()
        self.assertEqual(cfg.min_containment, 0.8)
        self.assertEqual(cfg.ambiguity_margin, 0.15)
        self.assertIsInstance(cfg.min_score, float)
        self.assertTrue(cfg.require_frame_index)

    def test_ints_are_normalized_to_float(self) -> None:
        self.assertIsInstance(AssociationConfig(min_containment=1).min_containment, float)

    def test_unit_interval_fields_reject_bad_values(self) -> None:
        for name in ("min_containment", "max_center_y_ratio", "ambiguity_margin", "border_factor", "jump_iou", "overlap_guard", "stationary_iou",
                     "min_face_height_ratio"):
            for bad in (0, 0.0, -0.1, 1.01, float("nan"), float("inf"), True, "0.5", None):
                with self.subTest(name=name, bad=bad), self.assertRaises(AssociationConfigError):
                    AssociationConfig(**{name: bad})

    def test_min_score_allows_zero_but_not_above_one(self) -> None:
        AssociationConfig(min_score=0.0)
        AssociationConfig(min_score=1.0)
        for bad in (-0.01, 1.01, float("nan"), True):
            with self.subTest(bad=bad), self.assertRaises(AssociationConfigError):
                AssociationConfig(min_score=bad)

    def test_height_ratio_order(self) -> None:
        with self.assertRaises(AssociationConfigError):
            AssociationConfig(min_face_height_ratio=0.5, max_face_height_ratio=0.4)
        with self.assertRaises(AssociationConfigError):
            AssociationConfig(min_face_height_ratio=0.5, max_face_height_ratio=0.5)

    def test_frame_counts_must_be_integers_in_range(self) -> None:
        for name, minimum in (("max_gap_frames", 0), ("overlap_cooldown_frames", 0), ("min_trusted_frames", 1), ("state_ttl_frames", 1), ("stationary_flag_frames", 1)):
            AssociationConfig(**{name: minimum + (5 if name == "state_ttl_frames" else 0)})
            for bad in (minimum - 1, 1.5, True, "3", None):
                with self.subTest(name=name, bad=bad), self.assertRaises(AssociationConfigError):
                    AssociationConfig(**{name: bad})

    def test_state_ttl_must_exceed_gap_tolerance(self) -> None:
        with self.assertRaises(AssociationConfigError):
            AssociationConfig(max_gap_frames=10, state_ttl_frames=10)
        AssociationConfig(max_gap_frames=10, state_ttl_frames=11)

    def test_require_frame_index_must_be_bool(self) -> None:
        with self.assertRaises(AssociationConfigError):
            AssociationConfig(require_frame_index=1)  # type: ignore[arg-type]

    def test_config_is_frozen(self) -> None:
        with self.assertRaises(Exception):
            AssociationConfig().min_score = 0.1  # type: ignore[misc]


class ThreadParsingTest(unittest.TestCase):
    def test_modes(self) -> None:
        self.assertIsNone(threads.parse_threads("keep"))
        self.assertEqual(threads.parse_threads("restore"), -1)
        self.assertEqual(threads.parse_threads(" RESTORE "), -1)
        self.assertEqual(threads.parse_threads("0"), 0)
        self.assertEqual(threads.parse_threads("4"), 4)

    def test_invalid(self) -> None:
        for bad in ("", "auto", "-1", "257", "1.5", "four"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                threads.parse_threads(bad)

    def test_argparse_type_normalizes_and_rejects(self) -> None:
        import argparse

        self.assertEqual(threads.threads_argument(" Restore "), "restore")
        with self.assertRaises(argparse.ArgumentTypeError):
            threads.threads_argument("-5")

    def test_keep_never_calls_setnumthreads(self) -> None:
        with mock.patch.object(threads.cv2, "setNumThreads") as spy:
            setting = threads.apply_opencv_threads("keep")
        spy.assert_not_called()
        self.assertFalse(setting.changed)
        self.assertIsNone(setting.value)

    def test_restore_and_number_call_setnumthreads_once(self) -> None:
        for text, expected in (("restore", -1), ("3", 3), ("0", 0)):
            with self.subTest(text=text), mock.patch.object(threads.cv2, "setNumThreads") as spy, mock.patch.object(threads.cv2, "getNumThreads", side_effect=[1, 10]):
                setting = threads.apply_opencv_threads(text)
            spy.assert_called_once_with(expected)
            self.assertTrue(setting.changed)
            self.assertEqual((setting.before, setting.after, setting.value), (1, 10, expected))

    def test_default_is_keep(self) -> None:
        with mock.patch.object(threads.cv2, "setNumThreads") as spy:
            threads.apply_opencv_threads()
        spy.assert_not_called()


class ImportIsolationTest(unittest.TestCase):
    """Paketin içe aktarılması model yüklemez, ultralytics/torch'u içe aktarmaz ve OpenCV iş parçacığı ayarına DOKUNMAZ (alt süreçte denetlenir)."""

    def test_importing_the_package_has_no_side_effects(self) -> None:
        import subprocess

        code = (
            "import sys, cv2\n"
            "before = cv2.getNumThreads()\n"
            "import isg_vision.face_association as fa\n"
            "import isg_vision.face_association.cli, isg_vision.face_association.pipeline, isg_vision.face_association.benchmark, isg_vision.face_association.scenes\n"
            "assert 'ultralytics' not in sys.modules, 'ultralytics içe aktarıldı'\n"
            "assert 'torch' not in sys.modules, 'torch içe aktarıldı'\n"
            "assert cv2.getNumThreads() == before, 'OpenCV iş parçacığı sayısı değişti'\n"
            "print('ok')\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ok", proc.stdout)


if __name__ == "__main__":
    unittest.main()
