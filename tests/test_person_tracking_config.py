"""TrackingConfig doğrulaması, eşik analizi (YOLO conf ↔ ByteTrack eşikleri) ve import yan etkileri. [Mock/birim; ağ/model/kamera yok]"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from isg_vision.tracking.config import (  # noqa: E402
    DEFAULT_MATCH_THRESH,
    DEFAULT_TRACK_BUFFER,
    MAX_TRACK_BUFFER,
    TrackingConfig,
    analyze_thresholds,
)
from isg_vision.tracking.errors import TrackingConfigError  # noqa: E402


class DefaultsTest(unittest.TestCase):
    def test_defaults_equal_installed_ultralytics_bytetrack_yaml(self) -> None:
        """Varsayılanlar rastgele seçilmedi: kurulu Ultralytics'in bytetrack.yaml dosyasıyla birebir aynı (dosya import edilmeden okunur)."""
        spec = importlib.util.find_spec("ultralytics")
        if spec is None or spec.origin is None:
            self.skipTest("ultralytics kurulu değil")
        path = Path(spec.origin).parent / "cfg" / "trackers" / "bytetrack.yaml"
        if not path.is_file():
            self.skipTest(f"{path} yok")
        shipped = yaml.safe_load(path.read_text(encoding="utf-8"))
        ours = TrackingConfig().bytetrack_args()
        for key in ("track_high_thresh", "track_low_thresh", "new_track_thresh", "track_buffer", "match_thresh", "fuse_score"):
            self.assertEqual(ours[key], shipped[key], key)
        self.assertEqual(ours["tracker_type"], shipped["tracker_type"])

    def test_default_values(self) -> None:
        cfg = TrackingConfig()
        self.assertEqual((cfg.track_high_thresh, cfg.track_low_thresh, cfg.new_track_thresh), (0.25, 0.1, 0.25))
        self.assertEqual((cfg.track_buffer, cfg.match_thresh, cfg.fuse_score), (DEFAULT_TRACK_BUFFER, DEFAULT_MATCH_THRESH, True))

    def test_config_is_frozen_and_floats_normalised(self) -> None:
        cfg = TrackingConfig(track_high_thresh=1, new_track_thresh=1, match_thresh=1)  # int değerler float'a çevrilir
        self.assertIsInstance(cfg.track_high_thresh, float)
        self.assertIsInstance(cfg.match_thresh, float)
        with self.assertRaises(Exception):
            cfg.track_buffer = 5  # type: ignore[misc]


class ValidationTest(unittest.TestCase):
    def assertRejected(self, **kwargs: object) -> None:
        with self.assertRaises(TrackingConfigError, msg=str(kwargs)):
            TrackingConfig(**kwargs)  # type: ignore[arg-type]

    def test_thresholds_must_be_in_unit_interval(self) -> None:
        for name in ("track_high_thresh", "track_low_thresh", "new_track_thresh", "match_thresh"):
            for bad in (0, 0.0, -0.1, 1.01, float("nan"), float("inf"), True, "0.5", None):
                extra = {"new_track_thresh": 0.9} if name == "track_high_thresh" else {}
                self.assertRejected(**{name: bad}, **extra)

    def test_low_must_be_below_high(self) -> None:
        self.assertRejected(track_low_thresh=0.25, track_high_thresh=0.25)
        self.assertRejected(track_low_thresh=0.4, track_high_thresh=0.3, new_track_thresh=0.3)

    def test_new_track_thresh_cannot_be_below_high(self) -> None:
        """Yeni iz yalnızca yüksek güvenli tespitten doğar; altındaki değer sessizce etkisiz kalırdı."""
        self.assertRejected(track_high_thresh=0.5, new_track_thresh=0.3)
        TrackingConfig(track_high_thresh=0.5, new_track_thresh=0.5)  # eşit serbest
        TrackingConfig(track_high_thresh=0.3, new_track_thresh=0.7)  # new > high serbest (yanlış pozitiften iz doğmasını azaltır)

    def test_track_buffer_is_bounded_integer(self) -> None:
        for bad in (0, -1, MAX_TRACK_BUFFER + 1, 30.0, True, "30", None):
            self.assertRejected(track_buffer=bad)
        TrackingConfig(track_buffer=1)
        TrackingConfig(track_buffer=MAX_TRACK_BUFFER)

    def test_fuse_score_must_be_bool(self) -> None:
        for bad in (1, 0, "true", None):
            self.assertRejected(fuse_score=bad)
        TrackingConfig(fuse_score=False)

    def test_error_messages_are_actionable(self) -> None:
        with self.assertRaises(TrackingConfigError) as ctx:
            TrackingConfig(track_high_thresh=0.5, new_track_thresh=0.3)
        self.assertIn("new_track_thresh", str(ctx.exception))
        self.assertIn("yüksek güvenli", str(ctx.exception))


class ThresholdAnalysisTest(unittest.TestCase):
    """§9: YOLO conf ByteTrack'in düşük güvenli aşamasını nasıl etkiler?"""

    def test_default_yolo_conf_disables_low_confidence_stage(self) -> None:
        a = analyze_thresholds(0.25, TrackingConfig())
        self.assertFalse(a.low_stage_active)
        self.assertIsNone(a.low_band)
        self.assertIn("KAPALI", " ".join(a.notes))
        self.assertEqual(a.effective_new_track_floor, 0.25)

    def test_lower_yolo_conf_opens_stage_with_exact_band(self) -> None:
        a = analyze_thresholds(0.10, TrackingConfig(track_high_thresh=0.35, new_track_thresh=0.35))
        self.assertTrue(a.low_stage_active)
        self.assertEqual(a.low_band, (0.10, 0.35))
        self.assertIn("AÇIK", " ".join(a.notes))
        self.assertEqual(a.effective_new_track_floor, 0.35)  # düşük güvenli aralık yeni iz doğuramaz

    def test_band_lower_edge_is_max_of_conf_and_low_thresh(self) -> None:
        self.assertEqual(analyze_thresholds(0.05, TrackingConfig()).low_band, (0.1, 0.25))
        self.assertEqual(analyze_thresholds(0.18, TrackingConfig()).low_band, (0.18, 0.25))

    def test_conf_above_high_thresh_is_reported(self) -> None:
        a = analyze_thresholds(0.35, TrackingConfig())
        self.assertFalse(a.low_stage_active)
        self.assertEqual(a.effective_new_track_floor, 0.35)
        self.assertTrue(any("etkin alt sınır" in n for n in a.notes))

    def test_new_track_floor_is_max_of_all(self) -> None:
        self.assertEqual(analyze_thresholds(0.25, TrackingConfig(new_track_thresh=0.6)).effective_new_track_floor, 0.6)

    def test_always_states_two_frame_confirmation(self) -> None:
        self.assertIn("İKİ karede", " ".join(analyze_thresholds(0.25, TrackingConfig()).notes))

    def test_invalid_conf_rejected(self) -> None:
        for bad in (0, -1, 1.5, float("nan"), True, "0.2"):
            with self.assertRaises(TrackingConfigError):
                analyze_thresholds(bad, TrackingConfig())  # type: ignore[arg-type]


class ImportSideEffectsTest(unittest.TestCase):
    def test_importing_tracking_package_does_not_load_ultralytics_or_torch(self) -> None:
        code = (
            "import sys; sys.path.insert(0, %r); import isg_vision.tracking, isg_vision.tracking.bytetrack_adapter;"
            "bad=[m for m in ('ultralytics','torch','lap','cv2') if m in sys.modules]; print('LOADED:', bad); sys.exit(1 if bad else 0)"
        ) % str(ROOT / "src")
        done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)


if __name__ == "__main__":
    unittest.main()
