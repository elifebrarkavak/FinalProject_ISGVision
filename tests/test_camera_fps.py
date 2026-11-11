"""Aşama 3 / Grup 2: FPS ölçümü ve çözünürlük ayarı testleri (gerçek kamera açılmaz)."""

from __future__ import annotations

import math
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from camera_test_support import FakeCapture, FakeFactory, make_frame, write_synthetic_video  # noqa: E402
from isg_vision.camera import FrameSource, SourceConfig, SourceConfigError  # noqa: E402
from isg_vision.camera.config import DEFAULT_HEIGHT, DEFAULT_WIDTH, MAX_DIMENSION, MIN_DIMENSION  # noqa: E402
from isg_vision.camera.fps import FpsMeter  # noqa: E402

W, H, FPS, COUNT = cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT


class FpsMeterTest(unittest.TestCase):
    def test_steady_30_fps(self) -> None:
        meter = FpsMeter(window=30)
        for i in range(100):
            fps = meter.tick(i / 30.0)
        self.assertAlmostEqual(fps, 30.0, places=6)
        self.assertAlmostEqual(meter.average_fps, 30.0, places=6)
        self.assertEqual(meter.frames, 100)
        self.assertAlmostEqual(meter.elapsed, 99 / 30.0, places=6)

    def test_zero_until_two_frames(self) -> None:
        meter = FpsMeter()
        self.assertEqual(meter.fps, 0.0)
        self.assertEqual(meter.average_fps, 0.0)
        self.assertEqual(meter.tick(5.0), 0.0)
        self.assertEqual(meter.average_fps, 0.0)
        self.assertGreater(meter.tick(5.1), 0.0)

    def test_identical_timestamps_do_not_divide_by_zero(self) -> None:
        meter = FpsMeter()
        meter.tick(1.0)
        self.assertEqual(meter.tick(1.0), 0.0)
        self.assertEqual(meter.average_fps, 0.0)

    def test_sliding_window_follows_recent_rate_while_average_lags(self) -> None:
        meter = FpsMeter(window=5)
        t = 0.0
        for _ in range(50):  # 10 FPS
            meter.tick(t)
            t += 0.1
        for _ in range(10):  # 50 FPS
            meter.tick(t)
            t += 0.02
        self.assertAlmostEqual(meter.fps, 50.0, places=6)  # pencere yalnızca son 5 kareyi görür
        self.assertLess(meter.average_fps, 25.0)  # genel ortalama hâlâ yavaş dönemi taşır

    def test_single_slow_frame_is_smoothed(self) -> None:
        meter = FpsMeter(window=30)
        t = 0.0
        for _ in range(29):
            meter.tick(t)
            t += 1 / 30
        t += 0.5  # tek kare 0.5 sn gecikti: anlık FPS 2'ye düşerdi
        fps = meter.tick(t)
        self.assertGreater(fps, 15.0)

    def test_decreasing_timestamps_rejected(self) -> None:
        meter = FpsMeter()
        meter.tick(2.0)
        with self.assertRaises(ValueError):
            meter.tick(1.0)

    def test_invalid_window_rejected(self) -> None:
        for bad in (0, 1, -3, 2.5, "30", True):
            with self.subTest(window=bad):
                with self.assertRaises(ValueError):
                    FpsMeter(window=bad)  # type: ignore[arg-type]

    def test_reset(self) -> None:
        meter = FpsMeter()
        for i in range(5):
            meter.tick(i * 0.1)
        meter.reset()
        self.assertEqual((meter.frames, meter.fps, meter.average_fps, meter.elapsed), (0, 0.0, 0.0, 0.0))

    def test_injected_clock_is_used_and_default_is_monotonic(self) -> None:
        times = iter([1.0, 1.5])
        meter = FpsMeter(clock=lambda: next(times))
        meter.tick()
        self.assertAlmostEqual(meter.tick(), 2.0)
        self.assertIs(FpsMeter()._clock, time.monotonic)

    def test_measured_fps_is_independent_of_reported_source_fps(self) -> None:
        cap = FakeCapture([make_frame(i) for i in range(3)], props={FPS: 60.0})
        with FrameSource(SourceConfig.camera(), capture_factory=FakeFactory(cap)) as source:
            meter = FpsMeter()
            for k in range(3):
                source.read()
                meter.tick(k * 0.1)  # uygulama gerçekte 10 FPS işliyor
        self.assertEqual(source.source_fps, 60.0)
        self.assertAlmostEqual(meter.average_fps, 10.0)


class ResolutionConfigTest(unittest.TestCase):
    def test_camera_default_request_is_1280x720(self) -> None:
        cfg = SourceConfig.camera()
        self.assertEqual((cfg.width, cfg.height), (DEFAULT_WIDTH, DEFAULT_HEIGHT))
        self.assertEqual((DEFAULT_WIDTH, DEFAULT_HEIGHT), (1280, 720))

    def test_custom_and_device_default_resolution(self) -> None:
        cfg = SourceConfig.camera(width=640, height=480)
        self.assertEqual((cfg.width, cfg.height), (640, 480))
        none_cfg = SourceConfig.camera(width=None, height=None)
        self.assertEqual((none_cfg.width, none_cfg.height), (None, None))

    def test_invalid_resolutions_rejected(self) -> None:
        bad_pairs = [
            (640, None), (None, 480),
            (0, 480), (640, 0), (-640, 480),
            (MIN_DIMENSION - 1, 480), (640, MAX_DIMENSION + 1),
            (640.5, 480), ("640", 480), (True, 480),
        ]
        for width, height in bad_pairs:
            with self.subTest(width=width, height=height):
                with self.assertRaises(SourceConfigError):
                    SourceConfig.camera(width=width, height=height)  # type: ignore[arg-type]

    def test_video_rejects_resolution(self) -> None:
        with self.assertRaisesRegex(SourceConfigError, "yalnızca kamera"):
            SourceConfig(kind="video", video_path=Path("x.mp4"), width=640, height=480)


class FrameSourceResolutionTest(unittest.TestCase):
    def test_requested_resolution_is_applied_and_read_back(self) -> None:
        cap = FakeCapture([make_frame(0, 1280, 720)], props={W: 640, H: 480}, accept_set=True)
        source = FrameSource(SourceConfig.camera(), capture_factory=FakeFactory(cap))
        with source:
            self.assertEqual(cap.set_calls, [(W, 1280.0), (H, 720.0)])
            self.assertEqual(source.requested_resolution, (1280, 720))
            self.assertEqual(source.actual_resolution, (1280, 720))  # cihaz isteneni uyguladı
            self.assertIsNone(source.resolution_mismatch)

    def test_device_ignoring_request_is_reported_as_mismatch(self) -> None:
        cap = FakeCapture([make_frame(0, 640, 480)], props={W: 640, H: 480}, accept_set=False)
        source = FrameSource(SourceConfig.camera(width=1920, height=1080), capture_factory=FakeFactory(cap))
        with source:
            self.assertEqual(source.actual_resolution, (640, 480))
            self.assertEqual(source.resolution_mismatch, ((1920, 1080), (640, 480)))
            source.read()
            self.assertEqual(source.resolution_mismatch, ((1920, 1080), (640, 480)))

    def test_real_frame_size_overrides_driver_report(self) -> None:
        cap = FakeCapture([make_frame(0, 1280, 720)], props={}, accept_set=False)  # sürücü boyut bildirmiyor
        source = FrameSource(SourceConfig.camera(), capture_factory=FakeFactory(cap))
        with source:
            self.assertIsNone(source.actual_resolution)
            self.assertIsNone(source.resolution_mismatch)  # bilinmeyen -> sahte uyarı üretilmez
            packet = source.read()
            self.assertEqual(source.actual_resolution, (packet.width, packet.height))
            self.assertEqual(source.actual_resolution, (1280, 720))

    def test_no_request_means_no_set_calls(self) -> None:
        cap = FakeCapture([make_frame(0)], props={W: 640, H: 480})
        with FrameSource(SourceConfig.camera(width=None, height=None), capture_factory=FakeFactory(cap)) as source:
            self.assertEqual(cap.set_calls, [])
            self.assertIsNone(source.requested_resolution)
            self.assertEqual(source.actual_resolution, (640, 480))

    def test_values_survive_close(self) -> None:
        cap = FakeCapture([make_frame(0, 640, 480)], props={W: 640, H: 480, FPS: 30.0})
        source = FrameSource(SourceConfig.camera(width=640, height=480), capture_factory=FakeFactory(cap))
        with source:
            source.read()
        self.assertEqual(source.actual_resolution, (640, 480))
        self.assertEqual(source.source_fps, 30.0)

    def test_unreliable_fps_values_become_none(self) -> None:
        for bad in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(fps=bad):
                cap = FakeCapture([make_frame(0)], props={FPS: bad})
                with FrameSource(SourceConfig.camera(), capture_factory=FakeFactory(cap)) as source:
                    self.assertIsNone(source.source_fps)

    def test_camera_has_no_frame_count(self) -> None:
        cap = FakeCapture([make_frame(0)], props={COUNT: 500.0})
        with FrameSource(SourceConfig.camera(), capture_factory=FakeFactory(cap)) as source:
            self.assertIsNone(source.frame_count)

    def test_real_video_reports_resolution_fps_and_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_synthetic_video(Path(tmp) / "v.mp4", n_frames=20, fps=25.0, size=(128, 96))
            with FrameSource(SourceConfig.video(path)) as source:
                self.assertEqual(source.actual_resolution, (128, 96))
                self.assertAlmostEqual(source.source_fps, 25.0, places=3)
                self.assertEqual(source.frame_count, 20)
                self.assertIsNone(source.requested_resolution)
                self.assertEqual(len(list(source.frames())), 20)


if __name__ == "__main__":
    unittest.main()
