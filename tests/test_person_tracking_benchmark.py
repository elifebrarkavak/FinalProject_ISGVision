"""Takip benchmark'ı: yöntem (dönüşümlü sıra, ayrı yükleme/ısınma/burn-in), cihaz atlama, hata yayılımı, özet matematiği, ölçeklenme yardımcıları.

[Mock/birim] Sahte tespitçi ve sahte backend'li takipçi kullanılır: gerçek model/ByteTrack/kamera/ağ yoktur; bu testler HIZ DEĞERLERİNİ değil,
ölçüm yönteminin doğruluğunu sınar. Gerçek ölçüm: ``scripts/benchmark_person_tracking.py``. ``measure_tracker_scaling``'in gerçek ByteTrack testi
``lap`` kuruluysa çalışır.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from camera_test_support import write_synthetic_video  # noqa: E402
from person_detection_test_support import FakeDetector  # noqa: E402
from tracking_test_support import echo_tracker  # noqa: E402

from isg_vision.human_detection.config import DetectionConfig, ResolvedDevice  # noqa: E402
from isg_vision.human_detection.errors import DeviceUnavailableError, ModelLoadError  # noqa: E402
from isg_vision.human_detection.pipeline import LatencyStats  # noqa: E402
from isg_vision.tracking.benchmark import (  # noqa: E402
    MODES,
    DeviceTrackingBenchmark,
    ModeRun,
    TrackingBenchmarkReport,
    measure_tracker_scaling,
    overhead_summary,
    report_to_dict,
    report_to_markdown,
    run_tracking_benchmark,
    scripted_detections,
)
from isg_vision.tracking.config import TrackingConfig  # noqa: E402

LAP_AVAILABLE = importlib.util.find_spec("lap") is not None and importlib.util.find_spec("ultralytics") is not None


def device_check(requested: str) -> ResolvedDevice:
    if requested == "mps":
        raise DeviceUnavailableError("MPS yok (test)")
    return ResolvedDevice(requested, "cpu")


class Harness:
    def __init__(self) -> None:
        self.detectors: list[FakeDetector] = []
        self.trackers: list = []

    def detector_factory(self, cfg: DetectionConfig) -> FakeDetector:
        detector = FakeDetector(counts=[2])
        detector.requested = cfg.device  # type: ignore[attr-defined]
        self.detectors.append(detector)
        return detector

    def tracker_factory(self, cfg: TrackingConfig):  # type: ignore[no-untyped-def]
        tracker = echo_tracker()
        self.trackers.append(tracker)
        return tracker


def stats(median: float) -> LatencyStats:
    return LatencyStats(10, median, median, median, median, median)


class RunBenchmarkTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.video = write_synthetic_video(Path(self._tmp.name) / "v.mp4", n_frames=12, size=(64, 48))

    def run_it(self, devices=("cpu",), passes=3, burn_in=2, max_frames=None, harness=None):  # type: ignore[no-untyped-def]
        harness = harness or Harness()
        report = run_tracking_benchmark(
            self.video, devices, DetectionConfig(), TrackingConfig(), passes=passes, burn_in=burn_in, max_frames=max_frames,
            detector_factory=harness.detector_factory, tracker_factory=harness.tracker_factory, device_check=device_check,
        )
        return report, harness

    def test_every_pass_runs_all_three_modes_with_full_frame_count(self) -> None:
        report, _ = self.run_it(passes=3, burn_in=0)
        entry = report.devices[0]
        self.assertEqual(len(entry.runs), 3)
        for runs in entry.runs:
            self.assertEqual(sorted(r.mode for r in runs), sorted(MODES))
            for r in runs:
                self.assertEqual(r.frames, 12)

    def test_mode_order_rotates_between_passes(self) -> None:
        report, _ = self.run_it(passes=4, burn_in=0)
        self.assertEqual([m[0] for m in report.mode_order], ["detect_only", "track", "track_draw", "detect_only"])
        for order, runs in zip(report.mode_order, report.devices[0].runs):
            self.assertEqual([r.mode for r in runs], order)

    def test_device_order_rotates_between_passes(self) -> None:
        report, _ = self.run_it(devices=("cpu", "auto"), passes=3, burn_in=0)
        self.assertEqual(report.pass_order, [["cpu", "auto"], ["auto", "cpu"], ["cpu", "auto"]])

    def test_unavailable_device_is_skipped_with_reason_and_never_run(self) -> None:
        report, harness = self.run_it(devices=("cpu", "mps"), passes=2, burn_in=0)
        mps = next(d for d in report.devices if d.requested == "mps")
        self.assertIn("MPS yok", mps.skipped_reason or "")
        self.assertEqual(mps.runs, [])
        self.assertEqual(len(harness.detectors), 1, "kullanılamayan cihaz için model yüklenmemeli (sessizce CPU'ya düşülmez)")
        self.assertEqual(report.pass_order, [["cpu"], ["cpu"]])

    def test_load_warmup_and_burn_in_are_recorded_separately(self) -> None:
        report, harness = self.run_it(passes=1, burn_in=3)
        entry = report.devices[0]
        self.assertEqual(entry.model_load_seconds, 0.01)
        self.assertIsNotNone(entry.tracker_load_seconds)
        self.assertIsNotNone(entry.warmup_seconds)
        self.assertEqual(entry.burn_in_frames, 3)
        self.assertIsNotNone(entry.burn_in_seconds)
        self.assertEqual(harness.detectors[0].warmup_calls, [(64, 48)], "ısınma gerçek kare boyutunda, ölçümden önce tek çağrı")
        # burn-in ölçüme girmez: ölçülen koşular 12'şer kare
        self.assertTrue(all(r.frames == 12 for r in entry.runs[0]))

    def test_max_frames_limits_each_run(self) -> None:
        report, _ = self.run_it(passes=1, burn_in=0, max_frames=5)
        self.assertEqual(report.frames_per_pass, 5)
        self.assertTrue(all(r.frames == 5 for r in report.devices[0].runs[0]))

    def test_resources_are_released_even_for_all_devices(self) -> None:
        _, harness = self.run_it(devices=("cpu", "auto"), passes=1, burn_in=0)
        self.assertTrue(all(d.close_calls == 1 for d in harness.detectors))
        self.assertTrue(all(not t.is_loaded for t in harness.trackers))

    def test_tracking_modes_report_tracking_and_detect_only_does_not(self) -> None:
        report, _ = self.run_it(passes=1, burn_in=0)
        by_mode = {r.mode: r for r in report.devices[0].runs[0]}
        self.assertIsNone(by_mode["detect_only"].tracking)
        self.assertIsNotNone(by_mode["track"].tracking)
        self.assertIsNone(by_mode["track"].draw, "çizim yalnızca track_draw modunda ölçülür")
        self.assertIsNotNone(by_mode["track_draw"].draw)
        self.assertEqual(by_mode["track"].distinct_tracks, 2)

    def test_model_load_error_stops_the_benchmark(self) -> None:
        harness = Harness()

        def failing(cfg: DetectionConfig) -> FakeDetector:
            return FakeDetector(load_error=ModelLoadError("ağırlık yok"))

        with self.assertRaises(ModelLoadError):
            run_tracking_benchmark(self.video, ("cpu",), DetectionConfig(), TrackingConfig(), passes=1, burn_in=0, detector_factory=failing,
                                   tracker_factory=harness.tracker_factory, device_check=device_check)

    def test_invalid_arguments(self) -> None:
        for kwargs in ({"passes": 0}, {"passes": True}, {"burn_in": -1}, {"burn_in": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_it(**kwargs)
        with self.assertRaises(ValueError):
            self.run_it(devices=())

    def test_report_is_json_serialisable_and_has_no_images(self) -> None:
        report, _ = self.run_it(devices=("cpu", "mps"), passes=2, burn_in=0)
        report.scaling = measure_tracker_scaling((1, 3), frames=5, burn_in=1, tracker_factory=lambda cfg: echo_tracker())
        text = json.dumps(report_to_dict(report), ensure_ascii=False)
        data = json.loads(text)
        self.assertEqual(len(data["devices"]), 2)
        self.assertEqual(data["devices"][1]["skipped_reason"], "MPS yok (test)")
        self.assertIsNone(data["devices"][1]["summary"])
        self.assertEqual(len(data["tracker_scaling"]), 2)
        self.assertIn("overhead_track_ms_per_frame", data["devices"][0]["summary"])

    def test_markdown_lists_devices_skips_and_scaling(self) -> None:
        report, _ = self.run_it(devices=("cpu", "mps"), passes=2, burn_in=0)
        report.scaling = measure_tracker_scaling((1, 3), frames=5, burn_in=1, tracker_factory=lambda cfg: echo_tracker())
        md = report_to_markdown(report)
        self.assertIn("Takip ek yükü", md)
        self.assertIn("**atlandı:** MPS yok (test)", md)
        self.assertIn("ByteTrack ölçeklenmesi", md)
        self.assertIn("| cpu |", md)

    def test_scaling_only_report_has_only_the_scaling_table(self) -> None:
        report = TrackingBenchmarkReport("", None, None, 0, {"tracking": {}}, [], [], [])
        report.scaling = measure_tracker_scaling((2,), frames=4, burn_in=0, tracker_factory=lambda cfg: echo_tracker())
        md = report_to_markdown(report)
        self.assertNotIn("Model:", md)
        self.assertIn("| 2 |", md)


class OverheadSummaryTest(unittest.TestCase):
    @staticmethod
    def run_of(mode: str, fps: float, inference: float = 10.0, tracking: float | None = None, draw: float | None = None) -> ModeRun:
        return ModeRun(mode, 100, 1.0, fps, stats(0.3), stats(inference), stats(tracking) if tracking is not None else None,
                       stats(inference + tracking) if tracking is not None else None, stats(inference + (tracking or 0.0) + 0.1),
                       stats(draw) if draw is not None else None)

    def test_overhead_is_difference_of_median_frame_times(self) -> None:
        entry = DeviceTrackingBenchmark("cpu", "cpu")
        entry.runs = [
            [self.run_of("detect_only", 100.0), self.run_of("track", 90.0, tracking=0.4), self.run_of("track_draw", 80.0, tracking=0.4, draw=0.5)],
            [self.run_of("detect_only", 100.0), self.run_of("track", 90.0, tracking=0.6), self.run_of("track_draw", 80.0, tracking=0.4, draw=0.7)],
            [self.run_of("detect_only", 125.0), self.run_of("track", 100.0, tracking=0.5), self.run_of("track_draw", 80.0, tracking=0.4, draw=0.6)],
        ]
        s = overhead_summary(entry)
        self.assertEqual((s["fps_detect_only"], s["fps_track"], s["fps_track_draw"]), (100.0, 90.0, 80.0))
        self.assertAlmostEqual(s["frame_ms_detect_only"], 10.0)
        self.assertAlmostEqual(s["frame_ms_track"], 1000.0 / 90.0)
        self.assertAlmostEqual(s["overhead_track_ms_per_frame"], 1000.0 / 90.0 - 10.0)
        self.assertAlmostEqual(s["overhead_draw_ms_per_frame"], 1000.0 / 80.0 - 1000.0 / 90.0)
        self.assertEqual(s["tracking_median_ms"], 0.5)
        self.assertEqual(s["draw_median_ms"], 0.6)

    def test_missing_modes_give_none_not_zero(self) -> None:
        entry = DeviceTrackingBenchmark("cpu", "cpu")
        entry.runs = [[self.run_of("detect_only", 50.0)]]
        s = overhead_summary(entry)
        self.assertIsNone(s["overhead_track_ms_per_frame"])
        self.assertIsNone(s["tracking_median_ms"])
        self.assertEqual(s["fps_detect_only"], 50.0)


class ScalingHelpersTest(unittest.TestCase):
    def test_scripted_detections_are_valid_in_bounds_and_non_overlapping(self) -> None:
        for persons in (0, 1, 5, 20, 100):
            for frame in (0, 7, 19, 20, 39, 40, 123):
                with self.subTest(persons=persons, frame=frame):
                    result = scripted_detections(persons, frame)
                    self.assertEqual(result.person_count, persons)
                    boxes = [d.bbox_xyxy for d in result.detections]
                    for x1, y1, x2, y2 in boxes:
                        self.assertTrue(0 <= x1 < x2 <= 1280 and 0 <= y1 < y2 <= 720)
                    for i, a in enumerate(boxes):
                        for b in boxes[i + 1 :]:
                            overlap = min(a[2], b[2]) - max(a[0], b[0]) > 0 and min(a[3], b[3]) - max(a[1], b[1]) > 0
                            self.assertFalse(overlap, "betiklenmiş kişiler üst üste binmemeli")

    def test_scripted_motion_is_continuous(self) -> None:
        """Üçgen dalga: ardışık karelerde en çok birkaç piksel kayma (ani sıçrama yok; iz kopmaz)."""
        for frame in range(0, 120):
            a, b = scripted_detections(5, frame).detections[0], scripted_detections(5, frame + 1).detections[0]
            self.assertLessEqual(abs(a.bbox_xyxy[0] - b.bbox_xyxy[0]), 1.0)

    def test_scaling_rows_and_validation(self) -> None:
        rows = measure_tracker_scaling((0, 2, 4), frames=6, burn_in=1, tracker_factory=lambda cfg: echo_tracker())
        self.assertEqual([r.persons for r in rows], [0, 2, 4])
        self.assertEqual([r.active_last for r in rows], [0, 2, 4])
        for r in rows:
            self.assertEqual(r.tracking.count, 6)  # type: ignore[union-attr]
        for bad in (0, -1, True):
            with self.assertRaises(ValueError):
                measure_tracker_scaling((1,), frames=bad)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            measure_tracker_scaling((-1,), frames=2)

    @unittest.skipUnless(LAP_AVAILABLE, "gerçek ByteTrack: 'lap' ve 'ultralytics' gerekir (atlandı)")
    def test_real_bytetrack_keeps_all_scripted_persons_active(self) -> None:
        rows = measure_tracker_scaling((1, 10, 50), frames=60, burn_in=10)
        for row in rows:
            self.assertEqual(row.active_last, row.persons, "betiklenmiş hareket izleri koparmamalı")
            self.assertIsNotNone(row.tracking)


if __name__ == "__main__":
    unittest.main()
