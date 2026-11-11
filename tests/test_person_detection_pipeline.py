"""Kare akışı -> tespit -> görselleştirme döngüsü testleri: FPS, gecikme, çıkış yolları, kaynak hataları (sahte kaynak/pencere/tespitçi)."""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from camera_test_support import FakeCapture, FakeClock, FakeFactory, FakeWindow, make_frame, write_synthetic_video  # noqa: E402

from isg_vision.camera import FrameSource, SourceConfig  # noqa: E402
from isg_vision.camera.display import PreviewWindow  # noqa: E402
from isg_vision.camera.errors import DisplayError, SourceDisconnectedError  # noqa: E402
from isg_vision.human_detection.errors import InvalidFrameError  # noqa: E402
from isg_vision.human_detection.visualization import WINDOW_NAME  # noqa: E402
from isg_vision.human_detection.pipeline import (  # noqa: E402
    REASON_END_OF_STREAM,
    REASON_INTERRUPTED,
    REASON_MAX_FRAMES,
    REASON_USER_QUIT,
    REASON_WINDOW_CLOSED,
    LatencyStats,
    run_detection,
)
from person_detection_test_support import FakeDetector  # noqa: E402


def camera_source(n_frames: int, clock: FakeClock | None = None, *, max_read_failures: int = 1) -> tuple[FrameSource, FakeCapture]:
    capture = FakeCapture([make_frame(i) for i in range(n_frames)], props={cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 48})
    kwargs = {"clock": clock} if clock is not None else {}
    source = FrameSource(
        SourceConfig.camera(0, width=64, height=48), capture_factory=FakeFactory(capture),
        max_read_failures=max_read_failures, sleep=lambda s: None, **kwargs,
    )
    source.open()
    return source, capture


class LatencyStatsTest(unittest.TestCase):
    def test_no_samples_gives_none(self) -> None:
        self.assertIsNone(LatencyStats.from_values([]))

    def test_single_sample(self) -> None:
        s = LatencyStats.from_values([12.5])
        assert s is not None
        self.assertEqual((s.count, s.mean_ms, s.median_ms, s.p95_ms, s.min_ms, s.max_ms), (1, 12.5, 12.5, 12.5, 12.5, 12.5))

    def test_odd_and_even_median_and_unsorted_input(self) -> None:
        odd = LatencyStats.from_values([30.0, 10.0, 20.0])
        even = LatencyStats.from_values([40.0, 10.0, 30.0, 20.0])
        assert odd is not None and even is not None
        self.assertEqual(odd.median_ms, 20.0)
        self.assertEqual(even.median_ms, 25.0)
        self.assertEqual((even.min_ms, even.max_ms, even.mean_ms), (10.0, 40.0, 25.0))

    def test_p95_is_nearest_rank(self) -> None:
        stats = LatencyStats.from_values([float(v) for v in range(1, 101)])  # 1..100
        assert stats is not None
        self.assertEqual(stats.p95_ms, 95.0)
        stats20 = LatencyStats.from_values([float(v) for v in range(1, 21)])  # ceil(0.95*20)=19 -> 19. değer
        assert stats20 is not None
        self.assertEqual(stats20.p95_ms, 19.0)

    def test_outlier_raises_p95_and_max_but_not_median(self) -> None:
        stats = LatencyStats.from_values([10.0] * 19 + [500.0])
        assert stats is not None
        self.assertEqual((stats.median_ms, stats.max_ms), (10.0, 500.0))
        self.assertEqual(stats.p95_ms, 10.0)  # 19. sıradaki değer


class CountingAndStopTest(unittest.TestCase):
    def test_processes_every_video_frame_and_counts_persons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = write_synthetic_video(Path(tmp) / "v.mp4", n_frames=6, fps=25.0, size=(64, 48))
            with FrameSource(SourceConfig.video(video)) as source:
                detector = FakeDetector(counts=[0, 2, 1, 0, 3, 1])
                summary = run_detection(source, detector)
        self.assertEqual((summary.reason, summary.frames_processed), (REASON_END_OF_STREAM, 6))
        self.assertEqual((summary.persons_total, summary.persons_max, summary.frames_with_persons), (7, 3, 4))
        self.assertAlmostEqual(summary.persons_mean, 7 / 6)
        self.assertEqual(len(detector.detect_calls), 6)

    def test_frames_without_people_are_handled(self) -> None:
        source, _ = camera_source(5)
        summary = run_detection(source, FakeDetector(counts=[0]), max_frames=5)
        self.assertEqual((summary.persons_total, summary.persons_max, summary.frames_with_persons, summary.persons_mean), (0, 0, 0, 0.0))
        self.assertEqual(summary.frames_processed, 5)

    def test_max_frames_stops_exactly_and_reads_no_extra_frame(self) -> None:
        source, capture = camera_source(10)
        summary = run_detection(source, FakeDetector(), max_frames=4)
        self.assertEqual((summary.reason, summary.frames_processed), (REASON_MAX_FRAMES, 4))
        self.assertEqual(capture.read_calls, 4)

    def test_invalid_max_frames_rejected(self) -> None:
        source, _ = camera_source(3)
        for bad in (0, -1, True, 1.5, "3"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                run_detection(source, FakeDetector(), max_frames=bad)  # type: ignore[arg-type]

    def test_detector_receives_frame_pixels_index_and_timestamp(self) -> None:
        clock = FakeClock(50.0)
        source, _ = camera_source(3, clock)
        detector = FakeDetector()
        run_detection(source, detector, max_frames=3, clock=clock)
        self.assertEqual([c["frame_index"] for c in detector.detect_calls], [0, 1, 2])
        self.assertEqual([c["timestamp"] for c in detector.detect_calls], [50.0, 50.0, 50.0])  # FrameSource damgası
        for i, call in enumerate(detector.detect_calls):
            self.assertTrue(np.array_equal(call["image"], make_frame(i)))

    def test_on_result_hook_gets_packet_and_result_for_every_frame(self) -> None:
        source, _ = camera_source(4)
        seen: list[tuple[int, int]] = []
        run_detection(source, FakeDetector(counts=[1, 2]), max_frames=4, on_result=lambda p, r: seen.append((p.frame_index, r.person_count)))
        self.assertEqual(seen, [(0, 1), (1, 2), (2, 2), (3, 2)])


class TimingTest(unittest.TestCase):
    def test_end_to_end_fps_and_latencies_with_window(self) -> None:
        clock = FakeClock(100.0)
        source, _ = camera_source(11, clock)
        window = FakeWindow(clock)  # show(delay=1 ms) saati 1 ms ilerletir
        detector = FakeDetector(clock=clock, inference_s=0.040)
        summary = run_detection(source, detector, window, max_frames=11, clock=clock)
        # her döngü: çıkarım 40 ms + gösterme 1 ms = 41 ms -> 10 aralık
        self.assertAlmostEqual(summary.end_to_end_fps, 1 / 0.041, places=6)
        self.assertAlmostEqual(summary.elapsed_s, 10 * 0.041, places=6)
        assert summary.inference and summary.frame_to_result and summary.frame_to_display and summary.read
        self.assertAlmostEqual(summary.inference.mean_ms, 40.0, places=6)
        self.assertAlmostEqual(summary.frame_to_result.mean_ms, 40.0, places=6)  # okuma bitişinden tespit bitişine
        self.assertAlmostEqual(summary.frame_to_display.mean_ms, 41.0, places=6)  # + çizim/gösterme
        self.assertAlmostEqual(summary.read.max_ms, 0.0, places=6)

    def test_end_to_end_fps_without_window_excludes_display_cost(self) -> None:
        clock = FakeClock(0.0)
        source, _ = camera_source(21, clock)
        summary = run_detection(source, FakeDetector(clock=clock, inference_s=0.050), max_frames=21, clock=clock)
        self.assertAlmostEqual(summary.end_to_end_fps, 20.0, places=6)
        self.assertIsNone(summary.frame_to_display)

    def test_fps_is_measured_not_taken_from_source_reported_fps(self) -> None:
        clock = FakeClock(0.0)
        capture = FakeCapture([make_frame(i) for i in range(11)], props={cv2.CAP_PROP_FPS: 15.0, cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 48})
        source = FrameSource(SourceConfig.camera(0, width=64, height=48), capture_factory=FakeFactory(capture), clock=clock, sleep=lambda s: None)
        source.open()
        self.assertEqual(source.source_fps, 15.0)
        summary = run_detection(source, FakeDetector(clock=clock, inference_s=0.0333), max_frames=11, clock=clock)
        self.assertAlmostEqual(summary.end_to_end_fps, 1 / 0.0333, places=4)  # sürücünün 15.0'ı değil
        self.assertGreater(summary.end_to_end_fps, 29.0)

    def test_inference_stats_use_detector_reported_time(self) -> None:
        clock = FakeClock(0.0)
        source, _ = camera_source(4, clock)

        class Varying(FakeDetector):
            def detect(self, image, **kw):  # type: ignore[no-untyped-def]
                self.inference_s = [0.010, 0.020, 0.030, 0.100][len(self.detect_calls)]
                return super().detect(image, **kw)

        summary = run_detection(source, Varying(clock=clock), max_frames=4, clock=clock)
        assert summary.inference
        self.assertAlmostEqual(summary.inference.mean_ms, 40.0)
        self.assertAlmostEqual(summary.inference.median_ms, 25.0)
        self.assertAlmostEqual(summary.inference.max_ms, 100.0)
        self.assertEqual(summary.inference.count, 4)

    def test_video_with_window_is_paced_and_omits_display_latency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = write_synthetic_video(Path(tmp) / "v.mp4", n_frames=30, fps=25.0, size=(64, 48))
            clock = FakeClock(0.0)
            window = FakeWindow(clock)
            with FrameSource(SourceConfig.video(video), clock=clock) as source:
                summary = run_detection(source, FakeDetector(clock=clock, inference_s=0.010), window, clock=clock)
        self.assertEqual(summary.frames_processed, 30)
        self.assertIsNone(summary.frame_to_display, "oynatma beklemesi 'görüntü gecikmesi' sayılmamalı")
        self.assertTrue(all(1 <= d <= 40 for d in window.delays), window.delays)  # 25 FPS -> en çok 40 ms
        self.assertGreater(max(window.delays), 20)
        # Video kaynağın hızında oynatılır. İlk karenin son teslim zamanı çıkarımdan sonra kurulduğu için ilk aralık 10 ms uzundur
        # (Aşama 3 run_preview ile aynı mantık); 30 karede bu sapma ~%0,8'dir (24,79 FPS).
        self.assertAlmostEqual(summary.end_to_end_fps, 25.0, delta=0.5)


class WindowAndExitTest(unittest.TestCase):
    def test_every_processed_frame_is_drawn_and_shown(self) -> None:
        source, _ = camera_source(3)
        window = FakeWindow()
        run_detection(source, FakeDetector(counts=[2, 0, 1]), window, max_frames=3)
        self.assertEqual(len(window.images), 3)
        for image in window.images:
            self.assertEqual(image.shape, (48, 64, 3))

    def test_overlay_shows_person_count_of_each_frame(self) -> None:
        source, _ = camera_source(3)
        texts: list[str] = []
        real = cv2.putText

        def spy(img, text, *a, **k):  # type: ignore[no-untyped-def]
            texts.append(text)
            return real(img, text, *a, **k)

        with mock.patch.object(cv2, "putText", side_effect=spy):
            run_detection(source, FakeDetector(counts=[2, 0, 1]), FakeWindow(), max_frames=3)
        counts = [t for t in texts if t.startswith("PERSONS:")]
        self.assertEqual(counts, ["PERSONS: 2", "PERSONS: 0", "PERSONS: 1"])
        self.assertTrue(any(t.startswith("INFERENCE:") for t in texts))
        self.assertTrue(any(t.startswith("FPS:") for t in texts))
        self.assertIn("SOURCE: CAMERA 0", texts)

    def test_source_frames_are_not_modified_by_drawing(self) -> None:
        source, _ = camera_source(2)
        detector = FakeDetector(counts=[3])
        window = FakeWindow()
        run_detection(source, detector, window, max_frames=2)
        for i, call in enumerate(detector.detect_calls):
            self.assertTrue(np.array_equal(call["image"], make_frame(i)), "kare üzerinde çizim yapılmamalı")
        self.assertFalse(np.array_equal(window.images[0], make_frame(0)))

    def test_exit_keys_q_upper_q_and_esc(self) -> None:
        for key in (ord("q"), ord("Q"), 27):
            with self.subTest(key=key):
                source, _ = camera_source(10)
                summary = run_detection(source, FakeDetector(), FakeWindow(keys={3: key}))
                self.assertEqual((summary.reason, summary.frames_processed), (REASON_USER_QUIT, 3))

    def test_other_keys_do_not_exit(self) -> None:
        source, _ = camera_source(5)
        summary = run_detection(source, FakeDetector(), FakeWindow(keys={2: ord("a"), 3: ord(" ")}), max_frames=5)
        self.assertEqual((summary.reason, summary.frames_processed), (REASON_MAX_FRAMES, 5))

    def test_window_closed_by_user_ends_loop(self) -> None:
        source, _ = camera_source(10)
        summary = run_detection(source, FakeDetector(), FakeWindow(close_after=2))
        self.assertEqual((summary.reason, summary.frames_processed), (REASON_WINDOW_CLOSED, 2))

    def test_ctrl_c_during_detection_returns_interrupted_summary(self) -> None:
        source, _ = camera_source(10)
        summary = run_detection(source, FakeDetector(detect_errors={3: KeyboardInterrupt()}))
        self.assertEqual((summary.reason, summary.frames_processed), (REASON_INTERRUPTED, 2))

    def test_ctrl_c_during_window_show_returns_interrupted_summary(self) -> None:
        source, _ = camera_source(10)
        summary = run_detection(source, FakeDetector(), FakeWindow(raise_on_call={2: KeyboardInterrupt()}))
        self.assertEqual((summary.reason, summary.frames_processed), (REASON_INTERRUPTED, 1))


class ErrorPropagationTest(unittest.TestCase):
    def test_camera_failure_mid_stream_propagates(self) -> None:
        source, capture = camera_source(2)  # 2 kare, sonra okuma başarısız
        with self.assertRaises(SourceDisconnectedError):
            run_detection(source, FakeDetector())
        self.assertGreaterEqual(capture.read_calls, 3)

    def test_detection_error_propagates(self) -> None:
        source, _ = camera_source(5)
        with self.assertRaises(InvalidFrameError):
            run_detection(source, FakeDetector(detect_errors={2: InvalidFrameError("bozuk kare")}))

    def test_display_error_propagates(self) -> None:
        source, _ = camera_source(5)
        with self.assertRaises(DisplayError):
            run_detection(source, FakeDetector(), FakeWindow(raise_on_call={2: DisplayError("pencere yok")}))


GUI_ENABLED = sys.platform == "darwin" and os.environ.get("ISG_GUI_TESTS") == "1"


@unittest.skipUnless(GUI_ENABLED, "Gerçek pencere testi: macOS + ISG_GUI_TESTS=1 + grafik oturumu gerekir")
class RealWindowDetectionLoopTest(unittest.TestCase):
    """Gerçek OpenCV/Cocoa penceresiyle tespit döngüsü: kırmızı düğme tıklaması (Aşama 3 düzeltmesi) tespit penceresinde de çalışmalı."""

    def test_red_button_click_ends_detection_loop_and_releases_camera(self) -> None:
        from isg_vision.camera.macos_window import CocoaRuntime

        frame = make_frame(0, 320, 240)
        capture = FakeCapture([frame] * 3000, props={cv2.CAP_PROP_FPS: 30.0})  # aynı dizi başvurusu: bellek küçük
        source = FrameSource(SourceConfig.camera(width=None, height=None), capture_factory=FakeFactory(capture))
        runtime = CocoaRuntime()
        window = PreviewWindow(WINDOW_NAME)

        def click() -> None:
            native = runtime.find_window(WINDOW_NAME)
            button = runtime._send(native, "standardWindowButton:", 0, argtypes=(ctypes.c_long,))
            runtime._send(
                button, "performSelectorOnMainThread:withObject:waitUntilDone:", runtime._objc.sel_registerName(b"performClick:"),
                None, False, argtypes=(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool),
            )

        timer = threading.Timer(0.5, click)
        timer.daemon = True
        timer.start()
        try:
            with source, window:
                summary = run_detection(source, FakeDetector(counts=[2]), window)
                self.assertIs(window.close_button_enabled, True)
        finally:
            timer.cancel()
        self.assertEqual(summary.reason, REASON_WINDOW_CLOSED)
        self.assertLess(summary.frames_processed, 3000)
        self.assertEqual(capture.release_calls, 1)

    def test_max_frames_run_with_real_window_draws_without_error(self) -> None:
        frame = make_frame(0, 320, 240)
        capture = FakeCapture([frame] * 30, props={cv2.CAP_PROP_FPS: 30.0})
        source = FrameSource(SourceConfig.camera(width=None, height=None), capture_factory=FakeFactory(capture))
        with source, PreviewWindow(WINDOW_NAME) as window:
            summary = run_detection(source, FakeDetector(counts=[0, 1, 3]), window, max_frames=30)
        self.assertEqual((summary.reason, summary.frames_processed), (REASON_MAX_FRAMES, 30))


if __name__ == "__main__":
    unittest.main()
