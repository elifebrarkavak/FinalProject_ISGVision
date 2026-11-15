"""Kare akışı -> yüz tespiti -> görselleştirme döngüsü testleri (sahte kaynak/pencere/tespitçi; gerçek kamera ve model yok).

Gerçek bir Cocoa penceresi gerektiren iki test isteğe bağlıdır (``ISG_GUI_TESTS=1``); varsayılan çalıştırmada açık nedenle ATLANIR.
"""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from camera_test_support import FakeCapture, FakeClock, FakeFactory, FakeWindow, make_frame, write_synthetic_video  # noqa: E402
from face_detection_test_support import FakeFaceDetector  # noqa: E402

from isg_vision.camera import FrameSource, SourceConfig  # noqa: E402
from isg_vision.camera.display import PreviewWindow  # noqa: E402  (modül başında: test_no_side_effects modülleri yeniden yükler)
from isg_vision.camera.errors import DisplayError, SourceDisconnectedError  # noqa: E402
from isg_vision.face_detection.errors import FaceInvalidFrameError  # noqa: E402
from isg_vision.face_detection.pipeline import (  # noqa: E402
    REASON_END_OF_STREAM,
    REASON_INTERRUPTED,
    REASON_MAX_FRAMES,
    REASON_USER_QUIT,
    REASON_WINDOW_CLOSED,
    LatencyStats,
    run_face_detection,
)
from isg_vision.face_detection.visualization import NOTICE, WINDOW_NAME  # noqa: E402


def camera_source(n_frames: int, clock: FakeClock | None = None) -> tuple[FrameSource, FakeCapture]:
    capture = FakeCapture([make_frame(i) for i in range(n_frames)], props={cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 48})
    kwargs = {"clock": clock} if clock is not None else {}
    source = FrameSource(SourceConfig.camera(0, width=64, height=48), capture_factory=FakeFactory(capture), max_read_failures=1, sleep=lambda s: None, **kwargs)
    source.open()
    return source, capture


class SummaryTest(unittest.TestCase):
    def test_counts_faces_and_frames(self) -> None:
        source, _ = camera_source(5)
        detector = FakeFaceDetector(counts=[0, 1, 2, 4, 0])
        summary = run_face_detection(source, detector, max_frames=5)
        self.assertEqual(summary.reason, REASON_MAX_FRAMES)
        self.assertEqual((summary.frames_processed, summary.faces_total, summary.faces_max, summary.frames_with_faces), (5, 7, 4, 3))
        self.assertAlmostEqual(summary.faces_mean, 7 / 5)
        self.assertEqual(len(detector.detect_calls), 5, "kare başına tam bir tespit çağrısı")

    def test_frames_without_any_face_are_handled(self) -> None:
        source, _ = camera_source(8)
        summary = run_face_detection(source, FakeFaceDetector(counts=[0]), max_frames=8)
        self.assertEqual((summary.frames_processed, summary.faces_total, summary.faces_max, summary.frames_with_faces, summary.faces_mean), (8, 0, 0, 0, 0.0))

    def test_dropped_and_truncated_boxes_accumulate(self) -> None:
        source, _ = camera_source(4)
        summary = run_face_detection(source, FakeFaceDetector(counts=[1], dropped=2, truncated=3), max_frames=4)
        self.assertEqual((summary.dropped_boxes, summary.truncated_faces), (8, 12))

    def test_latency_statistics_are_collected(self) -> None:
        clock = FakeClock()
        source, _ = camera_source(10, clock)
        summary = run_face_detection(source, FakeFaceDetector(counts=[1], clock=clock, inference_s=0.012), clock=clock, max_frames=10)
        for stats in (summary.inference, summary.read, summary.frame_to_result):
            self.assertIsInstance(stats, LatencyStats)
            self.assertEqual(stats.count, 10)
        self.assertAlmostEqual(summary.inference.median_ms, 12.0)  # type: ignore[union-attr]
        self.assertIsNone(summary.frame_to_display, "pencere yokken kare->görüntü ölçülmez")

    def test_end_to_end_fps_uses_the_measured_clock(self) -> None:
        clock = FakeClock()
        source, _ = camera_source(11, clock)
        # her karede 40 ms çıkarım; kare okuma saati ilerletmez -> (11-1)/(10*0.04) = 25 FPS
        summary = run_face_detection(source, FakeFaceDetector(counts=[1], clock=clock, inference_s=0.040), clock=clock, max_frames=11)
        self.assertAlmostEqual(summary.end_to_end_fps, 25.0, places=1)

    def test_summary_is_immutable(self) -> None:
        source, _ = camera_source(1)
        summary = run_face_detection(source, FakeFaceDetector(), max_frames=1)
        with self.assertRaises(Exception):
            summary.frames_processed = 99  # type: ignore[misc]


class ExitPathsTest(unittest.TestCase):
    def test_max_frames(self) -> None:
        source, capture = camera_source(50)
        summary = run_face_detection(source, FakeFaceDetector(), max_frames=7)
        self.assertEqual((summary.reason, summary.frames_processed), (REASON_MAX_FRAMES, 7))
        self.assertLess(capture.read_calls, 50)

    def test_q_and_esc_keys_end_the_loop(self) -> None:
        for key in (ord("q"), ord("Q"), 27):
            with self.subTest(key=key):
                source, _ = camera_source(30)
                window = FakeWindow(keys={4: key})
                summary = run_face_detection(source, FakeFaceDetector(), window)
                self.assertEqual((summary.reason, summary.frames_processed), (REASON_USER_QUIT, 4))

    def test_window_closed_by_user(self) -> None:
        source, _ = camera_source(30)
        summary = run_face_detection(source, FakeFaceDetector(), FakeWindow(close_after=6))
        self.assertEqual((summary.reason, summary.frames_processed), (REASON_WINDOW_CLOSED, 6))

    def test_end_of_stream_with_window_for_a_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = write_synthetic_video(Path(tmp) / "v.mp4", n_frames=3, fps=25.0, size=(64, 48))
            window = FakeWindow()
            with FrameSource(SourceConfig.video(video)) as source:
                summary = run_face_detection(source, FakeFaceDetector(), window)
        self.assertEqual((summary.reason, len(window.images)), (REASON_END_OF_STREAM, 3))

    def test_camera_running_out_of_frames_is_a_source_error_not_a_silent_end(self) -> None:
        source, _ = camera_source(2)
        with self.assertRaises(SourceDisconnectedError):
            run_face_detection(source, FakeFaceDetector())

    def test_ctrl_c_in_the_source_is_reported_not_raised(self) -> None:
        class InterruptingCapture(FakeCapture):  # FakeCapture yalnızca Exception alt sınıflarını fırlatır; Ctrl+C BaseException'dır
            def read(self):  # type: ignore[no-untyped-def]
                if self.read_calls >= 1:
                    raise KeyboardInterrupt
                return super().read()

        capture = InterruptingCapture([make_frame(0), make_frame(1)], props={cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 48})
        source = FrameSource(SourceConfig.camera(0, width=64, height=48), capture_factory=FakeFactory(capture), sleep=lambda s: None)
        source.open()
        summary = run_face_detection(source, FakeFaceDetector())
        self.assertEqual((summary.reason, summary.frames_processed), (REASON_INTERRUPTED, 1))

    def test_ctrl_c_in_the_detector_is_reported_not_raised(self) -> None:
        source, _ = camera_source(10)
        summary = run_face_detection(source, FakeFaceDetector(detect_errors={3: KeyboardInterrupt()}))
        self.assertEqual((summary.reason, summary.frames_processed), (REASON_INTERRUPTED, 2))

    def test_invalid_max_frames_rejected(self) -> None:
        source, _ = camera_source(2)
        for bad in (0, -3, 1.5, True, "5"):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                run_face_detection(source, FakeFaceDetector(), max_frames=bad)  # type: ignore[arg-type]


class ErrorPropagationTest(unittest.TestCase):
    def test_detector_errors_propagate_to_the_caller(self) -> None:
        source, _ = camera_source(5)
        with self.assertRaises(FaceInvalidFrameError):
            run_face_detection(source, FakeFaceDetector(detect_errors={2: FaceInvalidFrameError("bozuk kare")}))

    def test_source_errors_propagate(self) -> None:
        capture = FakeCapture([make_frame(0), None, None], props={cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 48})
        source = FrameSource(SourceConfig.camera(0, width=64, height=48), capture_factory=FakeFactory(capture), max_read_failures=1, sleep=lambda s: None)
        source.open()
        with self.assertRaises(SourceDisconnectedError):
            run_face_detection(source, FakeFaceDetector())

    def test_display_errors_propagate(self) -> None:
        source, _ = camera_source(5)
        with self.assertRaises(DisplayError):
            run_face_detection(source, FakeFaceDetector(), FakeWindow(raise_on_call={2: DisplayError("pencere öldü")}))


class OnResultHookTest(unittest.TestCase):
    def test_hook_receives_packet_and_result_for_the_same_frame_in_order(self) -> None:
        source, _ = camera_source(4)
        seen = []
        run_face_detection(source, FakeFaceDetector(counts=[1, 2, 0, 1]), max_frames=4, on_result=lambda packet, result: seen.append((packet, result)))
        self.assertEqual(len(seen), 4)
        for packet, result in seen:
            self.assertEqual((result.frame_index, result.timestamp), (packet.frame_index, packet.timestamp))
            self.assertEqual(result.image_size, (packet.width, packet.height), "5B: aynı karenin yüz ve kişi sonuçları image_size ile eşleşebilmeli")
        self.assertEqual([r.face_count for _, r in seen], [1, 2, 0, 1])
        self.assertEqual([p.frame_index for p, _ in seen], sorted(p.frame_index for p, _ in seen))

    def test_hook_runs_before_drawing_and_does_not_see_overlays(self) -> None:
        source, _ = camera_source(2)
        window = FakeWindow()
        frames = []
        run_face_detection(source, FakeFaceDetector(), window, max_frames=2, on_result=lambda packet, result: frames.append(packet.frame.copy()))
        for original, shown in zip(frames, window.images):
            self.assertFalse(np.array_equal(original, shown))  # gösterilen görüntü çizimli
            self.assertTrue((original == original[0, 0]).all(), "kanca kaynak kareyi çizimsiz almalı")


class WindowDrawingTest(unittest.TestCase):
    def test_source_frame_is_never_modified_by_drawing(self) -> None:
        source, _ = camera_source(3)
        originals = []
        window = FakeWindow()
        run_face_detection(source, FakeFaceDetector(counts=[2]), window, max_frames=3, on_result=lambda packet, result: originals.append((packet.frame, packet.frame.copy())))
        for frame, snapshot in originals:
            np.testing.assert_array_equal(frame, snapshot)
        self.assertEqual(len(window.images), 3)

    def test_displayed_image_has_the_same_size_and_the_notice_band(self) -> None:
        source, _ = camera_source(1)
        window = FakeWindow()
        run_face_detection(source, FakeFaceDetector(), window, max_frames=1)
        self.assertEqual(window.images[0].shape, (48, 64, 3))
        self.assertIsInstance(NOTICE, str)

    def test_frame_to_display_is_measured_for_camera_with_window(self) -> None:
        clock = FakeClock()
        source, _ = camera_source(5, clock)
        summary = run_face_detection(source, FakeFaceDetector(clock=clock), FakeWindow(clock), clock=clock, max_frames=5)
        self.assertIsNotNone(summary.frame_to_display)
        self.assertEqual(summary.frame_to_display.count, 5)  # type: ignore[union-attr]

    def test_video_with_window_is_paced_and_display_latency_is_not_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = write_synthetic_video(Path(tmp) / "v.mp4", n_frames=6, fps=25.0, size=(64, 48))
            clock = FakeClock()
            window = FakeWindow(clock)
            with FrameSource(SourceConfig.video(video), clock=clock) as source:
                summary = run_face_detection(source, FakeFaceDetector(clock=clock), window, clock=clock)
        self.assertEqual(summary.frames_processed, 6)
        self.assertIsNone(summary.frame_to_display)
        self.assertTrue(all(d >= 1 for d in window.delays))

    def test_video_without_window_runs_unpaced_to_the_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = write_synthetic_video(Path(tmp) / "v.mp4", n_frames=8, fps=25.0, size=(64, 48))
            with FrameSource(SourceConfig.video(video)) as source:
                summary = run_face_detection(source, FakeFaceDetector(counts=[1, 0, 1, 0, 1, 0, 1, 0]))
        self.assertEqual((summary.reason, summary.frames_processed, summary.frames_with_faces), (REASON_END_OF_STREAM, 8, 4))


class Stage5BReadinessTest(unittest.TestCase):
    """Aşama 5B için zemin: aynı karenin yüz sonucu ile 4B takip sonucu aynı ``frame_index``/``image_size`` ile eşleşebilir (eşleştirme YOK)."""

    def test_face_result_can_run_alongside_the_person_detector_and_tracker_hooks(self) -> None:
        from person_detection_test_support import FakeDetector

        source, _ = camera_source(3)
        person_detector = FakeDetector(counts=[2])
        pairs = []

        def hook(packet, face_result):  # type: ignore[no-untyped-def]
            person_result = person_detector.detect(packet.frame, frame_index=packet.frame_index, timestamp=packet.timestamp)
            pairs.append((face_result, person_result))

        run_face_detection(source, FakeFaceDetector(counts=[1]), max_frames=3, on_result=hook)
        for face_result, person_result in pairs:
            self.assertEqual((face_result.frame_index, face_result.image_size), (person_result.frame_index, person_result.image_size))
            self.assertEqual(face_result.timestamp, person_result.timestamp)


GUI_ENABLED = sys.platform == "darwin" and os.environ.get("ISG_GUI_TESTS") == "1"


@unittest.skipUnless(GUI_ENABLED, "Gerçek pencere testi: macOS + ISG_GUI_TESTS=1 + grafik oturumu gerekir")
class RealWindowFaceLoopTest(unittest.TestCase):
    """Gerçek OpenCV/Cocoa penceresiyle yüz döngüsü (kamera YOK; sahte kareler). Aşama 3'te çözülen macOS kırmızı kapatma düğmesi
    düzeltmesi yüz penceresinde de çalışmalı: düğme tıklanınca döngü biter ve kaynaklar serbest kalır."""

    def test_red_button_click_ends_face_loop_and_releases_camera(self) -> None:
        from isg_vision.camera.macos_window import CocoaRuntime

        frame = make_frame(0, 320, 240)
        capture = FakeCapture([frame] * 3000, props={cv2.CAP_PROP_FPS: 30.0})
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
                summary = run_face_detection(source, FakeFaceDetector(counts=[2]), window)
                self.assertIs(window.close_button_enabled, True)
        finally:
            timer.cancel()
        self.assertEqual(summary.reason, REASON_WINDOW_CLOSED)
        self.assertLess(summary.frames_processed, 3000)
        self.assertEqual(capture.release_calls, 1)

    def test_max_frames_run_with_real_window_draws_faces_without_error(self) -> None:
        frame = make_frame(0, 320, 240)
        capture = FakeCapture([frame] * 30, props={cv2.CAP_PROP_FPS: 30.0})
        source = FrameSource(SourceConfig.camera(width=None, height=None), capture_factory=FakeFactory(capture))
        with source, PreviewWindow(WINDOW_NAME) as window:
            summary = run_face_detection(source, FakeFaceDetector(counts=[0, 1, 3]), window, max_frames=30)
        self.assertEqual((summary.reason, summary.frames_processed), (REASON_MAX_FRAMES, 30))


if __name__ == "__main__":
    unittest.main()
