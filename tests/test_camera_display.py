"""Aşama 3 / Grup 2: overlay, pencere ve önizleme döngüsü testleri.

Hiçbir gerçek pencere açılmaz: OpenCV GUI çağrıları taklit edilir (``mock``) veya ``FakeWindow`` kullanılır.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from camera_test_support import FakeCapture, FakeClock, FakeFactory, FakeWindow, make_frame  # noqa: E402
from isg_vision.camera import FrameSource, SourceConfig  # noqa: E402
from isg_vision.camera.display import (  # noqa: E402
    EXIT_HINT,
    OverlayInfo,
    PreviewWindow,
    ascii_text,
    draw_overlay,
    is_exit_key,
)
from isg_vision.camera.errors import DisplayError  # noqa: E402
from isg_vision.camera.preview import (  # noqa: E402
    REASON_END_OF_STREAM,
    REASON_INTERRUPTED,
    REASON_MAX_FRAMES,
    REASON_USER_QUIT,
    REASON_WINDOW_CLOSED,
    run_preview,
)

FPS, COUNT = cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT


class AsciiAndKeysTest(unittest.TestCase):
    def test_turkish_characters_are_transliterated(self) -> None:
        self.assertEqual(ascii_text("ÇÖŞĞÜİ çöşğüı"), "COSGUI cosgui")
        self.assertEqual(ascii_text("kamera-görüntüsü.mp4"), "kamera-goruntusu.mp4")

    def test_unrenderable_characters_become_question_mark(self) -> None:
        self.assertEqual(ascii_text("a日b"), "a?b")

    def test_exit_keys(self) -> None:
        for key in (ord("q"), ord("Q"), 27, 0x100000 | ord("q")):  # son: üst bitlerde değiştirici olan tuş
            self.assertTrue(is_exit_key(key), key)
        for key in (-1, ord("a"), ord("w"), 13, 32):
            self.assertFalse(is_exit_key(key), key)


class OverlayTest(unittest.TestCase):
    INFO = OverlayInfo("CAMERA 0", 29.64, (1280, 720), 30.0, "LIVE")

    def texts_drawn(self, image: np.ndarray, info: OverlayInfo) -> list[str]:
        with mock.patch.object(cv2, "putText", wraps=cv2.putText) as spy:
            draw_overlay(image, info)
        return [call.args[1] for call in spy.call_args_list]

    def test_required_information_is_drawn(self) -> None:
        texts = self.texts_drawn(make_frame(200, 1280, 720), self.INFO)
        self.assertIn("SOURCE: CAMERA 0", texts)
        self.assertIn("FPS: 29.6", texts)
        self.assertIn("RESOLUTION: 1280x720", texts)
        self.assertIn("SOURCE FPS: 30.0", texts)
        self.assertIn("STATUS: LIVE", texts)
        self.assertIn(EXIT_HINT, texts)
        self.assertEqual(EXIT_HINT, "Press Q or ESC to exit")

    def test_unknown_values_and_note(self) -> None:
        info = OverlayInfo("VIDEO klip-görüntü.mp4", 0.0, None, None, "PLAYING 3/10", note="requested 1920x1080, got 640x480")
        texts = self.texts_drawn(make_frame(200, 640, 480), info)
        self.assertIn("SOURCE: VIDEO klip-goruntu.mp4", texts)  # ASCII'ye indirgenir
        self.assertIn("RESOLUTION: N/A", texts)
        self.assertIn("SOURCE FPS: N/A", texts)
        self.assertIn("NOTE: requested 1920x1080, got 640x480", texts)

    def test_input_image_is_not_modified_and_output_is_a_copy(self) -> None:
        original = make_frame(200, 640, 480)
        snapshot = original.copy()
        out = draw_overlay(original, self.INFO)
        self.assertTrue(np.array_equal(original, snapshot), "kaynak kare değiştirilmemeli")
        self.assertIsNot(out, original)
        self.assertEqual(out.shape, original.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertFalse(np.array_equal(out, original), "overlay çizilmiş olmalı")

    def test_text_band_darkens_top_and_bottom_but_not_middle(self) -> None:
        original = make_frame(200, 640, 480)
        out = draw_overlay(original, self.INFO)
        self.assertLess(float(out[:20].mean()), float(original[:20].mean()))
        self.assertLess(float(out[-20:].mean()), float(original[-20:].mean()))
        self.assertTrue(np.array_equal(out[200:280], original[200:280]))

    def test_various_sizes_do_not_crash(self) -> None:
        for w, h in ((32, 24), (64, 48), (320, 240), (640, 480), (1280, 720), (1920, 1080), (480, 1080)):
            with self.subTest(size=(w, h)):
                out = draw_overlay(make_frame(100, w, h), self.INFO)
                self.assertEqual(out.shape, (h, w, 3))


class PreviewWindowTest(unittest.TestCase):
    def setUp(self) -> None:
        patches = {
            "namedWindow": mock.patch.object(cv2, "namedWindow"),
            "imshow": mock.patch.object(cv2, "imshow"),
            "waitKey": mock.patch.object(cv2, "waitKey", return_value=-1),
            "getWindowProperty": mock.patch.object(cv2, "getWindowProperty", return_value=1.0),
            "destroyWindow": mock.patch.object(cv2, "destroyWindow"),
        }
        self.m = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

    def test_window_created_lazily_and_once(self) -> None:
        window = PreviewWindow("w")
        self.m["namedWindow"].assert_not_called()
        window.show(make_frame(0))
        window.show(make_frame(1))
        self.m["namedWindow"].assert_called_once()
        self.assertEqual(self.m["imshow"].call_count, 2)

    def test_show_returns_pressed_key_and_passes_delay(self) -> None:
        self.m["waitKey"].return_value = ord("q")
        window = PreviewWindow("w")
        self.assertEqual(window.show(make_frame(0), delay_ms=17), ord("q"))
        self.m["waitKey"].assert_called_with(17)

    def test_namedwindow_failure_becomes_display_error(self) -> None:
        self.m["namedWindow"].side_effect = cv2.error("grafik oturumu yok")
        with self.assertRaisesRegex(DisplayError, "penceresi oluşturulamadı"):
            PreviewWindow("w").open()

    def test_imshow_failure_becomes_display_error(self) -> None:
        self.m["imshow"].side_effect = cv2.error("GUI yok")
        with self.assertRaises(DisplayError):
            PreviewWindow("w").show(make_frame(0))

    def test_closed_by_user_detection(self) -> None:
        window = PreviewWindow("w")
        self.assertFalse(window.closed_by_user)  # henüz gösterilmedi
        window.show(make_frame(0))
        self.assertFalse(window.closed_by_user)  # görünür
        for gone in (0.0, -1.0):
            self.m["getWindowProperty"].return_value = gone
            self.assertTrue(window.closed_by_user)
        self.m["getWindowProperty"].side_effect = cv2.error("pencere yok")
        self.assertTrue(window.closed_by_user)

    def test_close_is_idempotent(self) -> None:
        window = PreviewWindow("w")
        window.close()  # hiç açılmadı
        self.m["destroyWindow"].assert_not_called()
        window.show(make_frame(0))
        window.close()
        window.close()
        self.m["destroyWindow"].assert_called_once_with("w")

    def test_context_manager_closes_window_on_exception(self) -> None:
        with self.assertRaises(RuntimeError):
            with PreviewWindow("w"):
                raise RuntimeError("hata")
        self.m["destroyWindow"].assert_called_once_with("w")

    def test_destroy_failure_is_swallowed(self) -> None:
        window = PreviewWindow("w")
        window.show(make_frame(0))
        self.m["destroyWindow"].side_effect = cv2.error("zaten kapalı")
        window.close()  # fırlatmamalı


class PreviewLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.video = Path(self._tmp.name) / "klip.mp4"
        self.video.write_bytes(b"x")  # yalnızca varlık denetimi için; FakeCapture içeriği okumaz

    def video_source(self, n: int, fps: float = 25.0, clock: FakeClock | None = None) -> FrameSource:
        cap = FakeCapture([make_frame(i) for i in range(n)], props={FPS: fps, COUNT: float(n)})
        return FrameSource(SourceConfig.video(self.video), capture_factory=FakeFactory(cap), clock=clock or FakeClock())

    def camera_source(self, n: int) -> tuple[FrameSource, FakeCapture]:
        cap = FakeCapture([make_frame(i) for i in range(n)], props={FPS: 30.0})
        return FrameSource(SourceConfig.camera(width=None, height=None), capture_factory=FakeFactory(cap)), cap

    def test_video_plays_to_end_and_reports_summary(self) -> None:
        clock = FakeClock()
        window = FakeWindow(clock)
        with self.video_source(10, clock=clock) as source:
            result = run_preview(source, window, clock=clock)
        self.assertEqual(result.reason, REASON_END_OF_STREAM)
        self.assertEqual(result.frames_processed, 10)
        self.assertEqual(len(window.images), 10)

    def test_video_playback_is_paced_to_source_fps_not_max_speed(self) -> None:
        clock = FakeClock()
        window = FakeWindow(clock)
        t0 = clock()
        with self.video_source(50, fps=25.0, clock=clock) as source:
            result = run_preview(source, window, clock=clock)
        self.assertTrue(all(38 <= d <= 40 for d in window.delays), window.delays[:5])  # ~40 ms = 1/25 s
        self.assertAlmostEqual(clock() - t0, 50 / 25.0, delta=0.1)  # 50 kare ~2 sn sürmeli
        self.assertAlmostEqual(result.measured_fps, 25.0, delta=0.7)

    def test_slow_processing_does_not_burst_or_drop_frames(self) -> None:
        clock = FakeClock()
        window = FakeWindow(clock, extra_cost_s=0.2)  # her kare 200 ms sürüyor; kaynak 25 FPS (40 ms)
        with self.video_source(10, fps=25.0, clock=clock) as source:
            result = run_preview(source, window, clock=clock)
        self.assertEqual(result.frames_processed, 10)  # kare atlanmadı
        self.assertEqual(window.delays[1:], [1] * 9)  # geri kalan karede bekleme yok, hızlanma (burst) yok
        self.assertAlmostEqual(result.measured_fps, 5.0, delta=0.2)  # gerçek performans olduğu gibi raporlanır

    def test_video_without_reported_fps_uses_default_30(self) -> None:
        clock = FakeClock()
        window = FakeWindow(clock)
        with self.video_source(5, fps=0.0, clock=clock) as source:
            run_preview(source, window, clock=clock)
        self.assertTrue(all(32 <= d <= 34 for d in window.delays), window.delays)

    def test_camera_is_not_paced(self) -> None:
        source, _ = self.camera_source(8)
        window = FakeWindow()
        with source:
            result = run_preview(source, window, max_frames=8)
        self.assertEqual(result.reason, REASON_MAX_FRAMES)
        self.assertEqual(window.delays, [1] * 8)

    def test_no_window_runs_without_waiting_and_respects_max_frames(self) -> None:
        with self.video_source(100) as source:
            result = run_preview(source, None, max_frames=7)
        self.assertEqual(result.reason, REASON_MAX_FRAMES)
        self.assertEqual(result.frames_processed, 7)
        self.assertEqual(source.frames_read, 7)  # fazladan kare okunmadı

    def test_no_window_video_runs_to_end(self) -> None:
        with self.video_source(12) as source:
            result = run_preview(source, None)
        self.assertEqual((result.reason, result.frames_processed), (REASON_END_OF_STREAM, 12))

    def test_q_key_stops_playback(self) -> None:
        window = FakeWindow(keys={3: ord("q")})
        source, _ = self.camera_source(20)
        with source:
            result = run_preview(source, window)
        self.assertEqual((result.reason, result.frames_processed), (REASON_USER_QUIT, 3))

    def test_esc_key_stops_playback(self) -> None:
        window = FakeWindow(keys={2: 27})
        source, _ = self.camera_source(20)
        with source:
            result = run_preview(source, window)
        self.assertEqual((result.reason, result.frames_processed), (REASON_USER_QUIT, 2))

    def test_other_keys_do_not_stop(self) -> None:
        window = FakeWindow(keys={1: ord("a"), 2: 13})
        with self.video_source(4) as source:
            result = run_preview(source, window)
        self.assertEqual((result.reason, result.frames_processed), (REASON_END_OF_STREAM, 4))

    def test_closing_window_stops_playback(self) -> None:
        window = FakeWindow(close_after=4)
        source, _ = self.camera_source(20)
        with source:
            result = run_preview(source, window)
        self.assertEqual((result.reason, result.frames_processed), (REASON_WINDOW_CLOSED, 4))

    def test_ctrl_c_is_handled_and_resources_are_released(self) -> None:
        window = FakeWindow(raise_on_call={3: KeyboardInterrupt()})
        source, cap = self.camera_source(20)
        with source:
            result = run_preview(source, window)
            self.assertEqual(result.reason, REASON_INTERRUPTED)
            self.assertEqual(result.frames_processed, 2)
        self.assertEqual(cap.release_calls, 1)

    def test_display_error_propagates_and_source_is_still_released(self) -> None:
        window = FakeWindow(raise_on_call={2: DisplayError("pencere öldü")})
        source, cap = self.camera_source(20)
        with self.assertRaises(DisplayError):
            with source:
                run_preview(source, window)
        self.assertEqual(cap.release_calls, 1)

    def test_frames_are_not_modified_by_overlay(self) -> None:
        window = FakeWindow()
        original = make_frame(77, 64, 48)
        cap = FakeCapture([original])
        with FrameSource(SourceConfig.camera(width=None, height=None), capture_factory=FakeFactory(cap)) as source:
            run_preview(source, window, max_frames=1)
        self.assertTrue(np.array_equal(original, make_frame(77, 64, 48)))
        self.assertFalse(np.array_equal(window.images[0], original))

    def test_video_overlay_shows_frame_position_and_camera_shows_live(self) -> None:
        with mock.patch.object(cv2, "putText", wraps=cv2.putText) as spy:
            with self.video_source(5) as source:
                run_preview(source, FakeWindow(), max_frames=2)
        statuses = [c.args[1] for c in spy.call_args_list if c.args[1].startswith("STATUS")]
        self.assertEqual(statuses, ["STATUS: PLAYING 1/5", "STATUS: PLAYING 2/5"])
        with mock.patch.object(cv2, "putText", wraps=cv2.putText) as spy:
            source, _ = self.camera_source(2)
            with source:
                run_preview(source, FakeWindow(), max_frames=1)
        self.assertIn("STATUS: LIVE", [c.args[1] for c in spy.call_args_list])

    def test_resolution_mismatch_is_shown_as_note(self) -> None:
        cap = FakeCapture([make_frame(0, 640, 480)], props={cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480},
                          accept_set=False)
        source = FrameSource(SourceConfig.camera(width=1920, height=1080), capture_factory=FakeFactory(cap))
        with mock.patch.object(cv2, "putText", wraps=cv2.putText) as spy:
            with source:
                run_preview(source, FakeWindow(), max_frames=1)
        self.assertIn("NOTE: requested 1920x1080, got 640x480", [c.args[1] for c in spy.call_args_list])

    def test_invalid_max_frames_rejected(self) -> None:
        source, _ = self.camera_source(1)
        for bad in (0, -1, 1.5, True):
            with self.subTest(max_frames=bad):
                with self.assertRaises(ValueError):
                    run_preview(source, None, max_frames=bad)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
