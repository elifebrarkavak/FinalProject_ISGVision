"""Aşama 3 / macOS pencere kapatma düzeltmesi testleri.

Kök neden (ölçüldü): OpenCV 5.0.0 Cocoa penceresi ``Closable`` stil biti olmadan açılır; kırmızı kapatma düğmesi devre dışıdır.
Düzeltme: ``isg_vision.camera.macos_window.enable_close_button``.

Varsayılan olarak **hiçbir gerçek pencere açılmaz**: Objective-C çalışma zamanı sahtelenir, OpenCV GUI çağrıları ``mock`` ile
taklit edilir. Gerçek Cocoa penceresiyle çalışan testler yalnızca ``ISG_GUI_TESTS=1`` ortam değişkeniyle ve macOS'ta çalışır
(grafik oturumu olan bir terminal gerekir); aksi hâlde atlanır.
"""

from __future__ import annotations

import contextlib
import ctypes
import io
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from camera_test_support import FakeCapture, FakeFactory, FakeWindow, make_frame, write_synthetic_video  # noqa: E402
from isg_vision.camera import FrameSource, SourceConfig  # noqa: E402
from isg_vision.camera import display as display_module  # noqa: E402
from isg_vision.camera import macos_window  # noqa: E402
from isg_vision.camera.cli import EXIT_OK, main  # noqa: E402
from isg_vision.camera.display import PreviewWindow  # noqa: E402
from isg_vision.camera.macos_window import (  # noqa: E402
    NS_WINDOW_STYLE_MASK_CLOSABLE,
    CocoaRuntime,
    enable_close_button,
)
from isg_vision.camera.preview import (  # noqa: E402
    REASON_END_OF_STREAM,
    REASON_MAX_FRAMES,
    REASON_USER_QUIT,
    REASON_WINDOW_CLOSED,
    run_preview,
)

OPENCV_COCOA_MASK = 0xD  # Titled | Miniaturizable | Resizable: gerçek OpenCV 5.0.0 penceresinde ölçülen değer (Closable yok)
FPS = cv2.CAP_PROP_FPS


class FakeRuntime:
    """Objective-C çalışma zamanının sahte sürümü: pencere = ``{başlık: styleMask}`` sözlüğü."""

    def __init__(self, windows: dict[str, int] | None = None, *, apply_mask: bool = True) -> None:
        self.windows = dict(windows or {})
        self.apply_mask = apply_mask  # False: setStyleMask çağrısı etkisiz (başarısız etkinleştirme taklidi)
        self.find_calls: list[str] = []
        self.set_calls: list[tuple[str, int]] = []
        self.find_error: BaseException | None = None

    def find_window(self, title: str) -> Any | None:
        self.find_calls.append(title)
        if self.find_error is not None:
            raise self.find_error
        return title if title in self.windows else None

    def style_mask(self, window: Any) -> int:
        return self.windows[window]

    def set_style_mask(self, window: Any, mask: int) -> None:
        self.set_calls.append((window, mask))
        if self.apply_mask:
            self.windows[window] = mask


class ClosableBitTest(unittest.TestCase):
    def test_measured_opencv_mask_lacks_closable_bit_and_fix_yields_0xf(self) -> None:
        self.assertEqual(NS_WINDOW_STYLE_MASK_CLOSABLE, 0x2)
        self.assertFalse(OPENCV_COCOA_MASK & NS_WINDOW_STYLE_MASK_CLOSABLE)  # kök neden: kapatma düğmesi devre dışı
        self.assertEqual(OPENCV_COCOA_MASK | NS_WINDOW_STYLE_MASK_CLOSABLE, 0xF)


class EnableCloseButtonTest(unittest.TestCase):
    def test_adds_closable_bit_when_missing(self) -> None:
        runtime = FakeRuntime({"w": OPENCV_COCOA_MASK})
        self.assertTrue(enable_close_button("w", runtime=runtime, platform="darwin"))
        self.assertEqual(runtime.set_calls, [("w", 0xF)])
        self.assertEqual(runtime.windows["w"], 0xF)

    def test_already_closable_window_is_left_untouched(self) -> None:
        runtime = FakeRuntime({"w": 0xF})
        self.assertTrue(enable_close_button("w", runtime=runtime, platform="darwin"))
        self.assertEqual(runtime.set_calls, [])

    def test_only_the_named_window_is_changed(self) -> None:
        runtime = FakeRuntime({"a": OPENCV_COCOA_MASK, "b": OPENCV_COCOA_MASK})
        enable_close_button("b", runtime=runtime, platform="darwin")
        self.assertEqual(runtime.windows, {"a": OPENCV_COCOA_MASK, "b": 0xF})

    def test_missing_window_returns_false_without_changes(self) -> None:
        runtime = FakeRuntime({"baska": OPENCV_COCOA_MASK})
        self.assertFalse(enable_close_button("w", runtime=runtime, platform="darwin"))
        self.assertEqual(runtime.set_calls, [])

    def test_ineffective_set_is_reported_as_failure(self) -> None:
        runtime = FakeRuntime({"w": OPENCV_COCOA_MASK}, apply_mask=False)
        self.assertFalse(enable_close_button("w", runtime=runtime, platform="darwin"))

    def test_non_macos_platforms_never_touch_the_runtime(self) -> None:
        for platform in ("linux", "win32", "freebsd"):
            with self.subTest(platform=platform):
                runtime = FakeRuntime({"w": OPENCV_COCOA_MASK})
                self.assertFalse(enable_close_button("w", runtime=runtime, platform=platform))
                self.assertEqual((runtime.find_calls, runtime.set_calls), ([], []))

    def test_non_main_thread_never_touches_the_runtime(self) -> None:
        runtime = FakeRuntime({"w": OPENCV_COCOA_MASK})
        result: list[bool] = []
        thread = threading.Thread(target=lambda: result.append(enable_close_button("w", runtime=runtime, platform="darwin")))
        thread.start()
        thread.join()
        self.assertEqual(result, [False])
        self.assertEqual((runtime.find_calls, runtime.set_calls), ([], []))

    def test_runtime_errors_never_propagate(self) -> None:
        for error in (RuntimeError("x"), OSError("libobjc yok"), AttributeError("y"), ctypes.ArgumentError("z")):
            with self.subTest(error=type(error).__name__):
                runtime = FakeRuntime({"w": OPENCV_COCOA_MASK})
                runtime.find_error = error
                self.assertFalse(enable_close_button("w", runtime=runtime, platform="darwin"))

    def test_library_load_failure_returns_false(self) -> None:
        with mock.patch.object(macos_window.ctypes.cdll, "LoadLibrary", side_effect=OSError("kütüphane yok")):
            self.assertFalse(enable_close_button("w", platform="darwin"))

    def test_default_platform_is_sys_platform(self) -> None:
        with mock.patch.object(macos_window.sys, "platform", "linux"):
            self.assertFalse(enable_close_button("w", runtime=FakeRuntime({"w": OPENCV_COCOA_MASK})))

    def test_importing_camera_modules_loads_no_system_library(self) -> None:
        code = (
            "import sys, ctypes\n"
            f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
            "import cv2, numpy\n"
            "calls = []\n"
            "original = ctypes.cdll.LoadLibrary\n"
            "ctypes.cdll.LoadLibrary = lambda *a, **k: calls.append(a) or original(*a, **k)\n"
            "import isg_vision.camera.macos_window, isg_vision.camera.display, isg_vision.camera.cli\n"
            "print('LOADLIBRARY_CALLS=%d' % len(calls))\n"
        )
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("LOADLIBRARY_CALLS=0", done.stdout)


class _CvPatched(unittest.TestCase):
    """OpenCV GUI çağrılarını taklit eder; gerçek pencere açılmaz."""

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
        enable = mock.patch.object(display_module, "enable_close_button", return_value=True)
        self.enable = enable.start()
        self.addCleanup(enable.stop)

    def camera_source(self, n: int) -> tuple[FrameSource, FakeCapture]:
        cap = FakeCapture([make_frame(i) for i in range(n)], props={FPS: 30.0})
        return FrameSource(SourceConfig.camera(width=None, height=None), capture_factory=FakeFactory(cap)), cap


class PreviewWindowCloseButtonTest(_CvPatched):
    def test_macos_open_enables_close_button_once(self) -> None:
        with mock.patch.object(display_module.sys, "platform", "darwin"):
            window = PreviewWindow("w")
            self.assertIsNone(window.close_button_enabled)  # henüz açılmadı
            window.show(make_frame(0))
            window.show(make_frame(1))
        self.enable.assert_called_once_with("w")
        self.assertIs(window.close_button_enabled, True)

    def test_failed_enable_is_reported_but_window_keeps_working(self) -> None:
        self.enable.return_value = False
        with mock.patch.object(display_module.sys, "platform", "darwin"):
            window = PreviewWindow("w")
            key = window.show(make_frame(0))
        self.assertIs(window.close_button_enabled, False)
        self.assertEqual(key, -1)
        self.m["imshow"].assert_called_once()

    def test_non_macos_never_calls_enable(self) -> None:
        with mock.patch.object(display_module.sys, "platform", "linux"):
            window = PreviewWindow("w")
            window.show(make_frame(0))
        self.enable.assert_not_called()
        self.assertIsNone(window.close_button_enabled)

    def test_close_resets_state_and_reopen_enables_again(self) -> None:
        with mock.patch.object(display_module.sys, "platform", "darwin"):
            window = PreviewWindow("w")
            window.open()
            window.close()
            self.assertIsNone(window.close_button_enabled)
            window.open()
        self.assertEqual(self.enable.call_count, 2)
        self.assertIs(window.close_button_enabled, True)


class WindowClosedDetectionTest(_CvPatched):
    """macOS'ta ``getWindowProperty`` davranışları: görünmez (0.0), -1.0, ``cv2.error``; ve çıkış yollarının korunması."""

    def run_with_property(self, values: list[Any], n: int = 20, max_frames: int | None = None):
        self.m["getWindowProperty"].side_effect = values
        source, cap = self.camera_source(n)
        window = PreviewWindow("w")
        with source, window:
            result = run_preview(source, window, max_frames=max_frames)
        return result, cap

    def test_invisible_window_ends_loop_and_releases_everything(self) -> None:
        result, cap = self.run_with_property([1.0, 1.0, 0.0] + [1.0] * 17)  # 3. karede pencere görünmez oldu
        self.assertEqual((result.reason, result.frames_processed), (REASON_WINDOW_CLOSED, 3))
        self.assertEqual(cap.release_calls, 1)  # kamera serbest bırakıldı
        self.m["destroyWindow"].assert_called_once_with("w")

    def test_cv2_error_from_get_window_property_ends_loop(self) -> None:
        result, cap = self.run_with_property([1.0, cv2.error("NULL window")] + [1.0] * 18)
        self.assertEqual((result.reason, result.frames_processed), (REASON_WINDOW_CLOSED, 2))
        self.assertEqual(cap.release_calls, 1)

    def test_negative_property_ends_loop(self) -> None:
        result, _ = self.run_with_property([1.0, -1.0] + [1.0] * 18)
        self.assertEqual((result.reason, result.frames_processed), (REASON_WINDOW_CLOSED, 2))

    def test_destroy_error_after_close_is_swallowed(self) -> None:
        self.m["destroyWindow"].side_effect = cv2.error("pencere yok")
        result, cap = self.run_with_property([0.0] + [1.0] * 19)
        self.assertEqual(result.reason, REASON_WINDOW_CLOSED)
        self.assertEqual(cap.release_calls, 1)

    def test_window_closed_during_first_frame_is_detected(self) -> None:
        result, _ = self.run_with_property([0.0] * 20)
        self.assertEqual((result.reason, result.frames_processed), (REASON_WINDOW_CLOSED, 1))

    def test_visible_window_does_not_stop_the_loop(self) -> None:
        result, cap = self.run_with_property([1.0] * 5, n=20, max_frames=5)
        self.assertEqual((result.reason, result.frames_processed), (REASON_MAX_FRAMES, 5))
        self.assertEqual(cap.release_calls, 1)

    def test_q_and_esc_still_exit_with_real_preview_window(self) -> None:
        for key, name in ((ord("q"), "q"), (ord("Q"), "Q"), (27, "Esc")):
            with self.subTest(key=name):
                self.m["waitKey"].side_effect = [-1, key] + [-1] * 18
                source, cap = self.camera_source(20)
                window = PreviewWindow("w")
                with source, window:
                    result = run_preview(source, window)
                self.assertEqual((result.reason, result.frames_processed), (REASON_USER_QUIT, 2))
                self.assertEqual(cap.release_calls, 1)

    def test_video_end_of_stream_with_real_preview_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = write_synthetic_video(Path(tmp) / "k.mp4", n_frames=6, fps=25.0)
            with FrameSource(SourceConfig.video(video)) as source:
                with PreviewWindow("w") as window:
                    result = run_preview(source, window)
        self.assertEqual((result.reason, result.frames_processed), (REASON_END_OF_STREAM, 6))


class CliWindowCloseTest(_CvPatched):
    def run_cli(self, argv: list[str], **kwargs: Any) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv, **kwargs)
        return code, out.getvalue(), err.getvalue()

    def test_closing_the_window_exits_normally_and_releases_camera(self) -> None:
        self.m["getWindowProperty"].side_effect = [1.0, 1.0, 0.0] + [1.0] * 50
        cap = FakeCapture([make_frame(i) for i in range(50)], props={FPS: 30.0})
        code, out, err = self.run_cli([], capture_factory=FakeFactory(cap), window_factory=PreviewWindow)
        self.assertEqual((code, err), (EXIT_OK, ""))
        self.assertIn("Bitiş nedeni: pencere kapatıldı", out)
        self.assertIn("İşlenen kare: 3", out)
        self.assertEqual(cap.release_calls, 1)
        self.m["destroyWindow"].assert_called_once()

    def test_failed_close_button_enable_prints_warning_on_macos(self) -> None:
        self.enable.return_value = False
        cap = FakeCapture([make_frame(i) for i in range(3)], props={FPS: 30.0})
        with mock.patch.object(display_module.sys, "platform", "darwin"):
            code, out, _ = self.run_cli(["--max-frames", "2"], capture_factory=FakeFactory(cap), window_factory=PreviewWindow)
        self.assertEqual(code, EXIT_OK)
        self.assertIn("[UYARI] Pencerenin kırmızı kapatma düğmesi etkinleştirilemedi", out)

    def test_no_warning_when_enabled_or_not_applicable(self) -> None:
        for platform, enabled in (("darwin", True), ("linux", True)):
            with self.subTest(platform=platform):
                self.enable.return_value = enabled
                cap = FakeCapture([make_frame(i) for i in range(3)], props={FPS: 30.0})
                with mock.patch.object(display_module.sys, "platform", platform):
                    _, out, _ = self.run_cli(["--max-frames", "2"], capture_factory=FakeFactory(cap), window_factory=PreviewWindow)
                self.assertNotIn("kapatma düğmesi etkinleştirilemedi", out)

    def test_fake_window_without_property_gives_no_warning(self) -> None:
        cap = FakeCapture([make_frame(i) for i in range(3)], props={FPS: 30.0})
        _, out, _ = self.run_cli(["--max-frames", "2"], capture_factory=FakeFactory(cap), window_factory=lambda: FakeWindow())
        self.assertNotIn("etkinleştirilemedi", out)

    def test_start_hint_mentions_window_close_button(self) -> None:
        cap = FakeCapture([make_frame(i) for i in range(3)], props={FPS: 30.0})
        _, out, _ = self.run_cli(["--max-frames", "2"], capture_factory=FakeFactory(cap), window_factory=lambda: FakeWindow())
        self.assertIn("kapatma düğmesine tıklayın", out)


# ------------------------------------------------------------------ isteğe bağlı: GERÇEK Cocoa penceresi
GUI_ENABLED = sys.platform == "darwin" and os.environ.get("ISG_GUI_TESTS") == "1"


@unittest.skipUnless(GUI_ENABLED, "Gerçek pencere testi: macOS + ISG_GUI_TESTS=1 + grafik oturumu gerekir")
class RealCocoaWindowTest(unittest.TestCase):
    """Gerçek bir OpenCV/Cocoa penceresi açar ve kırmızı düğme tıklamasını ``NSButton.performClick:`` ile taklit eder."""

    TITLE = "isg-test-close-window"

    @staticmethod
    def _frames(i: int):
        return make_frame(i, 320, 240)

    def _button(self, runtime: CocoaRuntime) -> tuple[Any, Any]:
        window = runtime.find_window(self.TITLE)
        self.assertIsNotNone(window, "OpenCV penceresi NSApp.windows içinde bulunamadı")
        button = runtime._send(window, "standardWindowButton:", 0, argtypes=(ctypes.c_long,))  # NSWindowCloseButton
        self.assertTrue(button)
        return window, button

    @staticmethod
    def _enabled(runtime: CocoaRuntime, button: Any) -> bool:
        return bool(runtime._send(button, "isEnabled", restype=ctypes.c_ubyte))

    def test_unpatched_opencv_window_has_no_closable_bit(self) -> None:
        """Kök neden testi: düzeltme uygulanmadan, ham OpenCV penceresinin kapatma düğmesi devre dışıdır."""
        runtime = CocoaRuntime()
        cv2.namedWindow(self.TITLE, cv2.WINDOW_AUTOSIZE)
        try:
            cv2.imshow(self.TITLE, self._frames(0))
            cv2.waitKey(30)
            window, button = self._button(runtime)
            self.assertFalse(runtime.style_mask(window) & NS_WINDOW_STYLE_MASK_CLOSABLE)
            self.assertFalse(self._enabled(runtime, button))
        finally:
            cv2.destroyWindow(self.TITLE)
            for _ in range(3):
                cv2.waitKey(1)

    def test_red_button_click_closes_preview_window_and_is_detected(self) -> None:
        runtime = CocoaRuntime()
        window = PreviewWindow(self.TITLE)
        try:
            window.show(self._frames(0), 30)
            self.assertIs(window.close_button_enabled, True)
            native, button = self._button(runtime)
            self.assertTrue(runtime.style_mask(native) & NS_WINDOW_STYLE_MASK_CLOSABLE)
            self.assertTrue(self._enabled(runtime, button))
            self.assertFalse(window.closed_by_user)

            runtime._send(  # kırmızı düğmeye tıklama (ana iş parçacığında, olay döngüsü waitKey içinde işlenir)
                button,
                "performSelectorOnMainThread:withObject:waitUntilDone:",
                runtime._objc.sel_registerName(b"performClick:"),
                None,
                False,
                argtypes=(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool),
            )
            detected_at = None
            for i in range(1, 60):
                window.show(self._frames(i), 30)
                if window.closed_by_user:
                    detected_at = i
                    break
            self.assertIsNotNone(detected_at, "Kırmızı düğme tıklaması 60 karede algılanmadı")
        finally:
            window.close()

    def test_full_loop_ends_with_window_closed_reason_and_releases_camera(self) -> None:
        frame = make_frame(0, 320, 240)
        cap = FakeCapture([frame] * 3000, props={FPS: 30.0})  # aynı dizi başvurusu: bellek kullanımı küçük
        source = FrameSource(SourceConfig.camera(width=None, height=None), capture_factory=FakeFactory(cap))
        runtime = CocoaRuntime()
        window = PreviewWindow(self.TITLE)

        def click() -> None:
            native = runtime.find_window(self.TITLE)
            button = runtime._send(native, "standardWindowButton:", 0, argtypes=(ctypes.c_long,))
            runtime._send(
                button,
                "performSelectorOnMainThread:withObject:waitUntilDone:",
                runtime._objc.sel_registerName(b"performClick:"),
                None,
                False,
                argtypes=(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool),
            )

        timer = threading.Timer(0.5, click)
        timer.start()
        try:
            with source, window:
                result = run_preview(source, window)
        finally:
            timer.cancel()
        self.assertEqual(result.reason, REASON_WINDOW_CLOSED)
        self.assertLess(result.frames_processed, 3000)
        self.assertEqual(cap.release_calls, 1)


if __name__ == "__main__":
    unittest.main()
