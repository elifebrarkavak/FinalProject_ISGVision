"""Aşama 3 / Grup 3: komut satırı arayüzü, çıkış kodları ve import yan etkisizliği testleri.

Gerçek kamera açılmaz (``capture_factory`` ile sahte kaynak verilir), gerçek pencere açılmaz, ağ kullanılmaz.
Alt süreç testleri yalnızca ``--help``, hatalı argüman ve penceresiz sentetik video çalıştırır.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from camera_test_support import FakeCapture, FakeFactory, FakeWindow, make_frame, write_synthetic_video  # noqa: E402
from isg_vision.camera import SourceConfig, SourceConfigError  # noqa: E402
from isg_vision.camera.cli import (  # noqa: E402
    EXIT_ERROR,
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_USAGE,
    build_parser,
    config_from_args,
    main,
)
from isg_vision.camera.errors import DisplayError  # noqa: E402

SCRIPT = ROOT / "scripts" / "run_camera.py"
CAMERA_MODULES = (
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
)


def parse(*argv: str):
    return build_parser().parse_args(list(argv))


def run_main(argv, **kwargs):
    """``main`` çalıştırır; (çıkış kodu, stdout, stderr) döndürür. ``SystemExit`` kodu da döndürülür."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main(argv, **kwargs)
        except SystemExit as exc:
            code = exc.code
    return code, out.getvalue(), err.getvalue()


class ConfigFromArgsTest(unittest.TestCase):
    def test_defaults_are_default_camera_1280x720(self) -> None:
        cfg = config_from_args(parse())
        self.assertEqual((cfg.kind, cfg.camera_index, cfg.width, cfg.height, cfg.backend), ("camera", 0, 1280, 720, "auto"))

    def test_camera_options(self) -> None:
        cfg = config_from_args(parse("--source", "camera", "--camera-index", "2", "--width", "640", "--height", "480",
                                     "--backend", "any"))
        self.assertEqual((cfg.camera_index, cfg.width, cfg.height, cfg.backend), (2, 640, 480, "any"))

    def test_video_options(self) -> None:
        cfg = config_from_args(parse("--source", "video", "--video-path", "/yol/a.mp4"))
        self.assertEqual((cfg.kind, str(cfg.video_path), cfg.width, cfg.height), ("video", "/yol/a.mp4", None, None))

    def test_conflicting_or_incomplete_arguments_rejected(self) -> None:
        bad = [
            ("--source", "video"),                                              # yol yok
            ("--source", "camera", "--video-path", "a.mp4"),                    # kamerada video yolu
            ("--width", "640"),                                                 # yalnızca genişlik
            ("--height", "480"),                                                # yalnızca yükseklik
            ("--source", "video", "--video-path", "a.mp4", "--camera-index", "1"),
            ("--source", "video", "--video-path", "a.mp4", "--width", "640", "--height", "480"),
            ("--camera-index", "-1"),
            ("--camera-index", "100"),
            ("--width", "0", "--height", "480"),
            ("--width", "640", "--height", "99999"),
        ]
        for argv in bad:
            with self.subTest(argv=argv):
                with self.assertRaises(SourceConfigError):
                    config_from_args(parse(*argv))


class ArgumentErrorsTest(unittest.TestCase):
    def test_invalid_arguments_exit_2_with_message_and_no_camera_access(self) -> None:
        cases = {
            ("--camera-index", "abc"): "tam sayı",
            ("--max-frames", "0"): "1 veya daha büyük",
            ("--source", "rtsp"): "invalid choice",
            ("--source", "video"): "--video-path gerekir",
            ("--width", "640"): "birlikte",
            ("--bilinmeyen",): "unrecognized",
        }
        factory = FakeFactory(FakeCapture())
        for argv, expected in cases.items():
            with self.subTest(argv=argv):
                code, _, err = run_main(list(argv), capture_factory=factory)
                self.assertEqual(code, EXIT_USAGE)
                self.assertIn(expected, err)
        self.assertEqual(factory.calls, [], "hatalı argümanlarda kaynak açılmamalı")

    def test_help_exits_0_and_documents_options(self) -> None:
        code, out, _ = run_main(["--help"])
        self.assertEqual(code, 0)
        for text in ("--source", "--camera-index", "--width", "--height", "--video-path", "--no-display",
                     "--max-frames", "Q veya Esc", "örnekler"):
            self.assertIn(text, out)


class RunFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.video = Path(self._tmp.name) / "klip.mp4"
        self.video.write_bytes(b"x")

    def camera_cap(self, n: int = 50, **kwargs) -> FakeCapture:
        return FakeCapture([make_frame(i) for i in range(n)], props={cv2.CAP_PROP_FPS: 30.0}, **kwargs)

    def test_camera_headless_run_reports_summary_and_releases(self) -> None:
        cap = self.camera_cap()
        factory = FakeFactory(cap)
        code, out, err = run_main(["--source", "camera", "--camera-index", "1", "--no-display", "--max-frames", "5"],
                                  capture_factory=factory)
        self.assertEqual((code, err), (EXIT_OK, ""))
        self.assertEqual(factory.calls, [(1, cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY)])
        self.assertIn("Kaynak açıldı: camera:1", out)
        self.assertIn("İşlenen kare: 5", out)
        self.assertIn("--max-frames", out)
        self.assertIn("kaynak FPS (bildirilen): 30.0", out)
        self.assertEqual(cap.release_calls, 1)

    def test_video_run_to_end_with_fake_window(self) -> None:
        cap = FakeCapture([make_frame(i) for i in range(6)], props={cv2.CAP_PROP_FPS: 25.0, cv2.CAP_PROP_FRAME_COUNT: 6.0})
        window = FakeWindow()
        code, out, err = run_main(["--source", "video", "--video-path", str(self.video)],
                                  capture_factory=FakeFactory(cap), window_factory=lambda: window)
        self.assertEqual((code, err), (EXIT_OK, ""))
        self.assertIn("video sona erdi", out)
        self.assertIn("toplam kare: 6", out)
        self.assertEqual((window.open_calls, window.close_calls, cap.release_calls), (1, 1, 1))
        self.assertEqual(len(window.images), 6)

    def test_q_key_exit_is_normal(self) -> None:
        cap, window = self.camera_cap(), FakeWindow(keys={4: ord("q")})
        code, out, _ = run_main([], capture_factory=FakeFactory(cap), window_factory=lambda: window)
        self.assertEqual(code, EXIT_OK)
        self.assertIn("kullanıcı çıkışı", out)
        self.assertIn("İşlenen kare: 4", out)
        self.assertEqual((window.close_calls, cap.release_calls), (1, 1))

    def test_ctrl_c_during_loop_exits_130_and_releases(self) -> None:
        cap = self.camera_cap()
        window = FakeWindow(raise_on_call={3: KeyboardInterrupt()})
        code, out, _ = run_main([], capture_factory=FakeFactory(cap), window_factory=lambda: window)
        self.assertEqual(code, EXIT_INTERRUPTED)
        self.assertIn("Ctrl+C", out)
        self.assertEqual((window.close_calls, cap.release_calls), (1, 1))

    def test_ctrl_c_while_opening_exits_130(self) -> None:
        window = FakeWindow()
        code, out, _ = run_main([], capture_factory=FakeFactory(KeyboardInterrupt()), window_factory=lambda: window)
        self.assertEqual(code, EXIT_INTERRUPTED)
        self.assertEqual(window.close_calls, 1)

    def test_camera_open_failure_exits_1_with_clear_message(self) -> None:
        cap = FakeCapture(opened=False)
        window = FakeWindow()
        code, out, err = run_main([], capture_factory=FakeFactory(cap), window_factory=lambda: window)
        self.assertEqual(code, EXIT_ERROR)
        self.assertIn("[HATA]", err)
        self.assertIn("Kamera 0 açılamadı", err)
        self.assertIn("izni", err)
        self.assertEqual(window.open_calls, 0, "kamera açılamadıysa pencere hiç açılmamalı")
        self.assertEqual(cap.release_calls, 1)

    def test_camera_read_failure_exits_1_and_releases(self) -> None:
        cap = FakeCapture([make_frame(0), make_frame(1)], props={})
        window = FakeWindow()
        with mock.patch("time.sleep"):  # yeniden deneme beklemeleri gerçek zaman harcamasın
            code, _, err = run_main([], capture_factory=FakeFactory(cap), window_factory=lambda: window)
        self.assertEqual(code, EXIT_ERROR)
        self.assertIn("bağlantısı kesildi", err)
        self.assertEqual((window.close_calls, cap.release_calls), (1, 1))

    def test_missing_video_exits_1_and_never_opens_capture(self) -> None:
        factory = FakeFactory(FakeCapture())
        code, _, err = run_main(["--source", "video", "--video-path", str(self.video.with_name("yok.mp4")), "--no-display"],
                                capture_factory=factory)
        self.assertEqual(code, EXIT_ERROR)
        self.assertIn("bulunamadı", err)
        self.assertEqual(factory.calls, [])

    def test_unopenable_video_exits_1(self) -> None:
        cap = FakeCapture(opened=False)
        code, _, err = run_main(["--source", "video", "--video-path", str(self.video), "--no-display"],
                                capture_factory=FakeFactory(cap))
        self.assertEqual(code, EXIT_ERROR)
        self.assertIn("açılamadı", err)
        self.assertEqual(cap.release_calls, 1)

    def test_window_creation_failure_exits_1_and_releases_camera(self) -> None:
        cap, window = self.camera_cap(), FakeWindow()
        window.open_error = DisplayError("OpenCV görüntü penceresi oluşturulamadı: test")
        code, _, err = run_main([], capture_factory=FakeFactory(cap), window_factory=lambda: window)
        self.assertEqual(code, EXIT_ERROR)
        self.assertIn("penceresi oluşturulamadı", err)
        self.assertEqual(cap.release_calls, 1)

    def test_unexpected_exception_still_releases_resources(self) -> None:
        cap = self.camera_cap()
        window = FakeWindow(raise_on_call={2: RuntimeError("beklenmeyen")})
        with self.assertRaises(RuntimeError), contextlib.redirect_stdout(io.StringIO()):
            main([], capture_factory=FakeFactory(cap), window_factory=lambda: window)
        self.assertEqual((window.close_calls, cap.release_calls), (1, 1))

    def test_resolution_mismatch_warning(self) -> None:
        cap = FakeCapture([make_frame(i, 640, 480) for i in range(3)], accept_set=False,
                          props={cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480})
        code, out, _ = run_main(["--width", "1920", "--height", "1080", "--no-display", "--max-frames", "2"],
                                capture_factory=FakeFactory(cap))
        self.assertEqual(code, EXIT_OK)
        self.assertIn("[UYARI] İstenen çözünürlük (1920x1080) uygulanamadı; kaynak 640x480 kullanıyor.", out)
        self.assertIn("gerçek çözünürlük: 640x480", out)

    def test_premature_video_end_warning(self) -> None:
        cap = FakeCapture([make_frame(i) for i in range(10)], props={cv2.CAP_PROP_FRAME_COUNT: 100.0, cv2.CAP_PROP_FPS: 25.0})
        code, _, err = run_main(["--source", "video", "--video-path", str(self.video), "--no-display"],
                                capture_factory=FakeFactory(cap))
        self.assertEqual(code, EXIT_OK)  # çökmez, kontrollü biter
        self.assertIn("beklenenden erken bitti", err)


class RealVideoThroughCliTest(unittest.TestCase):
    def test_real_synthetic_videos_headless(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for name, fourcc in (("a.mp4", "mp4v"), ("a.mov", "mp4v"), ("a.avi", "MJPG")):
                with self.subTest(file=name):
                    path = write_synthetic_video(Path(tmp) / name, n_frames=15, fourcc=fourcc)
                    code, out, err = run_main(["--source", "video", "--video-path", str(path), "--no-display"])
                    self.assertEqual((code, err), (EXIT_OK, ""))
                    self.assertIn("İşlenen kare: 15", out)
                    self.assertIn("video sona erdi", out)


class SubprocessScriptTest(unittest.TestCase):
    def run_script(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=60, cwd=ROOT)

    def test_script_help(self) -> None:
        result = self.run_script("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--video-path", result.stdout)

    def test_script_usage_error_code(self) -> None:
        result = self.run_script("--source", "video")
        self.assertEqual(result.returncode, EXIT_USAGE)
        self.assertIn("--video-path gerekir", result.stderr)

    def test_script_runs_synthetic_video_headless_and_leaves_file_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_synthetic_video(Path(tmp) / "v.mp4", n_frames=12)
            before = path.read_bytes()
            result = self.run_script("--source", "video", "--video-path", str(path), "--no-display")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("İşlenen kare: 12", result.stdout)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["v.mp4"], "hiçbir dosya kaydedilmemeli")

    def test_script_reports_missing_video_with_exit_1(self) -> None:
        result = self.run_script("--source", "video", "--video-path", "/yok/video.mp4", "--no-display")
        self.assertEqual(result.returncode, EXIT_ERROR)
        self.assertIn("bulunamadı", result.stderr)


class ImportSideEffectsTest(unittest.TestCase):
    def test_importing_camera_modules_opens_nothing(self) -> None:
        boom = AssertionError("import sırasında kamera/pencere kullanıldı!")
        with mock.patch.object(cv2, "VideoCapture", side_effect=boom), \
                mock.patch.object(cv2, "namedWindow", side_effect=boom), \
                mock.patch.object(cv2, "imshow", side_effect=boom), \
                mock.patch.object(cv2, "waitKey", side_effect=boom):
            for name in CAMERA_MODULES:
                sys.modules.pop(name, None)
                importlib.import_module(name)
            # Nesneleri oluşturmak da kaynak açmamalı; yalnızca open() açar.
            from isg_vision.camera import FrameSource
            from isg_vision.camera.display import PreviewWindow
            FrameSource(SourceConfig.camera())
            PreviewWindow()

    def test_script_module_import_does_not_run_main(self) -> None:
        spec = importlib.util.spec_from_file_location("run_camera_under_test", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.object(cv2, "VideoCapture", side_effect=AssertionError("kamera açıldı")):
            spec.loader.exec_module(module)
        self.assertTrue(callable(module.main))

    def test_camera_package_pulls_in_no_ai_libraries(self) -> None:
        code = (
            "import sys; sys.path.insert(0, %r)\n"
            "import isg_vision.camera, isg_vision.camera.cli\n"
            "bad = [m for m in ('torch', 'torchvision', 'ultralytics', 'mediapipe', 'onnxruntime') if m in sys.modules]\n"
            "print(bad); sys.exit(1 if bad else 0)\n" % str(ROOT / "src")
        )
        result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, f"yapay zekâ kütüphanesi yüklendi: {result.stdout}{result.stderr}")


if __name__ == "__main__":
    unittest.main()
