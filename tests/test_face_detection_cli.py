"""Yüz tespiti komut satırı testleri: argümanlar, hata kodları, penceresiz çalışma, kullanıcı çıkışı ve kaynak temizliği.

Sahte kamera/pencere/tespitçi kullanılır: gerçek kamera açılmaz, kamera izni istenmez, model yüklenmez, ağ kullanılmaz.
"""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from camera_test_support import FakeCapture, FakeFactory, FakeWindow, make_frame, write_synthetic_video  # noqa: E402
from face_detection_test_support import FakeFaceDetector  # noqa: E402

from isg_vision.camera.display import PreviewWindow  # noqa: E402  (modül başında: test_no_side_effects modülleri yeniden yükler)
from isg_vision.camera.errors import DisplayError  # noqa: E402
from isg_vision.face_detection import cli  # noqa: E402
from isg_vision.face_detection.config import FaceDetectionConfig  # noqa: E402
from isg_vision.face_detection.detector import FaceDetector  # noqa: E402
from isg_vision.face_detection.errors import FaceInvalidFrameError, FaceModelLoadError  # noqa: E402
from isg_vision.face_detection.visualization import WINDOW_NAME  # noqa: E402

SCRIPT = ROOT / "scripts" / "run_face_detection.py"
CAM_PROPS = {cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 48, cv2.CAP_PROP_FPS: 15.0}
CAM_ARGS = ["--source", "camera", "--width", "64", "--height", "48"]


def camera_factory(n_frames: int = 20, opened: bool = True) -> tuple[FakeFactory, FakeCapture]:
    capture = FakeCapture([make_frame(i) for i in range(n_frames)], opened=opened, props=dict(CAM_PROPS))
    return FakeFactory(capture), capture


def call(argv: list[str], **kwargs):  # type: ignore[no-untyped-def]
    """``cli.main``'i çağırır; ``(çıkış_kodu, stdout, stderr)`` döndürür. ``SystemExit`` kodu da yakalanır."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli.main(argv, **kwargs)
        except SystemExit as exc:
            code = int(exc.code) if exc.code is not None else 0
    return code, out.getvalue(), err.getvalue()


class ParserTest(unittest.TestCase):
    def test_defaults(self) -> None:
        args = cli.build_parser().parse_args([])
        self.assertEqual((args.source, args.conf, args.nms, args.max_faces, args.detect_width), ("camera", 0.6, 0.3, 100, None))
        self.assertIsNone(args.model)
        self.assertFalse(args.no_display)
        self.assertIsNone(args.max_frames)

    def test_documented_examples_parse(self) -> None:
        parser = cli.build_parser()
        for argv in (
            ["--source", "camera"],
            ["--source", "camera", "--camera-index", "0"],
            ["--source", "video", "--video-path", "/yol/video.mp4"],
            ["--source", "camera", "--conf", "0.7"],
            ["--source", "camera", "--detect-width", "960"],
            ["--source", "camera", "--no-display", "--max-frames", "100"],
            ["--source", "camera", "--nms", "0.4", "--max-faces", "10", "--model", "x.onnx"],
        ):
            with self.subTest(argv=argv):
                parser.parse_args(argv)

    def test_epilog_examples_in_help_all_parse(self) -> None:
        help_text = cli.build_parser().format_help()
        for line in help_text.splitlines():
            line = line.strip()
            if line.startswith("python scripts/run_face_detection.py"):
                argv = line.split("#")[0].split()[2:]
                with self.subTest(example=line):
                    cli.build_parser().parse_args(argv)

    def test_there_is_no_device_option_because_only_cpu_is_supported(self) -> None:
        code, _, err = call(["--device", "mps", "--no-display"])
        self.assertEqual(code, 2)
        self.assertIn("--device", err)

    def test_help_states_scope_and_limits(self) -> None:
        text = cli.build_parser().format_help()
        for expected in ("kimlik", "canlı insan", "download_face_model.py", "--detect-width", "--conf", "yüz"):
            self.assertIn(expected, text)

    def test_source_options_match_the_other_scripts(self) -> None:
        from isg_vision.human_detection import cli as person_cli

        face_opts = {a.dest for a in cli.build_parser()._actions}
        person_opts = {a.dest for a in person_cli.build_parser()._actions}
        for dest in ("source", "camera_index", "width", "height", "video_path", "backend", "no_display", "max_frames"):
            self.assertIn(dest, face_opts)
            self.assertIn(dest, person_opts)


class InvalidArgumentsTest(unittest.TestCase):
    def assert_usage_error(self, argv: list[str], needle: str) -> None:
        code, _, err = call(argv)
        self.assertEqual(code, 2, err)
        self.assertIn(needle, err)

    def test_invalid_values_exit_2_without_opening_anything(self) -> None:
        cases = {
            "--conf": ["0", "-0.1", "1.5", "abc", "nan", "inf"],
            "--nms": ["0", "2", "x"],
            "--max-faces": ["0", "-1", "1.5", "abc"],
            "--detect-width": ["10", "63", "9000", "abc", "1.5"],
            "--max-frames": ["0", "-5", "x"],
        }
        for option, values in cases.items():
            for value in values:
                with self.subTest(option=option, value=value):
                    factory, capture = camera_factory()
                    code, _, err = call(CAM_ARGS + [option, value], capture_factory=factory, detector_factory=lambda cfg: FakeFaceDetector())
                    self.assertEqual(code, 2, err)
                    self.assertEqual(factory.calls, [])
                    self.assertEqual(capture.read_calls, 0)

    def test_model_must_be_an_onnx_file(self) -> None:
        self.assert_usage_error(CAM_ARGS + ["--model", "agirlik.pt"], ".onnx")

    def test_video_source_requires_a_path(self) -> None:
        self.assert_usage_error(["--source", "video", "--no-display"], "video")

    def test_camera_with_video_path_is_rejected(self) -> None:
        self.assert_usage_error(["--source", "camera", "--video-path", "x.mp4"], "video")

    def test_width_without_height_is_rejected(self) -> None:
        self.assert_usage_error(["--source", "camera", "--width", "640"], "height")


class RunTest(unittest.TestCase):
    def run_with_window(self, window: FakeWindow, n_frames: int = 20, args: list[str] | None = None, detector: FakeFaceDetector | None = None):  # type: ignore[no-untyped-def]
        factory, capture = camera_factory(n_frames)
        detector = detector or FakeFaceDetector(counts=[2])
        code, out, err = call(CAM_ARGS + (args or []), capture_factory=factory, window_factory=lambda: window, detector_factory=lambda cfg: detector)
        return code, out, err, capture, detector

    def test_no_display_max_frames_exits_0_and_prints_summary(self) -> None:
        factory, capture = camera_factory(30)
        detector = FakeFaceDetector(counts=[0, 1, 3])
        code, out, err = call(CAM_ARGS + ["--no-display", "--max-frames", "12"], capture_factory=factory, detector_factory=lambda cfg: detector)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("--max-frames sınırına ulaşıldı", out)
        self.assertIn("İşlenen kare: 12", out)
        self.assertIn("Toplam yüz tespiti: 31", out)  # sayı listesi bitince son değer (3) tekrarlanır: 0 + 1 + 10 x 3
        self.assertIn("benzersiz kişi sayısı DEĞİLDİR", out)
        self.assertIn("en çok 3", out)
        self.assertIn("en az bir yüz olan kare: 11/12", out)
        self.assertIn("Pencere kapalıydı", out)
        for label in ("Çıkarım süresi", "Kare okuma süresi", "Kare -> sonuç gecikmesi", "uçtan uca FPS"):
            self.assertIn(label, out)
        self.assertEqual((capture.release_calls, detector.close_calls, detector.load_calls), (1, 1, 1))

    def test_model_info_and_scope_disclaimer_are_printed(self) -> None:
        factory, _ = camera_factory()
        code, out, _ = call(CAM_ARGS + ["--no-display", "--max-frames", "2", "--conf", "0.7", "--detect-width", "960"], capture_factory=factory,
                            detector_factory=lambda cfg: FakeFaceDetector())
        self.assertEqual(code, 0)
        self.assertIn("YuNet", out)
        self.assertIn("conf: 0.7", out)
        self.assertIn("960 px", out)
        self.assertIn("cihaz seçeneği yok", out)
        self.assertIn("kimlik veya canlı insan doğrulaması değildir", out)

    def test_config_is_passed_to_the_detector_factory(self) -> None:
        captured: list[FaceDetectionConfig] = []

        def make(cfg: FaceDetectionConfig) -> FakeFaceDetector:
            captured.append(cfg)
            return FakeFaceDetector()

        factory, _ = camera_factory()
        call(CAM_ARGS + ["--no-display", "--max-frames", "1", "--conf", "0.8", "--nms", "0.2", "--max-faces", "7", "--detect-width", "320"],
             capture_factory=factory, detector_factory=make)
        cfg = captured[0]
        self.assertEqual((cfg.conf, cfg.nms, cfg.max_faces, cfg.detect_width, cfg.verify_model_hash), (0.8, 0.2, 7, 320, True))

    def test_model_is_warmed_up_at_the_real_frame_size_before_the_loop(self) -> None:
        factory, _ = camera_factory()
        detector = FakeFaceDetector()
        call(CAM_ARGS + ["--no-display", "--max-frames", "2"], capture_factory=factory, detector_factory=lambda cfg: detector)
        self.assertEqual(detector.warmup_calls, [(64, 48)])

    def test_q_exit_closes_everything(self) -> None:
        window = FakeWindow(keys={3: ord("q")})
        code, out, _, capture, detector = self.run_with_window(window)
        self.assertEqual(code, 0)
        self.assertIn("kullanıcı çıkışı (Q/Esc)", out)
        self.assertEqual((window.open_calls, window.close_calls, capture.release_calls, detector.close_calls), (1, 1, 1, 1))
        self.assertEqual(len(window.images), 3)

    def test_esc_exit(self) -> None:
        code, out, _, capture, _ = self.run_with_window(FakeWindow(keys={2: 27}))
        self.assertEqual((code, capture.release_calls), (0, 1))
        self.assertIn("kullanıcı çıkışı", out)

    def test_window_close_button_exit(self) -> None:
        window = FakeWindow(close_after=4)
        code, out, _, capture, detector = self.run_with_window(window)
        self.assertEqual((code, window.close_calls, capture.release_calls, detector.close_calls), (0, 1, 1, 1))
        self.assertIn("pencere kapatıldı", out)

    def test_ctrl_c_during_the_loop_exits_130_and_releases_everything(self) -> None:
        window = FakeWindow()
        detector = FakeFaceDetector(detect_errors={3: KeyboardInterrupt()})
        code, out, _, capture, _ = self.run_with_window(window, detector=detector)
        self.assertEqual(code, 130)
        self.assertIn("Ctrl+C", out)
        self.assertEqual((window.close_calls, capture.release_calls, detector.close_calls), (1, 1, 1))

    def test_video_source_runs_to_the_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = write_synthetic_video(Path(tmp) / "v.mp4", n_frames=6, size=(64, 48))
            detector = FakeFaceDetector(counts=[1])
            code, out, err = call(["--source", "video", "--video-path", str(video), "--no-display"], detector_factory=lambda cfg: detector)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("video sona erdi", out)
        self.assertIn("İşlenen kare: 6", out)

    def test_default_window_uses_the_face_detection_title_and_close_button_fix(self) -> None:
        created: list[str] = []
        fake = FakeWindow(keys={1: ord("q")})

        def fake_preview_window(name: str) -> FakeWindow:
            created.append(name)
            return fake

        factory, _ = camera_factory()
        with mock.patch.object(cli, "PreviewWindow", fake_preview_window):
            code, _, err = call(CAM_ARGS, capture_factory=factory, detector_factory=lambda cfg: FakeFaceDetector())
        self.assertEqual(code, 0, err)
        self.assertEqual(created, [WINDOW_NAME])
        self.assertEqual(WINDOW_NAME, "ISG Vision - Face Detection")

    def test_default_window_class_is_the_camera_preview_window(self) -> None:
        self.assertIs(cli.PreviewWindow, PreviewWindow)

    def test_close_button_warning_is_shown_when_it_could_not_be_enabled(self) -> None:
        window = FakeWindow(keys={1: ord("q")})
        window.close_button_enabled = False  # type: ignore[attr-defined]
        _, out, _, _, _ = self.run_with_window(window)
        self.assertIn("kırmızı kapatma düğmesi etkinleştirilemedi", out)

    def test_warnings_are_printed_for_dropped_and_truncated_faces(self) -> None:
        factory, _ = camera_factory()
        code, out, _ = call(CAM_ARGS + ["--no-display", "--max-frames", "3"], capture_factory=factory, detector_factory=lambda cfg: FakeFaceDetector(dropped=1, truncated=2))
        self.assertEqual(code, 0)
        self.assertIn("3 geçersiz kutu atıldı", out)
        self.assertIn("6 yüz --max-faces sınırı nedeniyle kesildi", out)


class ErrorHandlingTest(unittest.TestCase):
    def test_model_load_failure_exits_1_and_never_opens_camera(self) -> None:
        factory, capture = camera_factory()
        detector = FakeFaceDetector(load_error=FaceModelLoadError("Yüz tespit modeli bulunamadı: yok.onnx"))
        code, _, err = call(CAM_ARGS + ["--no-display"], capture_factory=factory, detector_factory=lambda cfg: detector)
        self.assertEqual(code, 1)
        self.assertIn("bulunamadı", err)
        self.assertEqual(factory.calls, [], "model yüklenemezse kamera hiç açılmamalı")
        self.assertEqual(capture.read_calls, 0)
        self.assertEqual(detector.close_calls, 1)

    def test_real_detector_with_missing_model_file_exits_1_with_download_hint(self) -> None:
        factory, _ = camera_factory()
        with tempfile.TemporaryDirectory() as tmp:
            code, _, err = call(CAM_ARGS + ["--no-display", "--model", str(Path(tmp) / "yok.onnx")], capture_factory=factory)
        self.assertEqual(code, 1)
        self.assertIn("download_face_model.py", err)
        self.assertEqual(factory.calls, [])

    def test_model_failing_integrity_check_exits_1_without_opening_camera(self) -> None:
        factory, _ = camera_factory()
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "face_detection_yunet_2026may.onnx"  # resmî ad, yanlış içerik
            bad.write_bytes(b"x" * 1000)
            code, _, err = call(CAM_ARGS + ["--no-display", "--model", str(bad)], capture_factory=factory)
        self.assertEqual(code, 1)
        self.assertIn("bütünlük", err)
        self.assertEqual(factory.calls, [])

    def test_ctrl_c_during_model_loading_exits_130_without_opening_camera(self) -> None:
        factory, capture = camera_factory()
        detector = FakeFaceDetector(load_error=KeyboardInterrupt())
        code, out, _ = call(CAM_ARGS + ["--no-display"], capture_factory=factory, detector_factory=lambda cfg: detector)
        self.assertEqual(code, 130)
        self.assertIn("Ctrl+C", out)
        self.assertEqual((factory.calls, capture.read_calls, detector.close_calls), ([], 0, 1))

    def test_camera_open_failure_exits_1_and_closes_the_detector(self) -> None:
        factory, capture = camera_factory(opened=False)
        detector = FakeFaceDetector()
        code, _, err = call(CAM_ARGS + ["--no-display"], capture_factory=factory, detector_factory=lambda cfg: detector)
        self.assertEqual(code, 1)
        self.assertIn("[HATA]", err)
        self.assertEqual(detector.close_calls, 1)
        self.assertEqual(capture.read_calls, 0)

    def test_invalid_frame_during_the_loop_exits_1_and_releases_everything(self) -> None:
        factory, capture = camera_factory()
        window = FakeWindow()
        detector = FakeFaceDetector(detect_errors={2: FaceInvalidFrameError("bozuk kare")})
        code, _, err = call(CAM_ARGS, capture_factory=factory, window_factory=lambda: window, detector_factory=lambda cfg: detector)
        self.assertEqual(code, 1)
        self.assertIn("bozuk kare", err)
        self.assertEqual((window.close_calls, capture.release_calls, detector.close_calls), (1, 1, 1))

    def test_display_error_exits_1_and_releases_everything(self) -> None:
        window = FakeWindow()
        window.open_error = DisplayError("grafik oturumu yok")
        factory, capture = camera_factory()
        detector = FakeFaceDetector()
        code, _, err = call(CAM_ARGS, capture_factory=factory, window_factory=lambda: window, detector_factory=lambda cfg: detector)
        self.assertEqual(code, 1)
        self.assertIn("grafik oturumu yok", err)
        self.assertEqual((capture.release_calls, detector.close_calls), (1, 1))

    def test_no_image_is_written_anywhere(self) -> None:
        factory, _ = camera_factory()
        with mock.patch.object(cv2, "imwrite", side_effect=AssertionError("görüntü kaydedilmemeli")), \
                mock.patch.object(cv2, "VideoWriter", side_effect=AssertionError("video kaydedilmemeli")):
            code, _, err = call(CAM_ARGS + ["--no-display", "--max-frames", "3"], capture_factory=factory, detector_factory=lambda cfg: FakeFaceDetector(counts=[2]))
        self.assertEqual((code, err), (0, ""))


class ScriptTest(unittest.TestCase):
    def test_script_help_runs_without_opening_camera_or_loading_model(self) -> None:
        proc = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--detect-width", proc.stdout)
        self.assertIn("--source", proc.stdout)

    def test_script_usage_error_exits_2(self) -> None:
        proc = subprocess.run([sys.executable, str(SCRIPT), "--conf", "5"], capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 2)

    def test_script_with_missing_model_exits_1_before_touching_a_camera(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run([sys.executable, str(SCRIPT), "--source", "camera", "--no-display", "--model", str(Path(tmp) / "yok.onnx")],
                                  capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("download_face_model.py", proc.stderr)
        self.assertNotIn("Kaynak açıldı", proc.stdout)

    def test_download_script_help_does_not_use_the_network(self) -> None:
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "download_face_model.py"), "--help"], capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--verify-only", proc.stdout)

    def test_cli_module_uses_the_real_face_detector_by_default(self) -> None:
        import inspect

        self.assertIs(inspect.signature(cli.main).parameters["detector_factory"].default, FaceDetector)


if __name__ == "__main__":
    unittest.main()
