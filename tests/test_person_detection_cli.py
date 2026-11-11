"""İnsan tespiti komut satırı testleri: argümanlar, hata kodları, penceresiz çalışma, kullanıcı çıkışı ve kaynak temizliği.

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

from isg_vision.camera.display import PreviewWindow  # noqa: E402  (modül başında: test_no_side_effects modülleri yeniden yükler)
from isg_vision.camera.errors import DisplayError  # noqa: E402
from isg_vision.human_detection import cli  # noqa: E402
from isg_vision.human_detection.config import DetectionConfig  # noqa: E402
from isg_vision.human_detection.errors import DeviceUnavailableError, InvalidFrameError, ModelLoadError  # noqa: E402
from isg_vision.human_detection.visualization import WINDOW_NAME  # noqa: E402
from person_detection_test_support import FakeDetector  # noqa: E402

SCRIPT = ROOT / "scripts" / "run_person_detection.py"
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
        self.assertEqual((args.source, args.conf, args.iou, args.imgsz, args.device, args.max_det), ("camera", 0.25, 0.7, 640, "auto", 100))
        self.assertIsNone(args.model)
        self.assertFalse(args.no_display)
        self.assertIsNone(args.max_frames)

    def test_documented_examples_parse(self) -> None:
        parser = cli.build_parser()
        for argv in (
            ["--source", "camera"],
            ["--source", "camera", "--camera-index", "0"],
            ["--source", "video", "--video-path", "/yol/video.mp4"],
            ["--source", "camera", "--device", "mps"],
            ["--source", "camera", "--device", "cpu"],
            ["--source", "camera", "--conf", "0.35"],
            ["--source", "camera", "--no-display", "--max-frames", "100"],
        ):
            with self.subTest(argv=argv):
                parser.parse_args(argv)

    def test_help_documents_every_option(self) -> None:
        help_text = cli.build_parser().format_help()
        for option in ("--source", "--camera-index", "--width", "--height", "--video-path", "--backend", "--model", "--conf", "--iou",
                       "--imgsz", "--max-det", "--device", "--no-display", "--max-frames", "örnekler", "Çıkış kodları", "download_person_model"):
            self.assertIn(option, help_text)

    def test_help_exits_zero(self) -> None:
        code, out, _ = call(["--help"])
        self.assertEqual(code, 0)
        self.assertIn("--conf", out)

    def test_bad_arguments_exit_2_with_message(self) -> None:
        bad_cases = {
            "conf sıfır": ["--conf", "0"],
            "conf >1": ["--conf", "1.5"],
            "conf negatif": ["--conf", "-0.2"],
            "conf metin": ["--conf", "abc"],
            "conf nan": ["--conf", "nan"],
            "iou >1": ["--iou", "2"],
            "iou sıfır": ["--iou", "0"],
            "imgsz 32'nin katı değil": ["--imgsz", "100"],
            "imgsz küçük": ["--imgsz", "16"],
            "imgsz metin": ["--imgsz", "büyük"],
            "max-det sıfır": ["--max-det", "0"],
            "device cuda": ["--device", "cuda"],
            "device boş": ["--device", ""],
            "source geçersiz": ["--source", "rtsp"],
            "video yolu yok": ["--source", "video"],
            "video + kamera indeksi": ["--source", "video", "--video-path", "v.mp4", "--camera-index", "1"],
            "kamera + video yolu": ["--source", "camera", "--video-path", "v.mp4"],
            "width tek başına": ["--width", "640"],
            "kamera indeksi negatif": ["--camera-index", "-1"],
            "max-frames sıfır": ["--max-frames", "0"],
            "model uzantısı": ["--model", "model.onnx"],
            "bilinmeyen seçenek": ["--bilinmeyen"],
        }
        for label, argv in bad_cases.items():
            with self.subTest(label):
                code, out, err = call(argv)
                self.assertEqual(code, 2, f"{label}: {err}")
                self.assertTrue(err.strip(), "hata mesajı stderr'e yazılmalı")
                self.assertNotIn("Traceback", err)
                self.assertEqual(out, "", "hatalı argümanda normal çıktı olmamalı")

    def test_error_messages_name_the_problem(self) -> None:
        self.assertIn("--conf", call(["--conf", "5"])[2])
        self.assertIn("imgsz", call(["--imgsz", "100"])[2])
        self.assertIn("--video-path", call(["--source", "video"])[2])

    def test_detection_config_from_args(self) -> None:
        args = cli.build_parser().parse_args(["--conf", "0.35", "--iou", "0.5", "--imgsz", "320", "--device", "cpu", "--max-det", "5", "--model", "x.pt"])
        cfg = cli.detection_config_from_args(args)
        self.assertEqual((cfg.conf, cfg.iou, cfg.imgsz, cfg.device, cfg.max_det, cfg.model_name), (0.35, 0.5, 320, "cpu", 5, "x.pt"))


class NoDisplayRunTest(unittest.TestCase):
    def run_cli(self, extra: list[str], detector: FakeDetector | None = None, factory: FakeFactory | None = None):  # type: ignore[no-untyped-def]
        detector = detector or FakeDetector(counts=[2, 0, 1])
        factory = factory or camera_factory()[0]
        made: list[DetectionConfig] = []

        def detector_factory(config: DetectionConfig) -> FakeDetector:
            made.append(config)
            detector.config = config
            return detector

        window_factory = mock.Mock(side_effect=AssertionError("--no-display ile pencere oluşturulmamalı"))
        code, out, err = call(CAM_ARGS + ["--no-display"] + extra, capture_factory=factory, window_factory=window_factory,
                              detector_factory=detector_factory)
        return code, out, err, detector, made, window_factory

    def test_runs_without_window_and_stops_at_max_frames(self) -> None:
        factory, capture = camera_factory()
        code, out, err, detector, _, window_factory = self.run_cli(["--max-frames", "6"], factory=factory)
        self.assertEqual(code, 0, err)
        window_factory.assert_not_called()
        self.assertEqual(len(detector.detect_calls), 6)
        self.assertIn("--max-frames sınırına ulaşıldı", out)
        self.assertIn("İşlenen kare: 6", out)
        self.assertIn("uçtan uca FPS", out)
        self.assertIn("Çıkarım süresi", out)
        self.assertIn("Kişi sayısı", out)
        self.assertIn("Pencere kapalıydı", out)

    def test_resources_released_after_normal_finish(self) -> None:
        factory, capture = camera_factory()
        _, _, _, detector, _, _ = self.run_cli(["--max-frames", "3"], factory=factory)
        self.assertEqual(capture.release_calls, 1, "kamera serbest bırakılmalı")
        self.assertEqual(detector.close_calls, 1, "model serbest bırakılmalı")

    def test_model_loaded_once_warmed_up_at_real_frame_size_and_before_loop(self) -> None:
        _, _, _, detector, _, _ = self.run_cli(["--max-frames", "5"])
        self.assertEqual(detector.load_calls, 1)
        self.assertEqual(detector.warmup_calls, [(64, 48)])

    def test_options_reach_the_detector_configuration(self) -> None:
        _, _, _, _, made, _ = self.run_cli(["--max-frames", "1", "--conf", "0.35", "--iou", "0.5", "--imgsz", "320", "--device", "cpu", "--max-det", "9"])
        self.assertEqual(len(made), 1)
        self.assertEqual((made[0].conf, made[0].iou, made[0].imgsz, made[0].device, made[0].max_det), (0.35, 0.5, 320, "cpu", 9))

    def test_summary_reports_persons_and_device(self) -> None:
        _, out, _, _, _, _ = self.run_cli(["--max-frames", "3"], detector=FakeDetector(counts=[2, 0, 1], device="mps"))
        self.assertIn("en çok 2", out)
        self.assertIn("2/3", out)  # en az bir kişi olan kare
        self.assertIn("cihaz: mps", out)

    def test_source_reported_fps_is_labelled_as_information_only(self) -> None:
        _, out, _, _, _, _ = self.run_cli(["--max-frames", "3"])
        self.assertIn("yalnızca bilgi", out)
        self.assertIn("performans ölçüsü değildir", out)

    def test_device_fallback_warning_is_printed(self) -> None:
        detector = FakeDetector(device="cpu", device_warning="MPS kullanılamıyor (test); CPU'ya dönüldü.")
        _, out, _, _, _, _ = self.run_cli(["--max-frames", "2"], detector=detector)
        self.assertIn("[UYARI] MPS kullanılamıyor", out)

    def test_video_source_runs_to_the_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = write_synthetic_video(Path(tmp) / "v.mp4", n_frames=7, fps=25.0, size=(64, 48))
            detector = FakeDetector(counts=[1])
            code, out, err = call(["--source", "video", "--video-path", str(video), "--no-display"],
                                  detector_factory=lambda cfg: detector, window_factory=mock.Mock(side_effect=AssertionError))
        self.assertEqual(code, 0, err)
        self.assertIn("video sona erdi", out)
        self.assertEqual(len(detector.detect_calls), 7)
        self.assertEqual(detector.warmup_calls, [(64, 48)])


class DisplayRunTest(unittest.TestCase):
    def run_with_window(self, window: FakeWindow, detector: FakeDetector | None = None, extra: list[str] | None = None):  # type: ignore[no-untyped-def]
        factory, capture = camera_factory()
        detector = detector or FakeDetector()
        code, out, err = call(CAM_ARGS + (extra or []), capture_factory=factory, window_factory=lambda: window,
                              detector_factory=lambda cfg: detector)
        return code, out, err, capture, detector

    def test_q_exit_closes_window_camera_and_model(self) -> None:
        window = FakeWindow(keys={4: ord("q")})
        code, out, err, capture, detector = self.run_with_window(window)
        self.assertEqual(code, 0, err)
        self.assertIn("kullanıcı çıkışı (Q/Esc)", out)
        self.assertEqual((window.open_calls, window.close_calls, capture.release_calls, detector.close_calls), (1, 1, 1, 1))
        self.assertEqual(len(window.images), 4)

    def test_esc_exit(self) -> None:
        code, out, _, capture, _ = self.run_with_window(FakeWindow(keys={2: 27}))
        self.assertEqual(code, 0)
        self.assertIn("kullanıcı çıkışı", out)
        self.assertEqual(capture.release_calls, 1)

    def test_window_close_button_exit(self) -> None:
        window = FakeWindow(close_after=3)
        code, out, _, capture, detector = self.run_with_window(window)
        self.assertEqual(code, 0)
        self.assertIn("pencere kapatıldı", out)
        self.assertEqual((window.close_calls, capture.release_calls, detector.close_calls), (1, 1, 1))

    def test_ctrl_c_during_loop_exits_130_and_releases_everything(self) -> None:
        window = FakeWindow()
        detector = FakeDetector(detect_errors={3: KeyboardInterrupt()})
        code, out, _, capture, _ = self.run_with_window(window, detector)
        self.assertEqual(code, 130)
        self.assertEqual((window.close_calls, capture.release_calls, detector.close_calls), (1, 1, 1))

    def test_ctrl_c_during_model_loading_exits_130_without_opening_camera(self) -> None:
        factory, capture = camera_factory()
        detector = FakeDetector(load_error=KeyboardInterrupt())
        code, out, _ = call(CAM_ARGS, capture_factory=factory, window_factory=lambda: FakeWindow(), detector_factory=lambda cfg: detector)
        self.assertEqual(code, 130)
        self.assertEqual(factory.calls, [])
        self.assertEqual(detector.close_calls, 1)

    def test_close_button_warning_is_shown_when_it_could_not_be_enabled(self) -> None:
        window = FakeWindow(keys={1: ord("q")})
        window.close_button_enabled = False  # type: ignore[attr-defined]
        _, out, _, _, _ = self.run_with_window(window)
        self.assertIn("kırmızı kapatma düğmesi etkinleştirilemedi", out)

    def test_no_close_button_warning_when_enabled_or_not_applicable(self) -> None:
        for value in (True, None):
            with self.subTest(value=value):
                window = FakeWindow(keys={1: ord("q")})
                window.close_button_enabled = value  # type: ignore[attr-defined]
                _, out, _, _, _ = self.run_with_window(window)
                self.assertNotIn("etkinleştirilemedi", out)

    def test_display_error_exits_1_and_releases_everything(self) -> None:
        window = FakeWindow()
        window.open_error = DisplayError("grafik oturumu yok")
        code, out, err, capture, detector = self.run_with_window(window)
        self.assertEqual(code, 1)
        self.assertIn("grafik oturumu yok", err)
        self.assertEqual((capture.release_calls, detector.close_calls), (1, 1))

    def test_default_window_uses_person_detection_title(self) -> None:
        created: list[str] = []
        fake = FakeWindow(keys={1: ord("q")})

        def fake_preview_window(name: str) -> FakeWindow:
            created.append(name)
            return fake

        factory, _ = camera_factory()
        with mock.patch.object(cli, "PreviewWindow", fake_preview_window):
            code, _, err = call(CAM_ARGS, capture_factory=factory, detector_factory=lambda cfg: FakeDetector())
        self.assertEqual(code, 0, err)
        self.assertEqual(created, [WINDOW_NAME])
        self.assertEqual(WINDOW_NAME, "ISG Vision - Person Detection")

    def test_default_window_class_is_the_camera_preview_window_with_close_button_fix(self) -> None:
        self.assertIs(cli.PreviewWindow, PreviewWindow)


class ErrorHandlingTest(unittest.TestCase):
    def test_model_load_failure_exits_1_and_never_opens_camera(self) -> None:
        factory, capture = camera_factory()
        detector = FakeDetector(load_error=ModelLoadError("Model ağırlık dosyası bulunamadı: yok.pt"))
        code, out, err = call(CAM_ARGS + ["--no-display"], capture_factory=factory, detector_factory=lambda cfg: detector)
        self.assertEqual(code, 1)
        self.assertIn("bulunamadı", err)
        self.assertEqual(factory.calls, [], "model yüklenemezse kamera hiç açılmamalı")
        self.assertEqual(capture.read_calls, 0)
        self.assertEqual(detector.close_calls, 1)

    def test_explicit_mps_unavailable_exits_1_with_clear_message(self) -> None:
        factory, _ = camera_factory()
        detector = FakeDetector(load_error=DeviceUnavailableError("MPS istendi ancak kullanılamıyor: test"))
        code, out, err = call(CAM_ARGS + ["--no-display", "--device", "mps"], capture_factory=factory, detector_factory=lambda cfg: detector)
        self.assertEqual(code, 1)
        self.assertIn("MPS istendi ancak kullanılamıyor", err)
        self.assertEqual(factory.calls, [])
        self.assertNotIn("Bitiş nedeni", out, "CPU'da sessizce çalışıp başarılı gibi raporlanmamalı")

    def test_camera_open_failure_exits_1_and_releases_model(self) -> None:
        factory, capture = camera_factory(opened=False)
        detector = FakeDetector()
        code, _, err = call(CAM_ARGS + ["--no-display"], capture_factory=factory, detector_factory=lambda cfg: detector)
        self.assertEqual(code, 1)
        self.assertIn("Kamera 0 açılamadı", err)
        self.assertEqual(detector.close_calls, 1)
        self.assertEqual(detector.warmup_calls, [], "kamera açılamadıysa ısınma yapılmamalı")

    def test_camera_failure_mid_stream_exits_1_and_releases_everything(self) -> None:
        capture = FakeCapture([make_frame(i) for i in range(3)], props=dict(CAM_PROPS))
        detector = FakeDetector()
        window = FakeWindow()
        with mock.patch("time.sleep"):
            code, _, err = call(CAM_ARGS, capture_factory=FakeFactory(capture), window_factory=lambda: window,
                                detector_factory=lambda cfg: detector)
        self.assertEqual(code, 1)
        self.assertIn("Kamera bağlantısı kesildi", err)
        self.assertEqual((window.close_calls, capture.release_calls, detector.close_calls), (1, 1, 1))

    def test_invalid_frame_error_exits_1_and_releases_everything(self) -> None:
        factory, capture = camera_factory()
        detector = FakeDetector(detect_errors={2: InvalidFrameError("bozuk kare")})
        code, _, err = call(CAM_ARGS + ["--no-display"], capture_factory=factory, detector_factory=lambda cfg: detector)
        self.assertEqual(code, 1)
        self.assertIn("bozuk kare", err)
        self.assertEqual((capture.release_calls, detector.close_calls), (1, 1))

    def test_missing_video_file_exits_1(self) -> None:
        code, _, err = call(["--source", "video", "--video-path", "/olmayan/video.mp4", "--no-display"], detector_factory=lambda cfg: FakeDetector())
        self.assertEqual(code, 1)
        self.assertIn("bulunamadı", err)


class ImportAndScriptTest(unittest.TestCase):
    def run_python(self, code: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)

    def test_importing_cli_has_no_side_effects(self) -> None:
        code = (
            "import sys, cv2; sys.path.insert(0, %r)\n"
            "def boom(*a, **k): raise AssertionError('kamera açıldı')\n"
            "cv2.VideoCapture = boom\n"
            "import socket; socket.socket.connect = boom\n"
            "import isg_vision.human_detection.cli, isg_vision.human_detection.pipeline, isg_vision.human_detection.visualization\n"
            "bad = [m for m in ('torch', 'ultralytics') if m in sys.modules]\n"
            "sys.exit(1 if bad else 0)\n"
        ) % str(ROOT / "src")
        proc = self.run_python(code)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_script_help_runs(self) -> None:
        proc = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--device", proc.stdout)

    def test_script_rejects_bad_conf_with_exit_2(self) -> None:
        proc = subprocess.run([sys.executable, str(SCRIPT), "--conf", "5"], capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--conf", proc.stderr)

    def test_script_with_missing_weights_exits_1_and_points_to_download_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = write_synthetic_video(Path(tmp) / "v.mp4", n_frames=3, size=(64, 48))
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), "--source", "video", "--video-path", str(video), "--model", str(Path(tmp) / "yok.pt"), "--no-display"],
                capture_output=True, text=True, timeout=120,
            )
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("download_person_model.py", proc.stderr)
        self.assertNotIn("Kaynak açıldı", proc.stdout, "ağırlık yoksa kaynak açılmamalı")

    def test_stage3_camera_script_still_works(self) -> None:
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_camera.py"), "--help"], capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--no-display", proc.stdout)


if __name__ == "__main__":
    unittest.main()
