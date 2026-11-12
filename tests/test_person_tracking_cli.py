"""İnsan takibi komut satırı testleri: argümanlar, eşik doğrulaması, hata kodları, çıkış yolları ve kaynak temizliği.

[Mock/birim] Sahte kamera/pencere/tespitçi/takipçi: gerçek kamera açılmaz, kamera izni istenmez, model yüklenmez, ByteTrack/lap gerekmez,
ağ kullanılmaz. Betik düzeyi (alt süreç) testleri yalnızca ``--help``, hatalı argüman ve eksik ağırlık yollarını sınar.
"""

from __future__ import annotations

import contextlib
import io
import os
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
from person_detection_test_support import FakeDetector  # noqa: E402
from tracking_test_support import FakeTracker  # noqa: E402

from isg_vision.camera.display import PreviewWindow  # noqa: E402  (modül başında: test_no_side_effects modülleri yeniden yükler)
from isg_vision.camera.errors import DisplayError  # noqa: E402
from isg_vision.human_detection import cli as detection_cli  # noqa: E402
from isg_vision.human_detection.config import DetectionConfig  # noqa: E402
from isg_vision.human_detection.errors import ModelLoadError  # noqa: E402
from isg_vision.tracking import cli  # noqa: E402
from isg_vision.tracking.config import TrackingConfig  # noqa: E402
from isg_vision.tracking.errors import TrackerBackendError, TrackerDependencyError  # noqa: E402
from isg_vision.tracking.visualization import WINDOW_NAME  # noqa: E402

SCRIPT = ROOT / "scripts" / "run_person_tracking.py"
CAM_PROPS = {cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 48, cv2.CAP_PROP_FPS: 15.0}
CAM_ARGS = ["--source", "camera", "--width", "64", "--height", "48"]


def camera_factory(n_frames: int = 20) -> tuple[FakeFactory, FakeCapture]:
    capture = FakeCapture([make_frame(i) for i in range(n_frames)], props=dict(CAM_PROPS))
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
    def parse(self, argv: list[str]):  # type: ignore[no-untyped-def]
        return cli.build_parser().parse_args(argv)

    def test_defaults(self) -> None:
        args = self.parse([])
        self.assertEqual((args.source, args.conf, args.iou, args.imgsz, args.device, args.max_det), ("camera", 0.25, 0.7, 640, "auto", 100))
        self.assertEqual(
            (args.track_high_thresh, args.track_low_thresh, args.new_track_thresh, args.track_buffer, args.match_thresh, args.no_fuse_score),
            (0.25, 0.1, 0.25, 30, 0.8, False),
        )
        self.assertFalse(args.no_display)
        self.assertIsNone(args.max_frames)

    def test_default_conf_is_still_the_stage_4a_value_not_0_35(self) -> None:
        """conf=0.35 kendiliğinden genel varsayılan yapılmadı; 4A varsayılanı (0.25) korunur."""
        self.assertEqual(self.parse([]).conf, 0.25)
        self.assertEqual(detection_cli.build_parser().parse_args([]).conf, 0.25)

    def test_documented_task_commands_parse(self) -> None:
        for argv in (
            ["--source", "camera", "--camera-index", "0", "--device", "mps"],
            ["--source", "camera", "--camera-index", "0", "--device", "cpu"],
            ["--source", "video", "--video-path", "/yol/video.mp4", "--device", "mps"],
            ["--source", "camera", "--camera-index", "0", "--device", "mps", "--no-display", "--max-frames", "300"],
            ["--source", "camera", "--camera-index", "0", "--device", "mps", "--conf", "0.25"],
        ):
            with self.subTest(argv=argv):
                self.parse(argv)

    def test_help_documents_every_tracking_option(self) -> None:
        text = cli.build_parser().format_help()
        for option in ("--track-high-thresh", "--track-low-thresh", "--new-track-thresh", "--track-buffer", "--match-thresh", "--no-fuse-score",
                       "--source", "--video-path", "--camera-index", "--device", "--conf", "--no-display", "--max-frames"):
            self.assertIn(option, text)
        self.assertIn("Track ID", text)
        self.assertIn("run_person_tracking.py", text)

    def test_help_exits_zero(self) -> None:
        code, out, _ = call(["--help"])
        self.assertEqual(code, 0)
        self.assertIn("ByteTrack", out)

    def test_detection_parser_is_unchanged_by_tracking_parser(self) -> None:
        """4A betiğinin seçenekleri değişmedi: takip eşikleri 4A parser'ında YOK."""
        text = detection_cli.build_parser().format_help()
        self.assertNotIn("--track-", text)
        self.assertNotIn("--new-track-thresh", text)
        self.assertIn("run_person_detection.py", text)
        self.assertIn("Aşama 4A", text)
        cli.build_parser()  # ikinci çağrı da 4A parser'ını bozmamalı
        self.assertNotIn("--track-", detection_cli.build_parser().format_help())

    def test_bad_arguments_exit_2_with_message(self) -> None:
        cases = {
            "--track-high-thresh": ["0", "1.5", "abc", "-0.2", "nan"],
            "--track-low-thresh": ["0", "2", "x"],
            "--new-track-thresh": ["0", "1.1", "x"],
            "--match-thresh": ["0", "5", "x"],
            "--track-buffer": ["0", "-3", "1001", "2.5", "x"],
        }
        for option, values in cases.items():
            for value in values:
                with self.subTest(option=option, value=value):
                    code, _, err = call(CAM_ARGS + [option, value])
                    self.assertEqual(code, 2)
                    self.assertIn(option, err)

    def test_threshold_relationships_exit_2_with_actionable_message(self) -> None:
        for argv, needle in (
            (["--track-low-thresh", "0.5", "--track-high-thresh", "0.3", "--new-track-thresh", "0.3"], "track_low_thresh"),
            (["--track-low-thresh", "0.25"], "track_low_thresh"),  # high varsayılan 0.25 ile eşit
            (["--track-high-thresh", "0.5"], "new_track_thresh"),  # new varsayılan 0.25 < high
        ):
            with self.subTest(argv=argv):
                code, _, err = call(CAM_ARGS + argv)
                self.assertEqual(code, 2)
                self.assertIn(needle, err)

    def test_tracking_config_from_args(self) -> None:
        args = self.parse(["--track-high-thresh", "0.4", "--track-low-thresh", "0.15", "--new-track-thresh", "0.5", "--track-buffer", "45",
                           "--match-thresh", "0.7", "--no-fuse-score"])
        self.assertEqual(
            cli.tracking_config_from_args(args),
            TrackingConfig(track_high_thresh=0.4, track_low_thresh=0.15, new_track_thresh=0.5, track_buffer=45, match_thresh=0.7, fuse_score=False),
        )

    def test_detection_options_are_shared_with_stage_4a(self) -> None:
        args = self.parse(["--conf", "0.35", "--iou", "0.5", "--imgsz", "320", "--device", "cpu", "--max-det", "9", "--model", "a.pt"])
        cfg = detection_cli.detection_config_from_args(args)
        self.assertEqual((cfg.conf, cfg.iou, cfg.imgsz, cfg.device, cfg.max_det, cfg.model_name), (0.35, 0.5, 320, "cpu", 9, "a.pt"))


class RunTest(unittest.TestCase):
    def run_cli(self, extra: list[str], *, factory: FakeFactory | None = None, detector: FakeDetector | None = None,
                tracker: FakeTracker | None = None, window: FakeWindow | None = None):  # type: ignore[no-untyped-def]
        if factory is None:
            factory, capture = camera_factory()
        else:
            capture = factory.capture
        detector = detector or FakeDetector(counts=[2])
        tracker = tracker or FakeTracker()
        made: dict[str, list] = {"det": [], "trk": []}

        def detector_factory(cfg: DetectionConfig) -> FakeDetector:
            made["det"].append(cfg)
            return detector

        def tracker_factory(cfg: TrackingConfig) -> FakeTracker:
            made["trk"].append(cfg)
            return tracker

        args = CAM_ARGS + ([] if window is not None else ["--no-display"]) + extra
        window_factory = (lambda: window) if window is not None else mock.Mock(side_effect=AssertionError("--no-display ile pencere oluşturulmamalı"))
        code, out, err = call(args, capture_factory=factory, window_factory=window_factory, detector_factory=detector_factory,
                              tracker_factory=tracker_factory)
        return code, out, err, detector, tracker, capture, made

    def test_runs_without_window_and_stops_at_max_frames(self) -> None:
        code, out, err, detector, tracker, _, _ = self.run_cli(["--max-frames", "6"])
        self.assertEqual(code, 0, err)
        self.assertEqual(len(detector.detect_calls), 6)
        self.assertEqual(len(tracker.updates), 6)
        for needle in ("--max-frames sınırına ulaşıldı", "İşlenen kare: 6", "uçtan uca FPS", "YOLO çıkarım süresi", "ByteTrack takip süresi",
                       "YOLO + ByteTrack", "Kare -> takip sonucu gecikmesi", "Kare okuma süresi", "Pencere kapalıydı", "Görülen farklı Track ID"):
            self.assertIn(needle, out)

    def test_resources_released_after_normal_finish(self) -> None:
        _, _, _, detector, tracker, capture, _ = self.run_cli(["--max-frames", "3"])
        self.assertEqual((capture.release_calls, detector.close_calls, tracker.close_calls), (1, 1, 1))

    def test_model_and_tracker_loaded_once_and_warmup_before_loop(self) -> None:
        _, _, _, detector, tracker, _, _ = self.run_cli(["--max-frames", "5"])
        self.assertEqual((detector.load_calls, tracker.load_calls), (1, 1))
        self.assertEqual(detector.warmup_calls, [(64, 48)])
        self.assertEqual(tracker.events[:2], ["load", "reset"], "takipçi yüklenir, döngü başında sıfırlanır")

    def test_options_reach_detector_and_tracker_configuration(self) -> None:
        _, _, _, _, _, _, made = self.run_cli(["--max-frames", "1", "--conf", "0.35", "--device", "cpu", "--track-high-thresh", "0.4",
                                                "--new-track-thresh", "0.5", "--track-buffer", "12", "--no-fuse-score"])
        self.assertEqual((made["det"][0].conf, made["det"][0].device), (0.35, "cpu"))
        t = made["trk"][0]
        self.assertEqual((t.track_high_thresh, t.new_track_thresh, t.track_buffer, t.fuse_score), (0.4, 0.5, 12, False))

    def test_startup_prints_tracker_settings_and_threshold_analysis(self) -> None:
        _, out, _, _, _, _, _ = self.run_cli(["--max-frames", "1"])
        self.assertIn("Takip: ByteTrack", out)
        self.assertIn("tampon 30 kare", out)
        self.assertIn("KAPALI", out, "varsayılan conf=0.25: düşük güvenli kurtarma aşamasının fiilen kapalı olduğu söylenmeli")
        self.assertIn("Track ID geçici ve oturum kapsamlıdır", out)
        self.assertIn("YOLO her karede yalnızca bir kez çalışır", out)

    def test_threshold_analysis_changes_when_conf_is_lowered(self) -> None:
        _, out, _, _, _, _, _ = self.run_cli(["--max-frames", "1", "--conf", "0.10", "--track-high-thresh", "0.35", "--new-track-thresh", "0.35"])
        self.assertIn("AÇIK", out)
        self.assertNotIn("fiilen KAPALI", out)

    def test_version_note_is_shown_as_warning(self) -> None:
        tracker = FakeTracker()
        tracker.backend_info = {"versions": {"ultralytics": "9.0.0", "lap": "0.5.14"}, "version_note": "Ultralytics 9.0.0 kurulu; farklı sürüm"}
        _, out, _, _, _, _, _ = self.run_cli(["--max-frames", "1"], tracker=tracker)
        self.assertIn("[UYARI] Ultralytics 9.0.0 kurulu", out)
        self.assertIn("ultralytics 9.0.0, lap 0.5.14", out)

    def test_track_table_is_printed_with_ids_and_frame_ranges(self) -> None:
        _, out, _, _, _, _, _ = self.run_cli(["--max-frames", "6"], detector=FakeDetector(counts=[1, 2, 2, 0, 0, 1]))
        self.assertIn("İz özeti", out)
        self.assertIn("ID   1", out)
        self.assertIn("kare 0-5", out)
        self.assertIn("yanlış pozitif olabilir", out)

    def test_no_track_message_when_nobody_detected(self) -> None:
        _, out, _, _, _, _, _ = self.run_cli(["--max-frames", "3"], detector=FakeDetector(counts=[0]))
        self.assertIn("Hiç aktif iz oluşmadı", out)

    def test_source_reported_fps_is_labelled_as_information_only(self) -> None:
        _, out, _, _, _, _, _ = self.run_cli(["--max-frames", "2"])
        self.assertIn("kameranın bildirdiği FPS performans ölçüsü değildir", out)

    def test_nothing_is_written_to_disk(self) -> None:
        """Görüntü/kayıt arşivlenmez: çalışma dizininde yeni dosya oluşmaz."""
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                self.run_cli(["--max-frames", "4"])
                self.assertEqual(os.listdir(tmp), [])
            finally:
                os.chdir(cwd)

    def test_video_source_runs_to_the_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = write_synthetic_video(Path(tmp) / "v.mp4", n_frames=8, size=(64, 48))
            detector, tracker = FakeDetector(counts=[1]), FakeTracker()
            code, out, err = call(["--source", "video", "--video-path", str(video), "--no-display"],
                                  detector_factory=lambda cfg: detector, tracker_factory=lambda cfg: tracker)
        self.assertEqual(code, 0, err)
        self.assertIn("video sona erdi", out)
        self.assertIn("İşlenen kare: 8", out)
        self.assertEqual((len(tracker.updates), tracker.close_calls, detector.close_calls), (8, 1, 1))

    def run_with_window(self, window: FakeWindow, detector: FakeDetector | None = None, tracker: FakeTracker | None = None):  # type: ignore[no-untyped-def]
        return self.run_cli([], window=window, detector=detector, tracker=tracker)

    def test_q_exit_closes_everything(self) -> None:
        window = FakeWindow(keys={4: ord("q")})
        code, out, err, detector, tracker, capture, _ = self.run_with_window(window)
        self.assertEqual(code, 0, err)
        self.assertIn("kullanıcı çıkışı (Q/Esc)", out)
        self.assertEqual((window.open_calls, window.close_calls, capture.release_calls, detector.close_calls, tracker.close_calls), (1, 1, 1, 1, 1))
        self.assertEqual(len(window.images), 4)

    def test_esc_exit(self) -> None:
        code, out, _, _, tracker, capture, _ = self.run_with_window(FakeWindow(keys={2: 27}))
        self.assertEqual(code, 0)
        self.assertIn("kullanıcı çıkışı", out)
        self.assertEqual((capture.release_calls, tracker.close_calls), (1, 1))

    def test_window_close_button_exit(self) -> None:
        window = FakeWindow(close_after=3)
        code, out, _, detector, tracker, capture, _ = self.run_with_window(window)
        self.assertEqual(code, 0)
        self.assertIn("pencere kapatıldı", out)
        self.assertEqual((window.close_calls, capture.release_calls, detector.close_calls, tracker.close_calls), (1, 1, 1, 1))

    def test_ctrl_c_during_loop_exits_130_and_releases_everything(self) -> None:
        window = FakeWindow()
        code, out, _, detector, tracker, capture, _ = self.run_with_window(window, detector=FakeDetector(detect_errors={3: KeyboardInterrupt()}))
        self.assertEqual(code, 130)
        self.assertEqual((window.close_calls, capture.release_calls, detector.close_calls, tracker.close_calls), (1, 1, 1, 1))
        self.assertIn("Ctrl+C", out)

    def test_ctrl_c_inside_tracker_exits_130_and_releases_everything(self) -> None:
        window = FakeWindow()
        code, _, _, detector, tracker, capture, _ = self.run_with_window(window, tracker=FakeTracker(update_errors={2: KeyboardInterrupt()}))
        self.assertEqual(code, 130)
        self.assertEqual((window.close_calls, capture.release_calls, detector.close_calls, tracker.close_calls), (1, 1, 1, 1))

    def test_ctrl_c_during_model_loading_exits_130_without_opening_camera(self) -> None:
        factory, capture = camera_factory()
        detector, tracker = FakeDetector(load_error=KeyboardInterrupt()), FakeTracker()
        code, _, _ = call(CAM_ARGS, capture_factory=factory, window_factory=lambda: FakeWindow(), detector_factory=lambda cfg: detector,
                          tracker_factory=lambda cfg: tracker)
        self.assertEqual(code, 130)
        self.assertEqual(factory.calls, [])
        self.assertEqual((detector.close_calls, tracker.close_calls), (1, 1))

    def test_ctrl_c_during_tracker_loading_exits_130_without_opening_camera(self) -> None:
        factory, _ = camera_factory()
        tracker = FakeTracker(load_error=KeyboardInterrupt())
        code, _, _ = call(CAM_ARGS, capture_factory=factory, window_factory=lambda: FakeWindow(), detector_factory=lambda cfg: FakeDetector(),
                          tracker_factory=lambda cfg: tracker)
        self.assertEqual(code, 130)
        self.assertEqual(factory.calls, [])

    def test_display_error_exits_1_and_releases_everything(self) -> None:
        window = FakeWindow(raise_on_call={2: DisplayError("pencere yok")})
        code, _, err, detector, tracker, capture, _ = self.run_with_window(window)
        self.assertEqual(code, 1)
        self.assertIn("pencere yok", err)
        self.assertEqual((window.close_calls, capture.release_calls, detector.close_calls, tracker.close_calls), (1, 1, 1, 1))

    def test_close_button_warning_is_shown_when_it_could_not_be_enabled(self) -> None:
        window = FakeWindow(keys={1: ord("q")})
        window.close_button_enabled = False  # type: ignore[attr-defined]
        _, out, _, _, _, _, _ = self.run_with_window(window)
        self.assertIn("kırmızı kapatma düğmesi etkinleştirilemedi", out)

    def test_default_window_uses_the_tracking_title(self) -> None:
        factory, _ = camera_factory()
        created: list[tuple[tuple, dict]] = []

        class SpyWindow(FakeWindow):
            def __init__(self, *a, **k):  # type: ignore[no-untyped-def]
                super().__init__(keys={1: ord("q")})
                created.append((a, k))

        with mock.patch.object(cli, "PreviewWindow", SpyWindow):
            code, _, _ = call(CAM_ARGS, capture_factory=factory, detector_factory=lambda cfg: FakeDetector(), tracker_factory=lambda cfg: FakeTracker())
        self.assertEqual(code, 0)
        self.assertEqual(created[0][0], (WINDOW_NAME,))
        self.assertEqual(WINDOW_NAME, "ISG Vision - Person Tracking")

    def test_default_window_class_is_the_camera_preview_window_with_close_button_fix(self) -> None:
        """Aşama 3'te çözülen macOS kırmızı düğme düzeltmesi 4B penceresinde de korunur (aynı PreviewWindow sınıfı)."""
        self.assertIs(cli.PreviewWindow, PreviewWindow)


class ErrorPathTest(unittest.TestCase):
    def assert_everything_released(self, *objs: object) -> None:
        for obj in objs:
            self.assertEqual(getattr(obj, "close_calls"), 1)

    def test_model_load_failure_exits_1_and_never_opens_camera(self) -> None:
        factory, capture = camera_factory()
        detector, tracker = FakeDetector(load_error=ModelLoadError("ağırlık yok")), FakeTracker()
        code, _, err = call(CAM_ARGS, capture_factory=factory, detector_factory=lambda cfg: detector, tracker_factory=lambda cfg: tracker)
        self.assertEqual(code, 1)
        self.assertIn("ağırlık yok", err)
        self.assertEqual(factory.calls, [], "model yüklenemezse kamera hiç açılmamalı")
        self.assertEqual(tracker.load_calls, 0, "model başarısızsa takipçi de yüklenmemeli")
        self.assert_everything_released(detector, tracker)

    def test_missing_lap_exits_1_with_install_hint_and_never_opens_camera(self) -> None:
        factory, capture = camera_factory()
        detector = FakeDetector()
        tracker = FakeTracker(load_error=TrackerDependencyError("ByteTrack için 'lap' paketi kurulu değil. Kurulum: .venv/bin/python -m pip install lap==0.5.14"))
        code, out, err = call(CAM_ARGS, capture_factory=factory, detector_factory=lambda cfg: detector, tracker_factory=lambda cfg: tracker)
        self.assertEqual(code, 1)
        self.assertIn("lap", err)
        self.assertIn("pip install lap==", err)
        self.assertEqual(factory.calls, [], "bağımlılık eksikse kamera hiç açılmamalı")
        self.assertEqual(detector.warmup_calls, [])
        self.assert_everything_released(detector, tracker)
        self.assertNotIn("Kaynak açıldı", out)

    def test_tracker_backend_start_failure_exits_1(self) -> None:
        factory, _ = camera_factory()
        tracker = FakeTracker(load_error=TrackerBackendError("ByteTrack takipçisi başlatılamadı"))
        code, _, err = call(CAM_ARGS, capture_factory=factory, detector_factory=lambda cfg: FakeDetector(), tracker_factory=lambda cfg: tracker)
        self.assertEqual(code, 1)
        self.assertIn("başlatılamadı", err)
        self.assertEqual(factory.calls, [])

    def test_tracking_error_mid_stream_exits_1_and_releases_everything(self) -> None:
        factory, capture = camera_factory()
        detector = FakeDetector()
        tracker = FakeTracker(update_errors={3: TrackerBackendError("ByteTrack çöktü")})
        code, _, err = call(CAM_ARGS + ["--no-display"], capture_factory=factory, detector_factory=lambda cfg: detector, tracker_factory=lambda cfg: tracker)
        self.assertEqual(code, 1)
        self.assertIn("ByteTrack çöktü", err)
        self.assertEqual((capture.release_calls, detector.close_calls, tracker.close_calls), (1, 1, 1))

    def test_camera_open_failure_exits_1_and_releases_model_and_tracker(self) -> None:
        capture = FakeCapture([], opened=False, props=dict(CAM_PROPS))
        detector, tracker = FakeDetector(), FakeTracker()
        code, _, err = call(CAM_ARGS, capture_factory=FakeFactory(capture), detector_factory=lambda cfg: detector, tracker_factory=lambda cfg: tracker)
        self.assertEqual(code, 1)
        self.assertIn("açılamadı", err)
        self.assert_everything_released(detector, tracker)

    def test_camera_failure_mid_stream_exits_1_and_releases_everything(self) -> None:
        capture = FakeCapture([make_frame(0), make_frame(1)], props=dict(CAM_PROPS))  # sonra okuma başarısız
        detector, tracker = FakeDetector(), FakeTracker()
        code, _, err = call(CAM_ARGS + ["--no-display"], capture_factory=FakeFactory(capture), detector_factory=lambda cfg: detector,
                            tracker_factory=lambda cfg: tracker)
        self.assertEqual(code, 1)
        self.assertIn("bağlantısı kesildi", err)
        self.assertEqual((capture.release_calls, detector.close_calls, tracker.close_calls), (1, 1, 1))

    def test_missing_video_file_exits_1(self) -> None:
        code, _, err = call(["--source", "video", "--video-path", "/yok/olmayan.mp4", "--no-display"],
                            detector_factory=lambda cfg: FakeDetector(), tracker_factory=lambda cfg: FakeTracker())
        self.assertEqual(code, 1)
        self.assertTrue(err)


class ImportAndScriptTest(unittest.TestCase):
    def run_python(self, code: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)

    def test_importing_tracking_modules_has_no_side_effects(self) -> None:
        code = (
            "import sys, cv2; sys.path.insert(0, %r)\n"
            "def boom(*a, **k): raise AssertionError('kamera/ağ kullanıldı')\n"
            "cv2.VideoCapture = boom\n"
            "import socket; socket.socket.connect = boom\n"
            "import isg_vision.tracking.cli, isg_vision.tracking.pipeline, isg_vision.tracking.visualization, isg_vision.tracking.bytetrack_adapter\n"
            "bad = [m for m in ('torch', 'ultralytics', 'lap') if m in sys.modules]\n"
            "print('YUKLENEN:', bad)\n"
            "sys.exit(1 if bad else 0)\n"
        ) % str(ROOT / "src")
        proc = self.run_python(code)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_missing_lap_with_real_tracker_factory_fails_cleanly_without_pip_or_network(self) -> None:
        """CLI + GERÇEK PersonTracker fabrikası + gerçek bağımlılık denetimi; 'lap' kurulu değilmiş gibi (alt süreçte).
        pip/alt süreç ve ağ girişimi yasaktır (denenirse AssertionError); kamera hiç açılmamalı; çıkış kodu 1; kurulum komutu stderr'de."""
        code = (
            "import sys, importlib.metadata as m, subprocess, socket\n"
            "sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
            "real = m.version\n"
            "def fake(name):\n"
            "    if name == 'lap': raise m.PackageNotFoundError(name)\n"
            "    return real(name)\n"
            "m.version = fake\n"
            "def boom(*a, **k): raise AssertionError('pip/alt surec/ag denendi')\n"
            "subprocess.Popen = boom; subprocess.run = boom; subprocess.check_output = boom; socket.socket.connect = boom\n"
            "from camera_test_support import FakeCapture, FakeFactory, make_frame\n"
            "from person_detection_test_support import FakeDetector\n"
            "from isg_vision.tracking import cli\n"
            "factory = FakeFactory(FakeCapture([make_frame(i) for i in range(5)], props={3: 64, 4: 48}))\n"
            "detector = FakeDetector()\n"
            "rc = cli.main(['--source', 'camera', '--width', '64', '--height', '48', '--no-display', '--max-frames', '3'],\n"
            "              capture_factory=factory, detector_factory=lambda cfg: detector)\n"
            "print('RC', rc, 'CAMERA_CALLS', len(factory.calls), 'DETECT', len(detector.detect_calls))\n"
            "sys.exit(rc)\n"
        ) % (str(ROOT / "src"), str(ROOT / "tests"))
        proc = self.run_python(code)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("RC 1 CAMERA_CALLS 0 DETECT 0", proc.stdout)
        self.assertIn("'lap'", proc.stderr)
        self.assertIn("pip install lap==", proc.stderr)
        self.assertNotIn("AssertionError", proc.stderr)

    def test_script_help_runs(self) -> None:
        proc = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for option in ("--device", "--track-buffer", "--no-display"):
            self.assertIn(option, proc.stdout)

    def test_script_rejects_bad_threshold_with_exit_2(self) -> None:
        proc = subprocess.run([sys.executable, str(SCRIPT), "--track-high-thresh", "5"], capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--track-high-thresh", proc.stderr)

    def test_script_rejects_inconsistent_thresholds_with_exit_2(self) -> None:
        proc = subprocess.run([sys.executable, str(SCRIPT), "--new-track-thresh", "0.1"], capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("new_track_thresh", proc.stderr)

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

    def test_stage_4a_and_3_scripts_still_work(self) -> None:
        for name, option in (("run_person_detection.py", "--conf"), ("run_camera.py", "--no-display")):
            with self.subTest(script=name):
                proc = subprocess.run([sys.executable, str(ROOT / "scripts" / name), "--help"], capture_output=True, text=True, timeout=120)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn(option, proc.stdout)
                if name == "run_person_detection.py":
                    self.assertNotIn("--track-", proc.stdout)


if __name__ == "__main__":
    unittest.main()
