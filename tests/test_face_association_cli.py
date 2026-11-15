"""Yüz–kişi ilişkilendirme komut satırı: argümanlar, doğrulama, yükleme sırası, hata kodları, çıkış yolları, kaynak temizliği, ``--opencv-threads``.

[Mock/birim] Sahte kamera/pencere/modeller/takipçi: gerçek kamera açılmaz, kamera izni istenmez, model yüklenmez, ağ kullanılmaz. Betik düzeyi (alt süreç)
testleri yalnızca ``--help``, hatalı argüman ve eksik ağırlık yollarını sınar.
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

from camera_test_support import FakeCapture, FakeFactory, FakeWindow, make_frame  # noqa: E402
from face_association_test_support import ScriptedFaceDetector, ScriptedTracker, SizedFakeDetector, head_box, person_box  # noqa: E402

from isg_vision.camera.errors import DisplayError  # noqa: E402
from isg_vision.face_association import cli  # noqa: E402
from isg_vision.face_association.config import AssociationConfig  # noqa: E402
from isg_vision.face_association.threads import ThreadSetting  # noqa: E402
from isg_vision.face_association.visualization import WINDOW_NAME  # noqa: E402
from isg_vision.face_detection.errors import FaceModelLoadError  # noqa: E402
from isg_vision.human_detection.errors import ModelLoadError  # noqa: E402
from isg_vision.tracking import cli as tracking_cli  # noqa: E402
from isg_vision.tracking.errors import TrackerDependencyError  # noqa: E402

SCRIPT = ROOT / "scripts" / "run_face_person_association.py"
SIZE = (640, 480)
CAM_PROPS = {cv2.CAP_PROP_FRAME_WIDTH: SIZE[0], cv2.CAP_PROP_FRAME_HEIGHT: SIZE[1], cv2.CAP_PROP_FPS: 15.0}
CAM_ARGS = ["--source", "camera", "--width", str(SIZE[0]), "--height", str(SIZE[1])]
P = person_box(300, top=60, w=110, h=330)
F = head_box(P, size=34)


def camera_factory(n_frames: int = 20) -> tuple[FakeFactory, FakeCapture]:
    capture = FakeCapture([make_frame(i, *SIZE) for i in range(n_frames)], props=dict(CAM_PROPS))
    return FakeFactory(capture), capture


def call(argv: list[str], **kwargs):  # type: ignore[no-untyped-def]
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli.main(argv, **kwargs)
        except SystemExit as exc:
            code = int(exc.code) if exc.code is not None else 0
    return code, out.getvalue(), err.getvalue()


class Parts:
    """Fabrika sahteleri: oluşturulan nesneleri ve olay sırasını kaydeder."""

    def __init__(self, frames: int = 30, *, yolo=None, tracker=None, face=None, threads=None) -> None:  # type: ignore[no-untyped-def]
        self.events: list[str] = []
        self.yolo = yolo or SizedFakeDetector()
        self.tracker = tracker or ScriptedTracker([[(1, P)]] * frames)
        self.face = face or ScriptedFaceDetector([[F]] * frames)
        self.thread_requests: list[str] = []
        self.thread_result = threads or ThreadSetting("keep", None, 1, 1)
        self.window = FakeWindow()
        for name, obj in (("yolo", self.yolo), ("tracker", self.tracker), ("face", self.face)):
            original = obj.load
            obj.load = (lambda n=name, o=original: (self.events.append(f"load:{n}"), o())[1])  # type: ignore[method-assign]

    def kwargs(self, **extra):  # type: ignore[no-untyped-def]
        def apply(requested: str) -> ThreadSetting:
            self.events.append(f"threads:{requested}")
            self.thread_requests.append(requested)
            return self.thread_result

        kw = dict(detector_factory=lambda cfg: self.yolo, tracker_factory=lambda cfg: self.tracker, face_detector_factory=lambda cfg: self.face,
                  window_factory=lambda: self.window, thread_applier=apply)
        kw.update(extra)
        return kw


class ParserTest(unittest.TestCase):
    def parse(self, argv: list[str]):  # type: ignore[no-untyped-def]
        return cli.build_parser().parse_args(argv)

    def test_defaults_keep_4a_4b_5a_values(self) -> None:
        a = self.parse([])
        self.assertEqual((a.source, a.conf, a.iou, a.imgsz, a.device), ("camera", 0.25, 0.7, 640, "auto"))
        self.assertEqual((a.track_high_thresh, a.track_low_thresh, a.new_track_thresh, a.track_buffer, a.match_thresh), (0.25, 0.1, 0.25, 30, 0.8))
        self.assertEqual((a.face_conf, a.face_nms, a.face_max_faces, a.face_detect_width), (0.6, 0.3, 100, None))
        self.assertEqual((a.min_containment, a.min_score, a.ambiguity_margin, a.overlap_guard), (0.8, 0.5, 0.15, 0.25))
        self.assertEqual(a.opencv_threads, "keep", "iş parçacığı ayarı varsayılan olarak DEĞİŞTİRİLMEZ")
        self.assertFalse(a.no_display)

    def test_default_configs_equal_the_modules_own_defaults(self) -> None:
        from isg_vision.face_detection.config import FaceDetectionConfig
        from isg_vision.human_detection.config import DetectionConfig
        from isg_vision.human_detection.cli import detection_config_from_args
        from isg_vision.tracking.config import TrackingConfig

        a = self.parse([])
        self.assertEqual(cli.face_config_from_args(a), FaceDetectionConfig())
        self.assertEqual(cli.association_config_from_args(a), AssociationConfig())
        self.assertEqual(tracking_cli.tracking_config_from_args(a), TrackingConfig())
        self.assertEqual(detection_config_from_args(a), DetectionConfig())

    def test_documented_commands_parse(self) -> None:
        for argv in (
            ["--source", "camera", "--camera-index", "0", "--device", "mps"],
            ["--source", "video", "--video-path", "/yol/video.mp4", "--device", "mps"],
            ["--source", "camera", "--device", "mps", "--no-display", "--max-frames", "300"],
            ["--source", "camera", "--device", "mps", "--opencv-threads", "restore"],
            ["--source", "camera", "--face-conf", "0.7", "--ambiguity-margin", "0.2"],
            ["--opencv-threads", "4"],
        ):
            with self.subTest(argv=argv):
                self.parse(argv)

    def test_help_documents_the_new_options_and_the_limits(self) -> None:
        text = cli.build_parser().format_help()
        for option in ("--face-model", "--face-conf", "--face-nms", "--face-max-faces", "--face-detect-width", "--min-containment", "--min-score",
                       "--ambiguity-margin", "--overlap-guard", "--opencv-threads", "--source", "--video-path", "--device", "--no-display", "--max-frames",
                       "--track-buffer"):
            self.assertIn(option, text)
        for phrase in ("Track ID", "kimlik", "kare başına yalnızca bir kez", "run_face_person_association.py"):
            self.assertIn(phrase, text)

    def test_help_exits_zero(self) -> None:
        code, out, _ = call(["--help"])
        self.assertEqual(code, 0)
        self.assertIn("YuNet", out)

    def test_other_stage_parsers_are_untouched(self) -> None:
        from isg_vision.face_detection import cli as face_cli
        from isg_vision.human_detection import cli as det_cli

        for text in (det_cli.build_parser().format_help(), tracking_cli.build_parser().format_help(), face_cli.build_parser().format_help()):
            self.assertNotIn("--opencv-threads", text)
            self.assertNotIn("--ambiguity-margin", text)
        self.assertNotIn("--face-conf", tracking_cli.build_parser().format_help())

    def test_invalid_arguments_exit_with_usage_code_2(self) -> None:
        for argv in (["--min-score", "1.5"], ["--min-containment", "0"], ["--ambiguity-margin", "0"], ["--overlap-guard", "2"], ["--face-conf", "0"],
                     ["--face-detect-width", "10"], ["--opencv-threads", "-3"], ["--opencv-threads", "auto"], ["--min-score", "x"], ["--face-max-faces", "0"]):
            with self.subTest(argv=argv):
                code, _, err = call(argv)
                self.assertEqual(code, 2)
                self.assertIn("usage", err.lower())

    def test_invalid_values_are_rejected_before_any_factory_or_camera(self) -> None:
        parts = Parts()
        factory, capture = camera_factory()
        code, _, _ = call(CAM_ARGS + ["--min-score", "7"], capture_factory=factory, **parts.kwargs())
        self.assertEqual(code, 2)
        self.assertEqual(factory.calls, [])
        self.assertEqual(parts.events, [])


class RunFlowTest(unittest.TestCase):
    def test_normal_run_returns_zero_and_loads_in_the_documented_order_before_the_camera(self) -> None:
        parts = Parts()
        factory, capture = camera_factory()
        opened_after: list[list[str]] = []
        original_call = factory.__call__

        def spying_factory(*a):  # type: ignore[no-untyped-def]
            opened_after.append(list(parts.events))
            return original_call(*a)

        code, out, _ = call(CAM_ARGS + ["--max-frames", "5"], capture_factory=spying_factory, **parts.kwargs())
        self.assertEqual(code, 0)
        self.assertEqual(opened_after[0], ["load:yolo", "load:tracker", "load:face", "threads:keep"], "kamera, tüm modeller yüklendikten SONRA açılır")
        self.assertEqual(parts.yolo.warmup_calls, [SIZE])
        self.assertEqual(parts.face.warmup_calls, [SIZE], "yüz modeli de gerçek kare boyutunda ısıtılır")
        self.assertEqual((len(parts.yolo.calls), len(parts.tracker.updates), len(parts.face.calls)), (5, 5, 5))
        self.assertGreaterEqual(capture.release_calls, 1)
        self.assertEqual((parts.window.open_calls, parts.window.close_calls), (1, 1))
        self.assertEqual((parts.yolo.close_calls, parts.face.close_calls), (1, 1))
        self.assertIn("--max-frames", out)

    def test_summary_lists_every_required_measurement_and_count(self) -> None:
        parts = Parts()
        factory, _ = camera_factory()
        code, out, _ = call(CAM_ARGS + ["--max-frames", "6"], capture_factory=factory, **parts.kwargs())
        self.assertEqual(code, 0)
        for fragment in ("uçtan uca FPS", "YOLO çıkarım süresi", "ByteTrack takip süresi", "YuNet yüz çıkarım süresi", "İlişkilendirme süresi",
                         "Toplam hesaplama", "Kare -> ilişkilendirme sonucu gecikmesi", "Kare okuma süresi", "eşleşen 6", "eşleşmeyen yüz 0", "Süreklilik kesintileri",
                         "new_track: 1", "İz özeti", "ID   1", "benzersiz kişi sayısı DEĞİLDİR", "Track ID geçici", "kimlik veya canlı insan doğrulaması değildir"):
            self.assertIn(fragment, out)

    def test_no_display_never_creates_a_window(self) -> None:
        parts = Parts()
        factory, _ = camera_factory()
        wf = mock.Mock(side_effect=AssertionError("pencere oluşturulmamalı"))
        parts_kwargs = parts.kwargs(window_factory=wf)
        code, out, _ = call(CAM_ARGS + ["--no-display", "--max-frames", "4"], capture_factory=factory, **parts_kwargs)
        self.assertEqual(code, 0)
        wf.assert_not_called()
        self.assertIn("Pencere kapalıydı", out)

    def test_window_mode_uses_the_association_window_name(self) -> None:
        names = []
        parts = Parts()
        factory, _ = camera_factory()
        code, _, _ = call(CAM_ARGS + ["--max-frames", "3"], capture_factory=factory, **parts.kwargs(window_factory=lambda: (names.append("w"), parts.window)[1]))
        self.assertEqual(code, 0)
        self.assertEqual(len(parts.window.images), 3)
        self.assertIn("Association", WINDOW_NAME)

    def test_q_esc_and_window_close_exit_with_zero(self) -> None:
        for kwargs, expected in (({"keys": {2: ord("q")}}, "Q/Esc"), ({"keys": {2: 27}}, "Q/Esc"), ({"close_after": 2}, "pencere kapatıldı")):
            with self.subTest(kwargs=kwargs):
                parts = Parts()
                parts.window = FakeWindow(**kwargs)
                factory, capture = camera_factory()
                code, out, _ = call(CAM_ARGS, capture_factory=factory, **parts.kwargs())
                self.assertEqual(code, 0)
                self.assertIn(expected, out)
                self.assertGreaterEqual(capture.release_calls, 1)
                self.assertEqual(parts.window.close_calls, 1)

    def test_ctrl_c_during_the_loop_returns_130_and_releases_everything(self) -> None:
        class Interrupting(SizedFakeDetector):
            def detect(self, image, **kw):  # type: ignore[no-untyped-def]
                if len(self.calls) == 2:
                    raise KeyboardInterrupt
                return super().detect(image, **kw)

        parts = Parts(yolo=Interrupting())
        factory, capture = camera_factory()
        code, out, _ = call(CAM_ARGS, capture_factory=factory, **parts.kwargs())
        self.assertEqual(code, 130)
        self.assertGreaterEqual(capture.release_calls, 1)
        self.assertEqual((parts.window.close_calls, parts.yolo.close_calls, parts.face.close_calls), (1, 1, 1))

    def test_ctrl_c_during_model_loading_returns_130_without_opening_the_camera(self) -> None:
        class InterruptOnLoad(ScriptedFaceDetector):
            def load(self):  # type: ignore[no-untyped-def]
                raise KeyboardInterrupt

        parts = Parts(face=InterruptOnLoad([[F]]))
        factory, _ = camera_factory()
        code, _, _ = call(CAM_ARGS, capture_factory=factory, **parts.kwargs())
        self.assertEqual(code, 130)
        self.assertEqual(factory.calls, [])


class ErrorHandlingTest(unittest.TestCase):
    def assert_failure_before_camera(self, parts: Parts, expected_text: str) -> None:
        factory, capture = camera_factory()
        code, _, err = call(CAM_ARGS, capture_factory=factory, **parts.kwargs())
        self.assertEqual(code, 1)
        self.assertIn(expected_text, err)
        self.assertEqual(factory.calls, [], "model/takipçi yüklenemezse kamera HİÇ açılmaz")
        self.assertEqual(parts.thread_requests, [], "yükleme başarısızsa iş parçacığı ayarına gidilmez")

    def test_yolo_load_failure(self) -> None:
        class Bad(SizedFakeDetector):
            def load(self):  # type: ignore[no-untyped-def]
                raise ModelLoadError("yolo yok")

        self.assert_failure_before_camera(Parts(yolo=Bad()), "yolo yok")

    def test_tracker_dependency_failure(self) -> None:
        class Bad(ScriptedTracker):
            def load(self):  # type: ignore[no-untyped-def]
                raise TrackerDependencyError("lap yok")

        self.assert_failure_before_camera(Parts(tracker=Bad([[]])), "lap yok")

    def test_face_model_failure(self) -> None:
        class Bad(ScriptedFaceDetector):
            def load(self):  # type: ignore[no-untyped-def]
                raise FaceModelLoadError("yunet yok")

        self.assert_failure_before_camera(Parts(face=Bad([[]])), "yunet yok")

    def test_display_error_returns_1_and_releases_the_camera(self) -> None:
        parts = Parts()
        parts.window = FakeWindow(raise_on_call={2: DisplayError("pencere hatası")})
        factory, capture = camera_factory()
        code, _, err = call(CAM_ARGS, capture_factory=factory, **parts.kwargs())
        self.assertEqual(code, 1)
        self.assertIn("pencere hatası", err)
        self.assertGreaterEqual(capture.release_calls, 1)

    def test_inference_error_in_the_face_model_returns_1_and_closes_everything(self) -> None:
        from isg_vision.face_detection.errors import FaceInferenceError

        class Bad(ScriptedFaceDetector):
            def detect(self, image, **kw):  # type: ignore[no-untyped-def]
                raise FaceInferenceError("yunet çıkarımı bozuldu")

        parts = Parts(face=Bad([[F]]))
        factory, capture = camera_factory()
        code, _, err = call(CAM_ARGS, capture_factory=factory, **parts.kwargs())
        self.assertEqual(code, 1)
        self.assertIn("yunet çıkarımı bozuldu", err)
        self.assertGreaterEqual(capture.release_calls, 1)
        self.assertEqual((parts.yolo.close_calls, parts.face.close_calls, parts.window.close_calls), (1, 1, 1))


class OpenCvThreadsTest(unittest.TestCase):
    def run_with(self, *extra: str, setting: ThreadSetting | None = None):  # type: ignore[no-untyped-def]
        parts = Parts(threads=setting)
        factory, _ = camera_factory()
        code, out, _ = call(CAM_ARGS + ["--no-display", "--max-frames", "2", *extra], capture_factory=factory, **parts.kwargs())
        self.assertEqual(code, 0)
        return parts, out

    def test_default_requests_keep_and_warns_when_opencv_is_single_threaded(self) -> None:
        parts, out = self.run_with(setting=ThreadSetting("keep", None, 1, 1))
        self.assertEqual(parts.thread_requests, ["keep"])
        self.assertIn("getNumThreads()=1", out)
        self.assertIn("--opencv-threads restore", out)
        self.assertIn("Değiştirmedim", out)

    def test_keep_with_multithreaded_opencv_is_silent_about_slowdown(self) -> None:
        _, out = self.run_with(setting=ThreadSetting("keep", None, 10, 10))
        self.assertIn("değiştirilmedi", out)
        self.assertNotIn("[UYARI] OpenCV", out)

    def test_restore_and_number_are_passed_through_and_reported(self) -> None:
        parts, out = self.run_with("--opencv-threads", "restore", setting=ThreadSetting("restore", -1, 1, 10))
        self.assertEqual(parts.thread_requests, ["restore"])
        self.assertIn("cv2.setNumThreads(-1)", out)
        parts, _ = self.run_with("--opencv-threads", "4", setting=ThreadSetting("4", 4, 1, 10))
        self.assertEqual(parts.thread_requests, ["4"])

    def test_thread_setting_is_applied_after_all_models_are_loaded(self) -> None:
        parts, _ = self.run_with("--opencv-threads", "restore", setting=ThreadSetting("restore", -1, 1, 10))
        self.assertEqual(parts.events[:4], ["load:yolo", "load:tracker", "load:face", "threads:restore"], "ultralytics yüklendikten SONRA uygulanmalı")


class SideEffectTest(unittest.TestCase):
    def test_no_files_are_written_and_no_network_is_used(self) -> None:
        import socket

        parts = Parts()
        factory, _ = camera_factory()
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with mock.patch.object(socket.socket, "connect", side_effect=AssertionError("ağ kullanıldı")), \
                        mock.patch.object(cv2, "imwrite", side_effect=AssertionError("görüntü yazıldı")), \
                        mock.patch.object(cv2, "VideoWriter", side_effect=AssertionError("video yazıldı")):
                    code, _, _ = call(CAM_ARGS + ["--max-frames", "4"], capture_factory=factory, **parts.kwargs())
                self.assertEqual(code, 0)
                self.assertEqual(os.listdir(tmp), [], "çalışma dizinine hiçbir dosya yazılmamalı")
            finally:
                os.chdir(cwd)


class ScriptTest(unittest.TestCase):
    def run_script(self, *args: str) -> subprocess.CompletedProcess:
        env = dict(os.environ, YOLO_OFFLINE="1")
        return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=120, env=env)

    def test_help(self) -> None:
        proc = self.run_script("--help")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--opencv-threads", proc.stdout)

    def test_invalid_argument_exits_2(self) -> None:
        self.assertEqual(self.run_script("--min-score", "9").returncode, 2)

    def test_missing_yolo_weights_fail_cleanly_before_the_camera(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc = self.run_script("--source", "camera", "--model", str(Path(tmp) / "yok.pt"), "--no-display", "--max-frames", "1")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("bulunamadı", proc.stderr)

    def test_missing_face_model_fails_cleanly_before_the_camera(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc = self.run_script("--source", "camera", "--face-model", str(Path(tmp) / "yok.onnx"), "--no-display", "--max-frames", "1")
        self.assertEqual(proc.returncode, 1)
        self.assertTrue("bulunamadı" in proc.stderr or "ağırlık" in proc.stderr, proc.stderr)


if __name__ == "__main__":
    unittest.main()
