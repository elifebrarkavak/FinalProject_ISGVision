"""CPU/MPS karşılaştırma altyapısı testleri (sahte tespitçiyle; gerçek model ve gerçek ölçüm değil)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from camera_test_support import write_synthetic_video  # noqa: E402

from isg_vision.human_detection.benchmark import (  # noqa: E402
    SYNTHETIC_DISCLAIMER,
    make_synthetic_video,
    measure_capture,
    report_to_dict,
    report_to_markdown,
    run_benchmark,
)
from isg_vision.human_detection.config import DetectionConfig, ResolvedDevice  # noqa: E402
from isg_vision.human_detection.errors import DeviceUnavailableError  # noqa: E402
from person_detection_test_support import FakeDetector  # noqa: E402

SCRIPT = ROOT / "scripts" / "benchmark_person_detection.py"


class BenchmarkTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.video = write_synthetic_video(Path(self._tmp.name) / "v.mp4", n_frames=12, fps=25.0, size=(64, 48))
        self.made: dict[str, FakeDetector] = {}
        self.config = DetectionConfig(model_path="x.pt", conf=0.3, iou=0.6, imgsz=320)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def factory(self, config: DetectionConfig) -> FakeDetector:
        detector = FakeDetector(counts=[1], device=config.device)
        detector.config = config
        self.made[config.device] = detector
        return detector

    @staticmethod
    def check(requested: str) -> ResolvedDevice:
        return ResolvedDevice(requested, requested)

    def run_it(self, devices=("cpu", "mps"), **kw):  # type: ignore[no-untyped-def]
        return run_benchmark(self.video, list(devices), self.config, detector_factory=self.factory, device_check=self.check, **kw)

    def test_each_device_gets_every_pass_over_the_same_video(self) -> None:
        report = self.run_it(passes=3, burn_in=2)
        self.assertEqual([d.requested for d in report.devices], ["cpu", "mps"])
        for entry in report.devices:
            self.assertEqual(len(entry.passes), 3)
            self.assertTrue(all(p.frames_processed == 12 for p in entry.passes))
        self.assertEqual((report.video_size, report.video_frames, report.frames_per_pass), ((64, 48), 12, 12))

    def test_pass_order_rotates_between_devices(self) -> None:
        report = self.run_it(passes=3)
        self.assertEqual(report.pass_order, [["cpu", "mps"], ["mps", "cpu"], ["cpu", "mps"]])

    def test_load_warmup_and_burn_in_are_separate_from_measured_passes(self) -> None:
        report = self.run_it(passes=2, burn_in=3)
        for entry in report.devices:
            self.assertIsNotNone(entry.load_seconds)
            self.assertIsNotNone(entry.warmup_seconds)
            self.assertEqual(entry.burn_in_frames, 3)
            self.assertIsNotNone(entry.burn_in_seconds)
            self.assertEqual(len(entry.passes), 2)  # burn-in, ölçülen geçişlere karışmamalı
            self.assertTrue(all(p.frames_processed == 12 for p in entry.passes))
        for detector in self.made.values():
            self.assertEqual(detector.load_calls, 1, "model cihaz başına bir kez yüklenmeli")
            self.assertEqual(detector.warmup_calls, [(64, 48)], "ısınma gerçek kare boyutunda yapılmalı")
            self.assertEqual(detector.close_calls, 1)
            self.assertEqual(len(detector.detect_calls), 3 + 2 * 12)  # burn-in + 2 geçiş

    def test_zero_burn_in_skips_it(self) -> None:
        report = self.run_it(passes=1, burn_in=0)
        self.assertTrue(all(e.burn_in_frames == 0 and e.burn_in_seconds is None for e in report.devices))

    def test_same_configuration_is_used_for_all_devices(self) -> None:
        self.run_it(passes=1)
        for device, detector in self.made.items():
            cfg = detector.config
            self.assertEqual((cfg.device, cfg.conf, cfg.iou, cfg.imgsz), (device, 0.3, 0.6, 320))

    def test_max_frames_limits_each_pass(self) -> None:
        report = self.run_it(passes=2, burn_in=0, max_frames=5)
        self.assertEqual(report.frames_per_pass, 5)
        for entry in report.devices:
            self.assertTrue(all(p.frames_processed == 5 for p in entry.passes))

    def test_capture_only_baseline_is_measured_without_model(self) -> None:
        report = self.run_it(passes=2)
        self.assertEqual(len(report.capture), 2)
        for c in report.capture:
            self.assertEqual(c.frames, 12)
            self.assertGreater(c.fps, 0.0)
        self.assertEqual([c.frames for c in measure_capture(self.video, 1, max_frames=4)], [4])

    def test_unavailable_device_is_skipped_with_reason_not_silently_replaced(self) -> None:
        def check(requested: str) -> ResolvedDevice:
            if requested == "mps":
                raise DeviceUnavailableError("MPS istendi ancak kullanılamıyor: test")
            return ResolvedDevice(requested, requested)

        report = run_benchmark(self.video, ["cpu", "mps"], self.config, passes=2, burn_in=0, detector_factory=self.factory, device_check=check)
        cpu, mps = report.devices
        self.assertEqual(len(cpu.passes), 2)
        self.assertIn("MPS istendi", mps.skipped_reason or "")
        self.assertEqual(mps.passes, [])
        self.assertNotIn("mps", self.made, "kullanılamayan cihaz için tespitçi oluşturulmamalı")
        self.assertEqual(report.pass_order, [["cpu"], ["cpu"]])
        self.assertIn("atlandı", report_to_markdown(report))

    def test_model_load_failure_stops_the_benchmark_instead_of_skipping(self) -> None:
        from isg_vision.human_detection.errors import ModelLoadError

        broken = FakeDetector(load_error=ModelLoadError("ağırlık yok"))
        with self.assertRaises(ModelLoadError):
            run_benchmark(self.video, ["cpu"], self.config, passes=1, detector_factory=lambda cfg: broken, device_check=self.check)

    def test_invalid_arguments(self) -> None:
        for kw in ({"passes": 0}, {"passes": True}, {"burn_in": -1}, {"burn_in": 1.5}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.run_it(**kw)
        with self.assertRaises(ValueError):
            run_benchmark(self.video, [], self.config, detector_factory=self.factory, device_check=self.check)

    def test_report_is_json_serializable_and_contains_only_numbers(self) -> None:
        report = self.run_it(passes=2, burn_in=1)
        text = json.dumps(report_to_dict(report), ensure_ascii=False)
        data = json.loads(text)
        self.assertEqual(len(data["devices"]), 2)
        self.assertEqual(len(data["devices"][0]["passes"]), 2)
        self.assertIn("median_ms", data["devices"][0]["passes"][0]["inference_ms"])
        self.assertEqual(len(data["capture_only"]), 2)
        self.assertNotIn("image", text.lower().replace("imgsz", ""))

    def test_markdown_lists_each_device_and_pass(self) -> None:
        md = report_to_markdown(self.run_it(passes=2))
        for token in ("| cpu |", "| mps |", "Uçtan uca FPS", "Yakalama", "imgsz 320"):
            self.assertIn(token, md)


class SyntheticVideoTest(unittest.TestCase):
    def test_make_synthetic_video_is_readable_deterministic_and_correct_size(self) -> None:
        import importlib.util

        spec = importlib.util.find_spec("ultralytics")
        if spec is None or not (Path(spec.origin).parent / "assets" / "bus.jpg").is_file():  # type: ignore[arg-type]
            self.skipTest("ultralytics örnek görüntüleri yok")
        with tempfile.TemporaryDirectory() as tmp:
            first = make_synthetic_video(Path(tmp) / "a.mp4", frames=4)
            second = make_synthetic_video(Path(tmp) / "b.mp4", frames=4)
            frames = []
            for path in (first, second):
                cap = cv2.VideoCapture(str(path))
                ok, frame = cap.read()
                self.assertTrue(ok)
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
                cap.release()
                frames.append(frame)
            self.assertEqual(frames[0].shape, (720, 1280, 3))
            self.assertTrue((frames[0] == frames[1]).all(), "aynı parametrelerle aynı içerik üretilmeli")
        self.assertIn("SENTETİK", SYNTHETIC_DISCLAIMER)


class ScriptTest(unittest.TestCase):
    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=300)

    def test_help(self) -> None:
        proc = self.run_script("--help")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--devices", proc.stdout)
        self.assertIn("SENTETİK", proc.stdout)

    def test_requires_a_video_or_sample_flag(self) -> None:
        proc = self.run_script()
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--video-path", proc.stderr)

    def test_invalid_threshold_is_a_usage_error(self) -> None:
        proc = self.run_script("--video-path", "x.mp4", "--conf", "5")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("conf", proc.stderr)

    def test_missing_weights_is_a_hard_error_not_a_skipped_device(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = write_synthetic_video(Path(tmp) / "v.mp4", n_frames=3, size=(64, 48))
            proc = self.run_script("--video-path", str(video), "--model", str(Path(tmp) / "yok.pt"), "--devices", "cpu", "--output-dir", tmp)
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            self.assertIn("download_person_model.py", proc.stderr)
            self.assertEqual(list(Path(tmp).glob("*.json")), [], "hatalı çalıştırmada sonuç dosyası yazılmamalı")

    def test_missing_video_exits_1(self) -> None:
        proc = self.run_script("--video-path", "/olmayan/video.mp4", "--devices", "cpu")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
