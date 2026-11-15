"""Yüz tespiti benchmark yardımcılarının testleri (sahte tespitçi/pencere; gerçek model ve GUI yok; ölçüm değerleri uydurulmaz)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from face_detection_test_support import FakeFaceDetector  # noqa: E402

from isg_vision.face_detection import benchmark as bm  # noqa: E402
from isg_vision.face_detection.config import FaceDetectionConfig  # noqa: E402
from isg_vision.face_detection.errors import FaceDetectionError  # noqa: E402
from isg_vision.face_detection.results import FaceDetection, FaceDetectionResult  # noqa: E402


class PatchDetector:
    """Montaj için tek bir 'net yüz' bulan sahte tespitçi (kırpıntı üretimini sınamak için)."""

    def detect(self, image):  # type: ignore[no-untyped-def]
        h, w = image.shape[:2]
        face = FaceDetection(0.95, (w * 0.3, h * 0.2, w * 0.6, h * 0.6), image_size=(w, h), face_index=0)
        low = FaceDetection(0.5, (0.0, 0.0, 10.0, 10.0), image_size=(w, h), face_index=1)
        return FaceDetectionResult((face, low), (w, h))


def sample() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 255, (200, 300, 3), dtype=np.uint8)


class MontageTest(unittest.TestCase):
    def patches(self) -> list[np.ndarray]:
        return bm.extract_face_patches(PatchDetector(), sample())

    def test_extract_keeps_only_confident_faces_with_context(self) -> None:
        patches = self.patches()
        self.assertEqual(len(patches), 1)  # 0.5 güvenli yüz atlandı
        h, w = patches[0].shape[:2]
        self.assertGreater(h, int(200 * 0.4))  # kutudan büyük (bağlam payı)
        self.assertGreater(w, int(300 * 0.3))

    def test_extract_does_not_modify_the_source_image(self) -> None:
        image = sample()
        before = image.copy()
        bm.extract_face_patches(PatchDetector(), image)
        np.testing.assert_array_equal(image, before)

    def test_zero_faces_gives_a_plain_canvas(self) -> None:
        canvas = bm.make_face_montage([], 0, 96)
        self.assertEqual(canvas.shape, (720, 1280, 3))
        self.assertTrue((canvas == bm.MONTAGE_BACKGROUND).all())

    def test_montage_places_the_requested_number_of_cells(self) -> None:
        patches = self.patches()
        for n in (1, 2, 4, 8, 16):
            with self.subTest(n=n):
                canvas = bm.make_face_montage(patches, n, 96)
                self.assertEqual(canvas.shape, (720, 1280, 3))
                self.assertEqual(canvas.dtype, np.uint8)
                self.assertFalse((canvas == bm.MONTAGE_BACKGROUND).all())

    def test_more_faces_cover_more_pixels(self) -> None:
        patches = self.patches()
        covered = [int((bm.make_face_montage(patches, n, 96) != bm.MONTAGE_BACKGROUND).any(axis=2).sum()) for n in (1, 4, 16)]
        self.assertLess(covered[0], covered[1])
        self.assertLess(covered[1], covered[2])

    def test_overflowing_canvas_is_rejected_not_silently_truncated(self) -> None:
        with self.assertRaises(ValueError):
            bm.make_face_montage(self.patches(), 500, 96)

    def test_invalid_arguments(self) -> None:
        patches = self.patches()
        for n, px in ((-1, 96), (2, 4), (1, 0)):
            with self.subTest(n=n, px=px), self.assertRaises(ValueError):
                bm.make_face_montage(patches, n, px)
        with self.assertRaises(ValueError):
            bm.make_face_montage([], 2, 96)

    def test_custom_canvas_size(self) -> None:
        self.assertEqual(bm.make_face_montage(self.patches(), 2, 64, (640, 360)).shape, (360, 640, 3))

    def test_montage_is_deterministic(self) -> None:
        patches = self.patches()
        np.testing.assert_array_equal(bm.make_face_montage(patches, 4, 96), bm.make_face_montage(patches, 4, 96))


class MeasurementTest(unittest.TestCase):
    def test_measure_inference_counts_repeats_and_excludes_warmup(self) -> None:
        detector = FakeFaceDetector(counts=[2], inference_s=0.005)
        m = bm.measure_inference(detector, np.zeros((48, 64, 3), np.uint8), repeats=10, warmup=3, label="x", expected_faces=2)
        self.assertEqual(len(detector.detect_calls), 13)
        self.assertEqual((m.stats.count, m.detected_min, m.detected_max, m.expected_faces, m.image_size, m.label), (10, 2, 2, 2, (64, 48), "x"))
        self.assertAlmostEqual(m.stats.median_ms, 5.0)
        self.assertAlmostEqual(m.fps_inference_only, 200.0)

    def test_measure_inference_reports_count_spread(self) -> None:
        detector = FakeFaceDetector(counts=[1, 3, 2])
        m = bm.measure_inference(detector, np.zeros((48, 64, 3), np.uint8), repeats=6, warmup=0)
        self.assertEqual((m.detected_min, m.detected_max), (1, 3))

    def test_invalid_repeat_arguments(self) -> None:
        for repeats, warmup in ((0, 0), (-1, 0), (1, -1)):
            with self.subTest(repeats=repeats, warmup=warmup), self.assertRaises(ValueError):
                bm.measure_inference(FakeFaceDetector(), np.zeros((8, 8, 3), np.uint8), repeats=repeats, warmup=warmup)

    def test_model_load_measurement_uses_a_fresh_detector_each_time(self) -> None:
        created: list[FakeFaceDetector] = []

        def factory(cfg):  # type: ignore[no-untyped-def]
            d = FakeFaceDetector()
            created.append(d)
            return d

        report = bm.measure_model_load(FaceDetectionConfig(), 3, detector_factory=factory)
        self.assertEqual((len(created), report["loads"], len(report["load_seconds"]), len(report["warmup_seconds"])), (3, 3, 3, 3))
        self.assertTrue(all(d.load_calls == 1 and d.close_calls == 1 and d.warmup_calls == [(1280, 720)] for d in created))
        self.assertIn("SHA-256", report["note"])

    def test_gui_cost_measurement_calls_show_each_time(self) -> None:
        class Window:
            def __init__(self) -> None:
                self.shown = 0

            def show(self, image, delay_ms=1):  # type: ignore[no-untyped-def]
                self.shown += 1
                return -1

        window = Window()
        result = bm.measure_gui_cost(window, np.zeros((8, 8, 3), np.uint8), 25)
        self.assertEqual((window.shown, result["runs"]), (25, 25))
        self.assertGreaterEqual(result["median_ms"], 0.0)

    def test_peak_rss_is_a_positive_number(self) -> None:
        self.assertGreater(bm.peak_rss_mb(), 1.0)

    def test_environment_info_reports_opencv_and_import_state(self) -> None:
        info = bm.environment_info()
        self.assertEqual(info["opencv"], cv2.__version__)
        self.assertIn("ultralytics_imported", info)
        self.assertIn("opencv_threads", info)

    def test_sample_image_lookup_never_imports_ultralytics(self) -> None:
        import importlib.util
        import subprocess

        if importlib.util.find_spec("ultralytics") is None:
            self.skipTest("ultralytics kurulu değil")
        code = ("import sys; sys.path.insert(0, 'src'); from isg_vision.face_detection.benchmark import find_sample_images; "
                "r = find_sample_images(); assert 'ultralytics' not in sys.modules and 'torch' not in sys.modules; print(sorted(r))")
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("zidane", proc.stdout)

    def test_missing_ultralytics_gives_a_clear_error(self) -> None:
        from unittest import mock

        with mock.patch.object(bm.importlib.util, "find_spec", return_value=None), self.assertRaises(FaceDetectionError):
            bm.find_sample_images()


class ReportTest(unittest.TestCase):
    def sample_images(self) -> dict[str, np.ndarray]:
        return {"zidane": sample(), "bus": sample()}

    def run_report(self, **kw):  # type: ignore[no-untyped-def]
        class Detector(PatchDetector):
            def __init__(self, cfg=None) -> None:
                self.closed = 0
                self.loaded = 0
                self.load_seconds = 0.001

            def load(self) -> None:
                self.loaded += 1

            def warmup(self, size, runs=1) -> float:  # type: ignore[no-untyped-def]
                return 0.002

            def close(self) -> None:
                self.closed += 1

            def detect(self, image):  # type: ignore[no-untyped-def]
                r = super().detect(image)
                return FaceDetectionResult(r.faces, r.image_size, None, None, 4.0, "cpu", 0, 0, r.image_size)

        params = dict(repeats=3, face_counts=(0, 2), face_px=64, sizes=((640, 360), (1280, 720)), detect_widths=(640,), loads=2,
                      detector_factory=Detector, sample_images=self.sample_images(), log=lambda s: None)
        params.update(kw)
        return bm.run_benchmark(FaceDetectionConfig(), **params)

    def test_report_has_every_section_and_is_json_serializable(self) -> None:
        report = self.run_report()
        for key in ("environment", "disclaimer", "config", "model_load", "warmup_seconds_1280x720", "natural_images", "by_face_count",
                    "by_resolution", "by_detect_width", "gui", "peak_rss_mb", "tables_markdown"):
            self.assertIn(key, report)
        json.dumps(report, ensure_ascii=False)
        self.assertEqual([r["expected_faces"] for r in report["by_face_count"]["rows"]], [0, 2])
        self.assertEqual(len(report["by_resolution"]), 2)
        self.assertIn("SENTETİK", report["disclaimer"])

    def test_gui_is_not_measured_unless_requested_and_never_fabricated(self) -> None:
        self.assertIsNone(self.run_report()["gui"])
        self.assertIsNone(self.run_report(gui=True, window_factory=None)["gui"])

    def test_gui_cost_is_measured_with_the_given_window(self) -> None:
        class Window:
            opened = closed = shown = 0

            def open(self) -> None:
                Window.opened += 1

            def close(self) -> None:
                Window.closed += 1

            def show(self, image, delay_ms=1):  # type: ignore[no-untyped-def]
                Window.shown += 1
                return -1

        report = self.run_report(gui=True, gui_runs=5, window_factory=Window)
        self.assertEqual((Window.opened, Window.closed, Window.shown), (1, 1, 5))
        self.assertEqual(report["gui"]["show"]["runs"], 5)
        self.assertGreaterEqual(report["gui"]["draw_median_ms"], 0.0)

    def test_window_is_closed_even_if_measurement_fails(self) -> None:
        class Window:
            closed = 0

            def open(self) -> None:
                pass

            def close(self) -> None:
                Window.closed += 1

            def show(self, image, delay_ms=1):  # type: ignore[no-untyped-def]
                raise RuntimeError("gösterilemedi")

        with self.assertRaises(RuntimeError):
            self.run_report(gui=True, gui_runs=2, window_factory=Window)
        self.assertEqual(Window.closed, 1)

    def test_detect_width_rows_skip_widths_not_smaller_than_the_frame(self) -> None:
        report = self.run_report(detect_widths=(640, 1280, 4000))
        labels = [r["label"] for r in report["by_detect_width"]]
        self.assertTrue(any("1280x720, tespit genişliği: 640" in label for label in labels))
        self.assertFalse(any("1280x720, tespit genişliği: 1280" in label or label.endswith(": 4000") for label in labels))

    def test_markdown_tables_are_well_formed(self) -> None:
        md = self.run_report()["tables_markdown"]
        self.assertIn("| Ölçüm |", md)
        for line in md.splitlines():
            if line.startswith("|"):
                self.assertEqual(line.count("|"), 9, line)

    def test_write_json_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = bm.write_json(Path(tmp) / "a" / "b" / "r.json", {"x": 1, "ş": "ü"})
            self.assertEqual(json.loads(out.read_text(encoding="utf-8")), {"x": 1, "ş": "ü"})


if __name__ == "__main__":
    unittest.main()
