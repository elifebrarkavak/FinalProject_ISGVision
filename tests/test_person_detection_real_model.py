"""GERÇEK YOLO modeliyle entegrasyon testleri (mock YOLO yok).

Mock testleri gerçek çıkarımın çalıştığını kanıtlamaz; bu dosya ayrıca gerçek ``yolo11n.pt`` ağırlığıyla insan tespitini doğrular.

* Ağırlık yoksa (``scripts/download_person_model.py`` çalıştırılmadıysa) testler ATLANIR; hiçbir şey indirilmez, internet gerekmez.
* Görüntüler, yüklü ``ultralytics`` paketinin içinde gelen örnek dosyalardır (``bus.jpg``: 4 yaya + 1 otobüs, ``zidane.jpg``: 2 kişi);
  dışarıdan görüntü indirilmez. Kesitler (crop) bu görüntülerden üretilir.
* Beklenen sayılar, SHA-256'sı sabitlenmiş ağırlık ve ``ultralytics==8.4.210`` ile ölçülmüştür. Bu sayılar küçük bir örnek kümesindeki
  davranıştır; **model doğruluğu (precision/recall/mAP) iddiası değildir** (bağımsız etiketli test kümesi yoktur).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isg_vision.human_detection.config import DEFAULT_MODEL_PATH, DetectionConfig, torch_mps_status  # noqa: E402
from isg_vision.human_detection.person_detector import PersonDetector  # noqa: E402
from isg_vision.human_detection.visualization import PersonOverlayInfo, draw_person_detections  # noqa: E402

try:  # örnek görüntüler ultralytics paketiyle gelir; paketi import etmeden konumu bulmak için find_spec kullanılır
    import importlib.util

    _spec = importlib.util.find_spec("ultralytics")
    ASSETS = Path(_spec.origin).parent / "assets" if _spec and _spec.origin else None
except Exception:  # pragma: no cover
    ASSETS = None

WEIGHTS_PRESENT = DEFAULT_MODEL_PATH.is_file()
SAMPLES_PRESENT = bool(ASSETS and (ASSETS / "bus.jpg").is_file() and (ASSETS / "zidane.jpg").is_file())
SKIP_REASON = "Gerçek model testi: models/human_detection/yolo11n.pt (scripts/download_person_model.py) ve ultralytics örnek görüntüleri gerekir"


@unittest.skipUnless(WEIGHTS_PRESENT and SAMPLES_PRESENT, SKIP_REASON)
class RealModelCpuTest(unittest.TestCase):
    detector: PersonDetector
    bus: np.ndarray
    zidane: np.ndarray

    @classmethod
    def setUpClass(cls) -> None:
        assert ASSETS is not None
        cls.bus = cv2.imread(str(ASSETS / "bus.jpg"))
        cls.zidane = cv2.imread(str(ASSETS / "zidane.jpg"))
        cls.detector = PersonDetector(DetectionConfig(device="cpu"))
        cls.detector.load()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.detector.close()

    # ------------------------------------------------------------------ model / metadata
    def test_model_loaded_once_and_person_id_comes_from_metadata(self) -> None:
        d = self.detector
        self.assertTrue(d.is_loaded)
        self.assertEqual(d.load_count, 1)
        self.assertEqual(d.device, "cpu")
        self.assertEqual(d.person_class_id, 0)  # COCO'da 0; ama kimlik metadata'dan okunur
        self.assertEqual(d._model.names[d.person_class_id], "person")
        self.assertEqual(len(d._model.names), 80)
        self.assertIsNotNone(d.load_seconds)

    def test_repeated_detections_do_not_reload_the_model(self) -> None:
        for _ in range(3):
            self.detector.detect(self.zidane)
        self.assertEqual(self.detector.load_count, 1)

    # ------------------------------------------------------------------ Test A: insan yok
    def test_a_images_without_people_give_empty_result(self) -> None:
        samples = {
            "otobüs/bina kesiti (gerçek görüntü, insan yok)": np.ascontiguousarray(self.bus[0:380, :]),
            "düz gri": np.full((480, 640, 3), 128, np.uint8),
            "rastgele gürültü": np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8),
        }
        for label, image in samples.items():
            with self.subTest(label):
                result = self.detector.detect(image)
                self.assertEqual((result.person_count, result.detections), (0, ()))
                self.assertGreater(result.inference_ms, 0.0)
                out = draw_person_detections(image, result, PersonOverlayInfo("VIDEO test", 0.0, result.inference_ms, (image.shape[1], image.shape[0]), "cpu"))
                self.assertEqual(out.shape, image.shape)

    # ------------------------------------------------------------------ Test B: tek insan
    def test_b_single_person_crops(self) -> None:
        crops = {
            "bus: sağdaki yaya": np.ascontiguousarray(self.bus[380:1080, 640:810]),
            "bus: siyah ceketli yaya": np.ascontiguousarray(self.bus[400:870, 215:350]),
            "zidane: soldaki kişi": np.ascontiguousarray(self.zidane[0:720, 100:700]),
            "zidane: sağdaki kişi": np.ascontiguousarray(self.zidane[0:720, 700:1280]),
        }
        for label, image in crops.items():
            with self.subTest(label):
                result = self.detector.detect(image, frame_index=3, timestamp=1.5)
                self.assertEqual(result.person_count, 1)
                d = result.detections[0]
                self.assertGreaterEqual(d.confidence, 0.5)
                self.assertEqual((d.class_name, d.frame_index, d.timestamp), ("person", 3, 1.5))
                self._assert_valid_box(d.bbox_xyxy, image)

    # ------------------------------------------------------------------ Test C: çoklu insan
    def test_c_multiple_people(self) -> None:
        for label, image, expected in (("bus.jpg", self.bus, 4), ("zidane.jpg", self.zidane, 2)):
            with self.subTest(label):
                result = self.detector.detect(image)
                self.assertEqual(result.person_count, expected)
                self.assertEqual(len({d.bbox_xyxy for d in result.detections}), expected, "kutular birbirinden bağımsız olmalı")
                for d in result.detections:
                    self._assert_valid_box(d.bbox_xyxy, image)
                    self.assertEqual(d.class_name, "person")
                confs = [d.confidence for d in result.detections]
                self.assertEqual(confs, sorted(confs, reverse=True))

    def test_c_people_count_is_shown_on_the_overlay(self) -> None:
        result = self.detector.detect(self.bus)
        texts: list[str] = []
        real = cv2.putText

        def spy(img, text, *a, **k):  # type: ignore[no-untyped-def]
            texts.append(text)
            return real(img, text, *a, **k)

        with mock.patch.object(cv2, "putText", side_effect=spy):
            draw_person_detections(self.bus, result, PersonOverlayInfo("VIDEO bus.jpg", 20.0, result.inference_ms, (810, 1080), "cpu"))
        self.assertIn("PERSONS: 4", texts)
        self.assertEqual(sum(t.startswith("PERSON ") for t in texts), 4)

    def test_only_person_class_is_returned_even_though_model_sees_a_bus(self) -> None:
        raw = self.detector._model.predict(source=self.bus, verbose=False, device="cpu")[0]
        raw_classes = {raw.names[int(c)] for c in raw.boxes.cls.cpu().numpy()}
        self.assertIn("bus", raw_classes, "kontrol: süzülmemiş modelde otobüs görünmeli")
        self.assertIn("person", raw_classes)
        result = self.detector.detect(self.bus)
        self.assertEqual({d.class_name for d in result.detections}, {"person"})
        self.assertEqual(result.person_count, 4)

    # ------------------------------------------------------------------ güven eşiği
    def test_confidence_threshold_changes_the_detections(self) -> None:
        counts = {}
        for conf in (0.25, 0.5, 0.7, 0.9):
            detector = PersonDetector(DetectionConfig(device="cpu", conf=conf))
            counts[conf] = detector.detect(self.bus).person_count
        self.assertEqual(counts, {0.25: 4, 0.5: 4, 0.7: 3, 0.9: 0})
        values = list(counts.values())
        self.assertEqual(values, sorted(values, reverse=True), "eşik yükselince tespit sayısı artmamalı")

    # ------------------------------------------------------------------ genel
    def test_input_image_not_modified(self) -> None:
        before = self.bus.copy()
        self.detector.detect(self.bus)
        self.assertTrue(np.array_equal(self.bus, before))

    def test_inference_time_is_measured_and_plausible(self) -> None:
        result = self.detector.detect(self.zidane)
        self.assertTrue(0.5 < result.inference_ms < 5000, result.inference_ms)

    def test_warmup_is_measured_separately(self) -> None:
        detector = PersonDetector(DetectionConfig(device="cpu"))
        warm = detector.warmup((1280, 720))
        self.assertGreater(warm, 0.0)
        self.assertEqual(detector.warmup_seconds, warm)
        self.assertEqual(detector.load_count, 1)

    def _assert_valid_box(self, box: tuple[float, float, float, float], image: np.ndarray) -> None:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        self.assertTrue(0.0 <= x1 < x2 <= w and 0.0 <= y1 < y2 <= h, f"kutu görüntü sınırları dışında: {box} / {w}x{h}")
        self.assertGreater((x2 - x1) * (y2 - y1), 100.0)


@unittest.skipUnless(WEIGHTS_PRESENT and SAMPLES_PRESENT, SKIP_REASON)
@unittest.skipUnless(torch_mps_status()[0], "MPS bu makinede kullanılamıyor")
class RealModelMpsTest(unittest.TestCase):
    def test_mps_results_match_cpu(self) -> None:
        assert ASSETS is not None
        bus = cv2.imread(str(ASSETS / "bus.jpg"))
        cpu = PersonDetector(DetectionConfig(device="cpu"))
        mps = PersonDetector(DetectionConfig(device="mps"))
        mps.warmup((810, 1080), runs=2)
        cpu_result, mps_result = cpu.detect(bus), mps.detect(bus)
        self.assertEqual(mps.device, "mps")
        self.assertEqual(mps_result.device, "mps")
        self.assertEqual(mps_result.person_count, cpu_result.person_count)
        for a, b in zip(cpu_result.detections, mps_result.detections):
            self.assertLess(max(abs(p - q) for p, q in zip(a.bbox_xyxy, b.bbox_xyxy)), 2.0, "CPU/MPS kutuları 2 pikselden fazla ayrışıyor")
            self.assertLess(abs(a.confidence - b.confidence), 0.02)


@unittest.skipUnless(WEIGHTS_PRESENT and SAMPLES_PRESENT, SKIP_REASON)
class RealModelPrivacyTest(unittest.TestCase):
    def test_real_inference_makes_no_network_calls_and_writes_no_files(self) -> None:
        """Gerçek model yükleme + çıkarım sırasında ağ işlemi olmaz ve diskte dosya oluşmaz (alt süreçte, ağ sahtelenerek denetlenir)."""
        code = textwrap.dedent(
            """
            import os, socket, sys
            sys.path.insert(0, {src!r})
            def boom(*a, **k):
                raise AssertionError("AĞ ÇAĞRISI YAPILDI")
            socket.getaddrinfo = boom
            socket.create_connection = boom
            socket.socket.connect = boom
            import cv2, importlib.util
            from isg_vision.human_detection import DetectionConfig, PersonDetector
            assets = os.path.join(os.path.dirname(importlib.util.find_spec("ultralytics").origin), "assets")
            detector = PersonDetector(DetectionConfig(device="cpu"))
            result = detector.detect(cv2.imread(os.path.join(assets, "bus.jpg")))
            print("PERSONS=%d" % result.person_count, "OFFLINE=%s" % os.environ.get("YOLO_OFFLINE"))
            """
        ).format(src=str(ROOT / "src"))
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:  # çalışma dizini boş bir klasör: 'runs/' gibi çıktılar burada görünür
            proc = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=300, cwd=tmp)
            leftovers = sorted(p.name for p in Path(tmp).iterdir())
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("PERSONS=4", proc.stdout)
        self.assertIn("OFFLINE=1", proc.stdout)
        self.assertEqual(leftovers, [], f"çıkarım diske dosya yazdı: {leftovers}")
        self.assertFalse((ROOT / "runs").exists(), "proje kökünde 'runs/' oluşmamalı")


if __name__ == "__main__":
    unittest.main()
