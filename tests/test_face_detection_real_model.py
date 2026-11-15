"""GERÇEK YuNet modeliyle yüz tespiti testleri (mock DEĞİL): gerçek model yüklenir, gerçek görüntülerde gerçek çıkarım yapılır.

Kamera, internet ve izin gerekmez. Girdi görüntüleri Ultralytics paketinin içindeki iki örnek fotoğraftır (``zidane.jpg``: iki net yüz +
çok bulanık bir arka plan yüzü; ``bus.jpg``: iki net yüz) ve onlardan türetilen kırpma/ölçekleme/kaydırma/montajlar; dosyalar pakete
dâhildir (indirme yok), 4A/4B testlerinde de kullanılmıştır.

**Kanıt sınırı (rapora da yazılmıştır):** referans yüz kutuları (``ANCELOTTI``, ``ZIDANE``, ``BUS_FACES``) bağımsız bir etiketleme DEĞİLDİR;
modelin ürettiği kutular ``zidane.jpg``/``bus.jpg`` üzerine çizilip GÖZLE doğrulanmıştır (kutular yüzlerle örtüşüyor). Bu testler yüz
tespitinin bu görüntülerde çalıştığını, koordinatların geometrik olarak tutarlı olduğunu ve sınır durumlarında çökmediğini gösterir;
doğruluk (precision/recall/AP), profil/döndürülmüş/maskeli yüzler, farklı ten rengi/ışık koşulları ve gerçek kamera performansı hakkında
HİÇBİR iddia taşımaz.

Model dosyası (``models/face_detection/*.onnx``) veya örnek görüntüler yoksa testler açık nedenle ATLANIR (sahte başarı yok).
"""

from __future__ import annotations

import contextlib
import io
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from camera_test_support import FakeWindow  # noqa: E402

from isg_vision.camera import FrameSource, SourceConfig  # noqa: E402
from isg_vision.camera.display import PreviewWindow  # noqa: E402
from isg_vision.face_detection import cli  # noqa: E402
from isg_vision.face_detection.benchmark import find_sample_images  # noqa: E402
from isg_vision.face_detection.config import DEFAULT_MODEL_NAME, DEFAULT_MODEL_PATH, FaceDetectionConfig  # noqa: E402
from isg_vision.face_detection.detector import FaceDetector  # noqa: E402
from isg_vision.face_detection.errors import FaceInvalidFrameError, FaceModelLoadError  # noqa: E402
from isg_vision.face_detection.pipeline import REASON_END_OF_STREAM, REASON_MAX_FRAMES, run_face_detection  # noqa: E402
from isg_vision.face_detection.visualization import WINDOW_NAME  # noqa: E402
from isg_vision.face_detection.weights import OFFICIAL_MODELS  # noqa: E402
from isg_vision.human_detection.benchmark import make_synthetic_video  # noqa: E402

MODEL_PRESENT = DEFAULT_MODEL_PATH.is_file()
try:
    SAMPLES = find_sample_images()
except Exception:  # noqa: BLE001 (ultralytics yok / örnekler yok)
    SAMPLES = {}
SAMPLES_PRESENT = {"zidane", "bus"} <= set(SAMPLES)
READY = MODEL_PRESENT and SAMPLES_PRESENT
SKIP_REASON = (
    "gerçek yüz modeli veya örnek görüntüler yok (modeli indirin: .venv/bin/python scripts/download_face_model.py; "
    "örnekler ultralytics paketiyle gelir)"
)

# Gözle doğrulanmış referans kutular (bkz. modül açıklaması): (x1, y1, x2, y2), zidane.jpg 1280x720 ve bus.jpg 810x1080
ANCELOTTI = (917.0, 99.0, 1053.0, 273.0)
ZIDANE = (555.0, 261.0, 656.0, 422.0)
BUS_FACES = ((113.0, 416.0, 155.0, 470.0), (268.0, 423.0, 306.0, 473.0))
MIN_IOU = 0.6


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def best_iou(result, box):  # type: ignore[no-untyped-def]
    return max((iou(f.bbox_xyxy, box) for f in result.faces), default=0.0)


def scaled(box: tuple[float, ...], sx: float, sy: float) -> tuple[float, ...]:
    return (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)


_images: dict[str, np.ndarray] = {}


def image(name: str) -> np.ndarray:
    if name not in _images:
        loaded = cv2.imread(str(SAMPLES[name]))
        assert loaded is not None
        _images[name] = loaded
    return _images[name].copy()


def clear(result):  # type: ignore[no-untyped-def]
    """Net (güveni ≥ 0,8) yüzler; bulanık arka plan yüzünü (≈ 0,61) dışarıda bırakır."""
    return [f for f in result.faces if f.confidence >= 0.8]


@unittest.skipUnless(READY, SKIP_REASON)
class RealModelTestCase(unittest.TestCase):
    detector: FaceDetector

    @classmethod
    def setUpClass(cls) -> None:
        cls.detector = FaceDetector()
        cls.detector.load()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.detector.close()


class LoadingTest(RealModelTestCase):
    def test_real_model_loads_once_and_passes_the_integrity_check(self) -> None:
        d = FaceDetector()
        self.assertFalse(d.is_loaded)
        d.load()
        d.load()
        self.assertEqual((d.is_loaded, d.load_count, d.device, d.model_name), (True, 1, "cpu", DEFAULT_MODEL_NAME))
        self.assertGreater(d.load_seconds, 0.0)
        self.assertLess(d.load_seconds, 5.0)
        d.close()

    def test_default_model_file_matches_the_pinned_publisher_values(self) -> None:
        from isg_vision.face_detection.weights import sha256_file

        official = OFFICIAL_MODELS[DEFAULT_MODEL_NAME]
        self.assertEqual(DEFAULT_MODEL_PATH.stat().st_size, official.size_bytes)
        self.assertEqual(sha256_file(DEFAULT_MODEL_PATH), official.sha256)

    def test_tampered_copy_of_the_official_model_is_rejected_before_parsing(self) -> None:
        data = bytearray(DEFAULT_MODEL_PATH.read_bytes())
        data[len(data) // 2] ^= 0xFF
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / DEFAULT_MODEL_NAME
            path.write_bytes(bytes(data))
            with self.assertRaises(FaceModelLoadError) as ctx:
                FaceDetector(FaceDetectionConfig(model_path=path)).load()
        self.assertIn("bütünlük", str(ctx.exception))

    def test_corrupt_onnx_with_a_custom_name_fails_in_opencv_and_is_wrapped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bozuk.onnx"
            path.write_bytes(np.random.default_rng(0).bytes(5000))
            with self.assertRaises(FaceModelLoadError) as ctx:
                FaceDetector(FaceDetectionConfig(model_path=path)).load()
        self.assertIn("yüklenemedi", str(ctx.exception))

    def test_detection_and_loading_use_no_network(self) -> None:
        def no_network(*a: object, **k: object) -> None:
            raise AssertionError("ağ bağlantısı kurulmamalı")

        with mock.patch.object(socket.socket, "connect", no_network), mock.patch.object(socket, "create_connection", no_network):
            d = FaceDetector()
            r = d.detect(image("zidane"))
            d.close()
        self.assertGreaterEqual(r.face_count, 2)


class SingleFaceTest(RealModelTestCase):
    def test_single_face_ancelotti_crop(self) -> None:
        crop = image("zidane")[40:330, 850:1100]  # yalnızca Ancelotti'nin yüzü ve bağlamı
        result = self.detector.detect(crop, frame_index=1, timestamp=0.5)
        self.assertEqual(result.face_count, 1, [(f.bbox_xyxy, f.confidence) for f in result.faces])
        face = result.faces[0]
        expected = (ANCELOTTI[0] - 850, ANCELOTTI[1] - 40, ANCELOTTI[2] - 850, ANCELOTTI[3] - 40)
        self.assertGreaterEqual(iou(face.bbox_xyxy, expected), MIN_IOU)
        self.assertGreaterEqual(face.confidence, 0.8)
        self.assertEqual((face.frame_index, face.timestamp, face.image_size, face.face_index), (1, 0.5, (250, 290), 0))

    def test_single_face_zidane_crop(self) -> None:
        crop = image("zidane")[200:470, 480:720]
        result = self.detector.detect(crop)
        self.assertEqual(result.face_count, 1)
        self.assertGreaterEqual(iou(result.faces[0].bbox_xyxy, (ZIDANE[0] - 480, ZIDANE[1] - 200, ZIDANE[2] - 480, ZIDANE[3] - 200)), MIN_IOU)

    def test_single_face_on_a_large_canvas(self) -> None:
        patch = image("zidane")[40:330, 850:1100]
        canvas = np.full((720, 1280, 3), 114, np.uint8)
        canvas[200:490, 500:750] = patch
        result = self.detector.detect(canvas)
        self.assertEqual(result.face_count, 1)
        expected = (ANCELOTTI[0] - 850 + 500, ANCELOTTI[1] - 40 + 200, ANCELOTTI[2] - 850 + 500, ANCELOTTI[3] - 40 + 200)
        self.assertGreaterEqual(iou(result.faces[0].bbox_xyxy, expected), MIN_IOU)


class MultipleFacesTest(RealModelTestCase):
    def test_zidane_has_two_clear_faces_matching_the_reference_boxes(self) -> None:
        result = self.detector.detect(image("zidane"), frame_index=0)
        self.assertEqual(result.image_size, (1280, 720))
        faces = clear(result)
        self.assertEqual(len(faces), 2, [(f.bbox_xyxy, f.confidence) for f in result.faces])
        for ref in (ANCELOTTI, ZIDANE):
            self.assertGreaterEqual(best_iou(result, ref), MIN_IOU, ref)
        self.assertEqual([f.face_index for f in result.faces], list(range(result.face_count)))
        self.assertEqual([f.confidence for f in result.faces], sorted((f.confidence for f in result.faces), reverse=True))

    def test_bus_has_two_clear_faces_matching_the_reference_boxes(self) -> None:
        result = self.detector.detect(image("bus"))
        self.assertEqual(result.image_size, (810, 1080))
        self.assertEqual(len(clear(result)), 2)
        for ref in BUS_FACES:
            self.assertGreaterEqual(best_iou(result, ref), MIN_IOU, ref)

    def test_faces_are_distinct_boxes_not_duplicates(self) -> None:
        result = self.detector.detect(image("zidane"))
        for i, a in enumerate(result.faces):
            for b in result.faces[i + 1:]:
                self.assertLess(iou(a.bbox_xyxy, b.bbox_xyxy), 0.3)

    def test_montage_of_many_faces_finds_each_face(self) -> None:
        from isg_vision.face_detection.benchmark import extract_face_patches, make_face_montage

        patches = extract_face_patches(self.detector, image("zidane")) + extract_face_patches(self.detector, image("bus"))
        self.assertEqual(len(patches), 4)
        for n in (1, 2, 4, 8, 16):
            with self.subTest(n=n):
                result = self.detector.detect(make_face_montage(patches, n, 96))
                self.assertEqual(result.face_count, n, "sentetik montaj: her yüz tam bir kez bulunmalı")

    def test_confidences_are_valid_and_boxes_inside_the_image(self) -> None:
        for name in ("zidane", "bus"):
            result = self.detector.detect(image(name))
            w, h = result.image_size
            for f in result.faces:
                with self.subTest(image=name):
                    self.assertTrue(0.0 <= f.confidence <= 1.0)
                    self.assertTrue(0.0 <= f.x1 < f.x2 <= w and 0.0 <= f.y1 < f.y2 <= h, f.bbox_xyxy)
                    self.assertEqual(f.image_size, (w, h))


class NoFaceTest(RealModelTestCase):
    def test_frames_without_faces_give_empty_results_without_errors(self) -> None:
        bus, zid = image("bus"), image("zidane")
        cases = {
            "gri düz": np.full((720, 1280, 3), 114, np.uint8),
            "siyah": np.zeros((720, 1280, 3), np.uint8),
            "beyaz": np.full((480, 640, 3), 255, np.uint8),
            "gürültü": np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8),
            "bus: bina": np.ascontiguousarray(bus[0:300, :]),
            "bus: kaldırım": np.ascontiguousarray(bus[760:1080, :]),
            "zidane: gövde": np.ascontiguousarray(zid[430:720, 0:1000]),
        }
        for name, frame in cases.items():
            with self.subTest(name=name):
                result = self.detector.detect(frame, frame_index=0)
                self.assertEqual((result.faces, result.is_empty, result.face_count, result.dropped_boxes), ((), True, 0, 0))
                self.assertGreater(result.inference_ms, 0.0)


class PartialFaceAndBorderTest(RealModelTestCase):
    def shifted(self, dx: int, dy: int) -> np.ndarray:
        src = image("zidane")
        out = np.zeros_like(src)
        h, w = src.shape[:2]
        x0, x1 = max(0, -dx), min(w, w - dx)
        y0, y1 = max(0, -dy), min(h, h - dy)
        out[max(0, dy):max(0, dy) + (y1 - y0), max(0, dx):max(0, dx) + (x1 - x0)] = src[y0:y1, x0:x1]
        return out

    def test_face_cut_by_the_left_edge_is_clipped_flagged_and_inside_the_frame(self) -> None:
        result = self.detector.detect(self.shifted(-980, 0))  # Ancelotti'nin yüzü sol kenarda kısmen görünür
        edge = [f for f in result.faces if f.at_border and f.x1 == 0.0]
        self.assertTrue(edge, [(f.bbox_xyxy, f.confidence, f.at_border) for f in result.faces])
        for f in result.faces:
            self.assertTrue(0.0 <= f.x1 < f.x2 <= 1280.0 and 0.0 <= f.y1 < f.y2 <= 720.0)

    def test_face_cut_by_the_top_edge_is_flagged(self) -> None:
        result = self.detector.detect(self.shifted(0, -150))
        self.assertTrue(any(f.at_border and f.y1 <= 2.0 for f in result.faces), [(f.bbox_xyxy, f.at_border) for f in result.faces])

    def test_face_almost_completely_outside_is_not_reported(self) -> None:
        result = self.detector.detect(self.shifted(-1040, 0))
        self.assertFalse(any(iou(f.bbox_xyxy, (0.0, 107.0, 75.0, 274.0)) > 0.5 and f.confidence > 0.8 for f in result.faces))

    def test_clear_interior_faces_are_not_flagged_as_border_faces(self) -> None:
        result = self.detector.detect(image("zidane"))
        for f in clear(result):
            self.assertFalse(f.at_border, f.bbox_xyxy)

    def test_all_boxes_stay_inside_the_frame_for_every_shift(self) -> None:
        for dx, dy in ((-1100, 0), (-600, 0), (400, 0), (0, -300), (0, 300), (-500, -200), (300, 200)):
            with self.subTest(shift=(dx, dy)):
                result = self.detector.detect(self.shifted(dx, dy))
                for f in result.faces:
                    self.assertTrue(0.0 <= f.x1 < f.x2 <= 1280.0 and 0.0 <= f.y1 < f.y2 <= 720.0, f.bbox_xyxy)


class ResolutionTest(RealModelTestCase):
    def test_two_clear_faces_are_found_at_many_resolutions_with_scaled_coordinates(self) -> None:
        for w, h in ((640, 360), (960, 540), (1280, 720), (1920, 1080)):
            with self.subTest(size=(w, h)):
                frame = cv2.resize(image("zidane"), (w, h), interpolation=cv2.INTER_AREA if w < 1280 else cv2.INTER_CUBIC)
                result = self.detector.detect(frame)
                self.assertEqual(result.image_size, (w, h))
                self.assertEqual(len(clear(result)), 2)
                for ref in (ANCELOTTI, ZIDANE):
                    self.assertGreaterEqual(best_iou(result, scaled(ref, w / 1280, h / 720)), MIN_IOU)

    def test_small_and_distant_faces_down_to_about_32_pixels(self) -> None:
        for face_h in (32, 48, 64, 96, 128):
            with self.subTest(face_px=face_h):
                scale = face_h / (ANCELOTTI[3] - ANCELOTTI[1])
                small = cv2.resize(image("zidane"), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                canvas = np.full((720, 1280, 3), 114, np.uint8)
                canvas[: small.shape[0], : small.shape[1]] = small
                result = self.detector.detect(canvas)
                self.assertGreaterEqual(best_iou(result, scaled(ANCELOTTI, scale, scale)), 0.5)

    def test_large_close_faces(self) -> None:
        for face_h in (300, 450, 600):
            with self.subTest(face_px=face_h):
                scale = face_h / (ANCELOTTI[3] - ANCELOTTI[1])
                big = cv2.resize(image("zidane"), None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                cx, cy = (ANCELOTTI[0] + ANCELOTTI[2]) / 2 * scale, (ANCELOTTI[1] + ANCELOTTI[3]) / 2 * scale
                x0, y0 = int(cx - 640), int(cy - 360)
                canvas = np.full((720, 1280, 3), 114, np.uint8)
                sx0, sy0 = max(0, x0), max(0, y0)
                w = min(big.shape[1] - sx0, 1280 - max(0, -x0))
                h = min(big.shape[0] - sy0, 720 - max(0, -y0))
                canvas[max(0, -y0):max(0, -y0) + h, max(0, -x0):max(0, -x0) + w] = big[sy0:sy0 + h, sx0:sx0 + w]
                expected = (ANCELOTTI[0] * scale - x0, ANCELOTTI[1] * scale - y0, ANCELOTTI[2] * scale - x0, ANCELOTTI[3] * scale - y0)
                self.assertGreaterEqual(best_iou(self.detector.detect(canvas), expected), 0.5)

    def test_extreme_sizes_do_not_crash(self) -> None:
        for w, h in ((1, 1), (16, 16), (33, 17), (64, 48), (3000, 40), (40, 1200), (4000, 3000)):
            with self.subTest(size=(w, h)):
                result = self.detector.detect(np.full((h, w, 3), 100, np.uint8))
                self.assertEqual((result.image_size, result.is_empty), ((w, h), True))

    def test_detect_width_still_finds_the_clear_faces_and_maps_boxes_to_the_original_frame(self) -> None:
        d = FaceDetector(FaceDetectionConfig(detect_width=640))
        try:
            result = d.detect(image("zidane"))
        finally:
            d.close()
        self.assertEqual((result.image_size, result.inference_size), ((1280, 720), (640, 360)))
        self.assertEqual(len(clear(result)), 2)
        for ref in (ANCELOTTI, ZIDANE):
            self.assertGreaterEqual(best_iou(result, ref), MIN_IOU)


class ConfigBehaviourTest(RealModelTestCase):
    def count(self, **kw) -> int:  # type: ignore[no-untyped-def]
        d = FaceDetector(FaceDetectionConfig(**kw))
        try:
            return d.detect(image("zidane")).face_count
        finally:
            d.close()

    def test_higher_confidence_threshold_never_adds_faces(self) -> None:
        counts = [self.count(conf=c) for c in (0.3, 0.6, 0.9, 0.97)]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertGreaterEqual(counts[0], 3)
        self.assertLessEqual(counts[-1], 1)

    def test_max_faces_keeps_the_most_confident_and_reports_truncation(self) -> None:
        d = FaceDetector(FaceDetectionConfig(conf=0.3, max_faces=1))
        try:
            capped = d.detect(image("zidane"))
        finally:
            d.close()
        full = self.detector.detect(image("zidane"))
        self.assertEqual((capped.face_count, capped.truncated_faces >= 1), (1, True))
        self.assertEqual(capped.faces[0].bbox_xyxy, full.faces[0].bbox_xyxy)


class RobustnessTest(RealModelTestCase):
    def test_invalid_inputs_are_rejected_with_the_real_model_loaded(self) -> None:
        for bad in (None, np.zeros((48, 64), np.uint8), np.zeros((48, 64, 4), np.uint8), np.zeros((48, 64, 3), np.float32), np.zeros((0, 64, 3), np.uint8), "x.jpg"):
            with self.subTest(kind=type(bad).__name__), self.assertRaises(FaceInvalidFrameError):
                self.detector.detect(bad)  # type: ignore[arg-type]

    def test_detection_is_deterministic_and_does_not_modify_the_input(self) -> None:
        frame = image("zidane")
        before = frame.copy()
        a = self.detector.detect(frame)
        b = self.detector.detect(frame)
        np.testing.assert_array_equal(frame, before)
        self.assertEqual([(f.bbox_xyxy, f.confidence) for f in a.faces], [(f.bbox_xyxy, f.confidence) for f in b.faces])

    def test_noncontiguous_and_readonly_frames(self) -> None:
        frame = image("zidane")
        readonly = frame.copy()
        readonly.flags.writeable = False
        self.assertEqual(len(clear(self.detector.detect(readonly))), 2)
        wide = np.repeat(frame, 2, axis=1)[:, ::2]  # bitişik olmayan görünüm
        self.assertFalse(wide.flags.c_contiguous)
        self.assertEqual(len(clear(self.detector.detect(wide))), 2)

    def test_many_consecutive_frames_with_changing_sizes(self) -> None:
        # En-boy oranı korunur (bozulmuş oran yüzleri gerçekçi olmayan biçimde ezer; o durum burada sınanmaz). Boyut değişimi ardışık karelerde
        # modelin girdi boyutunu doğru yeniden ayarladığını gösterir.
        sequence = [("zidane", (1280, 720)), ("zidane", (640, 360)), ("zidane", (1280, 720)), ("bus", (810, 1080)), ("bus", (405, 540)), ("zidane", (1920, 1080))]
        for name, (w, h) in sequence * 3:
            frame = cv2.resize(image(name), (w, h), interpolation=cv2.INTER_AREA)
            result = self.detector.detect(frame)
            self.assertEqual(result.image_size, (w, h))
            self.assertGreaterEqual(len(clear(result)), 2, (name, (w, h)))


class RecordedVideoTest(RealModelTestCase):
    """Kayıtlı video üzerinde gerçek model (4A'nın sentetik montajı: bus.jpg + kayan zidane.jpg; kare başına 4 net yüz beklenir)."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._tmp = tempfile.TemporaryDirectory()
        cls.video = make_synthetic_video(Path(cls._tmp.name) / "montaj.mp4", frames=45)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()
        super().tearDownClass()

    def test_pipeline_over_the_recorded_video_finds_faces_in_every_frame(self) -> None:
        seen = []
        with FrameSource(SourceConfig.video(self.video)) as source:
            summary = run_face_detection(source, self.detector, on_result=lambda packet, result: seen.append((packet, result)))
        self.assertEqual((summary.reason, summary.frames_processed), (REASON_END_OF_STREAM, 45))
        self.assertEqual(summary.frames_with_faces, 45)
        self.assertEqual(summary.faces_max, 4)
        self.assertGreaterEqual(summary.faces_mean, 3.5)
        self.assertEqual((summary.dropped_boxes, summary.truncated_faces), (0, 0))
        self.assertEqual([p.frame_index for p, _ in seen], list(range(45)))
        for packet, result in seen:
            self.assertEqual((result.frame_index, result.timestamp, result.image_size), (packet.frame_index, packet.timestamp, (1280, 720)))
        self.assertGreater(summary.inference.median_ms, 0.0)  # type: ignore[union-attr]

    def test_pipeline_with_a_window_draws_the_real_faces_without_changing_the_source_frames(self) -> None:
        window = FakeWindow()
        originals = []
        with FrameSource(SourceConfig.video(self.video)) as source:
            summary = run_face_detection(source, self.detector, window, max_frames=10,
                                         on_result=lambda packet, result: originals.append((packet.frame, packet.frame.copy())))
        self.assertEqual((summary.reason, len(window.images)), (REASON_MAX_FRAMES, 10))
        for (frame, snapshot), shown in zip(originals, window.images):
            np.testing.assert_array_equal(frame, snapshot)
            self.assertEqual(shown.shape, snapshot.shape)
            self.assertFalse(np.array_equal(shown, snapshot))

    def test_cli_end_to_end_with_the_real_model_on_the_video(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["--source", "video", "--video-path", str(self.video), "--no-display", "--max-frames", "20"])
        self.assertEqual((code, err.getvalue()), (0, ""))
        text = out.getvalue()
        self.assertIn("İşlenen kare: 20", text)
        self.assertIn("en çok 4", text)
        self.assertIn("en az bir yüz olan kare: 20/20", text)
        self.assertIn("YuNet", text)

    def test_cli_with_a_window_closes_the_window_and_releases_the_video(self) -> None:
        window = FakeWindow(keys={5: ord("q")})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(["--source", "video", "--video-path", str(self.video)], window_factory=lambda: window)
        self.assertEqual((code, window.open_calls, window.close_calls, len(window.images)), (0, 1, 1, 5))
        self.assertIn("kullanıcı çıkışı", out.getvalue())

    def test_script_runs_as_a_subprocess_and_exits_0(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_face_detection.py"), "--source", "video", "--video-path", str(self.video),
             "--no-display", "--max-frames", "10", "--detect-width", "960"],
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("İşlenen kare: 10", proc.stdout)
        self.assertIn("960 px", proc.stdout)

    def test_no_frames_or_faces_are_written_to_disk(self) -> None:
        before = sorted(p.name for p in Path(self._tmp.name).iterdir())
        with mock.patch.object(cv2, "imwrite", side_effect=AssertionError("görüntü kaydedilmemeli")):
            with FrameSource(SourceConfig.video(self.video)) as source:
                run_face_detection(source, self.detector, max_frames=5)
        self.assertEqual(sorted(p.name for p in Path(self._tmp.name).iterdir()), before)


GUI_ENABLED = sys.platform == "darwin" and os.environ.get("ISG_GUI_TESTS") == "1"


@unittest.skipUnless(READY and GUI_ENABLED, "Gerçek pencere + gerçek model testi: macOS + ISG_GUI_TESTS=1 + grafik oturumu + model gerekir")
class RealWindowRealModelTest(unittest.TestCase):
    def test_real_window_with_the_real_model_on_the_recorded_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = make_synthetic_video(Path(tmp) / "montaj.mp4", frames=30)
            detector = FaceDetector()
            detector.load()
            try:
                with FrameSource(SourceConfig.video(video)) as source, PreviewWindow(WINDOW_NAME) as window:
                    summary = run_face_detection(source, detector, window, max_frames=20)
            finally:
                detector.close()
        self.assertEqual((summary.reason, summary.frames_processed), (REASON_MAX_FRAMES, 20))
        self.assertEqual(summary.faces_max, 4)


if __name__ == "__main__":
    unittest.main()
