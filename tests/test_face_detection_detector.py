"""``FaceDetector`` testleri — SAHTE YuNet modeliyle (gerçek model yüklenmez, ağırlık dosyası okunmaz, ağ yok).

Bu testler yalnızca ``FaceDetector``'ın mantığını (doğrulama, koordinat dönüşümü, kırpma, hata yönetimi, tek yükleme, süre ölçümü)
sınar. Gerçek yüz tespitinin çalıştığını KANITLAMAZ; o, ``test_face_detection_real_model.py`` dosyasındadır.
"""

from __future__ import annotations

import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from face_detection_test_support import FakeYuNet, LoaderSpy, blank_frame, fake_model_file, make_detector, yunet_rows  # noqa: E402
from person_detection_test_support import SequenceClock, TempDirMixin  # noqa: E402

from isg_vision.face_detection.config import FaceDetectionConfig  # noqa: E402
from isg_vision.face_detection.detector import DOWNLOAD_HINT, FaceDetector  # noqa: E402
from isg_vision.face_detection.errors import (  # noqa: E402
    FaceDetectionError,
    FaceInferenceError,
    FaceInvalidFrameError,
    FaceModelLoadError,
)
from isg_vision.face_detection.weights import OFFICIAL_MODELS  # noqa: E402


class SingleAndMultipleFacesTest(TempDirMixin, unittest.TestCase):
    def test_single_face_box_confidence_and_metadata(self) -> None:
        detector, _ = make_detector(FakeYuNet([[(10, 20, 30, 40, 0.93)]]), self.tmp)
        result = detector.detect(blank_frame(100, 80), frame_index=7, timestamp=12.5)
        self.assertEqual((result.face_count, result.image_size, result.frame_index, result.timestamp), (1, (100, 80), 7, 12.5))
        face = result.faces[0]
        self.assertEqual(face.bbox_xyxy, (10.0, 20.0, 40.0, 60.0))  # (x, y, w, h) -> (x1, y1, x2, y2)
        self.assertAlmostEqual(face.confidence, 0.93, places=5)
        self.assertEqual((face.frame_index, face.timestamp, face.image_size, face.face_index, face.at_border), (7, 12.5, (100, 80), 0, False))
        self.assertEqual((result.device, result.dropped_boxes, result.truncated_faces, result.inference_size), ("cpu", 0, 0, (100, 80)))

    def test_multiple_faces_sorted_by_confidence(self) -> None:
        rows = [(5, 5, 10, 10, 0.7), (30, 5, 12, 12, 0.95), (60, 40, 14, 14, 0.8), (80, 10, 8, 8, 0.65)]
        detector, _ = make_detector(FakeYuNet([rows]), self.tmp)
        result = detector.detect(blank_frame(100, 80))
        self.assertEqual(result.face_count, 4)
        self.assertEqual([round(f.confidence, 2) for f in result.faces], [0.95, 0.8, 0.7, 0.65])
        self.assertEqual([f.face_index for f in result.faces], [0, 1, 2, 3])
        self.assertEqual(result.faces[0].bbox_xyxy, (30.0, 5.0, 42.0, 17.0))

    def test_many_faces(self) -> None:
        rows = [(2 + 6 * i, 4, 5, 5, 0.9 - 0.001 * i) for i in range(15)]
        detector, _ = make_detector(FakeYuNet([rows]), self.tmp)
        self.assertEqual(detector.detect(blank_frame(100, 80)).face_count, 15)

    def test_every_frame_calls_the_model_exactly_once(self) -> None:
        model = FakeYuNet([[(5, 5, 10, 10, 0.9)]])
        detector, _ = make_detector(model, self.tmp)
        for i in range(6):
            detector.detect(blank_frame(), frame_index=i)
        self.assertEqual(len(model.detect_calls), 6)

    def test_script_per_frame_changes_results(self) -> None:
        detector, _ = make_detector(FakeYuNet([[(5, 5, 10, 10, 0.9)], None, [(5, 5, 10, 10, 0.9), (30, 5, 10, 10, 0.8)]]), self.tmp)
        counts = [detector.detect(blank_frame(100, 80)).face_count for _ in range(3)]
        self.assertEqual(counts, [1, 0, 2])


class NoFaceTest(TempDirMixin, unittest.TestCase):
    def test_model_returning_none_is_an_empty_result_not_an_error(self) -> None:
        detector, _ = make_detector(FakeYuNet([None]), self.tmp)
        result = detector.detect(blank_frame(), frame_index=2, timestamp=1.0)
        self.assertEqual((result.faces, result.is_empty, result.face_count, result.frame_index, result.timestamp), ((), True, 0, 2, 1.0))
        self.assertGreaterEqual(result.inference_ms, 0.0)

    def test_empty_array_is_also_an_empty_result(self) -> None:
        detector, _ = make_detector(FakeYuNet([np.zeros((0, 15), dtype=np.float32)]), self.tmp)
        self.assertTrue(detector.detect(blank_frame()).is_empty)

    def test_all_black_and_all_white_frames(self) -> None:
        detector, _ = make_detector(FakeYuNet([None]), self.tmp)
        for value in (0, 255):
            self.assertTrue(detector.detect(np.full((48, 64, 3), value, np.uint8)).is_empty)


class PartialAndBorderFacesTest(TempDirMixin, unittest.TestCase):
    def test_box_extending_past_the_left_edge_is_clipped_and_flagged(self) -> None:
        detector, _ = make_detector(FakeYuNet([[(-20, 10, 40, 40, 0.88)]]), self.tmp)
        face = detector.detect(blank_frame(100, 80)).faces[0]
        self.assertEqual(face.bbox_xyxy, (0.0, 10.0, 20.0, 50.0))
        self.assertTrue(face.at_border)

    def test_box_extending_past_right_and_bottom_edges(self) -> None:
        detector, _ = make_detector(FakeYuNet([[(80, 60, 40, 40, 0.8)]]), self.tmp)
        face = detector.detect(blank_frame(100, 80)).faces[0]
        self.assertEqual(face.bbox_xyxy, (80.0, 60.0, 100.0, 80.0))
        self.assertTrue(face.at_border)

    def test_box_touching_the_top_edge_is_flagged_without_clipping(self) -> None:
        detector, _ = make_detector(FakeYuNet([[(30, 0, 20, 25, 0.9)]]), self.tmp)
        face = detector.detect(blank_frame(100, 80)).faces[0]
        self.assertEqual(face.bbox_xyxy, (30.0, 0.0, 50.0, 25.0))
        self.assertTrue(face.at_border)

    def test_face_close_to_but_not_touching_the_border_is_not_flagged(self) -> None:
        detector, _ = make_detector(FakeYuNet([[(10, 10, 20, 20, 0.9)]]), self.tmp)
        self.assertFalse(detector.detect(blank_frame(100, 80)).faces[0].at_border)

    def test_box_entirely_outside_is_dropped_and_counted(self) -> None:
        detector, _ = make_detector(FakeYuNet([[(150, 10, 20, 20, 0.9), (30, 10, 20, 20, 0.8)]]), self.tmp)
        result = detector.detect(blank_frame(100, 80))
        self.assertEqual((result.face_count, result.dropped_boxes), (1, 1))

    def test_sliver_inside_the_image_is_dropped(self) -> None:
        detector, _ = make_detector(FakeYuNet([[(-39.5, 10, 40, 40, 0.9)]]), self.tmp)  # içeride yalnızca 0,5 px
        result = detector.detect(blank_frame(100, 80))
        self.assertEqual((result.face_count, result.dropped_boxes), (0, 1))

    def test_nonfinite_values_from_the_model_are_dropped(self) -> None:
        rows = yunet_rows([(5, 5, 10, 10, 0.9), (30, 5, 10, 10, 0.9), (60, 5, 10, 10, 0.9)])
        assert rows is not None
        rows[0, 0] = np.nan
        rows[1, 14] = np.inf
        detector, _ = make_detector(FakeYuNet([rows]), self.tmp)
        result = detector.detect(blank_frame(100, 80))
        self.assertEqual((result.face_count, result.dropped_boxes), (1, 2))

    def test_max_faces_truncates_and_counts(self) -> None:
        rows = [(2 + 12 * i, 4, 8, 8, 0.5 + 0.05 * i) for i in range(6)]
        detector, _ = make_detector(FakeYuNet([rows]), self.tmp, max_faces=2)
        result = detector.detect(blank_frame(100, 80))
        self.assertEqual((result.face_count, result.truncated_faces), (2, 4))
        self.assertEqual([round(f.confidence, 2) for f in result.faces], [0.75, 0.7])


class ResolutionTest(TempDirMixin, unittest.TestCase):
    def test_various_image_sizes_set_the_model_input_size_and_report_it(self) -> None:
        sizes = [(16, 16), (33, 17), (64, 48), (320, 240), (640, 480), (1280, 720), (1920, 1080), (3000, 40), (1, 500)]
        model = FakeYuNet([None])
        detector, _ = make_detector(model, self.tmp)
        for w, h in sizes:
            with self.subTest(size=(w, h)):
                result = detector.detect(blank_frame(w, h))
                self.assertEqual((result.image_size, result.inference_size), ((w, h), (w, h)))
        self.assertEqual(model.input_sizes, sizes)

    def test_input_size_is_set_only_when_the_size_changes(self) -> None:
        model = FakeYuNet([None])
        detector, _ = make_detector(model, self.tmp)
        for _ in range(4):
            detector.detect(blank_frame(64, 48))
        detector.detect(blank_frame(80, 60))
        detector.detect(blank_frame(80, 60))
        self.assertEqual(model.input_sizes, [(64, 48), (80, 60)])

    def test_detect_width_downscales_wider_frames_and_maps_boxes_back(self) -> None:
        model = FakeYuNet([[(10, 10, 20, 20, 0.9)]])
        detector, _ = make_detector(model, self.tmp, detect_width=100)
        frame = blank_frame(200, 100)
        result = detector.detect(frame)
        self.assertEqual(model.input_sizes, [(100, 50)])
        self.assertEqual(model.detect_calls[0].shape, (50, 100, 3))
        self.assertEqual((result.image_size, result.inference_size), ((200, 100), (100, 50)))
        self.assertEqual(result.faces[0].bbox_xyxy, (20.0, 20.0, 60.0, 60.0))  # 2x geri ölçek
        self.assertEqual(result.faces[0].image_size, (200, 100))

    def test_detect_width_does_not_upscale_narrower_frames(self) -> None:
        model = FakeYuNet([[(10, 10, 20, 20, 0.9)]])
        detector, _ = make_detector(model, self.tmp, detect_width=640)
        result = detector.detect(blank_frame(200, 100))
        self.assertEqual((model.input_sizes, result.inference_size, result.faces[0].bbox_xyxy), ([(200, 100)], (200, 100), (10.0, 10.0, 30.0, 30.0)))

    def test_detect_width_equal_to_frame_width_does_not_resize(self) -> None:
        model = FakeYuNet([None])
        detector, _ = make_detector(model, self.tmp, detect_width=200)
        detector.detect(blank_frame(200, 100))
        self.assertEqual(model.input_sizes, [(200, 100)])

    def test_detect_width_with_non_integer_scale_maps_back_to_original_pixels(self) -> None:
        model = FakeYuNet([[(30, 20, 30, 30, 0.9)]])
        detector, _ = make_detector(model, self.tmp, detect_width=100)
        result = detector.detect(blank_frame(150, 90))  # ölçek 1,5; yükseklik 90*100/150 = 60
        self.assertEqual(result.inference_size, (100, 60))
        x1, y1, x2, y2 = result.faces[0].bbox_xyxy
        self.assertAlmostEqual(x1, 45.0)
        self.assertAlmostEqual(y1, 30.0)
        self.assertAlmostEqual(x2, 90.0)
        self.assertAlmostEqual(y2, 75.0)

    def test_extremely_wide_frame_with_detect_width_keeps_at_least_one_row(self) -> None:
        model = FakeYuNet([None])
        detector, _ = make_detector(model, self.tmp, detect_width=64)
        result = detector.detect(blank_frame(6400, 3))
        self.assertEqual(result.inference_size, (64, 1))


class InvalidImageTest(TempDirMixin, unittest.TestCase):
    def test_invalid_images_raise_before_touching_the_model(self) -> None:
        bad = {
            "None": None,
            "liste": [[1, 2, 3]],
            "metin": "resim.jpg",
            "gri 2B": np.zeros((48, 64), np.uint8),
            "4 kanal": np.zeros((48, 64, 4), np.uint8),
            "1 kanal 3B": np.zeros((48, 64, 1), np.uint8),
            "float32": np.zeros((48, 64, 3), np.float32),
            "int64": np.zeros((48, 64, 3), np.int64),
            "boş 0 yükseklik": np.zeros((0, 64, 3), np.uint8),
            "boş 0 genişlik": np.zeros((48, 0, 3), np.uint8),
            "4B": np.zeros((1, 48, 64, 3), np.uint8),
        }
        for name, image in bad.items():
            with self.subTest(name=name):
                model = FakeYuNet([None])
                detector, spy = make_detector(model, self.tmp)
                with self.assertRaises(FaceInvalidFrameError):
                    detector.detect(image)  # type: ignore[arg-type]
                self.assertEqual(model.detect_calls, [])
                self.assertEqual(spy.calls, [], "geçersiz görüntüde model yüklenmemeli")

    def test_invalid_frame_error_is_a_value_error_and_a_detection_error(self) -> None:
        self.assertTrue(issubclass(FaceInvalidFrameError, ValueError))
        self.assertTrue(issubclass(FaceInvalidFrameError, FaceDetectionError))

    def test_detector_recovers_after_an_invalid_frame(self) -> None:
        detector, _ = make_detector(FakeYuNet([[(5, 5, 10, 10, 0.9)]]), self.tmp)
        with self.assertRaises(FaceInvalidFrameError):
            detector.detect(None)  # type: ignore[arg-type]
        self.assertEqual(detector.detect(blank_frame(100, 80)).face_count, 1)

    def test_non_contiguous_and_readonly_frames_are_accepted(self) -> None:
        detector, _ = make_detector(FakeYuNet([None]), self.tmp)
        wide = np.zeros((48, 128, 3), np.uint8)
        readonly = np.zeros((48, 64, 3), np.uint8)
        readonly.flags.writeable = False
        self.assertTrue(detector.detect(wide[:, ::2]).is_empty)
        self.assertTrue(detector.detect(readonly).is_empty)


class InputIsNeverModifiedTest(TempDirMixin, unittest.TestCase):
    def test_input_image_is_not_modified(self) -> None:
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, (80, 100, 3), dtype=np.uint8)
        before = frame.copy()
        for detect_width in (None, 64):  # 100 px genişlikli kare 64'e küçültülür (en küçük geçerli tespit genişliği)
            detector, _ = make_detector(FakeYuNet([[(10, 10, 20, 20, 0.9)]]), self.tmp, detect_width=detect_width)
            detector.detect(frame)
            np.testing.assert_array_equal(frame, before)


class ModelLoadingTest(TempDirMixin, unittest.TestCase):
    def test_model_is_loaded_exactly_once_over_many_frames(self) -> None:
        detector, spy = make_detector(FakeYuNet([None]), self.tmp)
        self.assertFalse(detector.is_loaded)
        for _ in range(10):
            detector.detect(blank_frame())
        detector.load()
        detector.load()
        self.assertEqual((len(spy.calls), detector.load_count, detector.is_loaded), (1, 1, True))

    def test_loader_receives_path_and_config(self) -> None:
        detector, spy = make_detector(FakeYuNet([None]), self.tmp, conf=0.7, nms=0.4)
        detector.load()
        path, config = spy.calls[0]
        self.assertEqual((path, config.conf, config.nms), (detector.config.model_path, 0.7, 0.4))

    def test_load_time_is_recorded(self) -> None:
        detector, _ = make_detector(FakeYuNet([None]), self.tmp, clock=SequenceClock(step=0.25))
        self.assertIsNone(detector.load_seconds)
        detector.load()
        self.assertGreater(detector.load_seconds, 0.0)

    def test_missing_model_file_gives_actionable_error_and_no_load_attempt(self) -> None:
        spy = LoaderSpy(FakeYuNet())
        detector = FaceDetector(FaceDetectionConfig(model_path=self.tmp / "yok.onnx"), model_loader=spy)
        with self.assertRaises(FaceModelLoadError) as ctx:
            detector.load()
        self.assertIn("yok.onnx", str(ctx.exception))
        self.assertIn("download_face_model.py", str(ctx.exception))
        self.assertEqual(spy.calls, [])
        self.assertFalse(detector.is_loaded)

    def test_directory_instead_of_file_is_reported_as_missing(self) -> None:
        (self.tmp / "klasor.onnx").mkdir()
        detector = FaceDetector(FaceDetectionConfig(model_path=self.tmp / "klasor.onnx"), model_loader=LoaderSpy(FakeYuNet()))
        with self.assertRaises(FaceModelLoadError):
            detector.load()

    def test_loader_exception_is_wrapped_with_the_original_cause(self) -> None:
        detector, _ = make_detector(RuntimeError("bozuk onnx"), self.tmp)
        with self.assertRaises(FaceModelLoadError) as ctx:
            detector.load()
        self.assertIn("bozuk onnx", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)
        self.assertFalse(detector.is_loaded)

    def test_detect_wraps_load_failures_too(self) -> None:
        detector, _ = make_detector(RuntimeError("x"), self.tmp)
        with self.assertRaises(FaceModelLoadError):
            detector.detect(blank_frame())

    def test_model_without_expected_interface_is_rejected(self) -> None:
        for model in (object(), type("M", (), {"detect": lambda self, i: (1, None)})(), type("M", (), {"setInputSize": lambda self, s: None})()):
            with self.subTest(model=type(model).__name__):
                detector, _ = make_detector(model, self.tmp)
                with self.assertRaises(FaceModelLoadError):
                    detector.load()

    def test_failed_load_can_be_retried(self) -> None:
        spy = LoaderSpy(RuntimeError("geçici"))
        detector = FaceDetector(FaceDetectionConfig(model_path=fake_model_file(self.tmp)), model_loader=spy)
        with self.assertRaises(FaceModelLoadError):
            detector.load()
        spy.model = FakeYuNet([None])
        detector.load()
        self.assertTrue(detector.is_loaded)
        self.assertEqual(len(spy.calls), 2)

    def test_close_releases_the_model_and_is_idempotent(self) -> None:
        detector, spy = make_detector(FakeYuNet([None]), self.tmp)
        detector.load()
        detector.close()
        detector.close()
        self.assertFalse(detector.is_loaded)
        detector.detect(blank_frame())  # kapatıldıktan sonra yeniden yüklenir
        self.assertEqual(len(spy.calls), 2)

    def test_context_manager_loads_and_closes(self) -> None:
        detector, _ = make_detector(FakeYuNet([None]), self.tmp)
        with detector as d:
            self.assertTrue(d.is_loaded)
        self.assertFalse(detector.is_loaded)

    def test_context_manager_closes_on_exception(self) -> None:
        detector, _ = make_detector(FakeYuNet([None]), self.tmp)
        with self.assertRaises(RuntimeError), detector:
            raise RuntimeError("boom")
        self.assertFalse(detector.is_loaded)

    def test_device_is_always_cpu_without_warning(self) -> None:
        detector, _ = make_detector(FakeYuNet([None]), self.tmp)
        self.assertEqual((detector.device, detector.device_warning), ("cpu", None))


class IntegrityCheckTest(TempDirMixin, unittest.TestCase):
    OFFICIAL = "face_detection_yunet_2026may.onnx"

    def test_official_name_with_wrong_content_is_rejected_before_parsing(self) -> None:
        spy = LoaderSpy(FakeYuNet([None]))
        path = fake_model_file(self.tmp, self.OFFICIAL, b"x" * OFFICIAL_MODELS[self.OFFICIAL].size_bytes)  # boyut doğru, özet yanlış
        detector = FaceDetector(FaceDetectionConfig(model_path=path), model_loader=spy)
        with self.assertRaises(FaceModelLoadError) as ctx:
            detector.load()
        self.assertIn("bütünlük", str(ctx.exception))
        self.assertEqual(spy.calls, [], "doğrulanamayan dosya ONNX ayrıştırıcıya verilmemeli")
        self.assertFalse(detector.is_loaded)

    def test_official_name_with_wrong_size_is_rejected(self) -> None:
        spy = LoaderSpy(FakeYuNet([None]))
        path = fake_model_file(self.tmp, self.OFFICIAL, b"kisa")
        with self.assertRaises(FaceModelLoadError):
            FaceDetector(FaceDetectionConfig(model_path=path), model_loader=spy).load()
        self.assertEqual(spy.calls, [])

    def test_verification_can_be_disabled_explicitly(self) -> None:
        spy = LoaderSpy(FakeYuNet([None]))
        path = fake_model_file(self.tmp, self.OFFICIAL, b"kisa")
        detector = FaceDetector(FaceDetectionConfig(model_path=path, verify_model_hash=False), model_loader=spy)
        detector.load()
        self.assertEqual(len(spy.calls), 1)

    def test_custom_model_names_are_not_hash_checked(self) -> None:
        detector, spy = make_detector(FakeYuNet([None]), self.tmp)  # "fake.onnx": izin listesinde değil
        detector.load()
        self.assertEqual(len(spy.calls), 1)

    def test_correct_content_passes(self) -> None:
        # Sahte bir izin listesi girdisiyle (gerçek dosya gerekmeden) doğru özetin geçtiğini sınar.
        import hashlib

        from isg_vision.face_detection import detector as det_module
        from isg_vision.face_detection.weights import OfficialFaceModel

        content = b"merhaba-model" * 100
        entry = OfficialFaceModel("ozel_resmi.onnx", "https://x/y", hashlib.sha256(content).hexdigest(), len(content))
        path = fake_model_file(self.tmp, "ozel_resmi.onnx", content)
        spy = LoaderSpy(FakeYuNet([None]))
        original = dict(det_module.OFFICIAL_MODELS)
        det_module.OFFICIAL_MODELS["ozel_resmi.onnx"] = entry
        try:
            FaceDetector(FaceDetectionConfig(model_path=path), model_loader=spy).load()
        finally:
            det_module.OFFICIAL_MODELS.clear()
            det_module.OFFICIAL_MODELS.update(original)
        self.assertEqual(len(spy.calls), 1)

    def test_download_hint_mentions_the_only_network_step(self) -> None:
        self.assertIn("download_face_model.py", DOWNLOAD_HINT)


class InferenceErrorTest(TempDirMixin, unittest.TestCase):
    def test_model_exception_is_wrapped_and_detector_recovers(self) -> None:
        model = FakeYuNet([[(5, 5, 10, 10, 0.9)]], error=RuntimeError("DNN çöktü"))
        detector, _ = make_detector(model, self.tmp)
        with self.assertRaises(FaceInferenceError) as ctx:
            detector.detect(blank_frame(100, 80))
        self.assertIn("DNN çöktü", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)
        model.error = None
        self.assertEqual(detector.detect(blank_frame(100, 80)).face_count, 1)
        self.assertEqual(model.input_sizes, [(100, 80), (100, 80)], "hata sonrası girdi boyutu yeniden ayarlanmalı")

    def test_unexpected_output_shapes_are_rejected(self) -> None:
        for name, output in {"1B": np.zeros(15, np.float32), "dar": np.zeros((2, 5), np.float32), "3B": np.zeros((1, 2, 15), np.float32)}.items():
            with self.subTest(name=name):
                detector, _ = make_detector(FakeYuNet([output]), self.tmp)
                with self.assertRaises(FaceInferenceError):
                    detector.detect(blank_frame())

    def test_unconvertible_output_is_rejected(self) -> None:
        detector, _ = make_detector(FakeYuNet([None]), self.tmp)
        detector.load()
        detector._model.detect = lambda image: (1, "bu bir dizi değil")  # type: ignore[union-attr]
        with self.assertRaises(FaceInferenceError):
            detector.detect(blank_frame())

    def test_set_input_size_failure_is_wrapped(self) -> None:
        model = FakeYuNet([None])

        def broken(size):  # type: ignore[no-untyped-def]
            raise RuntimeError("boyut reddedildi")

        model.setInputSize = broken  # type: ignore[method-assign]
        detector, _ = make_detector(model, self.tmp)
        with self.assertRaises(FaceInferenceError):
            detector.detect(blank_frame())


class TimingTest(TempDirMixin, unittest.TestCase):
    def test_inference_ms_comes_from_the_injected_clock(self) -> None:
        detector, _ = make_detector(FakeYuNet([None]), self.tmp, clock=SequenceClock([0.0, 0.002, 1.0, 1.0123]))
        detector.load()  # saat 2 kez okunur (yükleme)
        result = detector.detect(blank_frame())
        self.assertAlmostEqual(result.inference_ms, 12.3, places=3)

    def test_default_clock_gives_nonnegative_time(self) -> None:
        detector, _ = make_detector(FakeYuNet([None]), self.tmp)
        self.assertGreaterEqual(detector.detect(blank_frame()).inference_ms, 0.0)

    def test_warmup_runs_blank_frames_at_the_requested_size_and_records_time(self) -> None:
        model = FakeYuNet([None])
        detector, spy = make_detector(model, self.tmp)
        seconds = detector.warmup((320, 240), runs=3)
        self.assertGreaterEqual(seconds, 0.0)
        self.assertEqual(detector.warmup_seconds, seconds)
        self.assertEqual(len(model.detect_calls), 3)
        self.assertTrue(all(call.shape == (240, 320, 3) and not call.any() for call in model.detect_calls))
        self.assertEqual(len(spy.calls), 1, "ısınma modeli yüklemeli")

    def test_warmup_rejects_invalid_run_counts(self) -> None:
        detector, _ = make_detector(FakeYuNet([None]), self.tmp)
        for bad in (0, -1, 1.5, True, None):
            with self.subTest(runs=bad), self.assertRaises(ValueError):
                detector.warmup(runs=bad)  # type: ignore[arg-type]


class ResultImmutabilityTest(TempDirMixin, unittest.TestCase):
    def test_results_are_frozen_and_hold_no_reference_to_the_input(self) -> None:
        detector, _ = make_detector(FakeYuNet([[(5, 5, 10, 10, 0.9)]]), self.tmp)
        frame = blank_frame(100, 80)
        result = detector.detect(frame)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.faces = ()  # type: ignore[misc]
        frame[:] = 255  # sonradan değişen girdi, sonucu etkilememeli
        self.assertEqual(result.faces[0].bbox_xyxy, (5.0, 5.0, 15.0, 15.0))


class TempFileCleanupTest(unittest.TestCase):
    def test_tests_use_temporary_directories_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            detector, _ = make_detector(FakeYuNet([None]), Path(tmp))
            detector.detect(blank_frame())
        # Dizin silindikten sonra yeni bir yüklemede model dosyası bulunamaz (dosya sistemine kalıcı bir şey yazılmadı)
        detector.close()
        with self.assertRaises(FaceModelLoadError):
            detector.load()


if __name__ == "__main__":
    unittest.main()
