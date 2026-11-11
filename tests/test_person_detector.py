"""``PersonDetector`` testleri: sahte (mock) YOLO modeliyle; gerçek model, ağırlık indirme veya internet yoktur."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from isg_vision.human_detection.config import DetectionConfig, ResolvedDevice  # noqa: E402
from isg_vision.human_detection.errors import (  # noqa: E402
    DeviceUnavailableError,
    InferenceError,
    InvalidFrameError,
    ModelLoadError,
)
from isg_vision.human_detection.person_detector import PersonDetector  # noqa: E402
from person_detection_test_support import (  # noqa: E402
    FakeYoloModel,
    LoaderSpy,
    SequenceClock,
    TempDirMixin,
    blank_frame,
    fake_weights_file,
    make_detector,
)

ONE_PERSON = [((10, 10, 40, 40), 0.91, 0)]
THREE_PEOPLE = [((5, 5, 20, 40), 0.7, 0), ((100, 20, 150, 100), 0.95, 0), ((200, 30, 260, 120), 0.85, 0)]


class ModelLoadingTest(TempDirMixin, unittest.TestCase):
    def test_model_is_loaded_only_once_across_many_detections(self) -> None:
        detector, spy = make_detector(FakeYoloModel([ONE_PERSON]), self.tmp)
        self.assertFalse(detector.is_loaded)
        self.assertEqual(spy.calls, [], "yapıcı model yüklememeli")
        for _ in range(5):
            detector.detect(blank_frame(320, 240))
        detector.load()
        detector.load()
        self.assertEqual(len(spy.calls), 1)
        self.assertEqual(detector.load_count, 1)
        self.assertTrue(detector.is_loaded)

    def test_loader_receives_weight_path_and_resolved_device(self) -> None:
        detector, spy = make_detector(FakeYoloModel(), self.tmp, device="cpu")
        detector.load()
        self.assertEqual(spy.calls, [(detector.config.model_path, "cpu")])

    def test_missing_weight_file_gives_clear_error_without_loading_or_downloading(self) -> None:
        spy = LoaderSpy(FakeYoloModel())
        detector = PersonDetector(DetectionConfig(model_path=self.tmp / "yok.pt", device="cpu"), model_loader=spy)
        with self.assertRaises(ModelLoadError) as ctx:
            detector.load()
        self.assertIn("bulunamadı", str(ctx.exception))
        self.assertIn("download_person_model.py", str(ctx.exception))  # çözüm yolu söylenir
        self.assertEqual(spy.calls, [])

    def test_directory_instead_of_file_is_rejected(self) -> None:
        folder = self.tmp / "klasor.pt"
        folder.mkdir()
        detector = PersonDetector(DetectionConfig(model_path=folder, device="cpu"), model_loader=LoaderSpy(FakeYoloModel()))
        with self.assertRaises(ModelLoadError):
            detector.load()

    def test_loader_failure_is_wrapped_with_cause(self) -> None:
        detector = PersonDetector(
            DetectionConfig(model_path=fake_weights_file(self.tmp, "bozuk.pt"), device="cpu"),
            model_loader=LoaderSpy(RuntimeError("pickle bozuk")),
        )
        with self.assertRaises(ModelLoadError) as ctx:
            detector.load()
        self.assertIn("pickle bozuk", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)
        self.assertFalse(detector.is_loaded)

    def test_failed_load_can_be_retried(self) -> None:
        spy = LoaderSpy(RuntimeError("geçici"))
        detector = PersonDetector(DetectionConfig(model_path=fake_weights_file(self.tmp), device="cpu"), model_loader=spy)
        for _ in range(2):
            with self.assertRaises(ModelLoadError):
                detector.load()
        self.assertEqual(len(spy.calls), 2)
        self.assertEqual(detector.load_count, 0)

    def test_model_without_person_class_is_rejected(self) -> None:
        detector, _ = make_detector(FakeYoloModel(names={0: "helmet", 1: "vest"}), self.tmp)
        with self.assertRaisesRegex(ModelLoadError, "person"):
            detector.load()
        self.assertFalse(detector.is_loaded)

    def test_ambiguous_person_classes_rejected(self) -> None:
        detector, _ = make_detector(FakeYoloModel(names={0: "person", 1: "Person"}), self.tmp)
        with self.assertRaises(ModelLoadError):
            detector.load()

    def test_unreadable_class_map_rejected(self) -> None:
        detector = PersonDetector(
            DetectionConfig(model_path=fake_weights_file(self.tmp), device="cpu"),
            model_loader=LoaderSpy(type("NamelessModel", (), {"names": 5})()),
        )
        with self.assertRaises(ModelLoadError):
            detector.load()

    def test_person_class_id_comes_from_metadata_not_assumed_zero(self) -> None:
        model = FakeYoloModel([[((0, 0, 30, 30), 0.9, 7), ((40, 0, 70, 30), 0.99, 0)]], names={0: "helmet", 7: "person"})
        detector, _ = make_detector(model, self.tmp)
        result = detector.detect(blank_frame())
        self.assertEqual(detector.person_class_id, 7)
        self.assertEqual(model.predict_calls[0]["classes"], [7])  # modele yalnızca person kimliği istenir
        self.assertEqual([(d.class_id, d.class_name) for d in result.detections], [(7, "person")])

    def test_class_names_given_as_list_are_supported(self) -> None:
        detector, _ = make_detector(FakeYoloModel([ONE_PERSON], names=["person", "car"]), self.tmp)
        self.assertEqual(detector.detect(blank_frame()).person_count, 1)

    def test_person_class_id_requires_loaded_model(self) -> None:
        detector, _ = make_detector(FakeYoloModel(), self.tmp)
        with self.assertRaises(ModelLoadError):
            _ = detector.person_class_id

    def test_close_is_safe_and_idempotent(self) -> None:
        detector, _ = make_detector(FakeYoloModel(), self.tmp)
        detector.load()
        detector.close()
        detector.close()
        self.assertFalse(detector.is_loaded)

    def test_context_manager_loads_and_closes(self) -> None:
        detector, spy = make_detector(FakeYoloModel(), self.tmp)
        with detector as d:
            self.assertTrue(d.is_loaded)
        self.assertFalse(detector.is_loaded)
        self.assertEqual(len(spy.calls), 1)

    def test_load_and_warmup_durations_are_recorded_separately(self) -> None:
        # load: 2 saat okuması (başlangıç, bitiş); warmup: 2 okuma
        clock = SequenceClock([1.0, 1.5, 10.0, 10.25])
        detector, _ = make_detector(FakeYoloModel(), self.tmp, clock=clock)
        self.assertIsNone(detector.load_seconds)
        self.assertEqual(detector.warmup((64, 48)), 0.25)
        self.assertEqual(detector.load_seconds, 0.5)
        self.assertEqual(detector.warmup_seconds, 0.25)


class DetectionOutputTest(TempDirMixin, unittest.TestCase):
    def test_no_person_returns_empty_result_without_error(self) -> None:
        detector, _ = make_detector(FakeYoloModel([[]]), self.tmp)
        result = detector.detect(blank_frame(), frame_index=0)
        self.assertEqual((result.person_count, result.is_empty, result.detections), (0, True, ()))
        self.assertEqual(result.dropped_boxes, 0)

    def test_model_returning_no_boxes_object_is_handled(self) -> None:
        detector, _ = make_detector(FakeYoloModel([None]), self.tmp)
        self.assertTrue(detector.detect(blank_frame()).is_empty)

    def test_model_returning_empty_result_list_is_handled(self) -> None:
        model = FakeYoloModel()
        model.predict = lambda **kw: []  # type: ignore[method-assign,assignment]
        detector, _ = make_detector(model, self.tmp)
        self.assertTrue(detector.detect(blank_frame()).is_empty)

    def test_single_person(self) -> None:
        detector, _ = make_detector(FakeYoloModel([ONE_PERSON]), self.tmp)
        result = detector.detect(blank_frame(), frame_index=12, timestamp=34.5)
        self.assertEqual(result.person_count, 1)
        d = result.detections[0]
        self.assertEqual((d.class_id, d.class_name), (0, "person"))
        self.assertEqual(d.bbox_xyxy, (10.0, 10.0, 40.0, 40.0))
        self.assertAlmostEqual(d.confidence, 0.91)
        self.assertEqual((d.frame_index, d.timestamp, result.frame_index, result.timestamp), (12, 34.5, 12, 34.5))

    def test_multiple_people_in_one_frame_sorted_by_confidence(self) -> None:
        detector, _ = make_detector(FakeYoloModel([THREE_PEOPLE]), self.tmp)
        result = detector.detect(blank_frame(320, 240))
        self.assertEqual(result.person_count, 3)
        self.assertEqual([d.confidence for d in result.detections], [0.95, 0.85, 0.7])
        self.assertEqual(len({d.bbox_xyxy for d in result.detections}), 3)

    def test_non_person_classes_filtered_even_if_model_ignores_classes_argument(self) -> None:
        mixed = [((0, 0, 30, 30), 0.99, 2), ((40, 0, 90, 60), 0.8, 0), ((100, 0, 130, 50), 0.97, 1)]
        detector, _ = make_detector(FakeYoloModel([mixed], honor_classes=False), self.tmp)
        result = detector.detect(blank_frame(320, 240))
        self.assertEqual([d.class_name for d in result.detections], ["person"])
        self.assertEqual(result.detections[0].bbox_xyxy, (40.0, 0.0, 90.0, 60.0))

    def test_only_person_requested_from_model(self) -> None:
        model = FakeYoloModel([ONE_PERSON], honor_classes=True)
        detector, _ = make_detector(model, self.tmp)
        detector.detect(blank_frame())
        self.assertEqual(model.predict_calls[0]["classes"], [0])

    def test_confidence_scores_preserved_exactly(self) -> None:
        detector, _ = make_detector(FakeYoloModel([[((0, 0, 30, 30), 0.123456, 0), ((40, 0, 90, 60), 0.987654, 0)]]), self.tmp)
        confs = sorted(d.confidence for d in detector.detect(blank_frame(320, 240)).detections)
        self.assertEqual(confs, [0.123456, 0.987654])

    def test_boxes_outside_image_are_clipped_and_invalid_ones_counted(self) -> None:
        raw = [((-10, -10, 20, 40), 0.9, 0), ((60, 40, 90, 70), 0.8, 0), ((300, 300, 400, 400), 0.7, 0), ((5, 5, 5, 30), 0.6, 0)]
        detector, _ = make_detector(FakeYoloModel([raw]), self.tmp)
        result = detector.detect(blank_frame(64, 48))
        self.assertEqual(result.image_size, (64, 48))
        self.assertEqual(result.dropped_boxes, 2)  # tamamen dışarıda + dejenere
        boxes = sorted(d.bbox_xyxy for d in result.detections)
        self.assertEqual(boxes, [(0.0, 0.0, 20.0, 40.0), (60.0, 40.0, 64.0, 48.0)])

    def test_torch_like_tensor_outputs_are_converted_via_cpu(self) -> None:
        model = FakeYoloModel([THREE_PEOPLE], as_tensor=True)
        detector, _ = make_detector(model, self.tmp)
        result = detector.detect(blank_frame(320, 240))
        self.assertEqual(result.person_count, 3)

    def test_input_image_is_not_modified(self) -> None:
        image = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
        before = image.copy()
        detector, _ = make_detector(FakeYoloModel([ONE_PERSON]), self.tmp)
        detector.detect(image)
        self.assertTrue(np.array_equal(image, before))

    def test_model_receives_bgr_image_and_config_options(self) -> None:
        model = FakeYoloModel([ONE_PERSON])
        spy = LoaderSpy(model)
        cfg = DetectionConfig(model_path=fake_weights_file(self.tmp), conf=0.4, iou=0.55, imgsz=320, device="cpu", max_det=7)
        detector = PersonDetector(cfg, model_loader=spy)
        image = blank_frame(64, 48)
        detector.detect(image)
        call = model.predict_calls[0]
        self.assertIs(call["source"], image)
        self.assertEqual((call["conf"], call["iou"], call["imgsz"], call["device"], call["max_det"]), (0.4, 0.55, 320, "cpu", 7))
        self.assertIs(call["verbose"], False)

    def test_changing_confidence_changes_what_is_requested(self) -> None:
        low_model, high_model = FakeYoloModel([ONE_PERSON]), FakeYoloModel([ONE_PERSON])
        make_detector(low_model, self.tmp, conf=0.1)[0].detect(blank_frame())
        make_detector(high_model, self.tmp, conf=0.9)[0].detect(blank_frame())
        self.assertEqual((low_model.predict_calls[0]["conf"], high_model.predict_calls[0]["conf"]), (0.1, 0.9))

    def test_each_frame_triggers_exactly_one_predict(self) -> None:
        model = FakeYoloModel([ONE_PERSON])
        detector, _ = make_detector(model, self.tmp)
        for _ in range(4):
            detector.detect(blank_frame())
        self.assertEqual(len(model.predict_calls), 4)


class InvalidFrameTest(TempDirMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.model = FakeYoloModel([ONE_PERSON])
        self.detector, self.spy = make_detector(self.model, self.tmp)

    def test_invalid_images_rejected_before_model_is_touched(self) -> None:
        bad_images = {
            "None": None,
            "list": [[0, 0, 0]],
            "empty 0x0": np.zeros((0, 0, 3), dtype=np.uint8),
            "zero height": np.zeros((0, 10, 3), dtype=np.uint8),
            "zero width": np.zeros((10, 0, 3), dtype=np.uint8),
            "2-D gray": np.zeros((10, 10), dtype=np.uint8),
            "4 channels": np.zeros((10, 10, 4), dtype=np.uint8),
            "1 channel": np.zeros((10, 10, 1), dtype=np.uint8),
            "float32": np.zeros((10, 10, 3), dtype=np.float32),
            "1-D": np.zeros(30, dtype=np.uint8),
        }
        for label, image in bad_images.items():
            with self.subTest(label), self.assertRaises(InvalidFrameError):
                self.detector.detect(image)  # type: ignore[arg-type]
        self.assertEqual(self.model.predict_calls, [])
        self.assertEqual(self.spy.calls, [], "geçersiz görüntü modeli yüklemeyi de tetiklememeli")

    def test_invalid_frame_error_is_also_a_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.detector.detect(None)  # type: ignore[arg-type]

    def test_detector_keeps_working_after_invalid_frame(self) -> None:
        with self.assertRaises(InvalidFrameError):
            self.detector.detect(None)  # type: ignore[arg-type]
        self.assertEqual(self.detector.detect(blank_frame()).person_count, 1)


class DeviceBehaviourTest(TempDirMixin, unittest.TestCase):
    def test_explicit_mps_when_unavailable_raises_and_never_loads_model(self) -> None:
        def resolver(requested: str) -> ResolvedDevice:
            raise DeviceUnavailableError("MPS yok")

        detector, spy = make_detector(FakeYoloModel(), self.tmp, device="mps", device_resolver=resolver)
        with self.assertRaises(DeviceUnavailableError):
            detector.load()
        self.assertEqual(spy.calls, [], "MPS yoksa sessizce CPU'da model yüklenmemeli")
        self.assertFalse(detector.is_loaded)

    def test_auto_fallback_exposes_device_and_warning(self) -> None:
        resolver = lambda requested: ResolvedDevice("auto", "cpu", "MPS kullanılamıyor (test)")  # noqa: E731
        detector, spy = make_detector(FakeYoloModel([ONE_PERSON]), self.tmp, device="auto", device_resolver=resolver)
        result = detector.detect(blank_frame())
        self.assertEqual((detector.device, result.device), ("cpu", "cpu"))
        self.assertIn("MPS kullanılamıyor", detector.device_warning or "")
        self.assertEqual(spy.calls[0][1], "cpu")

    def test_mps_device_is_used_for_loading_prediction_and_result(self) -> None:
        resolver = lambda requested: ResolvedDevice("mps", "mps")  # noqa: E731
        synced: list[str] = []
        model = FakeYoloModel([ONE_PERSON])
        detector, spy = make_detector(model, self.tmp, device="mps", device_resolver=resolver, synchronize=synced.append)
        result = detector.detect(blank_frame())
        self.assertEqual((spy.calls[0][1], model.predict_calls[0]["device"], result.device), ("mps", "mps", "mps"))
        self.assertEqual(synced, ["mps"], "ölçüm bitmeden önce asenkron cihaz senkronize edilmeli")

    def test_synchronize_called_also_when_nothing_detected(self) -> None:
        synced: list[str] = []
        resolver = lambda requested: ResolvedDevice("mps", "mps")  # noqa: E731
        detector, _ = make_detector(FakeYoloModel([[]]), self.tmp, device="mps", device_resolver=resolver, synchronize=synced.append)
        detector.detect(blank_frame())
        self.assertEqual(synced, ["mps"])


class TimingTest(TempDirMixin, unittest.TestCase):
    def test_inference_ms_is_measured_between_two_clock_reads_in_milliseconds(self) -> None:
        # load: (0.0, 0.1); detect: başlangıç 5.000, bitiş 5.0375 -> 37.5 ms
        clock = SequenceClock([0.0, 0.1, 5.000, 5.0375])
        detector, _ = make_detector(FakeYoloModel([ONE_PERSON]), self.tmp, clock=clock)
        result = detector.detect(blank_frame())
        self.assertAlmostEqual(result.inference_ms, 37.5, places=6)

    def test_timing_covers_synchronize(self) -> None:
        events: list[str] = []
        clock_values = iter([0.0, 0.1, 1.0, 1.2])

        def clock() -> float:
            value = next(clock_values)
            events.append(f"clock:{value}")
            return value

        resolver = lambda requested: ResolvedDevice("mps", "mps")  # noqa: E731
        detector, _ = make_detector(
            FakeYoloModel([ONE_PERSON]), self.tmp, device="mps", device_resolver=resolver, clock=clock,
            synchronize=lambda device: events.append("sync"),
        )
        result = detector.detect(blank_frame())
        self.assertLess(events.index("sync"), events.index("clock:1.2"), "senkronizasyon, bitiş okumasından önce olmalı")
        self.assertAlmostEqual(result.inference_ms, 200.0, places=6)

    def test_warmup_runs_do_not_count_as_detections(self) -> None:
        model = FakeYoloModel([ONE_PERSON])
        detector, _ = make_detector(model, self.tmp)
        detector.warmup((320, 240), runs=3)
        self.assertEqual(len(model.predict_calls), 3)
        self.assertEqual(model.predict_calls[0]["source"].shape, (240, 320, 3))
        self.assertTrue(np.all(model.predict_calls[0]["source"] == 0))

    def test_warmup_rejects_bad_run_count(self) -> None:
        detector, _ = make_detector(FakeYoloModel(), self.tmp)
        for bad in (0, -1, True, 1.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                detector.warmup(runs=bad)  # type: ignore[arg-type]


class InferenceErrorTest(TempDirMixin, unittest.TestCase):
    def test_model_exception_is_wrapped(self) -> None:
        detector, _ = make_detector(FakeYoloModel(error=RuntimeError("MPS bellek yetersiz")), self.tmp)
        with self.assertRaises(InferenceError) as ctx:
            detector.detect(blank_frame())
        self.assertIn("MPS bellek yetersiz", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)

    def test_malformed_model_output_is_wrapped(self) -> None:
        model = FakeYoloModel()
        model.predict = lambda **kw: [type("R", (), {"boxes": object()})()]  # type: ignore[method-assign,assignment]
        detector, _ = make_detector(model, self.tmp)
        with self.assertRaises(InferenceError):
            detector.detect(blank_frame())


if __name__ == "__main__":
    unittest.main()
