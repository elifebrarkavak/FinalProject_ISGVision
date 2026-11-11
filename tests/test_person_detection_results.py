"""İnsan tespiti sonuç veri yapıları testleri: koordinatlar, güven skorları, kırpma, geçersiz kutular, person süzmesi."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isg_vision.human_detection.results import (  # noqa: E402
    MIN_BOX_SIDE_PX,
    DetectionResult,
    PersonDetection,
    build_detections,
    clip_box_xyxy,
)

NAMES = {0: "person", 1: "bicycle", 2: "car"}
W, H = 640, 480


def build(boxes, confs, classes, **kw):  # type: ignore[no-untyped-def]
    return build_detections(boxes, confs, classes, NAMES, person_class_id=0, image_size=(W, H), **kw)


class PersonDetectionTest(unittest.TestCase):
    def make(self, **overrides):  # type: ignore[no-untyped-def]
        values = dict(class_id=0, class_name="person", confidence=0.9, bbox_xyxy=(10, 20, 110, 220))
        values.update(overrides)
        return PersonDetection(**values)

    def test_fields_and_derived_values(self) -> None:
        d = self.make(frame_index=7, timestamp=12.5)
        self.assertEqual((d.class_id, d.class_name, d.confidence), (0, "person", 0.9))
        self.assertEqual(d.bbox_xyxy, (10.0, 20.0, 110.0, 220.0))
        self.assertEqual((d.x1, d.y1, d.x2, d.y2), (10.0, 20.0, 110.0, 220.0))
        self.assertEqual((d.width, d.height, d.area), (100.0, 200.0, 20000.0))
        self.assertEqual(d.center, (60.0, 120.0))
        self.assertEqual((d.frame_index, d.timestamp), (7, 12.5))

    def test_no_identity_fields_in_this_stage(self) -> None:
        names = set(PersonDetection.__dataclass_fields__)
        self.assertFalse({"track_id", "employee", "employee_name", "face_id", "identity"} & names)

    def test_is_immutable(self) -> None:
        with self.assertRaises(Exception):
            self.make().confidence = 0.1  # type: ignore[misc]

    def test_invalid_confidence_rejected(self) -> None:
        for bad in (-0.01, 1.01, math.nan, math.inf):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.make(confidence=bad)

    def test_invalid_boxes_rejected(self) -> None:
        for bad in ((5, 5, 5, 10), (5, 5, 10, 5), (10, 10, 5, 20), (0, 0, math.nan, 5), (0, 0, 1), (0, 0, math.inf, 3)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.make(bbox_xyxy=bad)

    def test_invalid_class_id_and_frame_index_rejected(self) -> None:
        for kw in ({"class_id": -1}, {"class_id": True}, {"class_id": 0.5}, {"frame_index": -3}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.make(**kw)


class DetectionResultTest(unittest.TestCase):
    def test_count_iteration_and_emptiness(self) -> None:
        a = PersonDetection(0, "person", 0.9, (0, 0, 10, 10))
        b = PersonDetection(0, "person", 0.8, (20, 20, 40, 60))
        result = DetectionResult((a, b), (W, H), frame_index=3, timestamp=1.0, inference_ms=12.5, device="cpu")
        self.assertEqual((result.person_count, len(result), result.is_empty), (2, 2, False))
        self.assertEqual(list(result), [a, b])
        self.assertEqual((result.image_size, result.frame_index, result.timestamp, result.inference_ms), ((W, H), 3, 1.0, 12.5))

    def test_empty_result_is_valid(self) -> None:
        result = DetectionResult((), (W, H))
        self.assertEqual((result.person_count, len(result), result.is_empty, list(result)), (0, 0, True, []))


class ClipBoxTest(unittest.TestCase):
    def test_inside_box_unchanged(self) -> None:
        self.assertEqual(clip_box_xyxy((10, 20, 110, 220), W, H), (10.0, 20.0, 110.0, 220.0))

    def test_partially_outside_box_is_clipped_not_dropped(self) -> None:
        self.assertEqual(clip_box_xyxy((-30, -10, 100, 50), W, H), (0.0, 0.0, 100.0, 50.0))
        self.assertEqual(clip_box_xyxy((600, 400, 700, 520), W, H), (600.0, 400.0, 640.0, 480.0))

    def test_box_covering_whole_image_is_clipped_to_image(self) -> None:
        self.assertEqual(clip_box_xyxy((-100, -100, 900, 900), W, H), (0.0, 0.0, 640.0, 480.0))

    def test_fully_outside_box_dropped(self) -> None:
        for box in ((-50, 10, -5, 60), (700, 10, 800, 60), (10, -80, 60, -5), (10, 500, 60, 600)):
            with self.subTest(box=box):
                self.assertIsNone(clip_box_xyxy(box, W, H))

    def test_degenerate_boxes_dropped(self) -> None:
        for box in ((10, 10, 10, 50), (10, 10, 50, 10), (50, 10, 10, 50), (10, 50, 50, 10)):
            with self.subTest(box=box):
                self.assertIsNone(clip_box_xyxy(box, W, H))

    def test_sliver_smaller_than_min_side_dropped_after_clipping(self) -> None:
        self.assertIsNone(clip_box_xyxy((639.5, 10, 700, 100), W, H))  # kırpmadan sonra 0.5 px genişlik
        self.assertIsNotNone(clip_box_xyxy((640 - MIN_BOX_SIDE_PX, 10, 700, 100), W, H))

    def test_nonfinite_and_malformed_boxes_dropped(self) -> None:
        for box in ((math.nan, 0, 10, 10), (0, 0, math.inf, 10), (0, -math.inf, 10, 10), (1, 2, 3), (1, 2, 3, 4, 5), ("a", 0, 1, 1), None):
            with self.subTest(box=box):
                self.assertIsNone(clip_box_xyxy(box, W, H))  # type: ignore[arg-type]


class BuildDetectionsTest(unittest.TestCase):
    def test_empty_output(self) -> None:
        self.assertEqual(build([], [], []), ((), 0))

    def test_single_person_keeps_confidence_and_coordinates(self) -> None:
        detections, dropped = build([(10, 20, 110, 220)], [0.8765], [0], frame_index=4, timestamp=9.5)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(detections), 1)
        d = detections[0]
        self.assertEqual(d.bbox_xyxy, (10.0, 20.0, 110.0, 220.0))
        self.assertAlmostEqual(d.confidence, 0.8765)  # güven skoru bozulmadan korunur
        self.assertEqual((d.class_id, d.class_name, d.frame_index, d.timestamp), (0, "person", 4, 9.5))

    def test_multiple_people_are_independent_boxes(self) -> None:
        boxes = [(10, 10, 60, 120), (200, 30, 280, 200), (400, 50, 470, 230)]
        detections, dropped = build(boxes, [0.7, 0.95, 0.85], [0, 0, 0])
        self.assertEqual(dropped, 0)
        self.assertEqual(len(detections), 3)
        self.assertEqual({d.bbox_xyxy for d in detections}, {tuple(map(float, b)) for b in boxes})

    def test_sorted_by_confidence_descending_with_stable_ties(self) -> None:
        detections, _ = build([(0, 0, 10, 10), (20, 20, 40, 40), (50, 50, 90, 90)], [0.5, 0.9, 0.5], [0, 0, 0])
        self.assertEqual([d.confidence for d in detections], [0.9, 0.5, 0.5])
        self.assertEqual([d.bbox_xyxy[0] for d in detections], [20.0, 0.0, 50.0])  # eşitlikte model sırası

    def test_non_person_classes_filtered_out(self) -> None:
        detections, dropped = build(
            [(0, 0, 50, 50), (60, 60, 120, 120), (130, 0, 200, 90), (210, 10, 260, 90)], [0.9, 0.99, 0.8, 0.7], [1, 0, 2, 0]
        )
        self.assertEqual(dropped, 0, "person dışı sınıflar 'geçersiz kutu' sayılmaz")
        self.assertEqual([d.confidence for d in detections], [0.99, 0.7])
        self.assertTrue(all(d.class_id == 0 and d.class_name == "person" for d in detections))

    def test_only_non_person_gives_empty(self) -> None:
        self.assertEqual(build([(0, 0, 50, 50)], [0.99], [2]), ((), 0))

    def test_float_class_ids_from_model_are_handled(self) -> None:
        detections, _ = build([(0, 0, 50, 50)], [0.9], [0.0])  # Ultralytics sınıfları float tensör olarak verir
        self.assertEqual(len(detections), 1)

    def test_nonstandard_person_id_from_metadata(self) -> None:
        names = {0: "helmet", 3: "person"}
        detections, _ = build_detections(
            [(0, 0, 50, 50), (60, 0, 120, 90)], [0.9, 0.8], [0, 3], names, person_class_id=3, image_size=(W, H)
        )
        self.assertEqual([(d.class_id, d.class_name) for d in detections], [(3, "person")])

    def test_invalid_boxes_dropped_and_counted_not_hidden(self) -> None:
        boxes = [(10, 10, 60, 120), (math.nan, 0, 5, 5), (30, 30, 30, 90), (-90, -90, -10, -10), (200, 20, 260, 100)]
        detections, dropped = build(boxes, [0.9, 0.9, 0.9, 0.9, 0.8], [0] * 5)
        self.assertEqual(len(detections), 2)
        self.assertEqual(dropped, 3)

    def test_nonfinite_confidence_dropped(self) -> None:
        detections, dropped = build([(0, 0, 50, 50), (60, 0, 120, 90)], [math.nan, 0.7], [0, 0])
        self.assertEqual((len(detections), dropped), (1, 1))

    def test_boxes_outside_image_are_clipped_to_bounds(self) -> None:
        detections, dropped = build([(-20, -5, 100, 80), (600, 400, 900, 700)], [0.9, 0.8], [0, 0])
        self.assertEqual(dropped, 0)
        for d in detections:
            self.assertTrue(0 <= d.x1 < d.x2 <= W and 0 <= d.y1 < d.y2 <= H, d.bbox_xyxy)

    def test_class_name_falls_back_to_person_if_missing_from_map(self) -> None:
        detections, _ = build_detections([(0, 0, 50, 50)], [0.9], [5], {}, person_class_id=5, image_size=(W, H))
        self.assertEqual(detections[0].class_name, "person")


if __name__ == "__main__":
    unittest.main()
