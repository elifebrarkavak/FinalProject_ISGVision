"""Yüz tespiti sonuç veri yapıları testleri: doğrulama, kırpma, kenar işareti, sıralama, kesme, 5B'ye uygunluk (model/OpenCV yok)."""

from __future__ import annotations

import dataclasses
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isg_vision.face_detection.results import BORDER_TOLERANCE_PX, FaceDetection, FaceDetectionResult, build_faces  # noqa: E402
from isg_vision.human_detection.results import PersonDetection  # noqa: E402
from isg_vision.tracking.results import TrackedPerson  # noqa: E402

SIZE = (100, 80)


def build(boxes, scores, **kw):  # type: ignore[no-untyped-def]
    return build_faces(boxes, scores, image_size=kw.pop("image_size", SIZE), **kw)


class FaceDetectionTest(unittest.TestCase):
    def test_valid_face_and_properties(self) -> None:
        f = FaceDetection(0.9, (10, 20, 50, 60), frame_index=3, timestamp=1.5, image_size=SIZE, face_index=0)
        self.assertEqual((f.x1, f.y1, f.x2, f.y2), (10.0, 20.0, 50.0, 60.0))
        self.assertEqual((f.width, f.height, f.area, f.center), (40.0, 40.0, 1600.0, (30.0, 40.0)))
        self.assertEqual((f.frame_index, f.timestamp, f.image_size, f.face_index, f.at_border), (3, 1.5, SIZE, 0, False))

    def test_all_values_are_floats_after_construction(self) -> None:
        f = FaceDetection(1, (1, 2, 3, 4))
        self.assertTrue(all(isinstance(v, float) for v in f.bbox_xyxy))

    def test_invalid_confidence_rejected(self) -> None:
        for bad in (-0.01, 1.01, float("nan"), float("inf"), "0.5", None, True):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                FaceDetection(bad, (1, 1, 5, 5))  # type: ignore[arg-type]

    def test_invalid_boxes_rejected(self) -> None:
        for bad in ((5, 5, 5, 9), (5, 5, 9, 5), (9, 5, 5, 9), (1, 2, 3), (1, 2, 3, 4, 5), (math.nan, 1, 5, 5), (1, 1, math.inf, 5)):
            with self.subTest(box=bad), self.assertRaises(ValueError):
                FaceDetection(0.5, bad)  # type: ignore[arg-type]

    def test_invalid_indices_rejected(self) -> None:
        for kw in ({"frame_index": -1}, {"face_index": -1}, {"frame_index": True}, {"face_index": True}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                FaceDetection(0.5, (1, 1, 5, 5), **kw)  # type: ignore[arg-type]

    def test_invalid_image_size_rejected(self) -> None:
        for bad in ((0, 10), (10, 0), (10,), (10, 10, 3), (10.5, 10), (True, 10), (-5, 10)):
            with self.subTest(size=bad), self.assertRaises(ValueError):
                FaceDetection(0.5, (1, 1, 5, 5), image_size=bad)  # type: ignore[arg-type]

    def test_box_must_be_inside_image_size(self) -> None:
        for box in ((-1, 0, 10, 10), (0, -1, 10, 10), (0, 0, 101, 10), (0, 0, 10, 81)):
            with self.subTest(box=box), self.assertRaises(ValueError):
                FaceDetection(0.5, box, image_size=SIZE)
        FaceDetection(0.5, (0, 0, 100, 80), image_size=SIZE)  # tam kare serbest

    def test_is_immutable(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            FaceDetection(0.5, (1, 1, 5, 5)).confidence = 0.1  # type: ignore[misc]


class BuildFacesTest(unittest.TestCase):
    def test_empty_input(self) -> None:
        self.assertEqual(build([], []), ((), 0, 0))

    def test_single_face(self) -> None:
        faces, dropped, truncated = build([(10, 20, 40, 60)], [0.93], frame_index=5, timestamp=2.0)
        self.assertEqual((dropped, truncated, len(faces)), (0, 0, 1))
        f = faces[0]
        self.assertEqual((f.bbox_xyxy, f.confidence, f.frame_index, f.timestamp, f.image_size, f.face_index), ((10.0, 20.0, 40.0, 60.0), 0.93, 5, 2.0, SIZE, 0))

    def test_multiple_faces_sorted_by_confidence_with_stable_ties_and_face_index(self) -> None:
        boxes = [(0, 0, 10, 10), (20, 0, 30, 10), (40, 0, 50, 10), (60, 0, 70, 10)]
        faces, _, _ = build(boxes, [0.7, 0.9, 0.7, 0.8])
        self.assertEqual([f.confidence for f in faces], [0.9, 0.8, 0.7, 0.7])
        self.assertEqual([f.face_index for f in faces], [0, 1, 2, 3])
        self.assertEqual([f.bbox_xyxy[0] for f in faces], [20.0, 60.0, 0.0, 40.0])  # eşitlikte giriş sırası korunur

    def test_partially_visible_face_is_clipped_not_dropped_and_marked(self) -> None:
        faces, dropped, _ = build([(-20, 10, 20, 50)], [0.8])
        self.assertEqual(dropped, 0)
        self.assertEqual(faces[0].bbox_xyxy, (0.0, 10.0, 20.0, 50.0))
        self.assertTrue(faces[0].at_border)

    def test_all_four_borders_are_detected(self) -> None:
        for box in ((0, 20, 20, 40), (30, 0, 50, 20), (80, 20, 100, 40), (30, 60, 50, 80), (-5, 20, 20, 40), (30, 70, 50, 95)):
            with self.subTest(box=box):
                faces, _, _ = build([box], [0.8])
                self.assertTrue(faces[0].at_border)

    def test_border_tolerance_boundary(self) -> None:
        inside, _, _ = build([(BORDER_TOLERANCE_PX + 1, 20, 40, 40)], [0.8])
        near, _, _ = build([(BORDER_TOLERANCE_PX, 20, 40, 40)], [0.8])
        self.assertFalse(inside[0].at_border)
        self.assertTrue(near[0].at_border)

    def test_face_well_inside_is_not_at_border(self) -> None:
        faces, _, _ = build([(30, 20, 60, 60)], [0.9])
        self.assertFalse(faces[0].at_border)

    def test_box_fully_outside_or_degenerate_is_dropped_and_counted(self) -> None:
        boxes = [(200, 10, 260, 50), (-50, 10, -10, 50), (10, 10, 10, 50), (30, 30, 20, 40), (10, 10, 10.5, 50), (math.nan, 1, 5, 5), (1, 1, math.inf, 5)]
        faces, dropped, _ = build(boxes, [0.9] * len(boxes))
        self.assertEqual((len(faces), dropped), (0, len(boxes)))

    def test_nonfinite_or_nonnumeric_scores_are_dropped(self) -> None:
        faces, dropped, _ = build([(0, 0, 10, 10)] * 3, [math.nan, math.inf, "abc"])  # type: ignore[list-item]
        self.assertEqual((len(faces), dropped), (0, 3))

    def test_scores_are_clamped_to_unit_interval(self) -> None:
        faces, _, _ = build([(0, 0, 10, 10), (20, 0, 30, 10)], [1.0000001, -0.0000001])
        self.assertEqual(sorted(f.confidence for f in faces), [0.0, 1.0])

    def test_max_faces_truncates_lowest_confidence_and_counts(self) -> None:
        boxes = [(i * 10, 0, i * 10 + 8, 8) for i in range(5)]
        faces, dropped, truncated = build(boxes, [0.5, 0.9, 0.6, 0.8, 0.7], max_faces=2)
        self.assertEqual((len(faces), dropped, truncated), (2, 0, 3))
        self.assertEqual([f.confidence for f in faces], [0.9, 0.8])
        self.assertEqual([f.face_index for f in faces], [0, 1])

    def test_max_faces_larger_than_count_truncates_nothing(self) -> None:
        faces, _, truncated = build([(0, 0, 10, 10)], [0.9], max_faces=50)
        self.assertEqual((len(faces), truncated), (1, 0))

    def test_every_face_carries_frame_metadata(self) -> None:
        faces, _, _ = build([(0, 0, 10, 10), (20, 0, 30, 10)], [0.9, 0.8], frame_index=9, timestamp=4.25)
        for f in faces:
            self.assertEqual((f.frame_index, f.timestamp, f.image_size), (9, 4.25, SIZE))

    def test_works_for_various_image_sizes(self) -> None:
        for size in ((1, 1), (16, 16), (640, 480), (1920, 1080), (4000, 30)):
            with self.subTest(size=size):
                faces, dropped, _ = build_faces([(0, 0, size[0], size[1])], [0.9], image_size=size)
                self.assertEqual((len(faces), dropped, faces[0].bbox_xyxy), (1, 0, (0.0, 0.0, float(size[0]), float(size[1]))))


class FaceDetectionResultTest(unittest.TestCase):
    def test_len_iter_and_flags(self) -> None:
        faces, _, _ = build([(0, 0, 10, 10), (20, 0, 30, 10)], [0.9, 0.8])
        r = FaceDetectionResult(faces, SIZE, frame_index=1, timestamp=0.5, inference_ms=7.0)
        self.assertEqual((r.face_count, len(r), r.is_empty, list(r) == list(faces)), (2, 2, False, True))
        self.assertEqual((r.device, r.dropped_boxes, r.truncated_faces, r.inference_size), ("cpu", 0, 0, None))

    def test_empty_result_is_not_an_error(self) -> None:
        r = FaceDetectionResult((), SIZE)
        self.assertEqual((r.face_count, r.is_empty, len(r), list(r)), (0, True, 0, []))

    def test_result_is_immutable(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            FaceDetectionResult((), SIZE).image_size = (1, 1)  # type: ignore[misc]


class StageFiveBContractTest(unittest.TestCase):
    """Aşama 5B'nin yüz ↔ insan kutusu ilişkilendirmesine uygun, ama onu YAPMAYAN çıktı sözleşmesi."""

    def test_face_box_uses_the_same_coordinate_convention_as_person_and_track_boxes(self) -> None:
        face = FaceDetection(0.9, (30, 10, 50, 30), frame_index=4, image_size=SIZE)
        person = PersonDetection(0, "person", 0.8, (20, 5, 70, 78), frame_index=4)
        track = TrackedPerson(1, 0, "person", 0.8, (20.0, 5.0, 70.0, 78.0), 4, None, 1, 0, 1)
        for other in (person, track):
            fx, fy = face.center
            self.assertTrue(other.bbox_xyxy[0] <= fx <= other.bbox_xyxy[2] and other.bbox_xyxy[1] <= fy <= other.bbox_xyxy[3])
        self.assertEqual(face.frame_index, person.frame_index)
        self.assertEqual(face.frame_index, track.frame_index)

    def test_result_exposes_the_required_fields(self) -> None:
        names = {f.name for f in dataclasses.fields(FaceDetection)}
        self.assertTrue({"bbox_xyxy", "confidence", "frame_index", "timestamp", "image_size"} <= names)
        self.assertTrue({"image_size", "frame_index", "timestamp", "faces"} <= {f.name for f in dataclasses.fields(FaceDetectionResult)})

    def test_no_identity_biometric_or_tracking_fields_exist(self) -> None:
        forbidden = ("track", "identity", "employee", "name", "embedding", "template", "landmark", "person", "recogn", "liveness")
        for cls in (FaceDetection, FaceDetectionResult):
            for field in dataclasses.fields(cls):
                with self.subTest(cls=cls.__name__, field=field.name):
                    self.assertFalse(any(word in field.name.lower() for word in forbidden), field.name)

    def test_face_index_matches_position_in_result(self) -> None:
        faces, _, _ = build([(0, 0, 10, 10), (20, 0, 30, 10), (40, 0, 50, 10)], [0.5, 0.9, 0.7])
        r = FaceDetectionResult(faces, SIZE)
        for position, face in enumerate(r.faces):
            self.assertEqual(face.face_index, position)


if __name__ == "__main__":
    unittest.main()
