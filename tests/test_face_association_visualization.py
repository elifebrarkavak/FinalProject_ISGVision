"""Yüz–kişi görselleştirme: çizim girdiyi değiştirmez, her boyutta çalışır, etiketler/uyarılar doğru, eşleşmeyen yüze Track ID yazılmaz [Mock/birim; pencere açılmaz]."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from face_association_test_support import faces, head_box, person_box, tracking  # noqa: E402

from isg_vision.face_association import AssociationConfig, FaceAssociationResult, FacePersonAssociator  # noqa: E402
from isg_vision.face_association import visualization as viz  # noqa: E402
from isg_vision.face_association.visualization import (  # noqa: E402
    LEGEND,
    NOTICE,
    AssociationOverlayInfo,
    draw_face_associations,
    face_label,
    person_label,
)
from isg_vision.tracking.visualization import color_for_track  # noqa: E402

INFO = AssociationOverlayInfo("CAMERA 0", 29.7, 12.3, 0.4, 10.8, 0.12, (1280, 720), "mps")
A, B, C = person_box(250), person_box(640), person_box(1030)


def textured(width: int = 1280, height: int = 720) -> np.ndarray:
    return np.random.default_rng(5).integers(0, 255, (height, width, 3), dtype=np.uint8)


class Spy:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self._real = cv2.putText

    def __call__(self, img, text, *a, **k):  # type: ignore[no-untyped-def]
        self.texts.append(text)
        return self._real(img, text, *a, **k)


def draw(image, result, info=INFO):  # type: ignore[no-untyped-def]
    spy = Spy()
    with mock.patch.object(viz.cv2, "putText", spy):
        out = draw_face_associations(image, result, info)
    return out, spy.texts


def mixed_result(**config) -> FaceAssociationResult:  # type: ignore[no-untyped-def]
    """A: eşleşir; B: yüzü yok; C: iki yinelenen kutu -> belirsiz; ayrıca eşleşmeyen bir yüz."""
    dup = person_box(1030)
    assoc = FacePersonAssociator(AssociationConfig(**config))
    people = [(1, A), (2, B), (3, dup), (4, dup)]
    return assoc.associate(tracking(people), faces([head_box(A), head_box(dup), (600.0, 600.0, 640.0, 640.0)]))


class LabelTest(unittest.TestCase):
    def test_person_labels_by_status(self) -> None:
        r = mixed_result()
        by_id = {p.track_id: p for p in r.persons}
        self.assertEqual(person_label(by_id[1]), "ID 1 FACE*")  # yeni iz: süreklilik henüz güvenilir değil
        self.assertEqual(person_label(by_id[2]), "ID 2 NO FACE")
        self.assertEqual(person_label(by_id[3]), "ID 3 FACE ?")

    def test_trusted_match_has_no_star_and_untrusted_does(self) -> None:
        assoc = FacePersonAssociator(AssociationConfig(min_trusted_frames=2))
        labels = []
        for f in range(3):
            r = assoc.associate(tracking([(1, A)], f), faces([head_box(A)], f))
            labels.append(person_label(r.persons[0]))
        self.assertEqual(labels, ["ID 1 FACE*", "ID 1 FACE", "ID 1 FACE"])

    def test_static_flag_is_shown_but_as_a_question(self) -> None:
        assoc = FacePersonAssociator(AssociationConfig(stationary_flag_frames=2))
        for f in range(4):
            r = assoc.associate(tracking([(1, A)], f), faces([head_box(A)], f))
        self.assertIn("STATIC?", person_label(r.persons[0]))

    def test_matched_face_label_names_the_track_and_others_never_do(self) -> None:
        r = mixed_result()
        by_idx = {f.face_index: f for f in r.faces}
        self.assertTrue(face_label(by_idx[0]).startswith("FACE>ID 1 "))
        for idx in (1, 2):  # belirsiz ve eşleşmeyen yüz
            label = face_label(by_idx[idx])
            self.assertNotRegex(label, r"ID \d", "eşleşmeyen/belirsiz yüze Track ID YAZILMAZ")
        self.assertIn("ambiguous", face_label(by_idx[1]))
        self.assertIn("unmatched", face_label(by_idx[2]))

    def test_edge_marker(self) -> None:
        r = FacePersonAssociator().associate(tracking([(1, person_box(120, top=0))]), faces([(90.0, 0.0, 140.0, 40.0)], at_border=[True]))
        self.assertIn("EDGE", face_label(r.faces[0]))

    def test_labels_are_plain_ascii(self) -> None:
        r = mixed_result()
        for text in [person_label(p) for p in r.persons] + [face_label(f) for f in r.faces] + [NOTICE, LEGEND]:
            self.assertTrue(text.isascii(), text)


class DrawTest(unittest.TestCase):
    def test_returns_a_copy_and_never_modifies_the_input(self) -> None:
        image = textured()
        before = image.copy()
        out = draw_face_associations(image, mixed_result(), INFO)
        np.testing.assert_array_equal(image, before)
        self.assertIsNot(out, image)
        self.assertEqual((out.shape, out.dtype), (image.shape, image.dtype))
        self.assertFalse(np.array_equal(out, image))

    def test_status_strip_shows_all_required_numbers_and_timings(self) -> None:
        _, texts = draw(textured(), mixed_result())
        joined = "\n".join(texts)
        for expected in ("PERSONS: 4", "FACES: 3", "MATCHED: 1", "NO FACE: 1", "AMBIGUOUS: 2", "UNMATCHED FACES: 1", "FPS: 29.7", "YOLO (MPS): 12 ms",
                         "BYTETRACK: 0.4 ms", "YUNET: 11 ms", "ASSOC: 0.12 ms", "SOURCE: CAMERA 0", "1280x720"):
            self.assertIn(expected, joined)

    def test_notice_and_exit_hint_are_always_drawn_even_when_empty(self) -> None:
        empty = FaceAssociationResult((), (), (1280, 720), 0, 0.0, 1)
        _, texts = draw(textured(), empty)
        self.assertIn(NOTICE, texts)
        self.assertIn(viz.EXIT_HINT, texts)
        self.assertTrue(any("PERSONS: 0" in t for t in texts))

    def test_labels_for_every_person_and_face_are_drawn(self) -> None:
        r = mixed_result()
        _, texts = draw(textured(), r)
        for text in [person_label(p) for p in r.persons] + [face_label(f) for f in r.faces]:
            self.assertIn(text, texts)

    def test_matched_face_is_drawn_in_its_persons_color(self) -> None:
        image = np.zeros((720, 1280, 3), np.uint8)
        assoc = FacePersonAssociator()
        r = assoc.associate(tracking([(3, A)]), faces([head_box(A)]))
        out = draw_face_associations(image, r, INFO)
        x1, y1 = (int(round(v)) for v in r.persons[0].face_bbox[:2])  # type: ignore[index]
        patch = out[y1 - 2 : y1 + 3, x1 - 2 : x1 + 12].reshape(-1, 3)
        color = color_for_track(3)
        self.assertTrue(any(tuple(int(v) for v in px) == color for px in patch), "yüz köşe ayracı kişinin renginde olmalı")

    def test_unmatched_and_ambiguous_faces_use_distinct_non_track_colors(self) -> None:
        self.assertNotEqual(viz._AMBER, viz._MAGENTA)
        palette = {tuple(c) for c in __import__("isg_vision.tracking.visualization", fromlist=["TRACK_PALETTE"]).TRACK_PALETTE}
        self.assertNotIn(viz._AMBER, palette)
        self.assertNotIn(viz._MAGENTA, palette)

    def test_invalid_input_result_is_drawn_with_a_warning_and_no_track_ids_on_faces(self) -> None:
        r = FacePersonAssociator().associate(tracking([(1, A)]), faces([(100.0, 50.0, 140.0, 90.0)], 0, size=(640, 480)))
        self.assertFalse(r.input_valid)
        out, texts = draw(textured(), r)
        self.assertTrue(any("MISMATCH" in t for t in texts))
        self.assertTrue(all("FACE>" not in t for t in texts))
        self.assertEqual(out.shape, (720, 1280, 3))

    def test_note_is_shown(self) -> None:
        _, texts = draw(textured(), mixed_result(), replace(INFO, note="requested 1920x1080, got 1280x720"))
        self.assertTrue(any("NOTE: requested 1920x1080" in t for t in texts))

    def test_missing_timings_render_as_dashes(self) -> None:
        info = replace(INFO, yolo_ms=None, tracking_ms=None, face_ms=None, association_ms=None, device=None)
        _, texts = draw(textured(), mixed_result(), info)
        self.assertEqual(sum(1 for t in texts if t.endswith("-- ms")), 4)

    def test_works_at_small_and_large_sizes_and_with_boxes_on_the_edges(self) -> None:
        for size in ((64, 48), (160, 120), (640, 480), (1920, 1080)):
            w, h = size
            p = (0.0, 0.0, w * 0.5, h * 1.0)
            f = (2.0, 0.0, w * 0.2, h * 0.15)
            r = FacePersonAssociator().associate(tracking([(1, p)], size=size), faces([f], size=size, at_border=[True]))
            with self.subTest(size=size):
                out = draw_face_associations(textured(w, h), r, replace(INFO, resolution=size))
                self.assertEqual(out.shape, (h, w, 3))

    def test_many_people_do_not_crash(self) -> None:
        people = [(i + 1, (10.0 + 40 * i, 100.0, 40.0 + 40 * i, 400.0)) for i in range(30)]
        r = FacePersonAssociator().associate(tracking(people), faces([]))
        self.assertEqual(draw_face_associations(textured(), r, INFO).shape, (720, 1280, 3))

    def test_no_window_is_opened(self) -> None:
        with mock.patch.object(cv2, "namedWindow", side_effect=AssertionError("pencere açıldı")), mock.patch.object(cv2, "imshow", side_effect=AssertionError("imshow")):
            draw_face_associations(textured(), mixed_result(), INFO)


if __name__ == "__main__":
    unittest.main()
