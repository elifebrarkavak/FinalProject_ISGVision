"""Sentetik sahne üretici ve ground-truth değerlendirme [model GEREKTİRMEZ: yalnızca OpenCV/NumPy; Ultralytics örnek görüntüleri (indirme yok)].

Sahneler MONTAJDIR (``scenes`` modül açıklaması): gerçek çoklu-kişi videosu değildir. Örnek görüntüler yoksa testler açık nedenle ATLANIR.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from face_association_test_support import faces as make_faces, tracking  # noqa: E402

from isg_vision.face_association import FaceAssociationResult, FacePersonAssociator, scenes  # noqa: E402
from isg_vision.face_association.evaluation import evaluate_scene  # noqa: E402
from isg_vision.face_association.geometry import iou  # noqa: E402
from isg_vision.face_association.results import REASON_GEOMETRY_OK, STATUS_MATCHED, AssociatedFace, AssociatedPerson  # noqa: E402

try:
    ACTORS = scenes.load_actors()
except scenes.SceneError as exc:  # örnek görüntüler yok
    ACTORS = None
    SKIP = str(exc)
else:
    SKIP = ""


@unittest.skipIf(ACTORS is None, f"örnek görüntüler yok: {SKIP}")
class RenderTest(unittest.TestCase):
    def test_actor_source_boxes_are_consistent_with_their_crops(self) -> None:
        for name, actor in ACTORS.items():  # type: ignore[union-attr]
            h, w = actor.crop.shape[:2]
            px1, py1, px2, py2 = actor.person_box
            fx1, fy1, fx2, fy2 = actor.face_box
            with self.subTest(actor=name):
                self.assertTrue(0 <= px1 < px2 <= w and 0 <= py1 < py2 <= h)
                self.assertTrue(px1 <= fx1 < fx2 <= px2 and py1 <= fy1 < fy2 <= py2, "yüz kutusu kişi kutusunun içinde olmalı")
                self.assertLess((fy1 + fy2) / 2 - py1, 0.3 * (py2 - py1), "yüz kişinin üst bölgesinde")

    def test_render_places_the_person_box_exactly_where_requested(self) -> None:
        frame = scenes.render_frame(ACTORS, [scenes.Placement("A", 300, 100, 400)])  # type: ignore[arg-type]
        truth = frame.truth["A"]
        x1, y1, x2, y2 = truth.person_box
        self.assertAlmostEqual(x1, 300, delta=1.5)
        self.assertAlmostEqual(y1, 100, delta=1.5)
        self.assertAlmostEqual(y2 - y1, 400, delta=2.0)
        self.assertEqual(truth.visible_fraction, 1.0)
        self.assertIsNotNone(truth.face_box)
        self.assertEqual(frame.image.shape, (720, 1280, 3))

    def test_truth_face_is_inside_the_truth_person_box_and_in_its_upper_region(self) -> None:
        for placement in (scenes.Placement("A", 100, 130), scenes.Placement("B", 700, 150, 380), scenes.Placement("C", 500, 120)):
            frame = scenes.render_frame(ACTORS, [placement])  # type: ignore[arg-type]
            t = frame.truth[placement.name]
            self.assertIsNotNone(t.face_box)
            fb, pb = t.face_box, t.person_box
            self.assertTrue(pb[0] <= fb[0] and fb[2] <= pb[2] and pb[1] <= fb[1] and fb[3] <= pb[3])  # type: ignore[index]
            self.assertLess(((fb[1] + fb[3]) / 2 - pb[1]) / (pb[3] - pb[1]), 0.3)  # type: ignore[index]

    def test_the_face_region_really_contains_the_face_pixels(self) -> None:
        """Gerçek (ground-truth) yüz kutusu, kaynak kırpıntıdaki yüz pikselleriyle örtüşüyor: maskesizken içerik, maskeliyken dolgu rengi."""
        open_face = scenes.render_frame(ACTORS, [scenes.Placement("A", 300, 130)])  # type: ignore[arg-type]
        masked = scenes.render_frame(ACTORS, [scenes.Placement("A", 300, 130, face_visible=False)])  # type: ignore[arg-type]
        x1, y1, x2, y2 = (int(v) for v in open_face.truth["A"].face_box)  # type: ignore[union-attr]
        patch_open = open_face.image[y1:y2, x1:x2].astype(float)
        patch_masked = masked.image[y1:y2, x1:x2].astype(float)
        self.assertGreater(float(np.abs(patch_open - patch_masked).mean()), 20.0, "maskeleme yüz piksellerini değiştirmeli")
        self.assertLess(float(patch_masked.mean()), 60.0, "maske koyu dolgu olmalı")
        self.assertIsNone(masked.truth["A"].face_box, "maskeli yüz GÖRÜNMEZ sayılır")
        self.assertEqual(masked.truth["A"].visible_fraction, 1.0, "beden hâlâ görünür")

    def test_a_person_in_front_hides_the_one_behind(self) -> None:
        frame = scenes.render_frame(ACTORS, [scenes.Placement("B", 560, 140, 440), scenes.Placement("A", 540, 120, 500)])  # type: ignore[arg-type]
        self.assertLess(frame.truth["B"].visible_fraction, 0.5)
        self.assertIsNone(frame.truth["B"].face_box, "yüzü örtülen kişinin gerçek yüzü görünmez sayılır")
        self.assertIsNotNone(frame.truth["A"].face_box)
        order = scenes.render_frame(ACTORS, [scenes.Placement("A", 540, 120, 500), scenes.Placement("B", 560, 140, 440)])  # type: ignore[arg-type]
        self.assertLess(order.truth["A"].visible_fraction, 0.5, "çizim sırası örtüşmeyi belirler")

    def test_person_outside_the_canvas_has_no_truth(self) -> None:
        frame = scenes.render_frame(ACTORS, [scenes.Placement("A", 1500, 100)])  # type: ignore[arg-type]
        self.assertEqual(frame.truth, {})
        partial = scenes.render_frame(ACTORS, [scenes.Placement("A", 1200, 100)])  # type: ignore[arg-type]
        self.assertIn("A", partial.truth)
        self.assertLessEqual(partial.truth["A"].person_box[2], 1280.0)

    def test_photo_and_poster_are_decoys_not_people(self) -> None:
        frame = scenes.render_frame(ACTORS, [scenes.Placement("photo:A", 900, 100, 120), scenes.Placement("poster:C", 50, 150, 250), scenes.Placement("B", 500, 130)])  # type: ignore[arg-type]
        self.assertEqual(sorted(frame.truth), ["B"])
        self.assertEqual(sorted(frame.decoys), ["photo:A", "poster:C"])

    def test_rendering_is_deterministic(self) -> None:
        a = scenes.render_frame(ACTORS, scenes.three_people(5)[2])  # type: ignore[arg-type]
        b = scenes.render_frame(ACTORS, scenes.three_people(5)[2])  # type: ignore[arg-type]
        np.testing.assert_array_equal(a.image, b.image)
        self.assertEqual(a.truth, b.truth)

    def test_every_named_scenario_renders_with_a_consistent_frame_count(self) -> None:
        for name, make in scenes.SCENARIOS.items():
            with self.subTest(scenario=name):
                scenario = make()
                frames = scenes.render_scenario(ACTORS, scenario)  # type: ignore[arg-type]
                self.assertEqual(len(frames), len(scenario))
                self.assertGreaterEqual(len(frames), 30)
                self.assertTrue(any(f.truth for f in frames))

    def test_leave_and_return_really_leaves_the_frame(self) -> None:
        frames = scenes.render_scenario(ACTORS, scenes.leave_and_return(80, 25))  # type: ignore[arg-type]
        present = [("A" in f.truth) for f in frames]
        self.assertTrue(present[0] and present[-1])
        self.assertGreaterEqual(present.count(False), 20, "A kadraj dışında kalmalı")

    def test_swap_places_makes_boxes_overlap_heavily_in_the_middle(self) -> None:
        frames = scenes.render_scenario(ACTORS, scenes.swap_places(60))  # type: ignore[arg-type]
        from isg_vision.face_association.geometry import overlap_fraction

        peak = max(overlap_fraction(f.truth["A"].person_box, f.truth["B"].person_box) for f in frames if len(f.truth) == 2)
        self.assertGreater(peak, 0.5)


def result_with(persons: list[AssociatedPerson], faces: list[AssociatedFace]) -> FaceAssociationResult:
    return FaceAssociationResult(tuple(persons), tuple(faces), (1280, 720), 0, 0.0, 1)


@unittest.skipIf(ACTORS is None, f"örnek görüntüler yok: {SKIP}")
class EvaluationTest(unittest.TestCase):
    def perfect(self, scenario: scenes.Scenario) -> tuple[list[scenes.SceneFrame], list[FaceAssociationResult]]:
        """Kusursuz takipçi + kusursuz yüz tespitçisi (gerçek kutular) -> ilişkilendirme sonuçları."""
        frames = scenes.render_scenario(ACTORS, scenario)  # type: ignore[arg-type]
        assoc, results = FacePersonAssociator(), []
        ids = {"A": 1, "B": 2, "C": 3}
        for i, f in enumerate(frames):
            people = [(ids[n], t.person_box) for n, t in f.truth.items()]
            face_boxes = [t.face_box for t in f.truth.values() if t.face_box is not None]
            results.append(assoc.associate(tracking(people, i), make_faces(face_boxes, i)))
        return frames, results

    def test_perfect_inputs_give_perfect_scores(self) -> None:
        for scenario in (scenes.single_person(10), scenes.side_by_side(10), scenes.three_people(10)):
            frames, results = self.perfect(scenario)
            ev = evaluate_scene(frames, results)
            n_faces = sum(1 for f in frames for t in f.truth.values() if t.face_box is not None)
            self.assertEqual((ev.wrong_pairs, ev.decoy_pairs, ev.decoy_on_person_pairs, ev.correct_pairs), (0, 0, 0, n_faces))
            self.assertEqual((ev.expected, ev.recalled), (n_faces, n_faces))
            self.assertTrue(ev.id_report.is_continuous)
            self.assertEqual(ev.recall, 1.0)
            self.assertEqual(ev.wrong_rate, 0.0)

    def test_face_not_detected_is_separated_from_association_misses(self) -> None:
        frames = scenes.render_scenario(ACTORS, scenes.single_person(6))  # type: ignore[arg-type]
        assoc = FacePersonAssociator()
        results = [assoc.associate(tracking([(1, f.truth["A"].person_box)], i), make_faces([], i)) for i, f in enumerate(frames)]
        ev = evaluate_scene(frames, results)
        self.assertEqual((ev.expected, ev.recalled, ev.face_not_detected, ev.missed, ev.ambiguous), (6, 0, 6, 0, 0))

    def test_a_manually_swapped_pair_is_counted_as_wrong(self) -> None:
        frames = scenes.render_scenario(ACTORS, scenes.side_by_side(3))  # type: ignore[arg-type]
        results = []
        for i, f in enumerate(frames):
            a, b = f.truth["A"], f.truth["B"]
            fa, fb = a.face_box, b.face_box
            persons = [
                AssociatedPerson(1, 1, i, 0.0, a.person_box, 0.9, STATUS_MATCHED, REASON_GEOMETRY_OK, 0, fb, 0.9, False, 0.9),   # A'nın izine B'nin yüzü
                AssociatedPerson(1, 2, i, 0.0, b.person_box, 0.9, STATUS_MATCHED, REASON_GEOMETRY_OK, 1, fa, 0.9, False, 0.9),   # B'nin izine A'nın yüzü
            ]
            fcs = [AssociatedFace(0, i, 0.0, fb, 0.9, False, STATUS_MATCHED, REASON_GEOMETRY_OK, 1, 1, 0.9),
                   AssociatedFace(1, i, 0.0, fa, 0.9, False, STATUS_MATCHED, REASON_GEOMETRY_OK, 1, 2, 0.9)]
            results.append(result_with(persons, fcs))
        ev = evaluate_scene(frames, results)
        self.assertEqual((ev.correct_pairs, ev.wrong_pairs), (0, 6))
        self.assertEqual(ev.wrong_rate, 1.0)
        self.assertEqual(len(ev.wrong_examples), 6)

    def test_decoy_face_on_a_real_person_is_reported_separately_from_wrong_pairs(self) -> None:
        frames = scenes.render_scenario(ACTORS, [[scenes.Placement("B", 500, 130), scenes.Placement("photo:A", 540, 300, 70)]])  # type: ignore[arg-type]
        f = frames[0]
        b, photo = f.truth["B"], f.decoys["photo:A"]
        person = AssociatedPerson(1, 1, 0, 0.0, b.person_box, 0.9, STATUS_MATCHED, REASON_GEOMETRY_OK, 0, photo, 0.9, False, 0.9)
        face = AssociatedFace(0, 0, 0.0, photo, 0.9, False, STATUS_MATCHED, REASON_GEOMETRY_OK, 1, 1, 0.9)
        ev = evaluate_scene(frames, [result_with([person], [face])])
        self.assertEqual((ev.wrong_pairs, ev.decoy_on_person_pairs, ev.decoy_pairs, ev.untrusted_decoy_on_person_pairs), (0, 1, 0, 1))

    def test_id_switch_is_counted_independently_of_association_correctness(self) -> None:
        frames = scenes.render_scenario(ACTORS, scenes.single_person(8))  # type: ignore[arg-type]
        assoc, results = FacePersonAssociator(), []
        for i, f in enumerate(frames):
            track_id = 1 if i < 4 else 7  # ID yarıda değişti
            results.append(assoc.associate(tracking([(track_id, f.truth["A"].person_box)], i), make_faces([f.truth["A"].face_box], i)))
        ev = evaluate_scene(frames, results)
        self.assertEqual(ev.id_report.switches, 1)
        self.assertEqual(ev.track_ids_per_actor["A"], (1, 7))
        self.assertEqual((ev.wrong_pairs, ev.correct_pairs), (0, 8), "ID switch, yüz–kişi ilişkisinin yanlış olduğu anlamına gelmez")

    def test_length_mismatch_is_rejected(self) -> None:
        frames, results = self.perfect(scenes.single_person(4))
        with self.assertRaises(ValueError):
            evaluate_scene(frames, results[:-1])

    def test_iou_helper_is_consistent_with_the_geometry_module(self) -> None:
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)


if __name__ == "__main__":
    unittest.main()
