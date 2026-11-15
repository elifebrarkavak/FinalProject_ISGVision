"""Kutu geometrisi, sert elemeler ve geometrik skor [Mock/birim; saf fonksiyonlar]."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from face_association_test_support import head_box, person_box  # noqa: E402

from isg_vision.face_association import geometry  # noqa: E402
from isg_vision.face_association.config import AssociationConfig  # noqa: E402

CFG = AssociationConfig()


class BoxMathTest(unittest.TestCase):
    def test_area_and_intersection(self) -> None:
        self.assertEqual(geometry.box_area((0, 0, 10, 20)), 200)
        self.assertEqual(geometry.box_area((5, 5, 5, 9)), 0)
        self.assertEqual(geometry.intersection_area((0, 0, 10, 10), (5, 5, 20, 20)), 25)
        self.assertEqual(geometry.intersection_area((0, 0, 10, 10), (10, 0, 20, 10)), 0)  # yalnızca kenar paylaşımı
        self.assertEqual(geometry.intersection_area((0, 0, 10, 10), (30, 30, 40, 40)), 0)

    def test_iou(self) -> None:
        self.assertAlmostEqual(geometry.iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(geometry.iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)
        self.assertEqual(geometry.iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertEqual(geometry.iou((0, 0, 0, 0), (0, 0, 0, 0)), 0.0)

    def test_containment_is_asymmetric(self) -> None:
        small, big = (4, 4, 6, 6), (0, 0, 10, 10)
        self.assertEqual(geometry.containment(small, big), 1.0)
        self.assertAlmostEqual(geometry.containment(big, small), 0.04)
        self.assertAlmostEqual(geometry.containment((8, 8, 12, 12), big), 0.25)
        self.assertEqual(geometry.containment((1, 1, 1, 5), big), 0.0)  # dejenere

    def test_overlap_fraction_uses_the_smaller_box_and_is_symmetric(self) -> None:
        a, b = (0, 0, 100, 100), (50, 50, 60, 60)
        self.assertEqual(geometry.overlap_fraction(a, b), 1.0)
        self.assertEqual(geometry.overlap_fraction(b, a), 1.0)
        self.assertEqual(geometry.overlap_fraction((0, 0, 10, 10), (10, 0, 20, 10)), 0.0)

    def test_center(self) -> None:
        self.assertEqual(geometry.center((0, 10, 20, 30)), (10.0, 20.0))


class GateTest(unittest.TestCase):
    P = person_box(400)

    def test_face_at_top_center_passes(self) -> None:
        self.assertIsNone(geometry.gate_failure(head_box(self.P), self.P, CFG))
        self.assertIsNotNone(geometry.score_pair(head_box(self.P), self.P, CFG))

    def test_face_outside_the_person_box_fails_containment(self) -> None:
        face = (self.P[2] + 10, self.P[1] + 20, self.P[2] + 60, self.P[1] + 70)
        self.assertEqual(geometry.gate_failure(face, self.P, CFG), "containment")
        self.assertIsNone(geometry.score_pair(face, self.P, CFG))

    def test_containment_threshold_is_inclusive_at_the_boundary(self) -> None:
        p = (0.0, 0.0, 100.0, 400.0)
        for inside_fraction, expected in ((0.79, "containment"), (0.80, None), (0.95, None)):
            width = 50.0
            outside = width * (1 - inside_fraction)
            face = (-outside, 20.0, width - outside, 70.0)  # soldan taşan yüz
            with self.subTest(inside=inside_fraction):
                self.assertEqual(geometry.gate_failure(face, p, CFG), expected)

    def test_face_in_the_lower_half_fails_the_vertical_gate(self) -> None:
        face = (self.P[0] + 40, self.P[1] + 0.7 * 420, self.P[0] + 94, self.P[1] + 0.7 * 420 + 54)
        self.assertEqual(geometry.gate_failure(face, self.P, CFG), "vertical")

    def test_vertical_gate_boundary(self) -> None:
        h = self.P[3] - self.P[1]
        for cy, expected in ((0.55, None), (0.56, "vertical")):
            top = self.P[1] + cy * h - 27
            face = (self.P[0] + 40, top, self.P[0] + 94, top + 54)
            with self.subTest(cy=cy):
                self.assertEqual(geometry.gate_failure(face, self.P, CFG), expected)

    def test_too_small_and_too_large_faces_fail_the_size_gate(self) -> None:
        tiny = (self.P[0] + 60, self.P[1] + 10, self.P[0] + 66, self.P[1] + 16)  # 6 px / 420 px = 0.014
        self.assertEqual(geometry.gate_failure(tiny, self.P, CFG), "size")
        huge = (self.P[0], self.P[1], self.P[2], self.P[1] + 0.9 * 420)
        self.assertIn(geometry.gate_failure(huge, self.P, CFG), ("size", "vertical"))
        self.assertEqual(geometry.gate_failure((self.P[0], self.P[1], self.P[2], self.P[1] + 0.85 * 420), self.P, CFG.__class__(max_center_y_ratio=1.0)), "size")

    def test_degenerate_person_box_is_rejected(self) -> None:
        self.assertEqual(geometry.gate_failure((1, 1, 5, 5), (0, 0, 10, 0), CFG), "containment")


class ScoreTest(unittest.TestCase):
    P = person_box(400)

    def test_ideal_head_scores_one(self) -> None:
        s = geometry.score_pair(head_box(self.P), self.P, CFG)
        assert s is not None
        self.assertAlmostEqual(s.score, 1.0)
        self.assertAlmostEqual(s.containment, 1.0)
        self.assertEqual(s.terms, (1.0, 1.0, 1.0, 1.0))

    def test_score_is_bounded_and_monotone_in_vertical_position(self) -> None:
        h = self.P[3] - self.P[1]
        scores = []
        for cy in (0.1, 0.3, 0.35, 0.4, 0.5, 0.55):
            top = self.P[1] + cy * h - 27
            s = geometry.score_pair((self.P[0] + 40, top, self.P[0] + 94, top + 54), self.P, CFG)
            assert s is not None
            self.assertTrue(0.0 <= s.score <= 1.0)
            scores.append(s.score)
        self.assertEqual(scores, sorted(scores, reverse=True), "yüz aşağı indikçe skor azalmalı")
        self.assertGreater(scores[0], scores[-1])

    def test_horizontal_offset_lowers_the_score(self) -> None:
        centered = geometry.score_pair(head_box(self.P), self.P, CFG)
        shifted = geometry.score_pair(head_box(self.P, dx=0.3), self.P, CFG)
        assert centered is not None and shifted is not None
        self.assertGreater(centered.score, shifted.score)
        self.assertAlmostEqual(shifted.center_x_offset, 0.3, places=2)

    def test_border_factor_scales_the_score(self) -> None:
        plain = geometry.score_pair(head_box(self.P), self.P, CFG)
        edge = geometry.score_pair(head_box(self.P), self.P, CFG, at_border=True)
        assert plain is not None and edge is not None
        self.assertAlmostEqual(edge.score, plain.score * CFG.border_factor)

    def test_size_term_penalizes_atypical_face_sizes_but_not_typical_ones(self) -> None:
        # kişi kutusu 420 px: yüz 30 px (r=0.07) tam puan; 15 px (r=0.036) ceza; 250 px (r=0.6) ceza
        wide = person_box(400, w=320)  # yüz genişliği kutuya sığsın

        def z(size: float) -> float:
            s = geometry.score_pair((wide[0] + 10, wide[1] + 10, wide[0] + 10 + size, wide[1] + 10 + size), wide, CFG)
            assert s is not None
            return s.terms[3]

        self.assertEqual(z(30), 1.0)
        self.assertEqual(z(180), 1.0)
        self.assertLess(z(15), 1.0)
        self.assertLess(z(250), 1.0)

    def test_score_is_translation_invariant(self) -> None:
        a = geometry.score_pair(head_box(self.P, down=0.1), self.P, CFG)
        dx, dy = 300.0, 77.0
        moved = (self.P[0] + dx, self.P[1] + dy, self.P[2] + dx, self.P[3] + dy)
        face = head_box(self.P, down=0.1)
        b = geometry.score_pair((face[0] + dx, face[1] + dy, face[2] + dx, face[3] + dy), moved, CFG)
        assert a is not None and b is not None
        self.assertAlmostEqual(a.score, b.score)

    def test_score_is_scale_invariant(self) -> None:
        face = head_box(self.P, down=0.1)
        a = geometry.score_pair(face, self.P, CFG)
        k = 0.5
        b = geometry.score_pair(tuple(v * k for v in face), tuple(v * k for v in self.P), CFG)  # type: ignore[arg-type]
        assert a is not None and b is not None
        self.assertAlmostEqual(a.score, b.score)

    def test_score_never_uses_face_confidence(self) -> None:
        import inspect

        self.assertNotIn("conf", inspect.signature(geometry.score_pair).parameters)


if __name__ == "__main__":
    unittest.main()
