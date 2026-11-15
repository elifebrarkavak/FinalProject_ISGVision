"""Tek karelik yüz–kişi eşleştirme kararı: geometri, çoklu aday, belirsizlik, tek-birebirlik, determinizm [Mock/birim; sahte kutular]."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from face_association_test_support import associate_once, head_box, person_box, person_by_id  # noqa: E402

from isg_vision.face_association import (  # noqa: E402
    STATUS_AMBIGUOUS,
    STATUS_MATCHED,
    STATUS_NO_FACE,
    STATUS_UNMATCHED_FACE,
    AssociationConfig,
)
from isg_vision.face_association.geometry import score_pair  # noqa: E402
from isg_vision.face_association.matching import FaceBox, PersonBox, match_faces_to_persons  # noqa: E402
from isg_vision.face_association.results import (  # noqa: E402
    REASON_CANDIDATES_TAKEN,
    REASON_CONTESTED,
    REASON_LOW_SCORE,
    REASON_NO_CANDIDATE,
)

CFG = AssociationConfig()


class BasicAssociationTest(unittest.TestCase):
    def test_single_person_with_head_face_is_matched(self) -> None:
        p = person_box(400)
        r = associate_once([(1, p)], [head_box(p)])
        person = person_by_id(r, 1)
        self.assertEqual(person.status, STATUS_MATCHED)
        self.assertEqual(person.face_index, 0)
        self.assertEqual(person.face_bbox, head_box(p))
        self.assertGreaterEqual(person.score, 0.99)
        self.assertIsNone(person.margin, "rakip aday yoksa margin None olmalı")
        self.assertEqual(r.faces[0].track_id, 1)
        self.assertEqual((r.person_count, r.face_count, r.matched_count), (1, 1, 1))

    def test_two_people_side_by_side_get_their_own_faces_regardless_of_order(self) -> None:
        a, b = person_box(300), person_box(800)
        for people, face_boxes in (([(1, a), (2, b)], [head_box(a), head_box(b)]), ([(2, b), (1, a)], [head_box(b), head_box(a)]),
                                   ([(1, a), (2, b)], [head_box(b), head_box(a)])):
            r = associate_once(people, face_boxes)
            self.assertEqual(person_by_id(r, 1).face_bbox, head_box(a))
            self.assertEqual(person_by_id(r, 2).face_bbox, head_box(b))
            self.assertEqual(r.matched_count, 2)

    def test_three_people_three_faces(self) -> None:
        people = [(i + 1, person_box(200 + 350 * i)) for i in range(3)]
        r = associate_once(people, [head_box(b) for _, b in people])
        self.assertEqual(r.matched_count, 3)
        self.assertEqual(sorted(r.pairs()), [(1, 0), (2, 1), (3, 2)])

    def test_person_without_a_face_is_reported_as_no_face_and_stays_tracked(self) -> None:
        a, b = person_box(300), person_box(800)
        r = associate_once([(1, a), (2, b)], [head_box(a)])
        self.assertEqual(person_by_id(r, 2).status, STATUS_NO_FACE)
        self.assertEqual(person_by_id(r, 2).reason, REASON_NO_CANDIDATE)
        self.assertEqual(r.person_count, 2, "yüzsüz kişi sonuçtan çıkarılmaz")
        self.assertEqual((r.matched_count, r.no_face_count), (1, 1))

    def test_face_inside_no_person_is_an_unmatched_face_with_no_track_id(self) -> None:
        a = person_box(300)
        stray = (900.0, 150.0, 950.0, 200.0)
        r = associate_once([(1, a)], [head_box(a), stray])
        face = r.face(1)
        assert face is not None
        self.assertEqual((face.status, face.reason), (STATUS_UNMATCHED_FACE, REASON_NO_CANDIDATE))
        self.assertIsNone(face.track_id)
        self.assertEqual(r.unmatched_face_count, 1)

    def test_no_people_all_faces_unmatched_and_no_faces_all_people_no_face(self) -> None:
        r = associate_once([], [(10.0, 10.0, 60.0, 60.0)])
        self.assertEqual((r.person_count, r.unmatched_face_count, r.matched_count), (0, 1, 0))
        a = person_box(300)
        r = associate_once([(1, a)], [])
        self.assertEqual((r.face_count, r.no_face_count, r.matched_count), (0, 1, 0))
        r = associate_once([], [])
        self.assertEqual((r.person_count, r.face_count, r.pairs()), (0, 0, ()))

    def test_face_in_the_lower_body_of_a_person_is_not_linked_to_that_person(self) -> None:
        adult = person_box(400, top=100, h=500)
        child_head = (370.0, 100 + 0.62 * 500, 424.0, 100 + 0.62 * 500 + 54)  # yetişkinin karnı hizası
        r = associate_once([(1, adult)], [child_head])
        self.assertEqual(person_by_id(r, 1).status, STATUS_NO_FACE)
        self.assertEqual(r.face(0).status, STATUS_UNMATCHED_FACE)  # type: ignore[union-attr]

    def test_child_in_front_of_adult_matches_the_child_not_the_adult(self) -> None:
        adult = person_box(400, top=100, h=520)            # 100..620
        child = person_box(400, top=330, w=100, h=290)     # çocuk önde, 330..620
        child_face = head_box(child, size=40, down=0.05)   # yetişkinin göğüs/karın hizası
        adult_face = head_box(adult)
        r = associate_once([(1, adult), (2, child)], [adult_face, child_face])
        self.assertEqual(person_by_id(r, 1).face_bbox, adult_face)
        self.assertEqual(person_by_id(r, 2).face_bbox, child_face)
        self.assertEqual(r.matched_count, 2)


class AmbiguityTest(unittest.TestCase):
    def test_duplicate_person_boxes_make_the_face_ambiguous_and_nothing_is_matched(self) -> None:
        p = person_box(400)
        r = associate_once([(1, p), (2, p)], [head_box(p)])
        self.assertEqual(r.matched_count, 0)
        for tid in (1, 2):
            self.assertEqual(person_by_id(r, tid).status, STATUS_AMBIGUOUS)
            self.assertEqual(person_by_id(r, tid).reason, REASON_CONTESTED)
            self.assertIsNone(person_by_id(r, tid).face_bbox, "belirsizde yüz atanmaz")
        face = r.face(0)
        assert face is not None
        self.assertEqual((face.status, face.reason, face.track_id), (STATUS_AMBIGUOUS, REASON_CONTESTED, None))
        self.assertEqual(sorted(face.candidate_track_ids), [1, 2])
        self.assertEqual((r.ambiguous_person_count, r.ambiguous_face_count, r.unmatched_face_count, r.no_face_count), (2, 1, 0, 0))

    def test_tie_is_ambiguous_for_either_input_order(self) -> None:
        p = person_box(400)
        for people in ([(1, p), (2, p)], [(2, p), (1, p)]):
            self.assertEqual(associate_once(people, [head_box(p)]).matched_count, 0)

    def test_nearby_heads_at_the_same_height_are_ambiguous(self) -> None:
        front, back = person_box(400, top=150), person_box(440, top=100)
        face = head_box(front)
        r = associate_once([(1, front), (2, back)], [face])
        self.assertEqual(r.matched_count, 0)
        self.assertEqual(r.face(0).status, STATUS_AMBIGUOUS)  # type: ignore[union-attr]

    def test_clearly_better_candidate_wins_over_a_poor_one(self) -> None:
        back = person_box(430, top=100, h=600)              # yetişkin: 100..700
        front = person_box(400, top=330, w=110, h=370)      # önündeki çocuk: 330..700; çocuğun yüzü yetişkinin göğüs hizasında (cy ≈ 0.47)
        face = head_box(front, size=40)
        a = score_pair(face, front, CFG)
        b = score_pair(face, back, CFG)
        assert a is not None
        if b is not None:
            self.assertGreaterEqual(a.score - b.score, CFG.ambiguity_margin, "senaryo belirsiz olmamalı")
        r = associate_once([(1, front), (2, back)], [face])
        self.assertEqual(person_by_id(r, 1).status, STATUS_MATCHED)
        self.assertEqual(person_by_id(r, 1).face_index, 0)
        self.assertIsNotNone(person_by_id(r, 1).margin)
        self.assertEqual(person_by_id(r, 2).status, STATUS_NO_FACE)
        self.assertEqual(person_by_id(r, 2).reason, REASON_CANDIDATES_TAKEN)

    def test_margin_threshold_decides_between_matched_and_ambiguous(self) -> None:
        front, back = person_box(400, top=150), person_box(440, top=100)
        face = head_box(front)
        sa, sb = score_pair(face, front, CFG), score_pair(face, back, CFG)
        assert sa is not None and sb is not None
        gap = sa.score - sb.score
        self.assertGreater(gap, 0.0)
        loose = associate_once([(1, front), (2, back)], [face], ambiguity_margin=max(gap / 2, 0.001))
        strict = associate_once([(1, front), (2, back)], [face], ambiguity_margin=min(gap * 2, 1.0))
        self.assertEqual(loose.matched_count, 1, "fark payı aşıyorsa eşleşmeli")
        self.assertEqual(strict.matched_count, 0, "fark payın altındaysa belirsiz kalmalı")

    def test_two_faces_for_one_person_with_clear_winner_leaves_the_other_unmatched(self) -> None:
        p = person_box(400, h=500)
        real, photo = head_box(p), (p[0] + 40, p[1] + 0.42 * 500, p[0] + 94, p[1] + 0.42 * 500 + 54)  # fotoğraf göğüs hizasında
        r = associate_once([(1, p)], [real, photo])
        self.assertEqual(person_by_id(r, 1).face_index, 0)
        extra = r.face(1)
        assert extra is not None
        self.assertEqual((extra.status, extra.reason), (STATUS_UNMATCHED_FACE, REASON_CANDIDATES_TAKEN))
        self.assertEqual(r.matched_count, 1)

    def test_two_equally_plausible_faces_for_one_person_are_ambiguous(self) -> None:
        p = person_box(400, w=240)
        left, right = (p[0] + 30, p[1] + 30, p[0] + 84, p[1] + 84), (p[2] - 84, p[1] + 30, p[2] - 30, p[1] + 84)
        r = associate_once([(1, p)], [left, right])
        self.assertEqual(person_by_id(r, 1).status, STATUS_AMBIGUOUS)
        self.assertEqual(r.matched_count, 0)
        self.assertEqual(r.ambiguous_face_count, 2)

    def test_face_is_never_assigned_by_elimination(self) -> None:
        """Yüz f: iki adaydan biri başka yüzle eşleşse bile f diğerine ATANMAZ (tahmin yok)."""
        p1, p2 = person_box(400, top=150), person_box(440, top=100)
        f_shared = head_box(p1)                  # her iki kutuda da aday; fark < margin
        g_own = head_box(p2, dx=0.35)            # yalnızca p2'nin aday yüzü olabilecek, ondan uzakta
        r = associate_once([(1, p1), (2, p2)], [f_shared, g_own])
        face = r.face(0)
        assert face is not None
        self.assertNotEqual(face.status, STATUS_MATCHED)
        self.assertIsNone(face.track_id)

    def test_low_score_candidate_is_ambiguous_not_matched(self) -> None:
        p = person_box(400, h=500)
        lowish = (p[0] + 40, p[1] + 0.45 * 500, p[0] + 94, p[1] + 0.45 * 500 + 54)
        score = score_pair(lowish, p, CFG)
        assert score is not None
        r = associate_once([(1, p)], [lowish], min_score=min(1.0, score.score + 0.05))
        self.assertEqual(person_by_id(r, 1).status, STATUS_AMBIGUOUS)
        self.assertEqual(person_by_id(r, 1).reason, REASON_LOW_SCORE)
        ok = associate_once([(1, p)], [lowish], min_score=max(0.0, score.score - 0.05))
        self.assertEqual(person_by_id(ok, 1).status, STATUS_MATCHED)

    def test_border_face_is_flagged_and_penalized(self) -> None:
        from face_association_test_support import faces as make_faces, tracking
        from isg_vision.face_association import FacePersonAssociator

        p = person_box(120, top=0)  # kadraj üst kenarında kişi
        face = (p[0] + 20, 0.0, p[0] + 70, 40.0)
        result = FacePersonAssociator().associate(tracking([(1, p)]), make_faces([face], at_border=[True]))
        person = person_by_id(result, 1)
        self.assertEqual(person.status, STATUS_MATCHED)
        self.assertTrue(person.face_at_border)
        plain = score_pair(face, p, CFG)
        assert plain is not None
        self.assertAlmostEqual(person.score, plain.score * CFG.border_factor)


class InvariantPropertyTest(unittest.TestCase):
    """Rastgele sahnelerde (sabit tohum) güvenlik değişmezleri ve girdi sırasından bağımsızlık."""

    @staticmethod
    def random_scene(rng: random.Random) -> tuple[list[PersonBox], list[FaceBox]]:
        persons = []
        for i in range(rng.randint(0, 6)):
            w, h = rng.uniform(60, 260), rng.uniform(150, 620)
            x, y = rng.uniform(0, 1280 - w), rng.uniform(0, 720 - h)
            persons.append(PersonBox(i + 1, (x, y, x + w, y + h)))
        faces = []
        for j in range(rng.randint(0, 8)):
            if persons and rng.random() < 0.7:
                p = rng.choice(persons)
                pw, ph = p.bbox[2] - p.bbox[0], p.bbox[3] - p.bbox[1]
                s = rng.uniform(0.05, 0.35) * ph
                x = p.bbox[0] + rng.uniform(-0.1, 1.0) * pw
                y = p.bbox[1] + rng.uniform(-0.05, 0.7) * ph
            else:
                s = rng.uniform(20, 120)
                x, y = rng.uniform(0, 1100), rng.uniform(0, 600)
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(1280.0, x + s), min(720.0, y + s)
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                faces.append(FaceBox(len(faces), (x1, y1, x2, y2), rng.random() < 0.1))
        return persons, faces

    def test_one_to_one_and_consistent_statuses_hold_for_many_random_scenes(self) -> None:
        rng = random.Random(20261115)
        seen_matched = seen_ambiguous = 0
        for _ in range(600):
            persons, faces = self.random_scene(rng)
            out = match_faces_to_persons(persons, faces, CFG)
            matched_faces = [f.face_index for f in out.faces.values() if f.status == STATUS_MATCHED]
            matched_persons = [p.track_id for p in out.persons.values() if p.status == STATUS_MATCHED]
            self.assertEqual(len(matched_faces), len(set(matched_faces)))
            self.assertEqual(len(matched_persons), len(set(matched_persons)))
            self.assertEqual(len(matched_faces), len(matched_persons))
            for p in out.persons.values():
                if p.status == STATUS_MATCHED:
                    f = out.faces[p.face_index]  # type: ignore[index]
                    self.assertEqual((f.status, f.track_id), (STATUS_MATCHED, p.track_id))
                    # kabul edilen çift GERÇEKTEN sert elemeleri geçti ve eşiği aştı
                    self.assertIn((p.face_index, p.track_id), out.pair_scores)
                    self.assertGreaterEqual(p.score, CFG.min_score - 1e-9)
                else:
                    self.assertIsNone(p.face_index)
            for f in out.faces.values():
                if f.status == STATUS_UNMATCHED_FACE:
                    self.assertTrue(all(out.persons[t].status == STATUS_MATCHED for t in f.candidates))
                if f.status == STATUS_AMBIGUOUS:
                    self.assertTrue(any(out.persons[t].status != STATUS_MATCHED for t in f.candidates))
            seen_matched += len(matched_faces)
            seen_ambiguous += sum(1 for f in out.faces.values() if f.status == STATUS_AMBIGUOUS)
        self.assertGreater(seen_matched, 100, "rastgele sahneler hiç eşleşme üretmediyse test anlamsız")
        self.assertGreater(seen_ambiguous, 0, "belirsiz durum hiç oluşmadıysa test anlamsız")

    def test_result_is_independent_of_input_order(self) -> None:
        rng = random.Random(7)
        for _ in range(200):
            persons, faces = self.random_scene(rng)
            base = match_faces_to_persons(persons, faces, CFG)
            base_pairs = {(persons[[p.track_id for p in persons].index(t)].bbox, faces[f].bbox) for t, f in
                          ((p.track_id, p.face_index) for p in base.persons.values() if p.face_index is not None)}
            shuffled_p, shuffled_f = persons[:], faces[:]
            rng.shuffle(shuffled_p)
            rng.shuffle(shuffled_f)
            out = match_faces_to_persons(shuffled_p, shuffled_f, CFG)
            by_id = {p.track_id: p for p in shuffled_p}
            by_idx = {f.index: f for f in shuffled_f}
            pairs = {(by_id[p.track_id].bbox, by_idx[p.face_index].bbox) for p in out.persons.values() if p.face_index is not None}
            self.assertEqual(pairs, base_pairs)

    def test_no_pair_below_the_hard_gates_is_ever_matched(self) -> None:
        rng = random.Random(99)
        for _ in range(300):
            persons, faces = self.random_scene(rng)
            out = match_faces_to_persons(persons, faces, CFG)
            by_id = {p.track_id: p for p in persons}
            by_idx = {f.index: f for f in faces}
            for p in out.persons.values():
                if p.face_index is None:
                    continue
                person, face = by_id[p.track_id], by_idx[p.face_index]
                inter_w = max(0.0, min(person.bbox[2], face.bbox[2]) - max(person.bbox[0], face.bbox[0]))
                inter_h = max(0.0, min(person.bbox[3], face.bbox[3]) - max(person.bbox[1], face.bbox[1]))
                area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
                self.assertGreaterEqual(inter_w * inter_h / area, CFG.min_containment - 1e-9)
                cy = ((face.bbox[1] + face.bbox[3]) / 2 - person.bbox[1]) / (person.bbox[3] - person.bbox[1])
                self.assertLessEqual(cy, CFG.max_center_y_ratio + 1e-9)


if __name__ == "__main__":
    unittest.main()
