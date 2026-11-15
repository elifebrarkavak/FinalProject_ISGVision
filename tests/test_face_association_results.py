"""Sonuç veri yapıları: alan doğrulama ve oluşturma anında denetlenen güvenlik değişmezleri [Mock/birim]."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from face_association_test_support import associate_once, head_box, person_box  # noqa: E402

from isg_vision.face_association import (  # noqa: E402
    STATUS_AMBIGUOUS,
    STATUS_INVALID,
    STATUS_MATCHED,
    STATUS_NO_FACE,
    STATUS_UNMATCHED_FACE,
    AssociatedFace,
    AssociatedPerson,
    AssociationResultError,
    FaceAssociationResult,
)
from isg_vision.face_association.results import REASON_GEOMETRY_OK, REASON_INPUT_INCONSISTENT, REASON_NO_CANDIDATE  # noqa: E402

PB = (100.0, 100.0, 240.0, 520.0)
FB = (150.0, 120.0, 190.0, 160.0)


def person(track_id: int = 1, **kw):  # type: ignore[no-untyped-def]
    base = dict(session_id=1, track_id=track_id, frame_index=3, timestamp=0.1, person_bbox=PB, person_confidence=0.9, status=STATUS_NO_FACE, reason=REASON_NO_CANDIDATE)
    base.update(kw)
    return AssociatedPerson(**base)


def matched_person(track_id: int = 1, face_index: int = 0):  # type: ignore[no-untyped-def]
    return person(track_id, status=STATUS_MATCHED, reason=REASON_GEOMETRY_OK, face_index=face_index, face_bbox=FB, face_confidence=0.9, score=0.95)


def face_obj(face_index: int = 0, **kw):  # type: ignore[no-untyped-def]
    base = dict(face_index=face_index, frame_index=3, timestamp=0.1, face_bbox=FB, face_confidence=0.9, at_border=False, status=STATUS_UNMATCHED_FACE, reason=REASON_NO_CANDIDATE)
    base.update(kw)
    return AssociatedFace(**base)


def matched_face(face_index: int = 0, track_id: int = 1):  # type: ignore[no-untyped-def]
    return face_obj(face_index, status=STATUS_MATCHED, reason=REASON_GEOMETRY_OK, session_id=1, track_id=track_id, score=0.95)


def result(persons, faces, **kw):  # type: ignore[no-untyped-def]
    base = dict(persons=tuple(persons), faces=tuple(faces), image_size=(1280, 720), frame_index=3, timestamp=0.1, session_id=1)
    base.update(kw)
    return FaceAssociationResult(**base)


class PersonValidationTest(unittest.TestCase):
    def test_valid_statuses(self) -> None:
        self.assertTrue(matched_person().is_matched)
        for status in (STATUS_NO_FACE, STATUS_AMBIGUOUS):
            self.assertFalse(person(status=status).is_matched)

    def test_matched_requires_all_face_fields_and_the_ok_reason(self) -> None:
        for missing in ("face_index", "face_bbox", "face_confidence", "score"):
            kw = dict(status=STATUS_MATCHED, reason=REASON_GEOMETRY_OK, face_index=0, face_bbox=FB, face_confidence=0.9, score=0.9)
            kw[missing] = None
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                person(**kw)
        with self.assertRaises(ValueError):
            person(status=STATUS_MATCHED, reason=REASON_NO_CANDIDATE, face_index=0, face_bbox=FB, face_confidence=0.9, score=0.9)

    def test_non_matched_must_not_carry_face_fields(self) -> None:
        for status in (STATUS_NO_FACE, STATUS_AMBIGUOUS):
            for kw in ({"face_index": 0}, {"face_bbox": FB}, {"face_confidence": 0.5}, {"score": 0.5}, {"face_at_border": True}, {"margin": 0.1}):
                with self.subTest(status=status, kw=kw), self.assertRaises(ValueError):
                    person(status=status, **kw)

    def test_non_matched_cannot_claim_the_ok_reason(self) -> None:
        with self.assertRaises(ValueError):
            person(status=STATUS_NO_FACE, reason=REASON_GEOMETRY_OK)

    def test_ids_boxes_and_scores_are_validated(self) -> None:
        for kw in ({"track_id": 0}, {"session_id": 0}, {"track_id": True}, {"frame_index": -1}, {"status": "eşleşti"}, {"reason": "?"},
                   {"person_bbox": (10, 10, 5, 5)}, {"person_bbox": (0, 0, float("nan"), 5)}, {"person_confidence": 1.5}, {"person_confidence": float("nan")}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                person(**kw)

    def test_invalid_status_has_no_continuity(self) -> None:
        from isg_vision.face_association import TrackContinuity

        with self.assertRaises(ValueError):
            person(status=STATUS_INVALID, reason=REASON_INPUT_INCONSISTENT, continuity=TrackContinuity(1, 1, False, ("new_epoch",)))

    def test_key_is_session_and_track(self) -> None:
        self.assertEqual(person(7, session_id=3).key, (3, 7))

    def test_trusted_match_requires_a_match_and_trusted_continuity(self) -> None:
        from isg_vision.face_association import TrackContinuity

        trusted = TrackContinuity(1, 5, True)
        self.assertTrue(replace(matched_person(), continuity=trusted).trusted_match)
        self.assertFalse(replace(matched_person(), continuity=TrackContinuity(1, 1, False, ("new_epoch",))).trusted_match)
        self.assertFalse(replace(matched_person(), continuity=None).trusted_match)
        self.assertFalse(person(continuity=trusted).trusted_match)


class FaceValidationTest(unittest.TestCase):
    def test_unmatched_must_not_carry_a_track_id(self) -> None:
        for status in (STATUS_UNMATCHED_FACE, STATUS_AMBIGUOUS):
            for kw in ({"track_id": 1}, {"session_id": 1}, {"score": 0.5}):
                with self.subTest(status=status, kw=kw), self.assertRaises(ValueError):
                    face_obj(status=status, **kw)

    def test_matched_requires_track_and_score(self) -> None:
        with self.assertRaises(ValueError):
            face_obj(status=STATUS_MATCHED, reason=REASON_GEOMETRY_OK)
        self.assertTrue(matched_face().is_matched)


class ResultInvariantTest(unittest.TestCase):
    def test_consistent_match_is_accepted(self) -> None:
        r = result([matched_person()], [matched_face()])
        self.assertEqual((r.matched_count, r.pairs()), (1, ((1, 0),)))

    def test_a_face_cannot_be_assigned_to_two_people(self) -> None:
        with self.assertRaises(AssociationResultError):
            result([matched_person(1, 0), matched_person(2, 0)], [matched_face(0, 1)])

    def test_matching_must_be_reciprocal(self) -> None:
        with self.assertRaises(AssociationResultError):  # kişi yüze işaret ediyor, yüz eşleşmemiş
            result([matched_person(1, 0)], [face_obj(0)])
        with self.assertRaises(AssociationResultError):  # yüz başka bir ize işaret ediyor
            result([matched_person(1, 0), person(2)], [matched_face(0, 2)])
        with self.assertRaises(AssociationResultError):  # yüz eşleşmiş ama kişi eşleşmemiş
            result([person(1)], [matched_face(0, 1)])
        with self.assertRaises(AssociationResultError):  # yüz yok
            result([matched_person(1, 5)], [matched_face(0, 1)])

    def test_duplicate_track_ids_or_face_indices_are_rejected(self) -> None:
        with self.assertRaises(AssociationResultError):
            result([person(1), person(1)], [])
        with self.assertRaises(AssociationResultError):
            result([], [face_obj(0), face_obj(0)])

    def test_matched_face_box_must_agree_with_the_person_record(self) -> None:
        other = replace(matched_face(), face_bbox=(0.0, 0.0, 10.0, 10.0))
        with self.assertRaises(AssociationResultError):
            result([matched_person()], [other])

    def test_invalid_input_result_must_have_no_matches(self) -> None:
        bad_person = person(status=STATUS_INVALID, reason=REASON_INPUT_INCONSISTENT)
        bad_face = face_obj(status=STATUS_INVALID, reason=REASON_INPUT_INCONSISTENT)
        r = result([bad_person], [bad_face], input_valid=False, invalid_reason="image_size")
        self.assertEqual((r.matched_count, r.pairs()), (0, ()))
        with self.assertRaises(AssociationResultError):
            result([matched_person()], [matched_face()], input_valid=False, invalid_reason="x")
        with self.assertRaises(AssociationResultError):
            result([], [], input_valid=False)  # neden yok
        with self.assertRaises(AssociationResultError):
            result([bad_person], [], input_valid=True)  # geçerli girdide invalid durumlu kişi

    def test_counts(self) -> None:
        r = result(
            [matched_person(1, 0), person(2, status=STATUS_NO_FACE), person(3, status=STATUS_AMBIGUOUS, reason="contested")],
            [matched_face(0, 1), face_obj(1, status=STATUS_UNMATCHED_FACE), face_obj(2, status=STATUS_AMBIGUOUS, reason="contested")],
        )
        self.assertEqual((r.person_count, r.face_count, r.matched_count, r.no_face_count), (3, 3, 1, 1))
        self.assertEqual((r.ambiguous_person_count, r.ambiguous_face_count, r.unmatched_face_count), (1, 1, 1))
        self.assertEqual(len(r), 3)
        self.assertEqual([p.track_id for p in r], [1, 2, 3])

    def test_results_are_immutable(self) -> None:
        r = associate_once([(1, person_box(300))], [head_box(person_box(300))])
        with self.assertRaises(Exception):
            r.persons = ()  # type: ignore[misc]
        with self.assertRaises(Exception):
            r.persons[0].status = "x"  # type: ignore[misc]

    def test_real_associator_output_satisfies_every_invariant(self) -> None:
        a, b = person_box(300), person_box(900)
        r = associate_once([(1, a), (2, b)], [head_box(a), head_box(b), (600.0, 600.0, 640.0, 640.0)])
        self.assertEqual((r.matched_count, r.unmatched_face_count), (2, 1))
        FaceAssociationResult(r.persons, r.faces, r.image_size, r.frame_index, r.timestamp, r.session_id)  # yeniden oluşturma denetimi


if __name__ == "__main__":
    unittest.main()
