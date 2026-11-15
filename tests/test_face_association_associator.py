"""FacePersonAssociator: girdi doğrulama (aynı kare/oturum), geçersiz girdide eşleştirmeme, süreklilik entegrasyonu, ID switch/devralma, sayaçlar [Mock/birim]."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from face_association_test_support import faces, head_box, person_box, person_by_id, tracking  # noqa: E402

from isg_vision.face_association import (  # noqa: E402
    STATUS_INVALID,
    STATUS_MATCHED,
    AssociationConfig,
    AssociationInputError,
    FacePersonAssociator,
)
from isg_vision.face_association.results import BREAK_JUMP, BREAK_NEW_TRACK, BREAK_SESSION_RESET, REASON_INPUT_INCONSISTENT  # noqa: E402

A, B = person_box(300), person_box(900)


def SMALL_FACES(frame: int):  # type: ignore[no-untyped-def]
    """Farklı görüntü boyutundan (640x480) gelen yüz sonucu: takip (1280x720) ile tutarsız."""
    return faces([(100.0, 50.0, 140.0, 90.0)], frame, size=(640, 480))


def one(assoc: FacePersonAssociator, frame: int, *, session: int = 1, people=((1, A),), boxes=None, **kw):  # type: ignore[no-untyped-def]
    boxes = [head_box(b) for _, b in people] if boxes is None else boxes
    return assoc.associate(tracking(list(people), frame, session=session, **kw), faces(boxes, frame, **{k: v for k, v in kw.items() if k == "size"}))


class StructuralValidationTest(unittest.TestCase):
    def test_none_and_wrong_objects_raise_a_clear_error(self) -> None:
        assoc = FacePersonAssociator()
        good_t, good_f = tracking([(1, A)]), faces([head_box(A)])
        for t, f in ((None, good_f), (good_t, None), (object(), good_f), (good_t, object()), ("x", "y")):
            with self.subTest(t=type(t).__name__, f=type(f).__name__), self.assertRaises(AssociationInputError):
                assoc.associate(t, f)

    def test_non_iterable_or_incomplete_items_raise(self) -> None:
        assoc = FacePersonAssociator()
        good_f = faces([head_box(A)])
        with self.assertRaises(AssociationInputError):
            assoc.associate(SimpleNamespace(tracks=5, image_size=(1, 1), frame_index=0, timestamp=None, session_id=1), good_f)
        with self.assertRaises(AssociationInputError):
            assoc.associate(SimpleNamespace(tracks=[object()], image_size=(1, 1), frame_index=0, timestamp=None, session_id=1), good_f)
        with self.assertRaises(AssociationInputError):
            assoc.associate(tracking([(1, A)]), SimpleNamespace(faces=[object()], image_size=(1280, 720), frame_index=0, timestamp=None))

    def test_bad_session_or_frame_index_types_raise(self) -> None:
        assoc = FacePersonAssociator()
        base = tracking([(1, A)])
        for kw in ({"session_id": 0}, {"session_id": True}, {"frame_index": -1}, {"frame_index": 1.5}):
            ns = SimpleNamespace(tracks=base.tracks, image_size=base.image_size, frame_index=base.frame_index, timestamp=base.timestamp, session_id=base.session_id)
            for k, v in kw.items():
                setattr(ns, k, v)
            with self.subTest(kw=kw), self.assertRaises(AssociationInputError):
                assoc.associate(ns, faces([head_box(A)]))

    def test_attribute_based_check_accepts_equivalent_objects_not_only_exact_classes(self) -> None:
        t = tracking([(1, A)])
        wrapped = SimpleNamespace(tracks=t.tracks, image_size=t.image_size, frame_index=t.frame_index, timestamp=t.timestamp, session_id=t.session_id)
        r = FacePersonAssociator().associate(wrapped, faces([head_box(A)]))
        self.assertEqual(r.matched_count, 1)


class InconsistentInputTest(unittest.TestCase):
    def assert_invalid(self, result, fragment: str) -> None:  # type: ignore[no-untyped-def]
        self.assertFalse(result.input_valid)
        self.assertIn(fragment, result.invalid_reason)
        self.assertEqual(result.matched_count, 0, "tutarsız girdide HİÇBİR eşleştirme yapılmaz")
        self.assertEqual(result.pairs(), ())
        self.assertTrue(all(p.status == STATUS_INVALID and p.reason == REASON_INPUT_INCONSISTENT for p in result.persons))
        self.assertTrue(all(f.status == STATUS_INVALID for f in result.faces))
        self.assertTrue(all(p.continuity is None for p in result.persons))

    def test_image_size_mismatch(self) -> None:
        r = FacePersonAssociator().associate(tracking([(1, A)]), SMALL_FACES(0))
        self.assert_invalid(r, "image_size")

    def test_frame_index_mismatch(self) -> None:
        r = FacePersonAssociator().associate(tracking([(1, A)], frame=5), faces([head_box(A)], frame=4))
        self.assert_invalid(r, "frame_index")

    def test_missing_face_frame_index_is_inconsistent_by_default_and_allowed_when_configured(self) -> None:
        t, f = tracking([(1, A)], frame=2), faces([head_box(A)], frame=None)
        self.assert_invalid(FacePersonAssociator().associate(t, f), "frame_index")
        lenient = FacePersonAssociator(AssociationConfig(require_frame_index=False)).associate(t, f)
        self.assertTrue(lenient.input_valid)
        self.assertEqual(lenient.matched_count, 1)

    def test_timestamp_mismatch(self) -> None:
        r = FacePersonAssociator().associate(tracking([(1, A)], frame=3, timestamp=0.1), faces([head_box(A)], frame=3, timestamp=0.2))
        self.assert_invalid(r, "timestamp")

    def test_one_missing_timestamp_is_not_a_mismatch(self) -> None:
        r = FacePersonAssociator().associate(tracking([(1, A)], frame=3, timestamp=None), faces([head_box(A)], frame=3, timestamp=0.2))
        self.assertTrue(r.input_valid)

    def test_box_outside_the_image(self) -> None:
        outside = (1200.0, 100.0, 1400.0, 500.0)
        t = SimpleNamespace(tracks=tracking([(1, A)]).tracks, image_size=(1280, 720), frame_index=0, timestamp=0.0, session_id=1)
        bad_face = SimpleNamespace(bbox_xyxy=(1200.0, 100.0, 1400.0, 140.0), confidence=0.9, face_index=0, frame_index=0, at_border=False)
        r = FacePersonAssociator().associate(t, SimpleNamespace(faces=[bad_face], image_size=(1280, 720), frame_index=0, timestamp=0.0))
        self.assert_invalid(r, "görüntü dışında")
        bad_track = SimpleNamespace(track_id=1, session_id=1, bbox_xyxy=outside, confidence=0.9)
        r = FacePersonAssociator().associate(SimpleNamespace(tracks=[bad_track], image_size=(1280, 720), frame_index=0, timestamp=0.0, session_id=1),
                                             faces([head_box(A)], frame=0, timestamp=0.0))
        self.assertFalse(r.input_valid)

    def test_duplicate_track_ids(self) -> None:
        t = tracking([(1, A), (2, B)])
        dup = SimpleNamespace(tracks=(t.tracks[0], t.tracks[0]), image_size=t.image_size, frame_index=0, timestamp=t.timestamp, session_id=1)
        self.assert_invalid(FacePersonAssociator().associate(dup, faces([head_box(A)])), "yinelenen track_id")

    def test_track_from_another_session(self) -> None:
        t = tracking([(1, A)], session=2)
        mixed = SimpleNamespace(tracks=t.tracks, image_size=t.image_size, frame_index=0, timestamp=t.timestamp, session_id=1)
        self.assert_invalid(FacePersonAssociator().associate(mixed, faces([head_box(A)])), "oturuma ait")

    def test_face_index_order_mismatch(self) -> None:
        f = faces([head_box(A), head_box(B)])
        swapped = SimpleNamespace(faces=tuple(reversed(f.faces)), image_size=f.image_size, frame_index=f.frame_index, timestamp=f.timestamp)
        self.assert_invalid(FacePersonAssociator().associate(tracking([(1, A), (2, B)]), swapped), "face_index")

    def test_face_from_another_frame(self) -> None:
        f = faces([head_box(A)], frame=0)
        wrong = SimpleNamespace(faces=(SimpleNamespace(bbox_xyxy=f.faces[0].bbox_xyxy, confidence=0.9, face_index=0, frame_index=9, at_border=False),),
                                image_size=f.image_size, frame_index=0, timestamp=f.timestamp)
        self.assert_invalid(FacePersonAssociator().associate(tracking([(1, A)]), wrong), "başka bir kareye")

    def test_frame_order_violation_in_the_same_session(self) -> None:
        assoc = FacePersonAssociator()
        one(assoc, 5)
        self.assert_invalid(one(assoc, 5), "kare sırası")
        self.assert_invalid(one(assoc, 3), "kare sırası")
        self.assertTrue(one(assoc, 6).input_valid)

    def test_invalid_frame_does_not_touch_continuity_state(self) -> None:
        assoc = FacePersonAssociator(AssociationConfig(min_trusted_frames=3))
        for f in range(3):
            r = one(assoc, f)
        self.assertTrue(person_by_id(r, 1).continuity.trusted)  # type: ignore[union-attr]
        bad = assoc.associate(tracking([(1, A)], frame=3), SMALL_FACES(3))
        self.assertFalse(bad.input_valid)
        nxt = one(assoc, 4)
        c = person_by_id(nxt, 1).continuity
        assert c is not None
        self.assertEqual((c.epoch, c.frames_in_epoch, c.breaks), (1, 4, ()), "geçersiz kare epoch'u bozmaz ve sayılmaz")

    def test_stats_count_invalid_frames_separately(self) -> None:
        assoc = FacePersonAssociator()
        one(assoc, 0)
        assoc.associate(tracking([(1, A)], frame=1), SMALL_FACES(1))
        s = assoc.stats
        self.assertEqual((s.frames, s.invalid_frames, s.matched), (2, 1, 1))


class AssociationFlowTest(unittest.TestCase):
    def test_result_carries_frame_session_size_and_timing(self) -> None:
        assoc = FacePersonAssociator()
        r = one(assoc, 7, session=3)
        self.assertEqual((r.frame_index, r.session_id, r.image_size), (7, 3, (1280, 720)))
        self.assertAlmostEqual(r.timestamp, 7 / 30)  # type: ignore[arg-type]
        self.assertGreaterEqual(r.association_ms, 0.0)
        p = r.persons[0]
        self.assertEqual((p.session_id, p.frame_index), (3, 7))
        self.assertEqual(p.person_bbox, A)

    def test_clock_is_used_for_the_timing(self) -> None:
        ticks = iter([1.0, 1.002, 5.0, 5.0])
        r = FacePersonAssociator(clock=lambda: next(ticks)).associate(tracking([(1, A)]), faces([head_box(A)]))
        self.assertAlmostEqual(r.association_ms, 2.0, places=3)

    def test_first_frame_match_is_not_trusted_and_later_ones_are(self) -> None:
        assoc = FacePersonAssociator(AssociationConfig(min_trusted_frames=3))
        results = [one(assoc, f) for f in range(5)]
        self.assertEqual([r.trusted_match_count for r in results], [0, 0, 1, 1, 1])
        self.assertEqual([r.matched_count for r in results], [1] * 5)
        self.assertEqual(results[0].breaks, ((1, BREAK_NEW_TRACK),))

    def test_face_loss_for_a_few_frames_keeps_the_person_tracked_and_reports_no_face(self) -> None:
        assoc = FacePersonAssociator()
        statuses = []
        for f in range(9):
            boxes = [] if 3 <= f < 6 else [head_box(A)]
            r = assoc.associate(tracking([(1, A)], f), faces(boxes, f))
            statuses.append((r.person_count, person_by_id(r, 1).status, person_by_id(r, 1).continuity.frames_since_match))  # type: ignore[union-attr]
        self.assertEqual([s[0] for s in statuses], [1] * 9, "yüz kaybı iz kaybı değildir")
        self.assertEqual([s[1] for s in statuses], ["matched"] * 3 + ["no_face"] * 3 + ["matched"] * 3)
        self.assertEqual([s[2] for s in statuses[3:6]], [1, 2, 3])
        last = person_by_id(r, 1).continuity
        assert last is not None
        self.assertEqual(last.epoch, 1, "yüz kaybı süreklilik dönemini bozmaz")

    def test_person_without_face_is_never_dropped_and_face_without_person_never_gets_an_id(self) -> None:
        r = FacePersonAssociator().associate(tracking([(1, A), (2, B)]), faces([head_box(A), (600.0, 650.0, 640.0, 700.0)]))
        self.assertEqual(r.person_count, 2)
        self.assertIsNone(r.face(1).track_id)  # type: ignore[union-attr]

    def test_reset_clears_the_history(self) -> None:
        assoc = FacePersonAssociator()
        for f in range(5):
            one(assoc, f)
        assoc.reset()
        r = one(assoc, 0)
        self.assertEqual(r.breaks, ((1, BREAK_NEW_TRACK),))
        self.assertTrue(r.input_valid)


class IdSwitchAndTakeoverTest(unittest.TestCase):
    def test_id_switch_between_two_people_never_carries_the_old_face_to_the_new_owner(self) -> None:
        """İki kişi Track ID'lerini değiş tokuş eder: yüz HER KARE yalnızca o karedeki kutuya bağlanır; süreklilik bozulur ve iki iz güvensiz olur."""
        assoc = FacePersonAssociator(AssociationConfig(min_trusted_frames=2))
        face_a, face_b = head_box(A), head_box(B)
        for f in range(6):
            r = assoc.associate(tracking([(1, A), (2, B)], f), faces([face_a, face_b], f))
        self.assertEqual(r.trusted_match_count, 2)
        # 6. karede ByteTrack numaraları değiştirdi: ID 1 artık B'nin konumunda, ID 2 A'nın konumunda
        r = assoc.associate(tracking([(1, B), (2, A)], 6), faces([face_a, face_b], 6))
        p1, p2 = person_by_id(r, 1), person_by_id(r, 2)
        self.assertEqual(p1.face_bbox, face_b, "ID 1'in kutusu B'nin konumunda: yüz o kutudakidir, eski (A) yüzü değil")
        self.assertEqual(p2.face_bbox, face_a)
        self.assertEqual(sorted(r.breaks), [(1, BREAK_JUMP), (2, BREAK_JUMP)])
        for p in (p1, p2):
            self.assertEqual(p.continuity.epoch, 2)  # type: ignore[union-attr]
            self.assertFalse(p.trusted_match, "şüpheli ID devrinden sonra iz güvenilir sayılmaz")
        self.assertEqual(r.trusted_match_count, 0)

    def test_takeover_by_a_newcomer_at_the_lost_position_is_flagged_when_the_gap_is_long(self) -> None:
        assoc = FacePersonAssociator()
        for f in range(8):
            assoc.associate(tracking([(1, A)], f), faces([head_box(A)], f))
        r = assoc.associate(tracking([(1, A)], 30), faces([head_box(A)], 30))  # 21 kare kayıp; yeni biri aynı konumda ID 1'i aldı
        self.assertEqual(r.breaks, ((1, "gap"),))
        self.assertFalse(person_by_id(r, 1).trusted_match)

    def test_new_track_id_for_a_returning_person_starts_clean(self) -> None:
        assoc = FacePersonAssociator()
        for f in range(8):
            assoc.associate(tracking([(1, A)], f), faces([head_box(A)], f))
        r = assoc.associate(tracking([(2, A)], 8), faces([head_box(A)], 8))  # aynı kişi yeni numarayla döndü
        c = person_by_id(r, 2).continuity
        assert c is not None
        self.assertEqual((c.epoch, c.frames_in_epoch, c.matched_streak), (1, 1, 1))
        self.assertFalse(c.trusted)
        self.assertEqual(r.breaks, ((2, BREAK_NEW_TRACK),))

    def test_session_change_resets_everything(self) -> None:
        assoc = FacePersonAssociator(AssociationConfig(min_trusted_frames=2))
        for f in range(5):
            assoc.associate(tracking([(1, A)], f), faces([head_box(A)], f))
        r = assoc.associate(tracking([(1, A)], 0, session=2), faces([head_box(A)], 0))
        self.assertEqual(r.breaks, ((1, BREAK_SESSION_RESET),))
        self.assertEqual(r.session_id, 2)
        self.assertFalse(person_by_id(r, 1).trusted_match)

    def test_tracker_reset_reason_resets_even_with_an_unchanged_session_number(self) -> None:
        assoc = FacePersonAssociator()
        for f in range(5):
            assoc.associate(tracking([(1, A)], f), faces([head_box(A)], f))
        r = assoc.associate(tracking([(1, A)], 5, reset_reason="source_changed"), faces([head_box(A)], 5))
        self.assertEqual(r.breaks, ((1, BREAK_SESSION_RESET),))

    def test_overlapping_people_keep_per_frame_geometry_but_lose_trust(self) -> None:
        front, back = person_box(400, top=150, w=150), person_box(470, top=60, w=150, h=600)
        assoc = FacePersonAssociator(AssociationConfig(min_trusted_frames=1))
        face_front, face_back = head_box(front), head_box(back)
        for f in range(4):
            r = assoc.associate(tracking([(1, front), (2, back)], f), faces([face_front, face_back], f))
        self.assertEqual(r.matched_count, 2, "yüzler farklı kafa bölgelerinde: geometri eşleştirir")
        self.assertEqual(r.trusted_match_count, 0, "ama örtüşen izlerin eşleşmesi kanıt biriktirmeye uygun (trusted) değildir")
        self.assertEqual(person_by_id(r, 1).continuity.overlapping_track_ids, (2,))  # type: ignore[union-attr]

    def test_possibly_static_track_is_flagged_but_still_matched(self) -> None:
        assoc = FacePersonAssociator(AssociationConfig(stationary_flag_frames=5))
        for f in range(8):
            r = assoc.associate(tracking([(1, A)], f), faces([head_box(A)], f))
        p = person_by_id(r, 1)
        self.assertTrue(p.continuity.possibly_static)  # type: ignore[union-attr]
        self.assertEqual(p.status, STATUS_MATCHED, "bayrak eşleşmeyi engellemez (yargı değildir)")
        self.assertEqual(r.possibly_static_count, 1)


class StatsTest(unittest.TestCase):
    def test_counters_accumulate_over_frames(self) -> None:
        assoc = FacePersonAssociator()
        for f in range(4):
            assoc.associate(tracking([(1, A), (2, B)], f), faces([head_box(A), (600.0, 650.0, 640.0, 700.0)], f))
        s = assoc.stats
        self.assertEqual((s.frames, s.invalid_frames, s.person_total, s.face_total), (4, 0, 8, 8))
        self.assertEqual((s.matched, s.no_face, s.unmatched_faces, s.ambiguous_persons), (4, 4, 4, 0))
        self.assertEqual(s.breaks, {"new_track": 2})


if __name__ == "__main__":
    unittest.main()
