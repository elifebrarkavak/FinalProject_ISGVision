"""Takip veri yapıları ve ID-switch sayacı. [Mock/birim; ağ/model/kamera yok]"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isg_vision.tracking.metrics import count_id_switches  # noqa: E402
from isg_vision.tracking.results import LostTrack, TrackedPerson, TrackingResult  # noqa: E402


def person(track_id: int = 1, **kw: object) -> TrackedPerson:
    base: dict[str, object] = dict(
        track_id=track_id, class_id=0, class_name="person", confidence=0.9, bbox_xyxy=(10.0, 20.0, 70.0, 190.0),
        frame_index=3, timestamp=0.1, session_id=1, detection_index=0, age_frames=4,
    )
    base.update(kw)
    return TrackedPerson(**base)  # type: ignore[arg-type]


class TrackedPersonTest(unittest.TestCase):
    def test_valid_person_exposes_required_fields(self) -> None:
        p = person()
        for name in ("track_id", "bbox_xyxy", "confidence", "class_id", "frame_index", "timestamp"):  # §8'in zorunlu alanları
            self.assertTrue(hasattr(p, name), name)
        self.assertEqual(p.key, (1, 1))
        self.assertEqual(p.center, (40.0, 105.0))

    def test_key_distinguishes_sessions(self) -> None:
        self.assertNotEqual(person(session_id=1).key, person(session_id=2).key)

    def test_track_id_must_be_positive_int(self) -> None:
        for bad in (0, -1, True, 1.0, "1", None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                person(track_id=bad)

    def test_session_id_must_be_positive_int(self) -> None:
        for bad in (0, True, 1.5):
            with self.assertRaises(ValueError):
                person(session_id=bad)

    def test_invalid_confidence_rejected(self) -> None:
        for bad in (-0.01, 1.01, math.nan, math.inf):
            with self.assertRaises(ValueError):
                person(confidence=bad)

    def test_invalid_boxes_rejected(self) -> None:
        for bad in ((10, 20, 10, 40), (10, 20, 5, 40), (0, 0, 5, 0), (math.nan, 0, 5, 5), (0, 0, 5), (0, 0, math.inf, 5)):
            with self.assertRaises(ValueError, msg=str(bad)):
                person(bbox_xyxy=bad)

    def test_negative_indices_and_zero_age_rejected(self) -> None:
        for field in ("frame_index", "detection_index"):
            with self.assertRaises(ValueError):
                person(**{field: -1})
        with self.assertRaises(ValueError):
            person(age_frames=0)

    def test_is_immutable(self) -> None:
        with self.assertRaises(Exception):
            person().track_id = 5  # type: ignore[misc]


class TrackingResultTest(unittest.TestCase):
    def make(self, ids: tuple[int, ...] = (1, 2), lost: tuple[LostTrack, ...] = (), detections: int = 3) -> TrackingResult:
        return TrackingResult(
            tracks=tuple(person(i, detection_index=n) for n, i in enumerate(ids)), lost_tracks=lost, tentative_count=1,
            detection_count=detections, image_size=(640, 480), frame_index=3, timestamp=0.1, session_id=1,
        )

    def test_counts_are_separate(self) -> None:
        """DETECTIONS, ACTIVE TRACKS ve LOST TRACKS birbirine karıştırılmamalıdır."""
        r = self.make(ids=(1, 2), lost=(LostTrack(7, 3),), detections=3)
        self.assertEqual((r.detection_count, r.active_count, r.lost_count), (3, 2, 1))
        self.assertEqual(r.tentative_count, 1)
        self.assertEqual(len(r), 2)
        self.assertEqual(r.track_ids, (1, 2))

    def test_lost_tracks_are_not_in_tracks(self) -> None:
        r = self.make(ids=(1,), lost=(LostTrack(9, 5),))
        self.assertNotIn(9, r.track_ids)
        self.assertEqual(r.active_count, 1)

    def test_empty_result(self) -> None:
        r = self.make(ids=(), detections=0)
        self.assertTrue(r.is_empty)
        self.assertEqual(r.active_count, 0)
        self.assertEqual(list(r), [])

    def test_defaults(self) -> None:
        r = self.make()
        self.assertEqual((r.tracking_ms, r.dropped_tracks, r.reset_reason), (0.0, 0, None))


class IdSwitchCounterTest(unittest.TestCase):
    def test_stable_ids_have_no_switch(self) -> None:
        rep = count_id_switches([{"A": 1, "B": 2}] * 5)
        self.assertTrue(rep.is_continuous)
        self.assertEqual(rep.track_ids_per_identity, {"A": (1,), "B": (2,)})
        self.assertEqual(rep.frames_tracked_per_identity, {"A": 5, "B": 5})

    def test_swap_counts_two_switches_and_two_takeovers(self) -> None:
        rep = count_id_switches([{"A": 1, "B": 2}, {"A": 2, "B": 1}])
        self.assertEqual(rep.switches, 2)
        self.assertEqual(rep.takeovers, 2)
        self.assertFalse(rep.is_continuous)

    def test_loss_and_return_with_new_id_is_a_switch(self) -> None:
        rep = count_id_switches([{"A": 1}, {}, {}, {"A": 5}])
        self.assertEqual(rep.switches, 1)
        self.assertEqual(rep.takeovers, 0)
        self.assertEqual(rep.track_ids_per_identity["A"], (1, 5))

    def test_loss_and_return_with_same_id_is_not_a_switch(self) -> None:
        self.assertTrue(count_id_switches([{"A": 1}, {}, {}, {"A": 1}]).is_continuous)

    def test_takeover_when_newcomer_gets_lost_persons_id(self) -> None:
        rep = count_id_switches([{"A": 1}, {}, {"C": 1}])
        self.assertEqual(rep.takeovers, 1)
        self.assertEqual(rep.switches, 0)  # C'nin önceki ID'si yoktu
        self.assertFalse(rep.is_continuous)

    def test_duplicate_assignment_in_one_frame_is_counted(self) -> None:
        rep = count_id_switches([{"A": 1, "B": 1}])
        self.assertEqual(rep.duplicate_frames, 1)

    def test_empty_input(self) -> None:
        rep = count_id_switches([])
        self.assertTrue(rep.is_continuous)
        self.assertEqual(rep.track_ids_per_identity, {})


if __name__ == "__main__":
    unittest.main()
