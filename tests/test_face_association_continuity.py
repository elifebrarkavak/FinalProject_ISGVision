"""Süreklilik izleyicisi: yeni iz, ID switch/devralma, zıplama, uzun kayıp, örtüşme, oturum değişimi, güven, hareketsizlik, bellek [Mock/birim].

Sahte kutu dizileri; gerçek ByteTrack burada ÇALIŞMAZ (gerçek ByteTrack + gerçek model davranışı ``test_face_association_real_model.py``'dedir).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from face_association_test_support import person_box  # noqa: E402

from isg_vision.face_association import AssociationConfig, ContinuityMonitor, TrackObservation  # noqa: E402
from isg_vision.face_association.results import (  # noqa: E402
    BREAK_GAP,
    BREAK_JUMP,
    BREAK_NEW_TRACK,
    BREAK_OVERLAP_END,
    BREAK_SESSION_RESET,
    UNTRUSTED_NEW_EPOCH,
    UNTRUSTED_OVERLAP,
    UNTRUSTED_OVERLAP_COOLDOWN,
)


def step(monitor: ContinuityMonitor, frame: int, tracks: dict[int, tuple[float, float, float, float]], *, matched: tuple[int, ...] = (), session: int = 1,
         reset_reason: str | None = None):  # type: ignore[no-untyped-def]
    return monitor.update(
        session_id=session, frame_index=frame, reset_reason=reset_reason, tracks=[TrackObservation(t, b) for t, b in tracks.items()], matched_track_ids=matched
    )


A = person_box(300)
B = person_box(900)


class NewTrackTest(unittest.TestCase):
    def test_new_track_starts_untrusted_in_epoch_one_and_becomes_trusted_after_min_frames(self) -> None:
        m = ContinuityMonitor(AssociationConfig(min_trusted_frames=3))
        trusted = []
        for f in range(5):
            cont, breaks = step(m, f, {1: A})
            trusted.append(cont[1].trusted)
            self.assertEqual(cont[1].epoch, 1)
            self.assertEqual(cont[1].frames_in_epoch, f + 1)
            self.assertEqual(breaks, ((1, BREAK_NEW_TRACK),) if f == 0 else ())
        self.assertEqual(trusted, [False, False, True, True, True])
        first, _ = step(ContinuityMonitor(), 0, {1: A})
        self.assertEqual(first[1].untrusted_reasons, (UNTRUSTED_NEW_EPOCH,))

    def test_a_new_track_id_never_inherits_history_from_a_vanished_one(self) -> None:
        m = ContinuityMonitor()
        for f in range(10):
            step(m, f, {1: A}, matched=(1,))
        cont, breaks = step(m, 10, {2: A})  # eski kişi kayboldu; AYNI konumda yeni numara
        self.assertEqual(cont[2].epoch, 1)
        self.assertEqual(cont[2].matched_streak, 0)
        self.assertIsNone(cont[2].frames_since_match)
        self.assertFalse(cont[2].trusted)
        self.assertEqual(breaks, ((2, BREAK_NEW_TRACK),))

    def test_two_tracks_have_independent_state(self) -> None:
        m = ContinuityMonitor()
        for f in range(4):
            step(m, f, {1: A})
        cont, _ = step(m, 4, {1: A, 2: B})
        self.assertEqual(cont[1].frames_in_epoch, 5)
        self.assertEqual(cont[2].frames_in_epoch, 1)


class MatchHistoryTest(unittest.TestCase):
    def test_matched_streak_and_frames_since_match(self) -> None:
        m = ContinuityMonitor()
        seq = [(1,), (1,), (), (), (1,)]
        out = [step(m, f, {1: A}, matched=mt)[0][1] for f, mt in enumerate(seq)]
        self.assertEqual([c.matched_streak for c in out], [1, 2, 0, 0, 1])
        self.assertEqual([c.frames_since_match for c in out], [0, 0, 1, 2, 0])

    def test_never_matched_has_no_match_age(self) -> None:
        m = ContinuityMonitor()
        for f in range(3):
            c = step(m, f, {1: A})[0][1]
        self.assertIsNone(c.frames_since_match)
        self.assertEqual(c.matched_streak, 0)


class JumpAndTakeoverTest(unittest.TestCase):
    def test_position_jump_starts_a_new_epoch_and_clears_the_match_history(self) -> None:
        """ID devri şüphesi: aynı Track ID bir anda başka konuma/kişiye geçti -> eski eşleşme geçmişi yeni kişiye ait SAYILMAZ."""
        m = ContinuityMonitor()
        for f in range(8):
            c = step(m, f, {1: A}, matched=(1,))[0][1]
        self.assertTrue(c.trusted)
        self.assertEqual(c.matched_streak, 8)
        cont, breaks = step(m, 8, {1: B}, matched=(1,))  # ID 1 artık başka kişide
        self.assertEqual(breaks, ((1, BREAK_JUMP),))
        self.assertEqual(cont[1].epoch, 2)
        self.assertEqual(cont[1].frames_in_epoch, 1)
        self.assertEqual(cont[1].matched_streak, 1, "yalnızca bu karedeki eşleşme sayılır; geçmiş taşınmaz")
        self.assertFalse(cont[1].trusted)
        self.assertIn(UNTRUSTED_NEW_EPOCH, cont[1].untrusted_reasons)

    def test_smooth_motion_does_not_break_continuity(self) -> None:
        m = ContinuityMonitor()
        for f in range(30):
            cont, breaks = step(m, f, {1: person_box(300 + 6 * f)})
            self.assertEqual(breaks, ((1, BREAK_NEW_TRACK),) if f == 0 else ())
        self.assertEqual(cont[1].epoch, 1)
        self.assertTrue(cont[1].trusted)

    def test_jump_iou_threshold_is_configurable(self) -> None:
        shifted = person_box(300 + 70)  # IoU ≈ 0.67
        m = ContinuityMonitor(AssociationConfig(jump_iou=0.9))
        step(m, 0, {1: A})
        self.assertEqual(step(m, 1, {1: shifted})[1], ((1, BREAK_JUMP),))
        m = ContinuityMonitor(AssociationConfig(jump_iou=0.3))
        step(m, 0, {1: A})
        self.assertEqual(step(m, 1, {1: shifted})[1], ())

    def test_swap_of_two_ids_is_reported_for_both_tracks(self) -> None:
        m = ContinuityMonitor()
        for f in range(6):
            step(m, f, {1: A, 2: B}, matched=(1, 2))
        cont, breaks = step(m, 6, {1: B, 2: A}, matched=(1, 2))  # iki kişi ID'lerini değiş tokuş etti
        self.assertEqual(sorted(breaks), [(1, BREAK_JUMP), (2, BREAK_JUMP)])
        self.assertEqual((cont[1].epoch, cont[2].epoch), (2, 2))
        self.assertFalse(cont[1].trusted or cont[2].trusted)


class GapTest(unittest.TestCase):
    def test_short_gap_with_continuous_position_keeps_the_epoch(self) -> None:
        m = ContinuityMonitor(AssociationConfig(max_gap_frames=3))
        for f in range(5):
            step(m, f, {1: A})
        cont, breaks = step(m, 8, {1: A})  # 3 kare görünmedi
        self.assertEqual(breaks, ())
        self.assertEqual(cont[1].epoch, 1)

    def test_gap_longer_than_tolerance_starts_a_new_epoch(self) -> None:
        m = ContinuityMonitor(AssociationConfig(max_gap_frames=3))
        for f in range(5):
            step(m, f, {1: A}, matched=(1,))
        cont, breaks = step(m, 9, {1: A}, matched=(1,))  # 4 kare görünmedi, AYNI konumda geri geldi
        self.assertEqual(breaks, ((1, BREAK_GAP),))
        self.assertEqual(cont[1].epoch, 2)
        self.assertEqual(cont[1].matched_streak, 1)

    def test_short_gap_but_different_position_is_a_gap_break(self) -> None:
        m = ContinuityMonitor(AssociationConfig(max_gap_frames=3))
        for f in range(5):
            step(m, f, {1: A})
        _, breaks = step(m, 6, {1: B})
        self.assertEqual(breaks, ((1, BREAK_GAP),))

    def test_newcomer_taking_over_a_lost_slot_after_a_long_gap_is_not_trusted(self) -> None:
        """ByteTrack KL-4B-001: kaybolan kişinin konumuna giren başkası eski ID'yi devralır; uzun kayıp sonrası güven verilmez."""
        m = ContinuityMonitor()
        for f in range(10):
            step(m, f, {1: A}, matched=(1,))
        cont, breaks = step(m, 25, {1: A}, matched=(1,))
        self.assertEqual(breaks, ((1, BREAK_GAP),))
        self.assertFalse(cont[1].trusted)
        self.assertEqual(cont[1].epoch, 2)


class OverlapTest(unittest.TestCase):
    P1 = person_box(400, w=150)
    P2_overlapping = person_box(460, w=150)  # büyük örtüşme
    P2_far = person_box(900, w=150)

    def test_overlapping_tracks_are_untrusted_even_with_long_history(self) -> None:
        m = ContinuityMonitor(AssociationConfig(min_trusted_frames=2))
        for f in range(5):
            step(m, f, {1: self.P1, 2: self.P2_far})
        cont, _ = step(m, 5, {1: self.P1, 2: self.P2_overlapping})
        for tid, other in ((1, 2), (2, 1)):
            self.assertFalse(cont[tid].trusted)
            self.assertIn(UNTRUSTED_OVERLAP, cont[tid].untrusted_reasons)
            self.assertEqual(cont[tid].overlapping_track_ids, (other,))
            self.assertGreater(cont[tid].overlap_fraction, 0.25)

    @staticmethod
    def p2(x: float) -> tuple[float, float, float, float]:
        return person_box(x, w=150)

    def test_overlap_end_starts_a_new_epoch_after_the_cooldown(self) -> None:
        cfg = AssociationConfig(min_trusted_frames=2, overlap_cooldown_frames=4)
        m = ContinuityMonitor(cfg)
        f = 0
        for x in (700, 700, 700, 640, 580, 520):  # örtüşme yok (düzgün yaklaşma; ardışık kareler arası IoU yüksek)
            step(m, f, {1: self.P1, 2: self.p2(x)}); f += 1
        for x in (460, 460, 460, 460):  # örtüşme (sabit)
            step(m, f, {1: self.P1, 2: self.p2(x)}); f += 1
        epochs, reasons, all_breaks = [], [], []
        for x in (485, 510, 535, 560, 585, 610, 635, 660):  # düzgün uzaklaşma: ilk 2 kare hâlâ örtüşür (485: dx=85, 510: dx=110)
            cont, breaks = step(m, f, {1: self.P1, 2: self.p2(x)}); f += 1
            epochs.append(cont[1].epoch)
            reasons.append(cont[1].untrusted_reasons)
            all_breaks.extend(breaks)
        self.assertEqual(epochs[:4], [1, 1, 1, 1], "örtüşme ve sakinleşme süresince epoch korunur")
        self.assertIn(UNTRUSTED_OVERLAP_COOLDOWN, reasons[2])
        self.assertEqual(sorted(all_breaks), [(1, BREAK_OVERLAP_END), (2, BREAK_OVERLAP_END)], "yalnızca örtüşme sonu kesintisi (zıplama yok)")
        self.assertGreaterEqual(epochs[-1], 2, "örtüşme bittikten sonra epoch yenilenir")
        self.assertTrue(all(e <= 2 for e in epochs), "tek örtüşme olayı tek epoch artışı üretir")

    def test_evidence_before_the_overlap_keeps_its_epoch(self) -> None:
        m = ContinuityMonitor(AssociationConfig(min_trusted_frames=2, overlap_cooldown_frames=3))
        f = 0
        for x in (700, 700, 700, 700, 700, 700):
            cont, _ = step(m, f, {1: self.P1, 2: self.p2(x)}, matched=(1, 2)); f += 1
        pre_epoch = cont[1].epoch
        for x in (640, 580, 520, 460, 460):  # düzgün yaklaşma + örtüşme
            cont, _ = step(m, f, {1: self.P1, 2: self.p2(x)}); f += 1
        self.assertEqual(cont[1].epoch, pre_epoch, "örtüşme sırasında epoch değişmez; yalnızca güvensiz")
        self.assertFalse(cont[1].trusted)

    def test_overlap_resuming_during_the_cooldown_does_not_create_an_extra_epoch(self) -> None:
        m = ContinuityMonitor(AssociationConfig(overlap_cooldown_frames=5))
        f = 0
        for x in (460, 460, 460, 485, 510, 535, 510, 485):  # örtüşme -> kısa ayrılma (3 sakin kare < 5) -> yeniden örtüşme
            cont, breaks = step(m, f, {1: self.P1, 2: self.p2(x)}); f += 1
            self.assertEqual([b for b in breaks if b[1] != BREAK_NEW_TRACK], [])
        self.assertEqual(cont[1].epoch, 1)

    def test_touching_boxes_do_not_count_as_overlap(self) -> None:
        m = ContinuityMonitor()
        a, b = (0.0, 100.0, 100.0, 400.0), (100.0, 100.0, 200.0, 400.0)
        cont, _ = step(m, 0, {1: a, 2: b})
        self.assertEqual(cont[1].overlapping_track_ids, ())
        self.assertEqual(cont[1].overlap_fraction, 0.0)


class SessionTest(unittest.TestCase):
    def test_session_change_clears_all_history(self) -> None:
        m = ContinuityMonitor(AssociationConfig(min_trusted_frames=2))
        for f in range(6):
            step(m, f, {1: A}, matched=(1,))
        cont, breaks = step(m, 0, {1: A}, matched=(1,), session=2)  # yeni oturum: AYNI numara AYNI konum
        self.assertEqual(breaks, ((1, BREAK_SESSION_RESET),))
        self.assertEqual((cont[1].epoch, cont[1].frames_in_epoch, cont[1].matched_streak), (1, 1, 1))
        self.assertFalse(cont[1].trusted)

    def test_reset_reason_clears_history_even_with_the_same_session_number(self) -> None:
        m = ContinuityMonitor()
        for f in range(6):
            step(m, f, {1: A})
        _, breaks = step(m, 6, {1: A}, reset_reason="source_changed")
        self.assertEqual(breaks, ((1, BREAK_SESSION_RESET),))

    def test_frame_index_may_restart_in_a_new_session(self) -> None:
        m = ContinuityMonitor()
        for f in range(5, 10):
            step(m, f, {1: A})
        self.assertFalse(m.accepts(1, 9, None), "aynı oturumda tekrar eden kare kabul edilmez")
        self.assertFalse(m.accepts(1, 3, None))
        self.assertTrue(m.accepts(1, 10, None))
        self.assertTrue(m.accepts(2, 0, None), "yeni oturumda kare numarası yeniden başlayabilir")
        self.assertTrue(m.accepts(1, 0, "image_size_changed"))

    def test_first_ever_frame_is_new_track_not_session_reset(self) -> None:
        self.assertEqual(step(ContinuityMonitor(), 0, {1: A})[1], ((1, BREAK_NEW_TRACK),))

    def test_reset_method_drops_state_but_keeps_break_counters(self) -> None:
        m = ContinuityMonitor()
        step(m, 0, {1: A})
        self.assertEqual(m.break_counts[BREAK_NEW_TRACK], 1)
        m.reset()
        self.assertEqual(m.tracked_count, 0)
        self.assertIsNone(m.session_id)
        self.assertEqual(step(m, 0, {1: A})[1], ((1, BREAK_NEW_TRACK),))
        self.assertEqual(m.break_counts[BREAK_NEW_TRACK], 2)


class StationaryAndMemoryTest(unittest.TestCase):
    def test_static_box_is_flagged_after_the_configured_frames_but_not_removed(self) -> None:
        m = ContinuityMonitor(AssociationConfig(stationary_flag_frames=10))
        flags = []
        for f in range(15):
            cont, _ = step(m, f, {1: A, 2: person_box(900 + 12 * f)})
            flags.append(cont[1].possibly_static)
            self.assertFalse(cont[2].possibly_static, "hareketli iz bayraklanmaz")
        self.assertFalse(flags[8])
        self.assertTrue(flags[10] and flags[14])

    def test_static_flag_resets_when_the_box_moves(self) -> None:
        m = ContinuityMonitor(AssociationConfig(stationary_flag_frames=5))
        for f in range(8):
            cont, _ = step(m, f, {1: A})
        self.assertTrue(cont[1].possibly_static)
        cont, _ = step(m, 8, {1: person_box(300 + 80)})
        self.assertFalse(cont[1].possibly_static)
        self.assertEqual(cont[1].stationary_frames, 0)

    def test_unseen_track_state_is_dropped_after_ttl(self) -> None:
        m = ContinuityMonitor(AssociationConfig(state_ttl_frames=20, max_gap_frames=3))
        step(m, 0, {1: A, 2: B})
        self.assertEqual(m.tracked_count, 2)
        for f in range(1, 25):
            step(m, f, {2: B})
        self.assertEqual(m.tracked_count, 1)
        cont, breaks = step(m, 25, {1: A, 2: B})
        self.assertEqual(breaks, ((1, BREAK_NEW_TRACK),), "TTL sonrası dönen aynı numara yeni sayılır")

    def test_memory_is_bounded_with_many_short_lived_tracks(self) -> None:
        m = ContinuityMonitor(AssociationConfig(state_ttl_frames=10))
        for f in range(300):
            step(m, f, {1000 + f: A})
        self.assertLessEqual(m.tracked_count, 12)


class ContinuityObjectTest(unittest.TestCase):
    def test_trusted_and_reasons_are_consistent(self) -> None:
        from isg_vision.face_association.results import TrackContinuity

        with self.assertRaises(ValueError):
            TrackContinuity(epoch=1, frames_in_epoch=1, trusted=True, untrusted_reasons=(UNTRUSTED_OVERLAP,))
        with self.assertRaises(ValueError):
            TrackContinuity(epoch=1, frames_in_epoch=1, trusted=False)
        with self.assertRaises(ValueError):
            TrackContinuity(epoch=0, frames_in_epoch=1, trusted=True)
        with self.assertRaises(ValueError):
            TrackContinuity(epoch=1, frames_in_epoch=1, trusted=True, breaks=("bilinmeyen",))


if __name__ == "__main__":
    unittest.main()
