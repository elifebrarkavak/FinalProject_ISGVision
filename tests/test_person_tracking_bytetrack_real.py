"""GERÇEK ByteTrack algoritmasıyla takip senaryoları (Ultralytics ``BYTETracker``; mock takipçi YOK).

Kanıt türü: **gerçek ByteTrack algoritması, SAHTE (betiklenmiş) tespit dizileri**. YOLO çalışmaz, kamera/video yoktur. Bu testler
ByteTrack entegrasyonunun ve takip mantığının beklenen biçimde davrandığını gösterir; **gerçek YOLO + gerçek kamera başarısının,
IDF1/HOTA/MOTA gibi doğruluk metriklerinin veya "%85 hedefinin" kanıtı DEĞİLDİR** (etiketli gerçek takip verisi yoktur).

İki tür beklenti ayrılmıştır ve test adı/açıklamasında belirtilmiştir:

* **Garanti**: tasarım gereği her zaman doğru olması gereken özellikler (aynı karede benzersiz ID, tespitsiz karede hata yok, ...).
* **Karakterizasyon / bilinen sınırlama**: kurulu ``ultralytics==8.4.210`` ile ÖLÇÜLEN davranış (ör. kurtarma penceresi, yanlış pozitifin
  iz alması). Algoritmanın her sürümde ve her gerçek sahnede aynı olacağının iddiası değildir; sürüm yükseltilirse yeniden gözden geçirilmelidir.
  Sentetik dizilerdeki "ID switch yok" sonuçları idealize hareketlere aittir (gürültüsüz/düzgün); gerçek videoda ID switch OLABİLİR.

``lap`` veya ``ultralytics`` kurulu değilse testler açıkça ATLANIR (sahte başarı üretilmez).
"""

from __future__ import annotations

import importlib.util
import os
import random
import unittest
from unittest import mock

os.environ.setdefault("YOLO_OFFLINE", "1")

from tracking_test_support import (  # noqa: E402
    H,
    W,
    box_at,
    lerp,
    make_result,
    make_tracker,
    run_frames,
)

from isg_vision.human_detection.results import clip_box_xyxy  # noqa: E402
from isg_vision.tracking.bytetrack_adapter import PersonTracker  # noqa: E402
from isg_vision.tracking.errors import (  # noqa: E402
    FrameOrderError,
    InvalidTrackingInputError,
    TrackerBackendError,
    TrackingConfigError,
)

AVAILABLE = importlib.util.find_spec("lap") is not None and importlib.util.find_spec("ultralytics") is not None
SKIP = "Gerçek ByteTrack testi: 'lap' ve 'ultralytics' kurulu olmalıdır (.venv/bin/python -m pip install lap==0.5.14); atlandı"
BUFFER = 30  # varsayılan track_buffer


def stay(name: str, cx: float, n: int, conf: float = 0.9) -> list[list[tuple]]:
    return [[(name, box_at(cx), conf)] for _ in range(n)]


@unittest.skipUnless(AVAILABLE, SKIP)
class RealBackendTest(unittest.TestCase):
    def test_real_bytetrack_loads_offline_without_network_or_pip(self) -> None:
        import socket
        import subprocess

        with mock.patch.object(socket.socket, "connect", side_effect=AssertionError("ağ!")), mock.patch.object(
            subprocess, "Popen", side_effect=AssertionError("alt süreç (pip?)!")
        ), mock.patch.object(subprocess, "check_output", side_effect=AssertionError("pip!")):
            tracker = make_tracker()
            tracker.load()
            run_frames(tracker, stay("A", 200, 5))
        info = tracker.backend_info
        self.assertEqual(info["versions"]["ultralytics"], "8.4.210")
        self.assertTrue(info["versions"]["lap"])
        self.assertIsNone(info["version_note"], "test edilen sürümde sürüm uyarısı olmamalı")
        self.assertEqual(os.environ.get("YOLO_AUTOINSTALL"), "False", "gizli paket kurulumu kapalı olmalı")
        self.assertEqual(os.environ.get("YOLO_OFFLINE"), "1")

    def test_real_tracker_matches_documented_bytetrack_output_contract(self) -> None:
        tracker = make_tracker()
        out = run_frames(tracker, stay("A", 200, 3))
        r = out.results[-1]
        self.assertEqual((r.active_count, r.detection_count, r.lost_count, r.tentative_count), (1, 1, 0, 0))
        t = r.tracks[0]
        self.assertEqual((t.track_id, t.session_id, t.class_id, t.class_name), (1, 1, 0, "person"))
        self.assertEqual(t.age_frames, 3)

    def test_identical_input_gives_identical_output(self) -> None:
        frames = [[("A", box_at(lerp(100, 400, i, 30)), 0.9), ("B", box_at(lerp(500, 150, i, 30)), 0.8)] for i in range(30)]
        first = run_frames(make_tracker(), frames)
        second = run_frames(make_tracker(), frames)
        for r1, r2 in zip(first.results, second.results):
            self.assertEqual([(t.track_id, t.bbox_xyxy) for t in r1.tracks], [(t.track_id, t.bbox_xyxy) for t in r2.tracks])

    def test_two_tracker_instances_do_not_share_id_counters(self) -> None:
        a, b = make_tracker(), make_tracker()
        run_frames(a, stay("A", 100, 3) + stay("B", 400, 3))
        out_b = run_frames(b, stay("X", 300, 3))
        self.assertEqual(out_b.results[-1].track_ids, (1,), "ikinci takipçinin ilk izi yine 1 olmalı (ID sayacı takipçiye özgü)")


@unittest.skipUnless(AVAILABLE, SKIP)
class SinglePersonTest(unittest.TestCase):
    """Test A — tek kişi takibi."""

    def test_person_in_first_frame_gets_id_and_keeps_it_while_moving(self) -> None:
        n, speed = 40, 5.0
        frames = [[("A", box_at(100 + speed * i), 0.9)] for i in range(n)]
        out = run_frames(make_tracker(), frames)
        self.assertEqual(out.ids("A"), [1] * n, "ilk karedeki kişi hemen ID almalı ve hareket boyunca korumalı")
        self.assertEqual(out.report.switches, 0)
        out.assert_unique_ids_every_frame(self)

    def test_box_moves_with_the_person(self) -> None:
        n, speed = 40, 5.0
        frames = [[("A", box_at(100 + speed * i), 0.9)] for i in range(n)]
        out = run_frames(make_tracker(), frames)
        errors = []
        for i, r in enumerate(out.results):
            tracked_cx = r.tracks[0].center[0]
            errors.append(abs(tracked_cx - (100 + speed * i)))
        # Ölçülen (5 px/kare, gürültüsüz): en büyük sapma 0,2 kare hareketi, oturduktan sonra ortalama 0,01 kare. Eşikler bunun birkaç katıdır.
        self.assertLess(max(errors), 0.5 * speed, f"Kalman kutusu ham harekete 0,5 kareden fazla uzaklaşmamalı; ölçülen en büyük sapma {max(errors):.2f}px")
        self.assertLess(sum(errors[10:]) / len(errors[10:]), 0.1 * speed, "oturduktan sonra ortalama sapma 0,1 kare hareketinden küçük olmalı")
        direction = [out.results[i + 1].tracks[0].center[0] - out.results[i].tracks[0].center[0] for i in range(n - 1)]
        self.assertTrue(all(d > 0 for d in direction), "kutu kişiyle aynı yönde ilerlemeli")

    def test_person_appearing_later_is_confirmed_on_second_consecutive_frame(self) -> None:
        out = run_frames(make_tracker(), [[]] * 3 + stay("A", 200, 5))
        self.assertEqual([r.active_count for r in out.results], [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(out.results[3].tentative_count, 1, "ilk görüldüğü karede onaylanmamış (tentative) iz olarak sayılır, görüntülenmez")
        self.assertEqual(out.ids("A")[4:], [1, 1, 1, 1])

    def test_confidence_and_detection_index_are_taken_from_matched_detection(self) -> None:
        tracker = make_tracker()
        out = run_frames(tracker, [[("N", box_at(500), 0.55), ("A", box_at(150), 0.93)]] * 3)  # A listede 2. sırada
        track = out.results[-1].tracks
        by_name = {("A" if t.detection_index == 1 else "N"): t for t in track}
        self.assertEqual(by_name["A"].confidence, 0.93)
        self.assertEqual(by_name["N"].confidence, 0.55)

    def test_age_increases_each_frame(self) -> None:
        out = run_frames(make_tracker(), stay("A", 200, 6))
        self.assertEqual([r.tracks[0].age_frames for r in out.results], [1, 2, 3, 4, 5, 6])


@unittest.skipUnless(AVAILABLE, SKIP)
class MultiPersonTest(unittest.TestCase):
    """Test B — üç kişi takibi."""

    def three(self, n: int = 40, rotate: bool = False) -> list[list[tuple]]:
        frames = []
        for i in range(n):
            items = [("R", box_at(500 + 2 * i), 0.95), ("L", box_at(100 + 2 * i), 0.80), ("M", box_at(300 - i), 0.60)]
            if rotate:
                k = i % 3
                items = items[k:] + items[:k]  # tespit sırası kare kare değişir (eşleme sıraya değil kutuya dayanmalı)
            frames.append(items)
        return frames

    def test_three_independent_unique_tracks(self) -> None:
        out = run_frames(make_tracker(), self.three())
        self.assertTrue(all(r.active_count == 3 for r in out.results))
        out.assert_unique_ids_every_frame(self)
        self.assertEqual(len({v for a in out.assignments for v in a.values()}), 3)
        self.assertEqual(out.report.switches, 0)

    def test_ids_follow_people_not_detection_order(self) -> None:
        out = run_frames(make_tracker(), self.three(rotate=True))
        self.assertEqual(out.report.switches, 0)
        self.assertEqual(out.report.takeovers, 0)
        self.assertEqual(len({v for a in out.assignments for v in a.values()}), 3)
        out.assert_unique_ids_every_frame(self)

    def test_ids_are_not_assigned_left_to_right(self) -> None:
        """§13 Test C: ID'ler soldan sağa sıralamayla verilmez; ilk karedeki tespit (güven) sırasıyla verilir."""
        out = run_frames(make_tracker(), self.three())
        first = out.assignments[0]
        self.assertEqual(first, {"R": 1, "L": 2, "M": 3})
        by_x = sorted(first, key=lambda name: {"L": 100, "M": 300, "R": 500}[name])
        self.assertNotEqual([first[n] for n in by_x], sorted(first.values()), "ID'ler soldan sağa artıyor olmamalı")

    def test_thirty_people_all_get_unique_ids(self) -> None:
        frames = [[(f"P{k}", box_at(20 + 20 * k + i, bottom=420, w=14, h=60), 0.9 - 0.005 * k) for k in range(30)] for i in range(10)]
        out = run_frames(make_tracker(), frames)
        self.assertEqual(out.results[-1].active_count, 30)
        out.assert_unique_ids_every_frame(self)
        self.assertEqual(out.report.switches, 0)


@unittest.skipUnless(AVAILABLE, SKIP)
class SwapTest(unittest.TestCase):
    """Test C — yer değiştirme."""

    def test_crossing_on_different_depths_keeps_ids_controlled_sequence(self) -> None:
        n = 40
        frames = [[("A", box_at(lerp(120, 520, i, n), bottom=330, w=40, h=110), 0.9),
                   ("B", box_at(lerp(520, 120, i, n), bottom=440, w=70, h=200), 0.85)] for i in range(n)]
        out = run_frames(make_tracker(), frames)
        self.assertTrue(out.report.is_continuous, out.report)
        out.assert_unique_ids_every_frame(self)

    def test_crossing_on_same_row_with_jitter_controlled_sequence(self) -> None:
        """İdealize (düzgün hızlı, 3 px gürültülü) kesişme: bu dizide ID switch yok. Gerçek videoda ID switch olabilir."""
        rng = random.Random(7)
        n = 60

        def jitter(box):  # type: ignore[no-untyped-def]
            return tuple(v + rng.uniform(-3, 3) for v in box)

        frames = [[("A", jitter(box_at(lerp(120, 520, i, n))), 0.9), ("B", jitter(box_at(lerp(520, 120, i, n))), 0.85)] for i in range(n)]
        out = run_frames(make_tracker(), frames)
        self.assertTrue(out.report.is_continuous, out.report)
        out.assert_unique_ids_every_frame(self)

    def test_instant_position_exchange_causes_id_switch_that_is_detected_and_reported(self) -> None:
        """Bilinen sınırlama: ByteTrack yalnızca konum/IoU kullanır, görünüm kullanmaz. İki kişi ardışık iki karede yer değiştirirse
        (kare kaybı, çok düşük FPS) numaralar KONUMA yapışır, kişiye değil: ID switch oluşur ve metrik bunu yakalar (gizlenmez)."""
        frames = [[("A", box_at(150), 0.9), ("B", box_at(450), 0.85)]] * 10 + [[("A", box_at(450), 0.9), ("B", box_at(150), 0.85)]] * 10
        out = run_frames(make_tracker(), frames)
        rep = out.report
        self.assertEqual((rep.switches, rep.takeovers), (2, 2), rep)
        self.assertFalse(rep.is_continuous)
        self.assertEqual(rep.track_ids_per_identity, {"A": (1, 2), "B": (2, 1)})
        out.assert_unique_ids_every_frame(self)  # karışmaya rağmen aynı karede aynı ID iki kişiye verilmez

    def test_three_people_rotating_positions_gradually(self) -> None:
        n = 60
        starts, ends = {"A": 100, "B": 300, "C": 500}, {"A": 300, "B": 500, "C": 100}
        frames = [[(k, box_at(lerp(starts[k], ends[k], i, n), bottom=400 + 8 * idx, w=56 + 6 * idx, h=160 + 14 * idx), 0.9 - 0.05 * idx)
                   for idx, k in enumerate("ABC")] for i in range(n)]
        out = run_frames(make_tracker(), frames)
        out.assert_unique_ids_every_frame(self)
        self.assertEqual(out.report.duplicate_frames, 0)
        self.assertEqual(out.report.takeovers, 0, out.report)


@unittest.skipUnless(AVAILABLE, SKIP)
class OcclusionTest(unittest.TestCase):
    """Test D — kısmi örtüşme."""

    N = 40

    def passing_frames(self, hidden: range | None = None) -> list[list[tuple]]:
        frames = []
        for i in range(self.N):
            items = [("B", box_at(lerp(100, 540, i, self.N), w=70, h=190), 0.88)]
            if hidden is None or i not in hidden:
                items.append(("A", box_at(320), 0.9))
            frames.append(items)
        return frames

    def test_person_passing_in_front_while_both_are_detected_controlled_sequence(self) -> None:
        out = run_frames(make_tracker(), self.passing_frames())
        self.assertTrue(out.report.is_continuous, out.report)
        self.assertTrue(all(r.active_count == 2 for r in out.results))

    def test_occluded_person_is_buffered_not_shown_and_resumes_with_same_id(self) -> None:
        out = run_frames(make_tracker(), self.passing_frames(hidden=range(16, 24)))
        a_id = out.ids("A")[0]
        hidden = out.results[16:24]
        self.assertTrue(all(r.active_count == 1 for r in hidden), "örtülen kişi AKTİF sayılmamalı")
        self.assertTrue(all(r.lost_count == 1 and r.lost_tracks[0].track_id == a_id for r in hidden), "tamponda kayıp iz olarak tutulmalı")
        self.assertEqual([r.lost_tracks[0].frames_since_seen for r in hidden], list(range(1, 9)))
        self.assertTrue(all(a_id not in r.track_ids for r in hidden))
        self.assertEqual(out.ids("A")[24:], [a_id] * (self.N - 24), "örtü bitince aynı ID ile devam etmeli (kısa örtüşme)")
        self.assertEqual(out.report.takeovers, 0)
        self.assertTrue(out.report.is_continuous, out.report)

    def test_detector_merging_two_overlapping_people_into_one_box_controlled_sequence(self) -> None:
        """YOLO NMS'i üst üste binen iki kişiden yalnızca birini bırakabilir (kutu birleşmesi)."""
        n = 60
        frames = []
        for i in range(n):
            ca, cb = lerp(120, 520, i, n), lerp(520, 120, i, n)
            items = [("A", box_at(ca), 0.9), ("B", box_at(cb), 0.85)]
            frames.append(items[:1] if abs(ca - cb) < 50 else items)
        out = run_frames(make_tracker(), frames)
        self.assertTrue(out.report.is_continuous, out.report)
        self.assertTrue(any(r.active_count == 1 for r in out.results), "birleşme anında yalnızca bir aktif iz olmalı")

    def test_long_occlusion_beyond_buffer_yields_new_id_not_a_silent_relink(self) -> None:
        frames = stay("A", 320, 10) + [[]] * (BUFFER + 5) + stay("A", 320, 5)
        out = run_frames(make_tracker(), frames)
        self.assertEqual(out.ids("A")[0], 1)
        self.assertEqual(out.ids("A")[-1], 2)
        self.assertEqual(out.report.switches, 1, "uzun kayıptan sonra yeni ID: ID switch olarak raporlanır (gizlenmez)")


@unittest.skipUnless(AVAILABLE, SKIP)
class ExitAndReenterTest(unittest.TestCase):
    """Test E — kadrajdan çıkma ve geri girme; tampon davranışı."""

    def gap_run(self, gap: int, cx: float = 200) -> object:
        return run_frames(make_tracker(), stay("A", 200, 5) + [[]] * gap + stay("A", cx, 3))

    def test_return_within_buffer_keeps_same_id(self) -> None:
        for gap in (1, 5, 15, BUFFER):
            with self.subTest(gap=gap):
                out = self.gap_run(gap)
                self.assertEqual(out.ids("A")[0], 1)
                self.assertEqual(out.ids("A")[5 + gap :], [1, 1, 1])
                self.assertEqual(len({v for a in out.assignments for v in a.values()}), 1)

    def test_return_after_buffer_gets_new_id(self) -> None:
        """Karakterizasyon (8.4.210): kurtarma penceresi track_buffer+1; track_buffer+2 kare kayıpta iz silinmiştir."""
        for gap in (BUFFER + 2, 60):
            with self.subTest(gap=gap):
                out = self.gap_run(gap)
                self.assertEqual(out.ids("A")[0], 1)
                self.assertEqual(out.ids("A")[-1], 2)
                self.assertEqual(out.ids("A")[5 + gap + 1 :], [2, 2], "yeni kişi gibi: önce onaylanmamış, sonra yeni ID ile aktif")

    def test_absent_person_is_lost_not_active_and_buffer_expires(self) -> None:
        out = run_frames(make_tracker(), stay("A", 200, 5) + [[]] * 40)
        gap = out.results[5:]
        self.assertTrue(all(r.active_count == 0 for r in gap), "görünmeyen kişi hiçbir karede aktif sayılmamalı")
        self.assertTrue(all(r.lost_count == 1 for r in gap[:BUFFER]), "tampon süresince kayıp iz olarak tutulur")
        self.assertEqual([r.lost_tracks[0].frames_since_seen for r in gap[:BUFFER]], list(range(1, BUFFER + 1)))
        self.assertTrue(all(r.lost_count == 0 for r in gap[BUFFER + 1 :]), "tampon dolunca iz bellekten silinir")
        self.assertEqual(out.results[-1].detection_count, 0)

    def test_newcomer_far_from_lost_person_gets_a_new_id(self) -> None:
        out = run_frames(make_tracker(), stay("A", 100, 5) + [[]] * 5 + stay("C", 540, 4))
        c_ids = [i for i in out.ids("C") if i is not None]
        self.assertEqual(c_ids, [2, 2, 2], "uzaktaki yeni kişi eski izi devralmamalı (IoU yok)")
        self.assertEqual(out.results[10].tentative_count, 1)

    def test_newcomer_at_lost_persons_location_inherits_id_known_limitation(self) -> None:
        """BİLİNEN SINIRLAMA: ByteTrack görünüm bilgisi kullanmaz. Tampon süresi içinde kaybolan kişinin son konumuna başka biri
        girerse eski ID'yi devralır (kimlik karışması). Bu test sınırlamayı belgeler; düzeltmez."""
        out = run_frames(make_tracker(), stay("A", 200, 5) + [[]] * 5 + stay("C", 200, 4))
        self.assertEqual([i for i in out.ids("C") if i is not None], [1, 1, 1, 1])
        self.assertEqual(out.report.takeovers, 1, "metrik bu devralmayı yakalar")

    def test_newcomer_at_same_location_after_buffer_gets_new_id(self) -> None:
        out = run_frames(make_tracker(), stay("A", 200, 5) + [[]] * (BUFFER + 3) + stay("C", 200, 4))
        self.assertEqual(out.ids("C")[-1], 2)

    def test_person_leaving_through_image_edge_and_returning_from_same_edge_gets_new_id_known_limitation(self) -> None:
        """BİLİNEN SINIRLAMA (ölçüldü: 2/4/6/10 px/kare hızları × 6/15/30 kare boşluk, hepsinde aynı sonuç): kişi kadrajın kenarından
        çıkıp AYNI kenardan, tampon süresi (30 kare) dolmadan geri girse bile YENİ ID alır. Nedeni: kayıp izin Kalman tahmini son
        hızıyla kadrajın DIŞINA ilerlemeyi sürdürür; kişi geri döndüğünde tahmin kutusu çoktan uzaklaşmıştır (IoU ≈ 0). Eski ID
        tamponda kayıp olarak bekler (ekranda görünmez). Hareket ederken ortada kaybolan kişi için bkz.
        ``test_moving_person_lost_briefly_returning_at_extrapolated_position_keeps_id``.

        Model varsayımı: görünür oranı %30'un altına düşen kısmi kutu (kenarda sıyrık şerit) YOLO tarafından tespit edilmez
        (gerçek YOLO eşiği ölçülmedi; bu, senaryonun varsayımıdır)."""
        step, min_visible = 4, 0.3

        def visible(cx: float):  # type: ignore[no-untyped-def]
            raw = box_at(cx)
            clipped = clip_box_xyxy(raw, W, H)
            if clipped is None:
                return None
            ratio = ((clipped[2] - clipped[0]) * (clipped[3] - clipped[1])) / ((raw[2] - raw[0]) * (raw[3] - raw[1]))
            return clipped if ratio >= min_visible else None

        frames: list[list[tuple]] = []
        for i in range(int((W + 60 - 300) / step) + 2):  # sağa yürür, sağ kenardan çıkar
            box = visible(300 + step * i)
            frames.append([("A", box, 0.9)] if box else [])
        left_at = next(i for i, f in enumerate(frames) if not f)
        frames += [[]] * 6
        reentry_start = len(frames)
        for i in range(int(120 / step) + 8):  # aynı kenardan geri girer
            box = visible(W + 40 - step * i)
            frames.append([("A", box, 0.9)] if box else [])
        out = run_frames(make_tracker(), frames)
        before = {i for i in out.ids("A")[:left_at] if i is not None}
        after = {i for i in out.ids("A")[reentry_start:] if i is not None}
        self.assertEqual(before, {1})
        self.assertEqual(after, {2}, "geri giren kişi yeni ID aldı (bilinen sınırlama)")
        self.assertEqual(out.report.switches, 1, "metrik bu kimlik kesintisini raporlar (gizlenmez)")
        out.assert_unique_ids_every_frame(self)
        entering = out.results[reentry_start + 1 :]
        self.assertTrue(any(r.active_count == 1 and r.lost_count == 1 for r in entering), "eski ID kayıp, yeni ID aktif: ayrı sayılır")

    def test_moving_person_lost_briefly_returning_at_extrapolated_position_keeps_id(self) -> None:
        """Karakterizasyon: hareket ederken kısa süre kaybolan kişi, sabit hızla ilerlemiş (tahmin edilen) konumda dönerse ID korunur."""
        for speed in (3, 6, 10):
            for gap in (5, 15):
                with self.subTest(speed=speed, gap=gap):
                    last = 100 + speed * 11
                    pre = [[("A", box_at(100 + speed * i), 0.9)] for i in range(12)]
                    post = [[("A", box_at(last + speed * (gap + 1 + j)), 0.9)] for j in range(4)]
                    out = run_frames(make_tracker(), pre + [[]] * gap + post)
                    self.assertEqual({v for a in out.assignments for v in a.values()}, {1})
                    self.assertEqual(out.report.switches, 0)

    def test_moving_person_returning_in_opposite_direction_gets_new_id_characterization(self) -> None:
        """Karakterizasyon (ölçüldü): kayıptan sonra ters yönde (tahminin tersine) dönen kişi 15 karelik boşlukta yeni ID alır."""
        speed, gap = 6, 15
        last = 100 + speed * 11
        pre = [[("A", box_at(100 + speed * i), 0.9)] for i in range(12)]
        post = [[("A", box_at(last - speed * (gap + 1) - speed * j), 0.9)] for j in range(4)]
        out = run_frames(make_tracker(), pre + [[]] * gap + post)
        self.assertEqual(out.ids("A")[-1], 2)
        self.assertEqual(out.report.switches, 1)


@unittest.skipUnless(AVAILABLE, SKIP)
class EmptyFrameTest(unittest.TestCase):
    """Test F — hiç insan bulunmayan kareler."""

    def test_only_empty_frames_are_handled_without_error(self) -> None:
        out = run_frames(make_tracker(), [[]] * 50)
        for r in out.results:
            self.assertEqual((r.detection_count, r.active_count, r.lost_count, r.tentative_count, r.dropped_tracks), (0, 0, 0, 0, 0))
            self.assertTrue(r.is_empty)
            self.assertEqual(r.tracks, ())

    def test_active_lost_separation_after_everyone_leaves(self) -> None:
        out = run_frames(make_tracker(), stay("A", 100, 5) + [[("A", box_at(100), 0.9), ("B", box_at(400), 0.8)]] * 3 + [[]] * 3)
        last = out.results[-1]
        self.assertEqual((last.detection_count, last.active_count), (0, 0), "kimse yokken ACTIVE TRACKS 0 olmalı")
        self.assertEqual(last.lost_count, 2, "kayıp izler ayrı sayılır")

    def test_empty_frames_between_people(self) -> None:
        out = run_frames(make_tracker(), stay("A", 100, 5) + [[]] * 4 + stay("A", 100, 5))
        self.assertEqual(out.ids("A")[:5] + out.ids("A")[9:], [1] * 10)


@unittest.skipUnless(AVAILABLE, SKIP)
class MissedDetectionTest(unittest.TestCase):
    """Test G — YOLO'nun bir veya birkaç karede kişiyi kaçırması."""

    def test_short_misses_keep_the_same_id_and_explain_why(self) -> None:
        for miss in (1, 2, 3, 8):
            with self.subTest(miss=miss):
                frames = stay("A", 200, 6) + [[]] * miss + stay("A", 200, 6)
                out = run_frames(make_tracker(), frames)
                self.assertEqual(out.report.switches, 0)
                self.assertEqual(len({v for a in out.assignments for v in a.values()}), 1, "yeni ID doğmamalı")
                during = out.results[6 : 6 + miss]
                self.assertTrue(all(r.active_count == 0 for r in during), "kaçırılan karede iz aktif GÖSTERİLMEZ (kaybolmuş, tamponda)")
                self.assertTrue(all(r.lost_count == 1 for r in during), "iz neden devam ediyor: tamponda bekleyen kayıp iz olarak")
                self.assertEqual(out.results[6 + miss].track_ids, (1,))
                self.assertEqual(out.results[6 + miss].lost_count, 0, "yeniden eşleşince kayıp sayısı düşer")

    def test_miss_while_another_person_is_visible(self) -> None:
        frames = [[("A", box_at(100), 0.9), ("B", box_at(450), 0.9)]] * 5 + [[("B", box_at(450), 0.9)]] * 2 + [[("A", box_at(100), 0.9), ("B", box_at(450), 0.9)]] * 5
        out = run_frames(make_tracker(), frames)
        self.assertTrue(out.report.is_continuous, out.report)
        self.assertEqual([r.active_count for r in out.results[5:7]], [1, 1])

    def test_missed_track_that_does_not_return_disappears_visibly_not_silently(self) -> None:
        out = run_frames(make_tracker(), stay("A", 200, 6) + [[]] * (BUFFER + 5))
        self.assertEqual(out.results[-1].lost_count, 0)
        self.assertEqual(out.results[-1].active_count, 0)
        self.assertGreater(max(r.lost_count for r in out.results), 0, "iz önce kayıp olarak görünür, sonra silinir (hata gizlenmez)")


@unittest.skipUnless(AVAILABLE, SKIP)
class FalsePositiveTest(unittest.TestCase):
    """Test H — yorgan türü yanlış pozitif (FP-4A-001).

    UYARI: Gerçek yorgan görüntüsü kaydedilmedi; olay BİREBİR YENİDEN ÜRETİLMEDİ. Aşağıdakiler kontrollü sahte tespit dizileridir ve
    yalnızca "ByteTrack böyle bir tespit dizisiyle ne yapar?" sorusuna cevap verir. **ByteTrack yanlış pozitifi ENGELLEMEZ**; yalnızca
    en az iki ardışık karede görülmeyen tespitlere iz vermez.
    """

    @staticmethod
    def blanket(pattern: list[int], conf: float = 0.5, **cfg: object):  # type: ignore[no-untyped-def]
        frames = [[("FP", box_at(560, bottom=450, w=120, h=60), conf)] if on else [] for on in pattern]
        return run_frames(make_tracker(**cfg), frames)

    def test_single_frame_false_positive_never_becomes_a_visible_track(self) -> None:
        for conf in (0.3, 0.6, 0.95):  # güven skoru onay şartını değiştirmez
            with self.subTest(conf=conf):
                out = self.blanket([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], conf)
                self.assertTrue(all(r.active_count == 0 for r in out.results))
                self.assertEqual(out.results[4].tentative_count, 1)
                self.assertEqual(out.results[5].tentative_count, 0, "onaylanmayan iz silinir")
                self.assertTrue(all(r.lost_count == 0 for r in out.results))

    def test_single_frame_false_positive_consumes_an_id_number(self) -> None:
        """Track ID'ler bitişik olmak zorunda değildir: onaylanmadan silinen geçici iz de numara tüketir."""
        frames = [[]] * 3 + [[("FP", box_at(560, bottom=450, w=120, h=60), 0.5)]] + [[]] * 2 + stay("A", 200, 4)
        out = run_frames(make_tracker(), frames)
        self.assertEqual(out.ids("A")[-1], 2, "yanlış pozitif 1 numarayı tüketti; gerçek kişi 2 aldı")

    def test_false_positive_in_the_first_frame_of_a_session_is_shown_immediately(self) -> None:
        """İSTİSNA (Ultralytics 8.4.210): oturumun 1. karesinde tespit edilenler onay beklemeden iz alır (her reset sonrası da)."""
        out = self.blanket([1, 0, 0, 0, 0, 0])
        self.assertEqual([r.active_count for r in out.results], [1, 0, 0, 0, 0, 0])
        self.assertEqual(out.results[1].lost_count, 1, "bir kare görünüp tamponda kaybolur")

    def test_two_and_three_frame_false_positives_are_shown_for_one_or_two_frames(self) -> None:
        self.assertEqual([r.active_count for r in self.blanket([0, 0, 1, 1, 0, 0, 0]).results], [0, 0, 0, 1, 0, 0, 0])
        self.assertEqual([r.active_count for r in self.blanket([0, 0, 1, 1, 1, 0, 0]).results], [0, 0, 0, 1, 1, 0, 0])

    def test_short_false_positive_leaves_a_lost_track_that_expires_after_the_buffer(self) -> None:
        out = self.blanket([0, 0, 1, 1, 1] + [0] * 40)
        self.assertEqual([r.lost_count for r in out.results[5 : 5 + BUFFER]], [1] * BUFFER)
        self.assertEqual(out.results[-1].lost_count, 0)

    def test_flickering_false_positive_every_other_frame_is_never_confirmed(self) -> None:
        out = self.blanket([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self.assertTrue(all(r.active_count == 0 for r in out.results))

    def test_intermittent_false_positive_with_two_consecutive_frames_keeps_one_stable_id(self) -> None:
        """Karakterizasyon: 2 ardışık kare + 1 boşluk deseni onaylanır ve kayıp tamponu sayesinde TEK, kararlı ID ile sürer."""
        out = self.blanket([0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1])
        ids = {v for a in out.assignments for v in a.values()}
        self.assertEqual(len(ids), 1)
        self.assertGreater(sum(r.active_count for r in out.results), 4)

    def test_persistent_static_false_positive_keeps_a_track_known_limitation(self) -> None:
        """BİLİNEN SINIRLAMA: ByteTrack kalıcı (statik) bir yanlış pozitifi elemez. Yorgan kalıcı olarak insan sanılırsa sürekli bir
        Track ID alır. Çözüm takipte değil, tespit kalitesinde/sonraki doğrulamalardadır (yüz, KKD, hareket süzgeci)."""
        out = self.blanket([1] * 60)
        self.assertEqual(out.ids("FP"), [1] * 60)
        self.assertEqual(out.results[-1].tracks[0].age_frames, 60)

    def test_low_confidence_detection_cannot_create_a_track(self) -> None:
        """ByteTrack'in temel fikri: düşük güvenli (0.10–0.25) tespit yeni iz doğurmaz."""
        out = run_frames(make_tracker(), stay("FP", 200, 10, conf=0.15))
        self.assertTrue(all(r.active_count == 0 and r.tentative_count == 0 for r in out.results))

    def test_low_confidence_detections_can_sustain_an_existing_track(self) -> None:
        frames = stay("A", 200, 4, conf=0.9) + stay("A", 200, 6, conf=0.15)
        out = run_frames(make_tracker(), frames)
        self.assertEqual(out.ids("A"), [1] * 10, "iz düşük güvenli ikinci eşleştirmeyle sürer")

    def test_raising_low_thresh_removes_the_recovery(self) -> None:
        frames = stay("A", 200, 4, conf=0.9) + stay("A", 200, 4, conf=0.15)
        out = run_frames(make_tracker(track_low_thresh=0.2), frames)  # 0.15 artık yok sayılır
        self.assertEqual(out.ids("A")[:4], [1] * 4)
        self.assertTrue(all(r.active_count == 0 for r in out.results[4:]))

    def test_new_track_thresh_separates_starting_from_continuing(self) -> None:
        """§9 soru 5: yeni iz başlatma eşiği (new_track_thresh) ile iz sürdürme eşiği (track_high_thresh) ayrıdır."""
        out = run_frames(make_tracker(new_track_thresh=0.5), stay("A", 200, 8, conf=0.4))
        self.assertTrue(all(r.active_count == 0 for r in out.results), "0.4 < new_track_thresh: yeni iz doğmaz")
        out = run_frames(make_tracker(new_track_thresh=0.5), stay("A", 200, 3, conf=0.9) + stay("A", 200, 5, conf=0.3))
        self.assertEqual(out.ids("A"), [1] * 8, "iz 0.9 ile başlamışsa 0.3 ile (>= track_high_thresh) sürebilir")

    def test_higher_new_track_thresh_blocks_mid_confidence_false_positive_but_not_real_high_confidence_person(self) -> None:
        frames = [[("FP", box_at(560, bottom=450, w=120, h=60), 0.45), ("A", box_at(200), 0.9)]] * 8
        loose = run_frames(make_tracker(), frames)
        strict = run_frames(make_tracker(new_track_thresh=0.6), frames)
        self.assertEqual(set(loose.results[-1].track_ids), {1, 2}, "varsayılan eşikte 0.45'lik yanlış pozitif de iz alır")
        self.assertEqual(strict.assignments[-1], {"A": 1}, "new_track_thresh=0.6: yalnızca 0.9'luk kişi iz alır")
        # Uyarı: bu yalnızca 0.45 güvenli sahte tespit için geçerlidir; gerçek yorganın skoru ölçülmedi.


@unittest.skipUnless(AVAILABLE, SKIP)
class SourceSwitchTest(unittest.TestCase):
    """Test I — kaynak değişimi / yeni oturum."""

    def test_source_change_does_not_carry_tracks_or_lost_buffer(self) -> None:
        tracker = make_tracker()
        first = run_frames(tracker, stay("A", 200, 6) + [[]] * 3, source="video:/a.mp4")
        self.assertEqual(first.results[-1].lost_count, 1)
        second = run_frames(tracker, stay("C", 200, 3), source="camera:0")  # aynı konumda başka kaynak: eski iz devralınmamalı
        r0 = second.results[0]
        self.assertEqual(r0.reset_reason, "source_changed")
        self.assertEqual((r0.session_id, r0.lost_count), (2, 0), "eski kaynağın kayıp izleri taşınmamalı")
        self.assertEqual(second.ids("C"), [1, 1, 1], "yeni oturumda numaralar 1'den başlar (oturum 2'de)")
        self.assertEqual({t.key for t in second.results[-1].tracks}, {(2, 1)})
        self.assertNotEqual({t.key for t in second.results[-1].tracks}, {t.key for t in first.results[0].tracks})

    def test_explicit_reset_clears_everything_and_restarts_ids(self) -> None:
        tracker = make_tracker()
        run_frames(tracker, stay("A", 200, 6) + [[]] * 3)
        tracker.reset()
        out = run_frames(tracker, stay("B", 200, 3))
        self.assertEqual((out.results[0].lost_count, out.results[0].session_id), (0, 2))
        self.assertEqual(out.ids("B"), [1, 1, 1])

    def test_image_size_change_resets_real_tracker(self) -> None:
        tracker = make_tracker()
        run_frames(tracker, stay("A", 200, 5), source="camera:0")
        result, names = make_result([("A", box_at(200), 0.9)], 5, size=(1280, 720))
        out = tracker.update(result, source="camera:0")
        self.assertEqual((out.reset_reason, out.session_id, out.lost_count), ("image_size_changed", 2, 0))

    def test_frame_numbers_restart_after_source_change_without_error(self) -> None:
        tracker = make_tracker()
        run_frames(tracker, stay("A", 200, 8), source="video:/a.mp4")
        run_frames(tracker, stay("A", 200, 3), start_index=0, source="video:/b.mp4")  # numara 0'a döndü: FrameOrderError OLMAMALI


@unittest.skipUnless(AVAILABLE, SKIP)
class InvalidInputRealTrackerTest(unittest.TestCase):
    """Test J — hatalı girişler gerçek takipçide durumu bozmaz."""

    def test_rejected_frames_do_not_age_or_corrupt_tracks(self) -> None:
        tracker = make_tracker()
        run_frames(tracker, stay("A", 200, 5) + [[]] * 2)  # A kayıp, frames_since_seen = 2
        before = tracker.stats
        with self.assertRaises(FrameOrderError):
            tracker.update(make_result([("A", box_at(200), 0.9)], 6, timestamp=6 / 30.0)[0])  # tekrar eden kare (6 zaten işlendi)
        bad, _ = make_result([("A", (600.0, 10.0, 700.0, 100.0), 0.9)], 7)
        with self.assertRaises(InvalidTrackingInputError):
            tracker.update(bad)
        self.assertEqual(tracker.stats, before)
        out = run_frames(tracker, stay("A", 200, 2), start_index=7)
        self.assertEqual(out.ids("A"), [1, 1], "reddedilen kareler izi yaşlandırmadı: A aynı ID ile döndü")

    def test_backend_crash_is_reported_as_tracker_backend_error(self) -> None:
        tracker = make_tracker()
        run_frames(tracker, stay("A", 200, 3))
        with mock.patch.object(tracker._backend._tracker, "update", side_effect=RuntimeError("iç hata")):  # type: ignore[union-attr]
            with self.assertRaisesRegex(TrackerBackendError, "iç hata"):
                run_frames(tracker, stay("A", 200, 1), start_index=3)
        tracker.reset()  # belgelenen kurtarma yolu
        out = run_frames(tracker, stay("A", 200, 2))
        self.assertEqual(out.ids("A"), [1, 1])

    def test_unexpected_backend_output_shape_is_rejected(self) -> None:
        import numpy as np

        tracker = make_tracker()
        run_frames(tracker, stay("A", 200, 2))
        with mock.patch.object(tracker._backend._tracker, "update", return_value=np.zeros((2, 9), dtype=np.float32)):  # type: ignore[union-attr]
            with self.assertRaisesRegex(TrackerBackendError, "beklenmeyen biçimde"):
                run_frames(tracker, stay("A", 200, 1), start_index=2)

    def test_invalid_config_fails_before_any_tracker_is_created(self) -> None:
        with self.assertRaises(TrackingConfigError):
            make_tracker(track_high_thresh=0.9, track_low_thresh=0.95)

    def test_config_values_actually_reach_the_real_tracker(self) -> None:
        tracker = make_tracker(track_buffer=7, match_thresh=0.6, new_track_thresh=0.5, track_high_thresh=0.4, track_low_thresh=0.2, fuse_score=False)
        run_frames(tracker, stay("A", 200, 2, conf=0.9))
        args = tracker._backend._tracker.args  # type: ignore[union-attr]
        self.assertEqual(
            (args.track_buffer, args.match_thresh, args.new_track_thresh, args.track_high_thresh, args.track_low_thresh, args.fuse_score),
            (7, 0.6, 0.5, 0.4, 0.2, False),
        )

    def test_small_track_buffer_expires_sooner(self) -> None:
        out = run_frames(make_tracker(track_buffer=3), stay("A", 200, 5) + [[]] * 6 + stay("A", 200, 3))
        self.assertEqual(out.ids("A")[0], 1)
        self.assertEqual(out.ids("A")[-1], 2)


if __name__ == "__main__":
    unittest.main()
