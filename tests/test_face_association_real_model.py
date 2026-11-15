"""GERÇEK YOLO11n + GERÇEK ByteTrack + GERÇEK YuNet birlikte (mock model/takipçi/yüz tespitçisi YOK): yüz–kişi ilişkilendirme entegrasyonu.

Mock testler gerçek zincirin çalıştığını kanıtlamaz; bu dosya ``PersonDetector`` (``yolo11n.pt``) + ``PersonTracker`` (Ultralytics ByteTrack) +
``FaceDetector`` (``yunet`` ONNX) + ``FacePersonAssociator`` zincirini AYNI süreçte ve ``run_face_association`` döngüsüyle sınar.

* Ağırlıklar, ``lap`` veya Ultralytics örnek görüntüleri yoksa testler açıkça ATLANIR; hiçbir şey indirilmez/kurulmaz, internet gerekmez.
* Girdiler: ``zidane.jpg`` / ``bus.jpg`` (gerçek fotoğraflar, tekrarlanan kare), ``scenes`` modülündeki SENTETİK MONTAJ senaryoları (gerçek kişi
  kırpıntıları bilinen konumlarda hareket eder; ground-truth bilinir) ve bunların mp4 kaydı.
* **KANIT SINIRI:** bunlar tek fotoğraf karesinden kırpılmış kişilerin montajıdır; gerçek farklı kişilerin hareket ettiği bir video DEĞİLDİR ve gerçek
  kamerada/gerçek çoklu-kişi sahnesinde ilişkilendirme doğruluğunu, ID switch oranını veya IDF1/HOTA/MOTA'yı kanıtlamaz. Beklenen sayılar SHA-256'sı sabit
  ağırlıklar, ``ultralytics==8.4.210`` ve CPU ile ölçülmüştür; sınır durumlarda (ör. fotoğraf tutan kişi) yalnızca GÖZLENEN davranış kaydedilir (garanti değil).
* Bu dosya süreç genelinde OpenCV iş parçacığı ayarını DEĞİŞTİRMEZ (alt süreç testleri hariç).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

os.environ.setdefault("YOLO_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from camera_test_support import FakeCapture, FakeFactory, FakeWindow  # noqa: E402

from isg_vision.camera.capture import FrameSource  # noqa: E402
from isg_vision.camera.config import SourceConfig  # noqa: E402
from isg_vision.face_association import AssociationConfig, FacePersonAssociator, cli, scenes  # noqa: E402
from isg_vision.face_association.evaluation import SceneEvaluation, evaluate_scene  # noqa: E402
from isg_vision.face_association.geometry import iou  # noqa: E402
from isg_vision.face_association.pipeline import REASON_END_OF_STREAM, REASON_MAX_FRAMES, run_face_association  # noqa: E402
from isg_vision.face_association.results import STATUS_AMBIGUOUS, STATUS_MATCHED, STATUS_NO_FACE  # noqa: E402
from isg_vision.face_detection.config import DEFAULT_MODEL_PATH as FACE_MODEL_PATH  # noqa: E402
from isg_vision.face_detection.detector import FaceDetector  # noqa: E402
from isg_vision.human_detection.config import DEFAULT_MODEL_PATH as YOLO_MODEL_PATH, DetectionConfig, torch_mps_status  # noqa: E402
from isg_vision.human_detection.person_detector import PersonDetector  # noqa: E402
from isg_vision.tracking.bytetrack_adapter import PersonTracker  # noqa: E402

_ultra = importlib.util.find_spec("ultralytics")
ASSETS = Path(_ultra.origin).parent / "assets" if _ultra and _ultra.origin else None
SAMPLES_PRESENT = bool(ASSETS and (ASSETS / "bus.jpg").is_file() and (ASSETS / "zidane.jpg").is_file())
READY = YOLO_MODEL_PATH.is_file() and FACE_MODEL_PATH.is_file() and SAMPLES_PRESENT and importlib.util.find_spec("lap") is not None
SKIP_REASON = (
    "Gerçek YOLO+ByteTrack+YuNet testi: models/human_detection/yolo11n.pt (scripts/download_person_model.py), "
    "models/face_detection/*.onnx (scripts/download_face_model.py), 'lap' ve ultralytics örnek görüntüleri gerekir; atlandı"
)
SCRIPT = ROOT / "scripts" / "run_face_person_association.py"


def run_frames(frames: list[np.ndarray], detector: PersonDetector, face_detector: FaceDetector, *, config: AssociationConfig | None = None, window=None):  # type: ignore[no-untyped-def]
    """Kareleri GERÇEK döngüden geçirir; ``(sonuçlar, özet, takipçi, ilişkilendirici)`` döndürür."""
    height, width = frames[0].shape[:2]
    capture = FakeCapture(list(frames), props={cv2.CAP_PROP_FRAME_WIDTH: width, cv2.CAP_PROP_FRAME_HEIGHT: height})
    source = FrameSource(SourceConfig.camera(0, width=width, height=height), capture_factory=FakeFactory(capture), max_read_failures=1, sleep=lambda s: None)
    tracker = PersonTracker()
    tracker.load()
    associator = FacePersonAssociator(config)
    results = []
    with source:
        summary = run_face_association(source, detector, tracker, face_detector, associator, window, max_frames=len(frames),
                                       on_result=lambda p, d, t, f, a: results.append(a))
    return results, summary, tracker, associator


@unittest.skipUnless(READY, SKIP_REASON)
class RealStackCase(unittest.TestCase):
    detector: PersonDetector
    face_detector: FaceDetector
    actors: dict[str, scenes.Actor]
    _cache: dict[str, tuple[list[scenes.SceneFrame], list, SceneEvaluation]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.detector = PersonDetector(DetectionConfig(device="cpu"))
        cls.detector.load()
        cls.face_detector = FaceDetector()
        cls.face_detector.load()
        cls.actors = scenes.load_actors()
        cls._cache = {}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.detector.close()
        cls.face_detector.close()

    @classmethod
    def scenario(cls, name: str, **kwargs):  # type: ignore[no-untyped-def]
        key = f"{name}:{kwargs}"
        if key not in cls._cache:
            frames = scenes.render_scenario(cls.actors, scenes.SCENARIOS[name](**kwargs))
            results, _, _, _ = run_frames([f.image for f in frames], cls.detector, cls.face_detector)
            cls._cache[key] = (frames, results, evaluate_scene(frames, results))
        return cls._cache[key]


class RealPhotographsTest(RealStackCase):
    """Gerçek fotoğraflar (``zidane.jpg``, ``bus.jpg``): tekrarlanan kare; ilişkilendirme sonucu görüntüde GÖZLE doğrulanmıştır (referans kutular)."""

    ANCELOTTI_FACE = (917.0, 99.0, 1053.0, 273.0)
    ZIDANE_FACE = (555.0, 261.0, 656.0, 422.0)
    BUS_FACES = ((113.0, 416.0, 155.0, 470.0), (268.0, 423.0, 306.0, 473.0))

    def run_image(self, name: str, n: int = 5):  # type: ignore[no-untyped-def]
        image = cv2.imread(str(ASSETS / f"{name}.jpg"))  # type: ignore[operator]
        return run_frames([image.copy() for _ in range(n)], self.detector, self.face_detector)

    def test_zidane_two_overlapping_people_are_matched_to_their_own_faces_and_the_blurry_background_face_stays_unmatched(self) -> None:
        results, summary, _, _ = self.run_image("zidane")
        last = results[-1]
        self.assertEqual((last.person_count, last.face_count, last.matched_count, last.unmatched_face_count), (2, 3, 2, 1))
        by_face = {}
        for p in last.persons:
            self.assertEqual(p.status, STATUS_MATCHED)
            best = max((self.ANCELOTTI_FACE, self.ZIDANE_FACE), key=lambda ref: iou(p.face_bbox, ref))  # type: ignore[arg-type]
            self.assertGreaterEqual(iou(p.face_bbox, best), 0.6)  # type: ignore[arg-type]
            by_face[best] = p
        self.assertEqual(len(by_face), 2, "iki farklı yüz iki farklı kişiye bağlanmalı")
        ancelotti, zidane = by_face[self.ANCELOTTI_FACE], by_face[self.ZIDANE_FACE]
        self.assertGreater(ancelotti.person_bbox[0], 600.0, "Ancelotti'nin yüzü sağdaki (x1≈748) kişiye")
        self.assertLess(zidane.person_bbox[0], 300.0, "Zidane'ın yüzü soldaki geniş (x1≈149) kişiye")
        unmatched = [f for f in last.faces if not f.is_matched]
        self.assertEqual(len(unmatched), 1)
        self.assertGreater(unmatched[0].face_bbox[0], 1000.0, "sağ kenardaki bulanık arka plan yüzü")
        self.assertIsNone(unmatched[0].track_id, "eşleşmeyen yüze Track ID yazılmaz")
        # iki kişi büyük ölçüde örtüşür: eşleşme doğru ama süreklilik güvenilir sayılmaz
        for p in last.persons:
            self.assertFalse(p.trusted_match)
            self.assertGreater(p.continuity.overlap_fraction, 0.5)  # type: ignore[union-attr]
        self.assertEqual(summary.reason, REASON_MAX_FRAMES)

    def test_bus_two_faces_matched_and_people_without_faces_reported_as_no_face(self) -> None:
        results, summary, _, _ = self.run_image("bus")
        last = results[-1]
        self.assertEqual((last.person_count, last.face_count, last.matched_count, last.no_face_count, last.unmatched_face_count), (4, 2, 2, 2, 0))
        owners = set()
        for p in last.persons:
            if p.is_matched:
                ref = max(self.BUS_FACES, key=lambda r: iou(p.face_bbox, r))  # type: ignore[arg-type]
                self.assertGreaterEqual(iou(p.face_bbox, ref), 0.6)  # type: ignore[arg-type]
                owners.add(ref)
                self.assertGreaterEqual(p.score, 0.99)
            else:
                self.assertEqual(p.status, STATUS_NO_FACE)
        self.assertEqual(owners, set(self.BUS_FACES))
        self.assertEqual(summary.persons_max, 4, "yüzü olmayan kişiler takipten çıkarılmadı")

    def test_a_face_is_never_assigned_to_two_people_in_any_real_frame(self) -> None:
        for name in ("zidane", "bus"):
            results, _, _, _ = self.run_image(name, 4)
            for r in results:
                faces_used = [p.face_index for p in r.persons if p.is_matched]
                self.assertEqual(len(faces_used), len(set(faces_used)))


class RealSingleCallTest(RealStackCase):
    def test_each_real_stage_runs_exactly_once_per_frame(self) -> None:
        class CountingModel:
            """Gerçek ``cv2.FaceDetectorYN``'i sarar (nesnenin ``detect``'i salt okunurdur); çağrıları sayar."""

            def __init__(self, real) -> None:  # type: ignore[no-untyped-def]
                self.real, self.calls = real, 0

            def setInputSize(self, size):  # type: ignore[no-untyped-def]  # noqa: N802
                return self.real.setInputSize(size)

            def detect(self, image):  # type: ignore[no-untyped-def]
                self.calls += 1
                return self.real.detect(image)

        frames = scenes.render_scenario(self.actors, scenes.side_by_side(12))
        images = [f.image for f in frames]
        real_yunet = self.face_detector._model
        counting = CountingModel(real_yunet)
        height, width = images[0].shape[:2]
        capture = FakeCapture(list(images), props={cv2.CAP_PROP_FRAME_WIDTH: width, cv2.CAP_PROP_FRAME_HEIGHT: height})
        source = FrameSource(SourceConfig.camera(0, width=width, height=height), capture_factory=FakeFactory(capture), max_read_failures=1, sleep=lambda s: None)
        tracker = PersonTracker()
        tracker.load()
        self.face_detector._model = counting
        try:
            with mock.patch.object(self.detector, "detect", wraps=self.detector.detect) as yolo, \
                    mock.patch.object(self.face_detector, "detect", wraps=self.face_detector.detect) as yunet, \
                    mock.patch.object(self.detector._model, "predict", wraps=self.detector._model.predict) as predict, \
                    mock.patch.object(tracker._backend, "update", wraps=tracker._backend.update) as byte, source:
                summary = run_face_association(source, self.detector, tracker, self.face_detector, FacePersonAssociator(), max_frames=len(images))
        finally:
            self.face_detector._model = real_yunet
        n = len(images)
        self.assertEqual(summary.frames_processed, n)
        self.assertEqual((yolo.call_count, predict.call_count), (n, n), "YOLO kare başına yalnızca bir kez")
        self.assertEqual((yunet.call_count, counting.calls), (n, n), "YuNet kare başına yalnızca bir kez")
        self.assertEqual(byte.call_count, n, "ByteTrack kare başına yalnızca bir kez")

    def test_models_are_loaded_once_and_not_reloaded_per_frame(self) -> None:
        loads = (self.detector.load_count, self.face_detector.load_count)
        self.scenario("single_person")
        self.assertEqual((self.detector.load_count, self.face_detector.load_count), loads)


class RealCoreScenarioTest(RealStackCase):
    """T1-T3: tek / iki yan yana / üç kişi. Yanlış eşleşme YOK, ID'ler sabit, her görünür yüz doğru kişiye bağlı."""

    def assert_clean(self, name: str, actors: int) -> SceneEvaluation:
        frames, results, ev = self.scenario(name)
        self.assertEqual(ev.invalid_frames, 0)
        self.assertEqual(ev.wrong_pairs, 0, ev.wrong_examples)
        self.assertEqual(ev.decoy_on_person_pairs, 0)
        self.assertGreaterEqual(ev.recall, 0.97, f"{name}: görünür yüzlerin ≥ %97'si doğru kişiye bağlanmalı")  # type: ignore[operator]
        self.assertTrue(ev.id_report.is_continuous, f"{name}: ID switch/devralma beklenmiyor: {ev.id_report}")
        self.assertEqual(len(ev.track_ids_per_actor), actors)
        for ids in ev.track_ids_per_actor.values():
            self.assertEqual(len(ids), 1)
        last = results[-1]
        self.assertEqual((last.matched_count, last.trusted_match_count, last.ambiguous_person_count, last.unmatched_face_count), (actors, actors, 0, 0))
        return ev

    def test_t1_single_person(self) -> None:
        self.assert_clean("single_person", 1)

    def test_t2_two_people_side_by_side_get_two_ids_and_their_own_faces(self) -> None:
        ev = self.assert_clean("side_by_side", 2)
        self.assertEqual(sorted(i[0] for i in ev.track_ids_per_actor.values()), [1, 2])

    def test_t3_three_people_at_once(self) -> None:
        ev = self.assert_clean("three_people", 3)
        self.assertEqual(sorted(i[0] for i in ev.track_ids_per_actor.values()), [1, 2, 3])


class RealOcclusionAndSwapTest(RealStackCase):
    """T4-T6: yer değiştirme, önünden geçme, kadrajdan çıkıp geri girme. İlişkilendirme YANLIŞ olmamalı; ID switch (varsa) AYRI raporlanır ve güveni düşürür."""

    def test_t4_two_people_swap_places_without_any_wrong_association(self) -> None:
        frames, results, ev = self.scenario("swap_places")
        self.assertEqual(ev.wrong_pairs, 0, ev.wrong_examples)
        self.assertEqual(ev.decoy_on_person_pairs, 0)
        self.assertGreaterEqual(ev.recall, 0.9)  # type: ignore[operator]
        peak = max(max((p.continuity.overlap_fraction for p in r.persons), default=0.0) for r in results)  # type: ignore[union-attr]
        self.assertGreater(peak, AssociationConfig().overlap_guard, "senaryo örtüşme eşiğini aşan takip kutuları üretmeli")
        self.assertIn("overlap_end", ev.breaks, "örtüşme sonu süreklilik kesintisi olarak işaretlenmeli")
        for r in results:  # örtüşme sırasında eşleşme olsa bile güvenilir (kanıt biriktirmeye uygun) sayılmaz
            for p in r.persons:
                if p.continuity.overlap_fraction >= 0.25:  # type: ignore[union-attr]
                    self.assertFalse(p.trusted_match)

    def test_t5_person_passing_in_front_hides_the_other_face_without_misassigning_it(self) -> None:
        frames, results, ev = self.scenario("pass_in_front")
        self.assertEqual(ev.wrong_pairs, 0, ev.wrong_examples)
        self.assertEqual(ev.decoy_on_person_pairs, 0)
        hidden = [i for i, f in enumerate(frames) if "B" in f.truth and f.truth["B"].face_box is None]
        self.assertGreater(len(hidden), 5, "B'nin yüzü örtülen kareler olmalı")
        for i in hidden:  # B'nin yüzü örtülüyken B'ye (veya başka bir kişiye) o yüz bağlanamaz: eşleşen her çift gerçek bir yüzle olmalı
            for p in results[i].persons:
                if p.is_matched:
                    owner_faces = [t.face_box for t in frames[i].truth.values() if t.face_box is not None]
                    self.assertTrue(any(iou(p.face_bbox, ref) >= 0.3 for ref in owner_faces), f"kare {i}: eşleşen yüz gerçek bir yüz değil")  # type: ignore[arg-type]
        self.assertGreater(ev.recall, 0.9)  # type: ignore[operator]

    def test_id_changes_are_reported_separately_and_new_ids_never_inherit_history(self) -> None:
        """ID switch yaşansa da (pass_in_front, leave_and_return) yeni Track ID eski kişinin eşleşme geçmişini DEVRALMAZ ve ilk karede güvenilir sayılmaz."""
        for name in ("pass_in_front", "leave_and_return"):
            frames, results, ev = self.scenario(name)
            self.assertEqual(ev.wrong_pairs, 0, f"{name}: {ev.wrong_examples}")
            first_seen: dict[int, int] = {}
            for i, r in enumerate(results):
                for p in r.persons:
                    first_seen.setdefault(p.track_id, i)
            for track_id, i in first_seen.items():
                p = results[i].person(track_id)
                assert p is not None and p.continuity is not None
                self.assertEqual((p.continuity.epoch, p.continuity.frames_in_epoch), (1, 1), f"{name}: yeni iz {track_id} geçmiş taşımamalı")
                self.assertFalse(p.trusted_match)
                self.assertLessEqual(p.continuity.matched_streak, 1)
            switched = [a for a, ids in ev.track_ids_per_actor.items() if len(ids) > 1]
            if switched:  # ByteTrack ID değiştirdiyse bu AYRI bir bulgudur (ilişkilendirme doğruluğundan bağımsız)
                self.assertGreaterEqual(ev.id_report.switches, 1)
                self.assertIn("new_track", ev.breaks)

    def test_t6_person_leaves_and_returns(self) -> None:
        frames, results, ev = self.scenario("leave_and_return")
        self.assertEqual(ev.wrong_pairs, 0, ev.wrong_examples)
        gone = [i for i, f in enumerate(frames) if "A" not in f.truth]
        self.assertGreaterEqual(len(gone), 20)
        for i in gone:  # kadrajdaki B hâlâ takip edilir ve eşleşir; A hakkında hiçbir sonuç yok
            self.assertTrue(all(p.person_bbox[0] < 1100 for p in results[i].persons if p.is_matched))
        self.assertGreaterEqual(ev.recall, 0.9)  # type: ignore[operator]
        self.assertGreater(len(ev.track_ids_per_actor["B"]) + len(ev.track_ids_per_actor["A"]), 2, "A en az bir kez ID değiştirdi veya B ile birlikte ≥ 3 farklı ID")


class RealFaceLossTest(RealStackCase):
    """T7-T8: arkasını dönme / yüz görünmüyor (yüz bölgesi maskelenerek TAKLİT edilir). İnsan takibi sürer, yüzsüz kişi doğru raporlanır."""

    def test_t7_person_keeps_being_tracked_while_the_face_disappears_and_reappears(self) -> None:
        frames, results, ev = self.scenario("turn_away")
        self.assertEqual(ev.wrong_pairs, 0)
        self.assertEqual(len(ev.track_ids_per_actor["A"]), 1, "yüz kaybolunca iz kaybolmamalı/değişmemeli")
        lo, hi = 20, 40
        for i in range(lo + 2, hi - 2):
            a = [p for p in results[i].persons if iou(p.person_bbox, frames[i].truth["A"].person_box) >= 0.5]
            self.assertEqual(len(a), 1, f"kare {i}: A takipte")
            self.assertNotEqual(a[0].status, STATUS_MATCHED, f"kare {i}: yüz görünmezken yüz eşleştirilmemeli")
        after = [p for p in results[hi + 5].persons if iou(p.person_bbox, frames[hi + 5].truth["A"].person_box) >= 0.5]
        self.assertEqual(after[0].status, STATUS_MATCHED, "yüz geri dönünce yeniden eşleşir")
        self.assertEqual(after[0].track_id, results[lo - 3].person(after[0].track_id).track_id)  # type: ignore[union-attr]
        self.assertEqual(after[0].continuity.epoch, 1, "yüz kaybı süreklilik dönemini bozmamalı")  # type: ignore[union-attr]

    def test_t8_person_without_any_visible_face_is_reported_as_no_face_every_frame(self) -> None:
        frames, results, ev = self.scenario("no_visible_face")
        self.assertEqual(ev.wrong_pairs, 0)
        for i, r in enumerate(results):
            a = [p for p in r.persons if iou(p.person_bbox, frames[i].truth["A"].person_box) >= 0.5]
            self.assertEqual(len(a), 1)
            self.assertEqual(a[0].status, STATUS_NO_FACE)
            self.assertIsNone(a[0].face_bbox)
        self.assertEqual(results[-1].no_face_count, 1)
        self.assertEqual(results[-1].matched_count, 1, "B'nin yüzü eşleşmeye devam eder")


class RealDecoyTest(RealStackCase):
    """T9: poster / fotoğraf. GÖZLENEN davranış (garanti değil): geometri sahte yüzü ayıramaz; hiçbir GERÇEK yüz yanlış kişiye bağlanmaz."""

    def test_t9_photo_and_poster_never_cause_a_real_person_to_receive_a_wrong_face(self) -> None:
        frames, results, ev = self.scenario("photo_decoys")
        self.assertEqual(ev.wrong_pairs, 0, ev.wrong_examples)
        self.assertEqual(ev.decoy_on_person_pairs, 0, "sahte yüzler gerçek kişi B'nin iziyle eşleşmemeli")
        self.assertGreaterEqual(ev.recall, 0.97)  # type: ignore[operator]
        self.assertEqual(ev.track_ids_per_actor["B"].__len__(), 1)

    def test_t9_decoys_do_get_tracks_and_faces_documenting_the_limitation(self) -> None:
        """Poster/fotoğraf kırpıntıları YOLO'da kişi gibi algılanabilir (FP-4B-001) ve içlerindeki yüz o izle eşleşir: geometri bunu AYIRAMAZ (tasarım sınırı)."""
        frames, results, ev = self.scenario("photo_decoys")
        self.assertGreater(ev.decoy_pairs + ev.correct_pairs, ev.correct_pairs, "gözlem: en az bir sahte yüz bir izle eşleşti (bu bir hata değil, belgelenen sınır)")

    def test_t9_static_decoys_are_flagged_possibly_static_but_the_moving_person_is_not(self) -> None:
        frames = scenes.render_scenario(self.actors, scenes.photo_decoys(40))
        config = AssociationConfig(stationary_flag_frames=15)
        results, _, _, _ = run_frames([f.image for f in frames], self.detector, self.face_detector, config=config)
        flagged = {p.track_id for r in results for p in r.persons if p.continuity.possibly_static}  # type: ignore[union-attr]
        truth_b_tracks = {p.track_id for i, r in enumerate(results) for p in r.persons if iou(p.person_bbox, frames[i].truth["B"].person_box) >= 0.5}
        self.assertTrue(flagged, "hareketsiz poster/fotoğraf izleri bayraklanmalı")
        self.assertTrue(flagged.isdisjoint(truth_b_tracks), "hareket eden gerçek kişi hareketsiz bayrağı almamalı")

    def test_t9b_photo_held_in_front_of_a_person_is_ambiguous_or_untrusted_never_trusted_silently(self) -> None:
        """B göğsünde bir fotoğraf tutuyor: YOLO B'yi iki kutuya bölebilir; sahte yüz o parçalardan birine bağlanabilir. Bu durumda kanıt biriktirmeye uygun (trusted) sayılmamalı."""
        for fraction in (0.45, 0.30, 0.20):
            frames, results, ev = self.scenario("photo_held_up", fraction=fraction)
            with self.subTest(fraction=fraction):
                self.assertEqual(ev.wrong_pairs, 0, ev.wrong_examples)
                self.assertEqual(ev.untrusted_decoy_on_person_pairs, ev.decoy_on_person_pairs, "sahte yüz gerçek kişinin iziyle eşleşirse bu eşleşme güvensiz olmalı")
                if ev.decoy_on_person_pairs == 0:
                    self.assertTrue(any(p.status == STATUS_AMBIGUOUS or not p.is_matched for r in results for p in r.persons))


class RealRecordedVideoTest(RealStackCase):
    """Kaydedilmiş video (mp4) üzerinde uçtan uca: dosyadan oku -> YOLO -> ByteTrack -> YuNet -> ilişkilendirme (``FrameSource`` video modu)."""

    def write_video(self, directory: Path, frames: list[np.ndarray]) -> Path:
        path = directory / "montaj.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (frames[0].shape[1], frames[0].shape[0]))
        self.assertTrue(writer.isOpened())
        for frame in frames:
            writer.write(frame)
        writer.release()
        return path

    def test_end_to_end_on_a_recorded_video_with_the_real_loop(self) -> None:
        scene = scenes.render_scenario(self.actors, scenes.three_people(24))
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_video(Path(tmp), [f.image for f in scene])
            tracker = PersonTracker()
            tracker.load()
            results = []
            with FrameSource(SourceConfig.video(path)) as source:
                summary = run_face_association(source, self.detector, tracker, self.face_detector, FacePersonAssociator(),
                                               on_result=lambda p, d, t, f, a: results.append(a))
        self.assertEqual(summary.reason, REASON_END_OF_STREAM)
        self.assertEqual(summary.frames_processed, 24)
        self.assertEqual(summary.distinct_tracks, 3)
        self.assertGreaterEqual(summary.matched_total, 3 * 22, "mp4 sıkıştırması sonrası da yüzlerin ≥ %90'ı eşleşmeli")
        self.assertEqual(summary.invalid_frames, 0)
        self.assertEqual((summary.persons_max, summary.faces_max), (3, 3))
        for r in results:
            self.assertEqual(len({p.face_index for p in r.persons if p.is_matched}), r.matched_count)
        self.assertIsNotNone(summary.yolo)
        self.assertGreater(summary.face.median_ms, 0.0)  # type: ignore[union-attr]

    def test_cli_main_runs_the_real_stack_headless_and_exits_zero(self) -> None:
        scene = scenes.render_scenario(self.actors, scenes.side_by_side(12))
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_video(Path(tmp), [f.image for f in scene])
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = cli.main(["--source", "video", "--video-path", str(path), "--device", "cpu", "--no-display", "--max-frames", "12"])
            self.assertEqual(code, 0, err.getvalue())
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["montaj.mp4"], "görüntü/çıktı dosyası yazılmamalı")
        text = out.getvalue()
        for fragment in ("YOLO:", "Takip: ByteTrack", "Yüz: face_detection_yunet", "eşleşen 24", "Her model kare başına yalnızca bir kez"):
            self.assertIn(fragment, text)

    def test_script_subprocess_runs_the_real_stack_end_to_end(self) -> None:
        scene = scenes.render_scenario(self.actors, scenes.side_by_side(10))
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_video(Path(tmp), [f.image for f in scene])
            proc = subprocess.run([sys.executable, str(SCRIPT), "--source", "video", "--video-path", str(path), "--device", "cpu", "--no-display"],
                                  capture_output=True, text=True, timeout=180, env=dict(os.environ, YOLO_OFFLINE="1"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Bitiş nedeni: video sona erdi", proc.stdout)
        self.assertIn("İşlenen kare: 10", proc.stdout)
        self.assertIn("eşleşen 20", proc.stdout)

    def test_windowed_loop_draws_every_frame_without_altering_the_source(self) -> None:
        scene = scenes.render_scenario(self.actors, scenes.single_person(6))
        window = FakeWindow()
        run_frames([f.image for f in scene], self.detector, self.face_detector, window=window)
        self.assertEqual(len(window.images), 6)
        for shown, original in zip(window.images, scene):
            self.assertEqual(shown.shape, original.image.shape)
            self.assertFalse(np.array_equal(shown, original.image))


@unittest.skipUnless(READY and torch_mps_status()[0], "MPS bu makinede kullanılamıyor (isteğe bağlı cihaz testi)")
class RealMpsInvariantTest(unittest.TestCase):
    """YOLO MPS'te: yalnızca DEĞİŞMEZLER denetlenir (cihazlar sınır tespitlerinde küçük farklar verebilir)."""

    def test_side_by_side_on_mps_has_no_wrong_association(self) -> None:
        detector = PersonDetector(DetectionConfig(device="mps"))
        detector.load()
        face = FaceDetector()
        face.load()
        try:
            frames = scenes.render_scenario(scenes.load_actors(), scenes.side_by_side(20))
            results, summary, _, _ = run_frames([f.image for f in frames], detector, face)
            ev = evaluate_scene(frames, results)
        finally:
            detector.close()
            face.close()
        self.assertEqual(ev.wrong_pairs, 0)
        self.assertGreaterEqual(ev.recall, 0.9)  # type: ignore[operator]
        self.assertEqual(summary.invalid_frames, 0)


if __name__ == "__main__":
    unittest.main()
