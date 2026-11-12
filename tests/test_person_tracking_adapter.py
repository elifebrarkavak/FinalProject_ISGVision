"""PersonTracker (adaptör) politikası: dönüştürme, eşleme, kare sırası, oturum/sıfırlama, hata ve bağımlılık yönetimi.

[Mock/birim] Backend SAHTEDİR (ByteTrack algoritması çalışmaz); gerçek ByteTrack davranışı ``test_person_tracking_bytetrack_real.py``'dadır.
Ağ ve kamera kullanılmaz; ``lap``/``ultralytics`` gerekmez.
"""

from __future__ import annotations

import importlib.metadata
import os
import socket
import subprocess
import unittest
from unittest import mock

import numpy as np

from tracking_test_support import (
    H,
    W,
    FakeBackend,
    SequenceClock,
    backend_frame,
    box_at,
    empty_backend_frame,
    fake_tracker,
    make_result,
)

from isg_vision.human_detection.results import DetectionResult, PersonDetection  # noqa: E402
from isg_vision.tracking import bytetrack_adapter as adapter  # noqa: E402
from isg_vision.tracking.bytetrack_adapter import PersonTracker  # noqa: E402
from isg_vision.tracking.config import TrackingConfig  # noqa: E402
from isg_vision.tracking.errors import (  # noqa: E402
    FrameOrderError,
    InvalidTrackingInputError,
    TrackerBackendError,
    TrackerDependencyError,
    TrackingError,
)


def one_person(frame_index: int | None = 0, conf: float = 0.9, **kw: object) -> DetectionResult:
    return make_result([("A", box_at(100), conf)], frame_index, **kw)[0]  # type: ignore[arg-type]


class ConversionTest(unittest.TestCase):
    def test_detections_are_converted_to_bytetrack_input(self) -> None:
        backend = FakeBackend()
        tracker = fake_tracker(backend)
        items = [("A", (10.0, 20.0, 70.0, 190.0), 0.9), ("B", (300.5, 40.0, 360.0, 200.25), 0.4)]
        result, _ = make_result(items, 0)
        tracker.update(result)
        call = backend.calls[0]
        self.assertEqual(call["xyxy"].dtype, np.float32)
        self.assertEqual(call["xyxy"].shape, (2, 4))
        np.testing.assert_allclose(call["xyxy"], [[10, 20, 70, 190], [300.5, 40, 360, 200.25]])
        np.testing.assert_allclose(call["conf"], [0.9, 0.4], rtol=1e-6)
        self.assertEqual(call["cls"].tolist(), [0.0, 0.0])
        self.assertEqual(call["image_hw"], (H, W), "görüntü boyutu (yükseklik, genişlik) sırasında verilmeli")

    def test_empty_detections_still_update_backend_exactly_once(self) -> None:
        """Tespitsiz kare de takipçiye verilir (yoksa kayıp izler yaşlanmaz); boş girdi biçimi korunur."""
        backend = FakeBackend()
        tracker = fake_tracker(backend)
        result, _ = make_result([], 0)
        out = tracker.update(result)
        self.assertEqual(len(backend.calls), 1)
        call = backend.calls[0]
        self.assertEqual((call["xyxy"].shape, call["conf"].shape, call["cls"].shape), ((0, 4), (0,), (0,)))
        self.assertEqual((out.detection_count, out.active_count), (0, 0))
        self.assertTrue(out.is_empty)

    def test_one_backend_update_per_frame(self) -> None:
        backend = FakeBackend()
        tracker = fake_tracker(backend)
        for i in range(5):
            tracker.update(one_person(i))
        self.assertEqual(len(backend.calls), 5)


class MappingTest(unittest.TestCase):
    def test_rows_become_tracked_persons_with_detection_data(self) -> None:
        # iki tespit; ByteTrack çıktısı 2. tespiti 7 numaralı, 1. tespiti 3 numaralı ize bağlıyor (sıra bilerek ters)
        backend = FakeBackend([backend_frame(
            [[301, 41, 361, 201, 7, 0.4, 0, 1], [11, 21, 71, 191, 3, 0.9, 0, 0]], ages={7: 5, 3: 2}, lost=[(9, 4)], tentative=1)])
        tracker = fake_tracker(backend)
        result, _ = make_result([("A", (10, 20, 70, 190), 0.9), ("B", (300, 40, 360, 200), 0.4)], 12, timestamp=1.5)
        out = tracker.update(result)
        self.assertEqual([t.track_id for t in out.tracks], [3, 7], "izler track_id'ye göre sıralı olmalı")
        by_id = {t.track_id: t for t in out.tracks}
        self.assertEqual(by_id[3].detection_index, 0)
        self.assertEqual(by_id[7].detection_index, 1)
        self.assertEqual(by_id[3].confidence, 0.9, "güven, eşleşen tespitten (tam değer) alınmalı")
        self.assertEqual(by_id[7].confidence, 0.4)
        self.assertEqual((by_id[3].age_frames, by_id[7].age_frames), (2, 5))
        for t in out.tracks:
            self.assertEqual((t.class_id, t.class_name, t.frame_index, t.timestamp, t.session_id), (0, "person", 12, 1.5, 1))
        self.assertEqual(by_id[3].bbox_xyxy, (11.0, 21.0, 71.0, 191.0), "kutu takipçinin (Kalman) kutusu olmalı")
        self.assertEqual((out.lost_count, out.lost_tracks[0].track_id, out.lost_tracks[0].frames_since_seen), (1, 9, 4))
        self.assertEqual((out.tentative_count, out.detection_count, out.active_count), (1, 2, 2))

    def test_kalman_boxes_are_clipped_to_image(self) -> None:
        backend = FakeBackend([backend_frame([[-15.0, -3.0, 50.0, 100.0, 1, 0.9, 0, 0], [600, 400, 700, 520, 2, 0.8, 0, 1]])])
        out = fake_tracker(backend).update(make_result([("A", (0, 0, 50, 100), 0.9), ("B", (600, 400, 640, 480), 0.8)], 0)[0])
        self.assertEqual(out.tracks[0].bbox_xyxy, (0.0, 0.0, 50.0, 100.0))
        self.assertEqual(out.tracks[1].bbox_xyxy, (600.0, 400.0, 640.0, 480.0))

    def test_degenerate_track_after_clipping_is_dropped_and_counted(self) -> None:
        backend = FakeBackend([backend_frame([[700, 10, 760, 100, 1, 0.9, 0, 0], [10, 10, 60, 100, 2, 0.9, 0, 1]])])
        tracker = fake_tracker(backend)
        out = tracker.update(make_result([("A", (600, 10, 640, 100), 0.9), ("B", (10, 10, 60, 100), 0.9)], 0)[0])
        self.assertEqual(out.track_ids, (2,))
        self.assertEqual(out.dropped_tracks, 1)
        self.assertEqual(tracker.stats.dropped_tracks, 1)

    def test_duplicate_track_id_in_one_frame_is_an_error(self) -> None:
        """Aynı karede aynı Track ID iki kişiye verilemez (§8 kural 3): iç tutarsızlık sessizce geçirilmez."""
        backend = FakeBackend([backend_frame([[10, 20, 70, 190, 4, 0.9, 0, 0], [300, 40, 360, 200, 4, 0.8, 0, 1]])])
        with self.assertRaisesRegex(TrackerBackendError, "Track ID 4"):
            fake_tracker(backend).update(make_result([("A", (10, 20, 70, 190), 0.9), ("B", (300, 40, 360, 200), 0.8)], 0)[0])

    def test_malformed_backend_rows_are_rejected(self) -> None:
        cases = {
            "nan": [[10, 20, 70, 190, 1, float("nan"), 0, 0]],
            "inf": [[10, 20, float("inf"), 190, 1, 0.9, 0, 0]],
            "id0": [[10, 20, 70, 190, 0, 0.9, 0, 0]],
            "id_too_large": [[10, 20, 70, 190, 2**24 + 2, 0.9, 0, 0]],
            "idx_out_of_range": [[10, 20, 70, 190, 1, 0.9, 0, 5]],
            "idx_negative": [[10, 20, 70, 190, 1, 0.9, 0, -1]],
        }
        for name, rows in cases.items():
            with self.subTest(name):
                tracker = fake_tracker(FakeBackend([backend_frame(rows)]))
                with self.assertRaises(TrackerBackendError):
                    tracker.update(one_person(0))


class FrameOrderTest(unittest.TestCase):
    def test_duplicate_frame_index_rejected_and_state_unchanged(self) -> None:
        backend = FakeBackend()
        tracker = fake_tracker(backend)
        tracker.update(one_person(0))
        tracker.update(one_person(1))
        before = tracker.stats
        with self.assertRaisesRegex(FrameOrderError, "bozuk"):
            tracker.update(one_person(1))
        self.assertEqual(len(backend.calls), 2, "reddedilen kare takipçiye GİTMEMELİ (iki kez yaşlandırma olmaz)")
        self.assertEqual(tracker.stats, before)
        tracker.update(one_person(2))  # akış kaldığı yerden sürer
        self.assertEqual(len(backend.calls), 3)

    def test_decreasing_frame_index_rejected(self) -> None:
        tracker = fake_tracker(FakeBackend())
        tracker.update(one_person(5))
        with self.assertRaises(FrameOrderError):
            tracker.update(one_person(3))

    def test_decreasing_timestamp_rejected(self) -> None:
        tracker = fake_tracker(FakeBackend())
        tracker.update(one_person(0, timestamp=10.0))
        with self.assertRaises(FrameOrderError):
            tracker.update(one_person(1, timestamp=9.5))
        tracker.update(one_person(1, timestamp=10.0))  # eşit zaman damgası serbest

    def test_gap_is_accepted_and_counted(self) -> None:
        backend = FakeBackend()
        tracker = fake_tracker(backend)
        tracker.update(one_person(0))
        tracker.update(one_person(4))  # 1, 2, 3 atlandı
        self.assertEqual(tracker.stats.gap_frames, 3)
        self.assertEqual(len(backend.calls), 2, "ByteTrack atlanan kareleri GÖRMEZ (belgelenmiş davranış)")

    def test_missing_frame_index_is_numbered_by_tracker(self) -> None:
        tracker = fake_tracker(FakeBackend())
        indices = [tracker.update(one_person(None)).frame_index for _ in range(3)]
        self.assertEqual(indices, [0, 1, 2])

    def test_error_is_a_value_error_and_tracking_error(self) -> None:
        self.assertTrue(issubclass(FrameOrderError, ValueError) and issubclass(FrameOrderError, TrackingError))


class SessionTest(unittest.TestCase):
    def test_first_session_is_one(self) -> None:
        tracker = fake_tracker(FakeBackend())
        self.assertEqual(tracker.session_id, 1)
        self.assertEqual(tracker.update(one_person(0)).session_id, 1)

    def test_source_change_resets_state_and_flags_reason(self) -> None:
        backend = FakeBackend()
        tracker = fake_tracker(backend)
        r1 = tracker.update(one_person(0), source="video:/a.mp4")
        tracker.update(one_person(1), source="video:/a.mp4")
        r2 = tracker.update(one_person(0), source="camera:0")  # yeni kaynakta kare numarası 0'dan başlar: hata DEĞİL
        self.assertIsNone(r1.reset_reason)
        self.assertEqual(r2.reset_reason, "source_changed")
        self.assertEqual(backend.reset_calls, 1)
        self.assertEqual((r1.session_id, r2.session_id), (1, 2))
        self.assertEqual(tracker.stats.sessions, 2)
        self.assertEqual(tracker.stats.resets, 1)

    def test_same_source_does_not_reset(self) -> None:
        backend = FakeBackend()
        tracker = fake_tracker(backend)
        for i in range(3):
            self.assertIsNone(tracker.update(one_person(i), source="camera:0").reset_reason)
        self.assertEqual(backend.reset_calls, 0)

    def test_source_none_never_triggers_reset(self) -> None:
        backend = FakeBackend()
        tracker = fake_tracker(backend)
        tracker.update(one_person(0), source="camera:0")
        self.assertIsNone(tracker.update(one_person(1)).reset_reason)
        self.assertEqual(backend.reset_calls, 0)

    def test_image_size_change_resets(self) -> None:
        backend = FakeBackend()
        tracker = fake_tracker(backend)
        tracker.update(one_person(0), source="camera:0")
        out = tracker.update(make_result([("A", box_at(100), 0.9)], 1, size=(1280, 720))[0], source="camera:0")
        self.assertEqual(out.reset_reason, "image_size_changed")
        self.assertEqual(out.session_id, 2)
        self.assertEqual(backend.reset_calls, 1)

    def test_explicit_reset_bumps_session_only_if_frames_were_processed(self) -> None:
        backend = FakeBackend()
        tracker = fake_tracker(backend)
        tracker.load()
        tracker.reset()  # hiç kare yok: oturum numarası boşuna artmaz
        self.assertEqual(tracker.session_id, 1)
        tracker.update(one_person(0))
        tracker.reset()
        self.assertEqual(tracker.session_id, 2)
        tracker.reset()
        self.assertEqual(tracker.session_id, 2)
        self.assertEqual(backend.reset_calls, 3, "backend her reset'te temizlenir")

    def test_after_reset_frame_numbering_may_restart(self) -> None:
        tracker = fake_tracker(FakeBackend())
        tracker.update(one_person(10))
        tracker.reset()
        out = tracker.update(one_person(0))  # sıfırlamadan sonra küçük numara serbest
        self.assertEqual((out.frame_index, out.session_id), (0, 2))

    def test_reset_before_load_is_safe(self) -> None:
        tracker = fake_tracker(FakeBackend())
        tracker.reset()
        self.assertFalse(tracker.is_loaded)

    def test_stats(self) -> None:
        backend = FakeBackend([backend_frame([[10, 20, 70, 190, 1, 0.9, 0, 0]]), backend_frame([[10, 20, 70, 190, 1, 0.9, 0, 0]]),
                               backend_frame([[10, 20, 70, 190, 1, 0.9, 0, 0]])])
        tracker = fake_tracker(backend)
        tracker.update(one_person(0), source="a")
        tracker.update(one_person(1), source="a")
        tracker.update(one_person(0), source="b")  # ikinci oturumda yine ID 1: farklı anahtar
        s = tracker.stats
        self.assertEqual((s.frames, s.sessions, s.resets, s.distinct_tracks, s.max_active), (3, 2, 1, 2, 1))


class InvalidInputTest(unittest.TestCase):
    def assertRejectedWithoutStateChange(self, bad: object) -> None:
        backend = FakeBackend()
        tracker = fake_tracker(backend)
        tracker.update(one_person(0))
        before = tracker.stats
        with self.assertRaises(InvalidTrackingInputError):
            tracker.update(bad)  # type: ignore[arg-type]
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(tracker.stats, before)
        tracker.update(one_person(1))  # sonraki geçerli kare işlenir

    def test_non_detection_result_rejected(self) -> None:
        for bad in (None, [], "x", {"detections": []}, object()):
            with self.subTest(repr(bad)):
                self.assertRejectedWithoutStateChange(bad)

    def test_invalid_image_size_rejected(self) -> None:
        for size in ((0, 480), (640, 0), (-1, 5), (640.0, 480), (640,), "640x480", (True, 480)):
            with self.subTest(size=size):
                bad = DetectionResult((), size, 1, None, 1.0, "cpu")  # type: ignore[arg-type]
                self.assertRejectedWithoutStateChange(bad)

    def test_box_outside_image_rejected(self) -> None:
        for box in ((600.0, 10.0, 700.0, 100.0), (-5.0, 10.0, 50.0, 100.0), (10.0, 10.0, 50.0, 500.0), (10.0, -2.0, 50.0, 100.0)):
            with self.subTest(box=box):
                det = PersonDetection(0, "person", 0.9, box, frame_index=1)
                self.assertRejectedWithoutStateChange(DetectionResult((det,), (W, H), 1, None, 1.0, "cpu"))

    def test_foreign_object_in_detections_rejected(self) -> None:
        self.assertRejectedWithoutStateChange(DetectionResult(("not-a-detection",), (W, H), 1, None, 1.0, "cpu"))  # type: ignore[arg-type]

    def test_invalid_boxes_and_confidences_cannot_even_be_constructed(self) -> None:
        """İlk savunma hattı 4A: geçersiz kutu/güven ``PersonDetection`` oluşturulurken reddedilir (tracker'a hiç ulaşmaz)."""
        for kwargs in (
            dict(confidence=1.2, bbox_xyxy=(1, 1, 5, 5)), dict(confidence=float("nan"), bbox_xyxy=(1, 1, 5, 5)),
            dict(confidence=0.5, bbox_xyxy=(5, 5, 1, 1)), dict(confidence=0.5, bbox_xyxy=(1, 1, float("nan"), 5)),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PersonDetection(0, "person", **kwargs)  # type: ignore[arg-type]

    def test_validation_is_attribute_based_not_class_identity_based(self) -> None:
        """Modül yeniden yüklense (hot-reload) sınıf kimliği değişir; aynı biçimdeki sonuç yine kabul edilmeli."""
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class OtherDetection:
            class_id: int
            class_name: str
            confidence: float
            bbox_xyxy: tuple

        @dataclass(frozen=True)
        class OtherResult:
            detections: tuple
            image_size: tuple
            frame_index: int | None = None
            timestamp: float | None = None

        backend = FakeBackend()
        out = fake_tracker(backend).update(OtherResult((OtherDetection(0, "person", 0.8, (1.0, 2.0, 30.0, 60.0)),), (W, H), 0))  # type: ignore[arg-type]
        self.assertEqual(out.detection_count, 1)
        self.assertEqual(len(backend.calls), 1)

    def test_structurally_broken_results_are_rejected(self) -> None:
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class NoSize:
            detections: tuple
            frame_index: int | None = None
            timestamp: float | None = None

        @dataclass(frozen=True)
        class Det:
            class_id: object
            class_name: str
            confidence: object
            bbox_xyxy: object

        bad_results = [
            NoSize(()),
            DetectionResult(None, (W, H), 0, None, 1.0, "cpu"),  # type: ignore[arg-type]
            DetectionResult((Det(0, "person", 0.5, (1, 2, 3)),), (W, H), 0, None, 1.0, "cpu"),  # type: ignore[arg-type]
            DetectionResult((Det(0, "person", 0.5, ("a", "b", "c", "d")),), (W, H), 0, None, 1.0, "cpu"),  # type: ignore[arg-type]
            DetectionResult((Det(0, "person", "yüksek", (1, 2, 30, 60)),), (W, H), 0, None, 1.0, "cpu"),  # type: ignore[arg-type]
            DetectionResult((Det(0, "person", float("nan"), (1, 2, 30, 60)),), (W, H), 0, None, 1.0, "cpu"),  # type: ignore[arg-type]
            DetectionResult((Det(0, "person", 1.5, (1, 2, 30, 60)),), (W, H), 0, None, 1.0, "cpu"),  # type: ignore[arg-type]
            DetectionResult((Det(0, "person", 0.5, (1, 2, float("inf"), 60)),), (W, H), 0, None, 1.0, "cpu"),  # type: ignore[arg-type]
        ]
        for index, bad in enumerate(bad_results):
            with self.subTest(case=index):
                self.assertRejectedWithoutStateChange(bad)

    def test_box_exactly_on_image_border_is_accepted(self) -> None:
        det = PersonDetection(0, "person", 0.9, (0.0, 0.0, float(W), float(H)), frame_index=0)
        fake_tracker(FakeBackend()).update(DetectionResult((det,), (W, H), 0, None, 1.0, "cpu"))


class BackendErrorTest(unittest.TestCase):
    def test_backend_exception_is_wrapped(self) -> None:
        tracker = fake_tracker(FakeBackend([RuntimeError("kaboom")]))
        with self.assertRaisesRegex(TrackerBackendError, "kaboom"):
            tracker.update(one_person(0))

    def test_tracking_error_from_backend_passes_through(self) -> None:
        tracker = fake_tracker(FakeBackend([TrackerBackendError("özel")]))
        with self.assertRaisesRegex(TrackerBackendError, "özel"):
            tracker.update(one_person(0))

    def test_failed_update_does_not_advance_bookkeeping(self) -> None:
        tracker = fake_tracker(FakeBackend([RuntimeError("x")]))
        with self.assertRaises(TrackerBackendError):
            tracker.update(one_person(0))
        self.assertEqual(tracker.stats.frames, 0)
        tracker.update(one_person(0))  # aynı numara hâlâ geçerli (kare işlenmemişti)

    def test_factory_failure_is_wrapped_and_tracker_stays_unloaded(self) -> None:
        def boom(_config: TrackingConfig) -> FakeBackend:
            raise OSError("disk")

        tracker = PersonTracker(backend_factory=boom)
        with self.assertRaisesRegex(TrackerBackendError, "başlatılamadı"):
            tracker.load()
        self.assertFalse(tracker.is_loaded)
        with self.assertRaises(TrackerBackendError):
            tracker.update(one_person(0))

    def test_dependency_error_from_factory_passes_through(self) -> None:
        def missing(_config: TrackingConfig) -> FakeBackend:
            raise TrackerDependencyError("lap yok")

        with self.assertRaises(TrackerDependencyError):
            PersonTracker(backend_factory=missing).load()

    def test_load_is_idempotent_and_lazy_load_works(self) -> None:
        backend = FakeBackend()
        tracker = fake_tracker(backend)
        self.assertFalse(tracker.is_loaded)
        tracker.load()
        tracker.load()
        tracker.update(one_person(0))
        self.assertEqual(tracker.load_count, 1)

    def test_close_is_idempotent_and_context_manager_loads(self) -> None:
        tracker = fake_tracker(FakeBackend())
        with tracker as t:
            self.assertTrue(t.is_loaded)
        self.assertFalse(tracker.is_loaded)
        tracker.close()
        tracker.close()


class TimingTest(unittest.TestCase):
    def test_tracking_ms_uses_injected_clock(self) -> None:
        tracker = fake_tracker(FakeBackend(), clock=SequenceClock([1.000, 1.0035]))
        self.assertAlmostEqual(tracker.update(one_person(0)).tracking_ms, 3.5, places=6)

    def test_negative_clock_delta_is_clamped(self) -> None:
        tracker = fake_tracker(FakeBackend(), clock=SequenceClock([2.0, 1.0]))
        self.assertEqual(tracker.update(one_person(0)).tracking_ms, 0.0)


class DependencyGuardTest(unittest.TestCase):
    """lap eksikse: açıklayıcı hata; ağ erişimi, alt süreç (pip) ve paket kurulumu YOK."""

    def setUp(self) -> None:
        self.net = mock.patch.object(socket.socket, "connect", side_effect=AssertionError("ağ bağlantısı denendi!"))
        self.run = mock.patch.object(subprocess, "run", side_effect=AssertionError("alt süreç (pip?) başlatıldı!"))
        self.popen = mock.patch.object(subprocess, "Popen", side_effect=AssertionError("alt süreç (pip?) başlatıldı!"))
        self.out = mock.patch.object(subprocess, "check_output", side_effect=AssertionError("alt süreç (pip?) başlatıldı!"))
        for p in (self.net, self.run, self.popen, self.out):
            p.start()
            self.addCleanup(p.stop)

    @staticmethod
    def version_without(missing: str):  # type: ignore[no-untyped-def]
        real = importlib.metadata.version

        def fake(name: str) -> str:
            if name == missing:
                raise importlib.metadata.PackageNotFoundError(name)
            return real(name)

        return mock.patch.object(importlib.metadata, "version", side_effect=fake)

    def test_missing_lap_raises_clear_error_with_install_hint(self) -> None:
        with self.version_without("lap"):
            with self.assertRaises(TrackerDependencyError) as ctx:
                adapter.check_tracker_dependencies()
        message = str(ctx.exception)
        self.assertIn("'lap'", message)
        self.assertIn("pip install lap==", message)
        self.assertIn("otomatik kurulum", message.lower())

    def test_missing_ultralytics_raises_clear_error(self) -> None:
        with self.version_without("ultralytics"):
            with self.assertRaisesRegex(TrackerDependencyError, "ultralytics"):
                adapter.check_tracker_dependencies()

    def test_broken_lap_binary_raises_clear_error(self) -> None:
        real_import = adapter.importlib.import_module

        def fake_import(name: str, *a: object, **k: object):  # type: ignore[no-untyped-def]
            if name == "lap":
                raise ImportError("dlopen: kötü ikili")
            return real_import(name, *a, **k)

        with mock.patch.object(adapter.importlib, "import_module", side_effect=fake_import):
            with self.assertRaisesRegex(TrackerDependencyError, "kullanılamıyor"):
                adapter.check_tracker_dependencies()

    def test_lap_without_version_attribute_is_rejected(self) -> None:
        fake_lap = mock.Mock(spec=[])  # __version__ yok (Ultralytics de bunu reddeder: 'dizin adı lap' durumu)
        real_import = adapter.importlib.import_module
        with mock.patch.object(adapter.importlib, "import_module", side_effect=lambda n, *a, **k: fake_lap if n == "lap" else real_import(n, *a, **k)):
            with self.assertRaises(TrackerDependencyError):
                adapter.check_tracker_dependencies()

    def test_default_tracker_with_missing_lap_fails_at_load_without_install_attempt(self) -> None:
        with self.version_without("lap"):
            with self.assertRaises(TrackerDependencyError):
                PersonTracker().load()
            with self.assertRaises(TrackerDependencyError):
                PersonTracker().update(one_person(0))
        # setUp'taki yamalar: socket.connect / subprocess çağrılsaydı AssertionError ile test düşerdi

    def test_prepare_offline_environment_blocks_autoinstall_and_network_probe(self) -> None:
        with mock.patch.dict(os.environ, {"YOLO_AUTOINSTALL": "True"}, clear=False):
            os.environ.pop("YOLO_OFFLINE", None)
            adapter.prepare_offline_environment()
            self.assertEqual(os.environ["YOLO_OFFLINE"], "1")
            self.assertEqual(os.environ["YOLO_AUTOINSTALL"], "False")

    def test_explicit_offline_choice_is_not_overridden(self) -> None:
        with mock.patch.dict(os.environ, {"YOLO_OFFLINE": "0"}, clear=False):
            adapter.prepare_offline_environment()
            self.assertEqual(os.environ["YOLO_OFFLINE"], "0")


class StructureTest(unittest.TestCase):
    def test_tracker_module_never_references_a_yolo_model(self) -> None:
        """Mimari güvence: takipçi modülü modeli yüklemez/çağırmaz (YOLO tek kez, PersonDetector'da çalışır)."""
        import ast
        from pathlib import Path

        tree = ast.parse(Path(adapter.__file__).read_text(encoding="utf-8"))  # yalnızca GERÇEK kod: docstring/yorumlar sayılmaz
        called: set[str] = set()
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                called.add(func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", ""))
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        self.assertFalse(called & {"predict", "track", "YOLO", "detect", "warmup"}, f"takipçi modeli çağırıyor: {called}")
        self.assertFalse(
            {n for n in imported if "person_detector" in n or n in {"PersonDetector", "YOLO"}}, "takipçi PersonDetector/YOLO import etmemeli"
        )
        # ultralytics yalnızca gecikmeli (fonksiyon içinde) ve yalnızca izleme sınıfları için import edilir
        top_level = {n.module for n in tree.body if isinstance(n, ast.ImportFrom) and n.module} | {
            a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names
        }
        self.assertFalse({m for m in top_level if m.startswith(("ultralytics", "torch", "lap", "cv2"))}, "modül düzeyinde ağır import var")

    def test_backend_info_before_load_is_empty(self) -> None:
        self.assertEqual(fake_tracker(FakeBackend()).backend_info, {"versions": {}, "version_note": None})

    def test_backend_info_after_load(self) -> None:
        backend = FakeBackend()
        backend.version_note = "uyarı"
        tracker = fake_tracker(backend)
        tracker.load()
        self.assertEqual(tracker.backend_info, {"versions": {"ultralytics": "fake", "lap": "fake"}, "version_note": "uyarı"})

    def test_empty_backend_frame_helper_shape(self) -> None:
        self.assertEqual(empty_backend_frame().rows.shape, (0, 8))


if __name__ == "__main__":
    unittest.main()
