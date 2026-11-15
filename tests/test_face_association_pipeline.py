"""Birleşik döngü (``run_face_association``): her model kare başına TEK çağrı, çağrı düzeni, özet/ölçüm, çıkış nedenleri, hata yayılımı, oturum sıfırlama,
``run_tracking(on_result=...)`` kancasıyla denklik [Mock/birim: sahte kamera/pencere/modeller; gerçek model/ByteTrack/kamera YOK].

"Kare başına tek YOLO / tek YuNet çağrısı" ayrıca GERÇEK ``PersonDetector`` + SAHTE YOLO modeli ve GERÇEK ``FaceDetector`` + SAHTE YuNet ile sınanır
(modellerin ``predict``/``detect`` çağrıları sayılır). Gerçek modellerle bütünleşik döngü ``test_face_association_real_model.py``'dedir.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("YOLO_OFFLINE", "1")

from camera_test_support import FakeCapture, FakeClock, FakeFactory, FakeWindow, make_frame  # noqa: E402
from face_association_test_support import (  # noqa: E402
    ScriptedFaceDetector,
    ScriptedTracker,
    SizedFakeDetector,
    head_box,
    person_box,
)
from face_detection_test_support import FakeYuNet, make_detector as make_face_detector  # noqa: E402
from person_detection_test_support import FakeYoloModel, make_detector as make_person_detector  # noqa: E402
from tracking_test_support import echo_tracker  # noqa: E402

from isg_vision.camera.capture import FrameSource  # noqa: E402
from isg_vision.camera.config import SourceConfig  # noqa: E402
from isg_vision.camera.errors import DisplayError  # noqa: E402
from isg_vision.face_association import FacePersonAssociator  # noqa: E402
from isg_vision.face_association.pipeline import (  # noqa: E402
    REASON_END_OF_STREAM,
    REASON_INTERRUPTED,
    REASON_MAX_FRAMES,
    REASON_USER_QUIT,
    REASON_WINDOW_CLOSED,
    run_face_association,
)
from isg_vision.face_detection.errors import FaceInferenceError  # noqa: E402
from isg_vision.human_detection.errors import InferenceError  # noqa: E402
from isg_vision.tracking.errors import TrackerBackendError  # noqa: E402
from isg_vision.tracking.pipeline import run_tracking  # noqa: E402

SIZE = (640, 480)
P1, P2 = person_box(150, top=60, w=110, h=330), person_box(450, top=60, w=110, h=330)
F1, F2 = head_box(P1, size=34), head_box(P2, size=34)
STRAY = (300.0, 400.0, 330.0, 430.0)  # hiçbir kişi kutusunda değil


def camera_source(n_frames: int, clock: FakeClock | None = None) -> FrameSource:
    capture = FakeCapture([make_frame(i, *SIZE) for i in range(n_frames)], props={cv2.CAP_PROP_FRAME_WIDTH: SIZE[0], cv2.CAP_PROP_FRAME_HEIGHT: SIZE[1]})
    kwargs = {"clock": clock} if clock is not None else {}
    source = FrameSource(SourceConfig.camera(0, width=SIZE[0], height=SIZE[1]), capture_factory=FakeFactory(capture), max_read_failures=1, sleep=lambda s: None, **kwargs)
    source.open()
    return source


def scripted(frames: int = 6, people=((1, P1), (2, P2)), faces_=(F1, F2, STRAY)):  # type: ignore[no-untyped-def]
    return SizedFakeDetector(), ScriptedTracker([list(people)] * frames), ScriptedFaceDetector([list(faces_)] * frames), FacePersonAssociator()


class CallPatternTest(unittest.TestCase):
    def test_every_stage_is_called_exactly_once_per_frame_in_order_on_the_same_frame(self) -> None:
        source = camera_source(8)
        det, trk, fd, assoc = scripted()
        summary = run_face_association(source, det, trk, fd, assoc, max_frames=5)
        self.assertEqual(summary.frames_processed, 5)
        self.assertEqual((len(det.calls), len(trk.updates), len(fd.calls)), (5, 5, 5))
        for i in range(5):
            self.assertEqual(det.calls[i]["frame_index"], i)
            self.assertEqual(trk.updates[i]["result"].frame_index, i, "takipçi aynı karenin tespitini almalı")
            self.assertEqual(fd.calls[i]["frame_index"], i, "yüz tespiti aynı kare için çalışmalı")
            self.assertEqual(fd.calls[i]["timestamp"], det.calls[i]["timestamp"], "aynı FramePacket zaman damgası")
        self.assertEqual(trk.events[:3], ["reset", "update", "update"])

    def test_tracker_and_associator_are_reset_once_before_the_first_frame(self) -> None:
        det, trk, fd, assoc = scripted()
        resets = []
        original = assoc.reset
        assoc.reset = lambda: (resets.append(len(det.calls)), original())[1]  # type: ignore[method-assign]
        run_face_association(camera_source(4), det, trk, fd, assoc, max_frames=2)
        self.assertEqual(trk.reset_calls, 1)
        self.assertEqual(resets, [0], "ilişkilendirici sıfırlaması ilk kareden ÖNCE bir kez")

    def test_real_wrappers_call_the_underlying_models_once_per_frame(self) -> None:
        """Gerçek PersonDetector + PersonTracker(sahte backend) + FaceDetector, SAHTE modellerle: model çağrıları sayılır."""
        with tempfile.TemporaryDirectory() as tmp:
            yolo = FakeYoloModel([[((100, 60, 220, 390), 0.9, 0)]])
            yunet = FakeYuNet([[(120, 80, 36, 36, 0.95)]])
            detector, _ = make_person_detector(yolo, Path(tmp))
            face_detector, _ = make_face_detector(yunet, Path(tmp), clock=None)
            tracker = echo_tracker()
            summary = run_face_association(camera_source(7), detector, tracker, face_detector, FacePersonAssociator(), max_frames=6)
        self.assertEqual(summary.frames_processed, 6)
        self.assertEqual(len(yolo.predict_calls), 6, "YOLO kare başına yalnızca bir kez")
        self.assertEqual(len(yunet.detect_calls), 6, "YuNet kare başına yalnızca bir kez")
        self.assertEqual(tracker.echo_backend.calls, 6, "ByteTrack adaptörü kare başına yalnızca bir kez")  # type: ignore[attr-defined]
        self.assertEqual(summary.persons_total, 6)
        self.assertEqual(summary.matched_total, 6, "kutular uyumlu: her karede 1 eşleşme")

    def test_models_see_the_same_unmodified_frame_and_drawing_never_mutates_it(self) -> None:
        source = camera_source(3)
        det, trk, fd, assoc = scripted(3)
        seen = []
        window = FakeWindow()
        run_face_association(source, det, trk, fd, assoc, window, max_frames=3, on_result=lambda p, d, t, f, a: seen.append(p.frame.copy()))
        for i, frame in enumerate(seen):
            np.testing.assert_array_equal(frame, make_frame(i, *SIZE), "çizim kaynak kareyi değiştirmemeli")
        self.assertEqual(len(window.images), 3)


class SummaryTest(unittest.TestCase):
    def test_counts_and_status_totals(self) -> None:
        det, trk, fd, assoc = scripted(6)
        s = run_face_association(camera_source(8), det, trk, fd, assoc, max_frames=6)
        self.assertEqual((s.persons_total, s.persons_max, s.faces_total, s.faces_max), (12, 2, 18, 3))
        self.assertEqual((s.matched_total, s.no_face_total, s.unmatched_faces_total), (12, 0, 6))
        self.assertEqual((s.ambiguous_persons_total, s.ambiguous_faces_total, s.invalid_frames), (0, 0, 0))
        self.assertEqual(s.frames_with_match, 6)
        self.assertEqual(s.distinct_tracks, 2)
        self.assertEqual(s.trusted_matched_total, 8, "min_trusted_frames=3: kare başına iki iz; ilk 2 kare güvensiz")
        self.assertEqual(s.break_counts, {"new_track": 2})
        self.assertAlmostEqual(s.persons_mean, 2.0)
        self.assertAlmostEqual(s.faces_mean, 3.0)

    def test_track_spans_report_matches_and_epochs(self) -> None:
        det, trk, fd, assoc = scripted(6)
        s = run_face_association(camera_source(8), det, trk, fd, assoc, max_frames=6)
        self.assertEqual([(sp.track_id, sp.frames_active, sp.frames_matched, sp.frames_trusted_matched, sp.epochs) for sp in s.tracks],
                         [(1, 6, 6, 4, 1), (2, 6, 6, 4, 1)])
        self.assertFalse(s.spans_truncated)

    def test_ambiguity_no_face_and_id_switch_are_counted(self) -> None:
        # kare 0-2: iki kişi + iki yüz; kare 3: ID'ler yer değiştirdi; kare 4: yüzler kayboldu; kare 5: aynı kutuda iki kopya iz
        frames = [[(1, P1), (2, P2)]] * 3 + [[(1, P2), (2, P1)], [(1, P2), (2, P1)], [(1, P1), (2, P1)]]
        face_frames = [[F1, F2]] * 4 + [[]] + [[F1]]
        det = SizedFakeDetector()
        s = run_face_association(camera_source(8), det, ScriptedTracker(frames), ScriptedFaceDetector(face_frames), FacePersonAssociator(), max_frames=6)
        self.assertEqual(s.break_counts.get("jump"), 3, "ID değişimi (2 iz) + kopya izin zıplaması")
        self.assertEqual(s.no_face_total, 2, "kare 4: yüzsüz iki kişi")
        self.assertEqual(s.ambiguous_persons_total, 2, "kare 5: aynı kutudaki iki iz tek yüz için belirsiz")
        self.assertEqual(s.ambiguous_faces_total, 1)
        self.assertEqual(s.frames_with_match, 4)

    def test_latency_statistics_are_separate_and_consistent(self) -> None:
        clock = FakeClock()
        source = camera_source(8, clock)
        det, trk, fd, assoc = scripted(6)
        s = run_face_association(source, det, trk, fd, assoc, max_frames=5, clock=clock)
        for name in ("read", "yolo", "tracking", "face", "association", "compute", "frame_to_result"):
            self.assertIsNotNone(getattr(s, name), name)
        self.assertEqual(s.yolo.median_ms, 12.0)  # type: ignore[union-attr]  (SizedFakeDetector.inference_ms)
        self.assertAlmostEqual(s.face.median_ms, 11.0)  # type: ignore[union-attr]  (faces() inference_ms)
        self.assertAlmostEqual(s.tracking.median_ms, 0.4)  # type: ignore[union-attr]
        self.assertAlmostEqual(s.compute.median_ms, 12.0 + 0.4 + 11.0 + s.association.median_ms, places=1)  # type: ignore[union-attr]
        self.assertIsNone(s.draw)
        self.assertIsNone(s.show)

    def test_drawing_and_show_are_measured_only_with_a_window(self) -> None:
        det, trk, fd, assoc = scripted(4)
        s = run_face_association(camera_source(6), det, trk, fd, assoc, FakeWindow(), max_frames=3)
        self.assertIsNotNone(s.draw)
        self.assertIsNotNone(s.show)
        self.assertIsNotNone(s.frame_to_display)

    def test_end_to_end_fps_uses_the_clock(self) -> None:
        clock = FakeClock()
        det = SizedFakeDetector()
        capture = FakeCapture([make_frame(i, *SIZE) for i in range(11)], props={cv2.CAP_PROP_FRAME_WIDTH: SIZE[0], cv2.CAP_PROP_FRAME_HEIGHT: SIZE[1]})
        source = FrameSource(SourceConfig.camera(0, width=SIZE[0], height=SIZE[1]), capture_factory=FakeFactory(capture), max_read_failures=1,
                             sleep=lambda s: None, clock=clock)
        source.open()
        original_detect = det.detect

        def slow_detect(image, **kw):  # type: ignore[no-untyped-def]
            clock.advance(0.1)  # kare başına 100 ms
            return original_detect(image, **kw)

        det.detect = slow_detect  # type: ignore[method-assign]
        s = run_face_association(source, det, ScriptedTracker([[(1, P1)]]), ScriptedFaceDetector([[F1]]), FacePersonAssociator(), max_frames=11, clock=clock)
        self.assertAlmostEqual(s.end_to_end_fps, 10.0, delta=0.5)


class ExitAndErrorTest(unittest.TestCase):
    def run_loop(self, n_frames: int = 10, window: FakeWindow | None = None, *, max_frames: int | None = None, **parts):  # type: ignore[no-untyped-def]
        det, trk, fd, assoc = scripted(n_frames)
        det = parts.get("det", det)
        trk = parts.get("trk", trk)
        fd = parts.get("fd", fd)
        return run_face_association(camera_source(n_frames), det, trk, fd, assoc, window, max_frames=max_frames)

    def test_end_of_stream_is_reported_for_video_like_sources(self) -> None:
        from camera_test_support import write_synthetic_video

        with tempfile.TemporaryDirectory() as tmp:
            path = write_synthetic_video(Path(tmp) / "v.mp4", 6, 25.0, SIZE)
            with FrameSource(SourceConfig.video(path)) as source:
                det, trk, fd, assoc = scripted(6)
                s = run_face_association(source, det, trk, fd, assoc)
        self.assertEqual(s.reason, REASON_END_OF_STREAM)
        self.assertEqual(s.frames_processed, 6)

    def test_max_frames(self) -> None:
        self.assertEqual(self.run_loop(10, max_frames=4).reason, REASON_MAX_FRAMES)

    def test_q_and_esc_stop_the_loop(self) -> None:
        for key in (ord("q"), ord("Q"), 27):
            with self.subTest(key=key):
                window = FakeWindow(keys={3: key})
                s = self.run_loop(10, window)
                self.assertEqual((s.reason, s.frames_processed), (REASON_USER_QUIT, 3))

    def test_closing_the_window_stops_the_loop(self) -> None:
        s = self.run_loop(10, FakeWindow(close_after=2))
        self.assertEqual((s.reason, s.frames_processed), (REASON_WINDOW_CLOSED, 2))

    def test_keyboard_interrupt_ends_cleanly_with_a_summary(self) -> None:
        class Interrupting(SizedFakeDetector):
            def detect(self, image, **kw):  # type: ignore[no-untyped-def]
                if len(self.calls) == 3:
                    raise KeyboardInterrupt
                return super().detect(image, **kw)

        s = self.run_loop(10, det=Interrupting())
        self.assertEqual(s.reason, REASON_INTERRUPTED)
        self.assertEqual(s.frames_processed, 3)

    def test_invalid_max_frames(self) -> None:
        for bad in (0, -1, True, 1.5, "3"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.run_loop(3, max_frames=bad)  # type: ignore[arg-type]

    def test_errors_from_each_stage_propagate_to_the_caller(self) -> None:
        class BadYolo(SizedFakeDetector):
            def detect(self, image, **kw):  # type: ignore[no-untyped-def]
                raise InferenceError("yolo bozuldu")

        class BadTracker(ScriptedTracker):
            def update(self, result, *, source=None):  # type: ignore[no-untyped-def]
                raise TrackerBackendError("bytetrack bozuldu")

        class BadFace(ScriptedFaceDetector):
            def detect(self, image, **kw):  # type: ignore[no-untyped-def]
                raise FaceInferenceError("yunet bozuldu")

        with self.assertRaises(InferenceError):
            self.run_loop(3, det=BadYolo())
        with self.assertRaises(TrackerBackendError):
            self.run_loop(3, trk=BadTracker([[(1, P1)]]))
        with self.assertRaises(FaceInferenceError):
            self.run_loop(3, fd=BadFace([[F1]]))

    def test_display_error_propagates(self) -> None:
        with self.assertRaises(DisplayError):
            self.run_loop(3, FakeWindow(raise_on_call={2: DisplayError("pencere")}))

    def test_a_face_failure_never_removes_tracked_people_before_the_error(self) -> None:
        """Yüz tespiti hata verince döngü durur; önceki karelerin sonuçları (on_result ile alınanlar) bozulmaz."""
        class FlakyFace(ScriptedFaceDetector):
            def detect(self, image, **kw):  # type: ignore[no-untyped-def]
                if len(self.calls) == 2:
                    raise FaceInferenceError("yunet bozuldu")
                return super().detect(image, **kw)

        got = []
        det, trk, _, assoc = scripted(5)
        with self.assertRaises(FaceInferenceError):
            run_face_association(camera_source(5), det, trk, FlakyFace([[F1, F2]]), assoc, on_result=lambda p, d, t, f, a: got.append(a))
        self.assertEqual(len(got), 2)
        self.assertTrue(all(a.person_count == 2 for a in got))


class HookEquivalenceTest(unittest.TestCase):
    def test_run_tracking_on_result_hook_gives_the_same_associations(self) -> None:
        """4B'nin ``run_tracking(..., on_result=...)`` kancası, yüz tespiti + ilişkilendirmeyi çalıştırabilir ve bu döngüyle AYNI eşleşmeleri verir."""
        frames = [[(1, P1), (2, P2)]] * 5 + [[(1, P2), (2, P1)]] * 3
        face_frames = [[F1, F2, STRAY]] * 8

        det_a, trk_a, fd_a, assoc_a = SizedFakeDetector(), ScriptedTracker(frames), ScriptedFaceDetector(face_frames), FacePersonAssociator()
        direct = []
        run_face_association(camera_source(8), det_a, trk_a, fd_a, assoc_a, max_frames=8, on_result=lambda p, d, t, f, a: direct.append(a))

        det_b, trk_b, fd_b, assoc_b = SizedFakeDetector(), ScriptedTracker(frames), ScriptedFaceDetector(face_frames), FacePersonAssociator()
        via_hook = []

        def hook(packet, detection, tracking):  # type: ignore[no-untyped-def]
            faces_ = fd_b.detect(packet.frame, frame_index=packet.frame_index, timestamp=packet.timestamp)
            via_hook.append(assoc_b.associate(tracking, faces_))

        run_tracking(camera_source(8), det_b, trk_b, None, max_frames=8, on_result=hook)
        self.assertEqual(len(direct), len(via_hook))
        for a, b in zip(direct, via_hook):
            self.assertEqual(a.pairs(), b.pairs())
            self.assertEqual([(p.track_id, p.status) for p in a.persons], [(p.track_id, p.status) for p in b.persons])
            self.assertEqual(a.breaks, b.breaks)


class InconsistencyInTheLoopTest(unittest.TestCase):
    def test_a_face_detector_reporting_another_frame_yields_invalid_frames_and_no_matches(self) -> None:
        class OffByOne(ScriptedFaceDetector):
            def detect(self, image, *, frame_index=None, timestamp=None):  # type: ignore[no-untyped-def]
                return super().detect(image, frame_index=None if frame_index is None else frame_index + 1, timestamp=timestamp)

        det, trk, _, assoc = scripted(4)
        s = run_face_association(camera_source(5), det, trk, OffByOne([[F1, F2]]), assoc, max_frames=4)
        self.assertEqual(s.invalid_frames, 4)
        self.assertEqual((s.matched_total, s.frames_with_match), (0, 0))
        self.assertEqual(s.persons_total, 8, "tutarsız girdide bile kişiler sayılır ve takipten çıkarılmaz")


if __name__ == "__main__":
    unittest.main()
