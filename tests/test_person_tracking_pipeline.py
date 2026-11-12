"""Takip döngüsü (``run_tracking``): tek YOLO çağrısı, takipçi çağrı düzeni, özet/ölçüm, çıkış nedenleri, hata yayılımı, oturum sıfırlama.

[Mock/birim] Sahte kamera/pencere/tespitçi/takipçi: gerçek kamera, model, ByteTrack veya ağ yoktur. "Kare başına tek YOLO çağrısı" testi
GERÇEK ``PersonDetector`` + SAHTE YOLO modeliyle yapılır (modelin ``predict`` çağrıları sayılır). Gerçek ByteTrack ile bütünleşik döngü
testleri dosyanın sonundaki ``RealTrackerLoopTest``'tedir (``lap`` yoksa atlanır).
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
import threading
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("YOLO_OFFLINE", "1")

from camera_test_support import FakeCapture, FakeClock, FakeFactory, FakeWindow, make_frame, write_synthetic_video  # noqa: E402
from person_detection_test_support import FakeDetector, FakeYoloModel, make_detector  # noqa: E402
from tracking_test_support import FakeTracker, echo_tracker, make_tracker  # noqa: E402

from isg_vision.camera.capture import FrameSource  # noqa: E402
from isg_vision.camera.config import SourceConfig  # noqa: E402
from isg_vision.camera.display import PreviewWindow  # noqa: E402  (modül başında: test_no_side_effects modülleri yeniden yükler)
from isg_vision.camera.errors import DisplayError, SourceDisconnectedError  # noqa: E402
from isg_vision.human_detection.errors import DetectionError, InvalidFrameError  # noqa: E402
from isg_vision.tracking import pipeline  # noqa: E402
from isg_vision.tracking.errors import TrackerBackendError  # noqa: E402
from isg_vision.tracking.visualization import WINDOW_NAME  # noqa: E402
from isg_vision.tracking.pipeline import (  # noqa: E402
    REASON_END_OF_STREAM,
    REASON_INTERRUPTED,
    REASON_MAX_FRAMES,
    REASON_USER_QUIT,
    REASON_WINDOW_CLOSED,
    run_tracking,
)

LAP_AVAILABLE = importlib.util.find_spec("lap") is not None and importlib.util.find_spec("ultralytics") is not None


def camera_source(n_frames: int, clock: FakeClock | None = None, *, size: tuple[int, int] = (64, 48)) -> tuple[FrameSource, FakeCapture]:
    capture = FakeCapture(
        [make_frame(i, *size) for i in range(n_frames)], props={cv2.CAP_PROP_FRAME_WIDTH: size[0], cv2.CAP_PROP_FRAME_HEIGHT: size[1]}
    )
    kwargs = {"clock": clock} if clock is not None else {}
    source = FrameSource(
        SourceConfig.camera(0, width=size[0], height=size[1]), capture_factory=FakeFactory(capture),
        max_read_failures=1, sleep=lambda s: None, **kwargs,
    )
    source.open()
    return source, capture


class CallPatternTest(unittest.TestCase):
    def test_detector_and_tracker_are_called_exactly_once_per_frame_in_order(self) -> None:
        source, _ = camera_source(8)
        detector, tracker = FakeDetector(counts=[2]), FakeTracker()
        summary = run_tracking(source, detector, tracker, max_frames=5)
        self.assertEqual(summary.frames_processed, 5)
        self.assertEqual(len(detector.detect_calls), 5)
        self.assertEqual(len(tracker.updates), 5)
        for i, (d, u) in enumerate(zip(detector.detect_calls, tracker.updates)):
            self.assertEqual(d["frame_index"], i)
            self.assertEqual(u["result"].frame_index, i, "takipçi, aynı karenin tespit sonucunu almalı")

    def test_tracker_is_reset_once_before_the_first_frame(self) -> None:
        source, _ = camera_source(4)
        tracker = FakeTracker()
        run_tracking(source, FakeDetector(), tracker, max_frames=2)
        self.assertEqual(tracker.events[0], "reset")
        self.assertEqual(tracker.events.count("reset"), 1)

    def test_frames_without_detections_are_still_given_to_the_tracker(self) -> None:
        source, _ = camera_source(6)
        tracker = FakeTracker()
        summary = run_tracking(source, FakeDetector(counts=[0, 2, 0, 0, 1, 0]), tracker, max_frames=6)
        self.assertEqual(len(tracker.updates), 6, "tespitsiz kareler de takipçiye verilmeli (kayıp izler yaşlansın)")
        self.assertEqual([len(u["result"].detections) for u in tracker.updates], [0, 2, 0, 0, 1, 0])
        self.assertEqual(summary.frames_with_tracks, 2)

    def test_source_label_is_passed_to_tracker(self) -> None:
        source, _ = camera_source(3)
        tracker = FakeTracker()
        run_tracking(source, FakeDetector(), tracker, max_frames=3)
        self.assertEqual({u["source"] for u in tracker.updates}, {"camera:0"})

    def test_yolo_model_runs_exactly_once_per_frame_with_real_detector_class(self) -> None:
        """Mimari güvence (mock model): kare başına modelin predict'i TEK kez çağrılır; takip ek çıkarım yapmaz."""
        with TempDirMixinCtx() as tmp:
            model = FakeYoloModel(script=[[((10, 5, 20, 40), 0.9, 0)], [((10, 5, 20, 40), 0.8, 0), ((30, 5, 40, 40), 0.7, 0)], []])
            detector, _ = make_detector(model, tmp)
            tracker = echo_tracker()
            source, _ = camera_source(7)
            summary = run_tracking(source, detector, tracker, max_frames=7)
        self.assertEqual(summary.frames_processed, 7)
        self.assertEqual(len(model.predict_calls), 7, "7 kare için tam 7 YOLO çıkarımı")
        self.assertEqual(detector.load_count, 1, "model yalnızca bir kez yüklenir")
        self.assertEqual(tracker.echo_backend.calls, 7)  # type: ignore[attr-defined]
        self.assertEqual(tracker.load_count, 1)

    def test_tracker_never_receives_the_image(self) -> None:
        """Takipçi yalnızca DetectionResult alır (görüntü kopyalanmaz/YOLO'ya verilmez)."""
        source, _ = camera_source(3)
        tracker = FakeTracker()
        run_tracking(source, FakeDetector(), tracker, max_frames=3)
        for u in tracker.updates:
            self.assertEqual(set(u), {"result", "source"})


class TempDirMixinCtx:
    def __enter__(self) -> Path:
        import tempfile

        self._t = tempfile.TemporaryDirectory()
        return Path(self._t.name)

    def __exit__(self, *exc: object) -> None:
        self._t.cleanup()


class SummaryTest(unittest.TestCase):
    def test_counts_and_spans(self) -> None:
        source, _ = camera_source(6)
        summary = run_tracking(source, FakeDetector(counts=[1, 2, 2, 0, 3, 1]), FakeTracker(lost=2), max_frames=6)
        self.assertEqual(summary.frames_processed, 6)
        self.assertEqual((summary.detections_total, summary.detections_max), (9, 3))
        self.assertEqual(summary.active_max, 3)
        self.assertEqual(summary.lost_max, 2)
        self.assertEqual(summary.frames_with_tracks, 5)
        self.assertEqual(summary.distinct_tracks, 3, "FakeTracker ID'leri 1..3")
        by_id = {s.track_id: s for s in summary.tracks}
        self.assertEqual((by_id[1].first_frame, by_id[1].last_frame, by_id[1].frames_active), (0, 5, 5))
        self.assertEqual((by_id[3].first_frame, by_id[3].last_frame, by_id[3].frames_active), (4, 4, 1))
        self.assertGreater(by_id[1].confidence_mean, 0.0)
        self.assertGreaterEqual(by_id[1].confidence_max, by_id[1].confidence_mean)
        self.assertEqual(summary.dropped_boxes, 0)
        self.assertEqual(summary.reset_events, 0)

    def test_span_duration_uses_packet_timestamps(self) -> None:
        clock = FakeClock()
        source, _ = camera_source(5, clock)
        detector = FakeDetector(counts=[1], clock=clock, inference_s=0.1)
        summary = run_tracking(source, detector, FakeTracker(), clock=clock, max_frames=5)
        span = summary.tracks[0]
        self.assertAlmostEqual(span.duration_s, 0.4, places=6)  # 5 kare, 0.1 sn aralıkla: ilk-son = 0.4 sn

    def test_span_with_gap_shows_fewer_active_frames_than_range(self) -> None:
        source, _ = camera_source(8)
        # FakeTracker her tespite ID=sıra+1 verir; ID 1 kare 0-1 ve 6-7'de, aradaki karelerde tespit yok
        summary = run_tracking(source, FakeDetector(counts=[1, 1, 0, 0, 0, 0, 1, 1]), FakeTracker(), max_frames=8)
        s = summary.tracks[0]
        self.assertEqual((s.first_frame, s.last_frame, s.frames_active), (0, 7, 4))

    def test_spans_are_capped_and_flagged(self) -> None:
        source, _ = camera_source(4)
        original = pipeline.MAX_TRACK_SPANS
        pipeline.MAX_TRACK_SPANS = 2
        try:
            summary = run_tracking(source, FakeDetector(counts=[5]), FakeTracker(), max_frames=4)
        finally:
            pipeline.MAX_TRACK_SPANS = original
        self.assertEqual(len(summary.tracks), 2)
        self.assertTrue(summary.spans_truncated)
        self.assertEqual(summary.distinct_tracks, 5, "özet sınırı sayımı etkilemez")

    def test_mean_property(self) -> None:
        source, _ = camera_source(4)
        summary = run_tracking(source, FakeDetector(counts=[2]), FakeTracker(), max_frames=4)
        self.assertEqual(summary.detections_mean, 2.0)


class LatencyTest(unittest.TestCase):
    def test_every_stage_is_measured_separately(self) -> None:
        clock = FakeClock()
        source, _ = camera_source(10, clock)
        detector = FakeDetector(counts=[1], clock=clock, inference_s=0.040)
        tracker = FakeTracker(clock=clock, track_s=0.003)
        summary = run_tracking(source, detector, tracker, FakeWindow(clock), max_frames=10, clock=clock)
        self.assertEqual(summary.inference.count, 10)  # type: ignore[union-attr]
        self.assertAlmostEqual(summary.inference.median_ms, 40.0, places=6)  # type: ignore[union-attr]
        self.assertAlmostEqual(summary.tracking.median_ms, 3.0, places=6)  # type: ignore[union-attr]
        self.assertAlmostEqual(summary.detect_plus_track.median_ms, 43.0, places=6)  # type: ignore[union-attr]
        self.assertAlmostEqual(summary.frame_to_result.median_ms, 43.0, places=6)  # type: ignore[union-attr]
        self.assertIsNotNone(summary.read)
        self.assertEqual(summary.draw.count, 10)  # type: ignore[union-attr]
        self.assertEqual(summary.show.count, 10)  # type: ignore[union-attr]
        # pencere 1 ms "bekler": kare -> görüntü = 43 + 1 ms
        self.assertAlmostEqual(summary.frame_to_display.median_ms, 44.0, places=6)  # type: ignore[union-attr]

    def test_end_to_end_fps_uses_measured_time_not_source_fps(self) -> None:
        clock = FakeClock()
        capture = FakeCapture([make_frame(i) for i in range(21)], props={cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 48, cv2.CAP_PROP_FPS: 999.0})
        source = FrameSource(SourceConfig.camera(0, width=64, height=48), capture_factory=FakeFactory(capture), clock=clock, sleep=lambda s: None)
        source.open()
        detector = FakeDetector(counts=[0], clock=clock, inference_s=0.050)
        summary = run_tracking(source, detector, FakeTracker(), clock=clock, max_frames=21)
        self.assertAlmostEqual(summary.end_to_end_fps, 20.0, places=6)  # 20 aralık / 1.0 sn (kaynağın bildirdiği 999 FPS kullanılmaz)

    def test_no_window_means_no_draw_or_display_samples(self) -> None:
        source, _ = camera_source(4)
        summary = run_tracking(source, FakeDetector(), FakeTracker(), max_frames=4)
        self.assertIsNone(summary.draw)
        self.assertIsNone(summary.show)
        self.assertIsNone(summary.frame_to_display)
        self.assertIsNotNone(summary.tracking)

    def test_video_with_window_leaves_display_latency_unmeasured_but_measures_drawing(self) -> None:
        with TempDirMixinCtx() as tmp:
            video = write_synthetic_video(Path(tmp) / "v.mp4", n_frames=5, size=(64, 48))
            with FrameSource(SourceConfig.video(video)) as source:
                summary = run_tracking(source, FakeDetector(), FakeTracker(), FakeWindow())
        self.assertIsNone(summary.show, "video + pencere: oynatma beklemesi gösterme süresine karışmaz")
        self.assertIsNone(summary.frame_to_display)
        self.assertEqual(summary.draw.count, 5)  # type: ignore[union-attr]
        self.assertEqual(summary.reason, REASON_END_OF_STREAM)


class ExitReasonTest(unittest.TestCase):
    def test_max_frames(self) -> None:
        source, _ = camera_source(10)
        summary = run_tracking(source, FakeDetector(), FakeTracker(), max_frames=4)
        self.assertEqual((summary.reason, summary.frames_processed), (REASON_MAX_FRAMES, 4))

    def test_invalid_max_frames(self) -> None:
        source, _ = camera_source(2)
        for bad in (0, -1, 1.5, True, "3"):
            with self.assertRaises(ValueError):
                run_tracking(source, FakeDetector(), FakeTracker(), max_frames=bad)  # type: ignore[arg-type]

    def test_user_quit_key(self) -> None:
        for key in (ord("q"), ord("Q"), 27):
            with self.subTest(key=key):
                source, _ = camera_source(10)
                window = FakeWindow(keys={3: key})
                summary = run_tracking(source, FakeDetector(), FakeTracker(), window)
                self.assertEqual((summary.reason, summary.frames_processed, len(window.images)), (REASON_USER_QUIT, 3, 3))

    def test_window_closed_by_user(self) -> None:
        source, _ = camera_source(10)
        summary = run_tracking(source, FakeDetector(), FakeTracker(), FakeWindow(close_after=2))
        self.assertEqual((summary.reason, summary.frames_processed), (REASON_WINDOW_CLOSED, 2))

    def test_end_of_video(self) -> None:
        with TempDirMixinCtx() as tmp:
            video = write_synthetic_video(Path(tmp) / "v.mp4", n_frames=6, size=(64, 48))
            with FrameSource(SourceConfig.video(video)) as source:
                summary = run_tracking(source, FakeDetector(), FakeTracker())
        self.assertEqual((summary.reason, summary.frames_processed), (REASON_END_OF_STREAM, 6))

    def test_ctrl_c_in_detector_is_reported_not_raised(self) -> None:
        source, _ = camera_source(10)
        summary = run_tracking(source, FakeDetector(detect_errors={3: KeyboardInterrupt()}), FakeTracker())
        self.assertEqual((summary.reason, summary.frames_processed), (REASON_INTERRUPTED, 2))

    def test_ctrl_c_in_tracker_is_reported_not_raised(self) -> None:
        source, _ = camera_source(10)
        summary = run_tracking(source, FakeDetector(), FakeTracker(update_errors={4: KeyboardInterrupt()}))
        self.assertEqual((summary.reason, summary.frames_processed), (REASON_INTERRUPTED, 3))


class ErrorPropagationTest(unittest.TestCase):
    def test_detection_error_propagates(self) -> None:
        source, _ = camera_source(5)
        with self.assertRaises(DetectionError):
            run_tracking(source, FakeDetector(detect_errors={2: InvalidFrameError("kare")}), FakeTracker())

    def test_tracking_error_propagates_and_stops_the_loop(self) -> None:
        source, _ = camera_source(5)
        tracker = FakeTracker(update_errors={2: TrackerBackendError("ByteTrack çöktü")})
        with self.assertRaisesRegex(TrackerBackendError, "çöktü"):
            run_tracking(source, FakeDetector(), tracker)
        self.assertEqual(len(tracker.updates), 2, "hatadan sonra döngü ilerlememeli")

    def test_display_error_propagates(self) -> None:
        source, _ = camera_source(5)
        with self.assertRaises(DisplayError):
            run_tracking(source, FakeDetector(), FakeTracker(), FakeWindow(raise_on_call={2: DisplayError("pencere yok")}))

    def test_source_error_propagates(self) -> None:
        capture = FakeCapture([make_frame(0), make_frame(1)], props={cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 48})
        source = FrameSource(SourceConfig.camera(0, width=64, height=48), capture_factory=FakeFactory(capture), max_read_failures=1, sleep=lambda s: None)
        source.open()
        with self.assertRaises(SourceDisconnectedError):
            run_tracking(source, FakeDetector(), FakeTracker())


class HookTest(unittest.TestCase):
    def test_on_result_receives_packet_detection_and_tracking(self) -> None:
        source, _ = camera_source(4)
        seen: list[tuple[int, int, int, int]] = []
        run_tracking(
            source, FakeDetector(counts=[2]), FakeTracker(), max_frames=4,
            on_result=lambda p, d, t: seen.append((p.frame_index, d.frame_index, t.frame_index, t.active_count)),
        )
        self.assertEqual(seen, [(i, i, i, 2) for i in range(4)])

    def test_drawing_does_not_modify_the_packet_frame(self) -> None:
        source, capture = camera_source(3)
        window = FakeWindow()
        refs: list[np.ndarray] = []  # KOPYA değil referans: çizim yerinde değiştirseydi burada görünürdü
        run_tracking(source, FakeDetector(counts=[2]), FakeTracker(), window, max_frames=3, on_result=lambda p, d, t: refs.append(p.frame))
        self.assertEqual(len(window.images), 3)
        for i, (ref, shown) in enumerate(zip(refs, window.images)):
            np.testing.assert_array_equal(ref, make_frame(i), "kaynak kare çizimle değişmemeli")
            self.assertFalse(np.array_equal(shown, make_frame(i)), "gösterilen görüntü çizim içermeli")
            self.assertIsNot(shown, ref)


class SessionTest(unittest.TestCase):
    def test_reusing_tracker_for_a_new_run_resets_state(self) -> None:
        tracker = FakeTracker()
        source1, _ = camera_source(4)
        run_tracking(source1, FakeDetector(counts=[1]), tracker, max_frames=4)
        source2, _ = camera_source(3)
        run_tracking(source2, FakeDetector(counts=[1]), tracker, max_frames=3)
        self.assertEqual(tracker.events.count("reset"), 2, "her çalıştırma takipçiyi sıfırlamalı (kaynak/oturum değişimi)")
        self.assertEqual(tracker.events[0], "reset")

    def test_resolution_change_mid_stream_is_counted_as_reset_event(self) -> None:
        frames = [make_frame(i, 64, 48) for i in range(3)] + [make_frame(i, 96, 72) for i in range(3, 6)]
        capture = FakeCapture(frames, props={cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 48})
        source = FrameSource(SourceConfig.camera(0, width=64, height=48), capture_factory=FakeFactory(capture), max_read_failures=1, sleep=lambda s: None)
        source.open()
        summary = run_tracking(source, FakeDetector(counts=[1]), echo_tracker(), max_frames=6)
        self.assertEqual(summary.reset_events, 1)
        self.assertEqual(summary.frames_processed, 6)


@unittest.skipUnless(LAP_AVAILABLE, "gerçek ByteTrack: 'lap' ve 'ultralytics' gerekir (atlandı)")
class RealTrackerLoopTest(unittest.TestCase):
    """[Gerçek ByteTrack + sahte tespitçi] Döngü ile gerçek takipçinin bütünleşmesi (YOLO yok)."""

    def test_loop_with_real_bytetrack_keeps_ids_for_moving_people(self) -> None:
        class MovingPeople(FakeDetector):
            def detect(self, image, *, frame_index=None, timestamp=None):  # type: ignore[no-untyped-def]
                from isg_vision.human_detection.results import DetectionResult, PersonDetection

                self.detect_calls.append({"image": image, "frame_index": frame_index, "timestamp": timestamp})
                i = frame_index or 0
                h, w = image.shape[:2]
                people = (
                    PersonDetection(0, "person", 0.9, (5.0 + i, 4.0, 19.0 + i, 40.0), frame_index=frame_index, timestamp=timestamp),
                    PersonDetection(0, "person", 0.8, (45.0 - i, 4.0, 59.0 - i, 40.0), frame_index=frame_index, timestamp=timestamp),
                )
                return DetectionResult(people, (w, h), frame_index, timestamp, 5.0, "cpu")

        source, _ = camera_source(10)
        tracker = make_tracker()
        summary = run_tracking(source, MovingPeople(), tracker, max_frames=10)
        self.assertEqual(summary.frames_processed, 10)
        self.assertEqual(summary.distinct_tracks, 2)
        self.assertEqual(summary.active_max, 2)
        self.assertEqual([s.frames_active for s in summary.tracks], [10, 10])
        self.assertIsNotNone(summary.tracking)
        self.assertEqual(tracker.stats.frames, 10)

    def test_real_tracker_reused_on_a_second_run_restarts_ids(self) -> None:
        tracker = make_tracker()
        s1, _ = camera_source(5)
        run_tracking(s1, FakeDetector(counts=[1]), tracker, max_frames=5)
        s2, _ = camera_source(5)
        summary = run_tracking(s2, FakeDetector(counts=[1]), tracker, max_frames=5)
        self.assertEqual([(t.session_id, t.track_id) for t in summary.tracks], [(2, 1)])


GUI_ENABLED = sys.platform == "darwin" and os.environ.get("ISG_GUI_TESTS") == "1"


@unittest.skipUnless(GUI_ENABLED, "Gerçek pencere testi: macOS + ISG_GUI_TESTS=1 + grafik oturumu gerekir")
class RealWindowTrackingLoopTest(unittest.TestCase):
    """Gerçek OpenCV/Cocoa penceresiyle takip döngüsü (kamera YOK; sahte kareler). Aşama 3'te çözülen macOS kırmızı kapatma düğmesi
    düzeltmesi 4B penceresinde de çalışmalı: düğme tıklanınca döngü biter ve kaynaklar serbest kalır."""

    def test_red_button_click_ends_tracking_loop_and_releases_camera(self) -> None:
        from isg_vision.camera.macos_window import CocoaRuntime

        frame = make_frame(0, 320, 240)
        capture = FakeCapture([frame] * 3000, props={cv2.CAP_PROP_FPS: 30.0})  # aynı dizi başvurusu: bellek küçük
        source = FrameSource(SourceConfig.camera(width=None, height=None), capture_factory=FakeFactory(capture))
        runtime = CocoaRuntime()
        window = PreviewWindow(WINDOW_NAME)
        tracker = FakeTracker()

        def click() -> None:
            native = runtime.find_window(WINDOW_NAME)
            button = runtime._send(native, "standardWindowButton:", 0, argtypes=(ctypes.c_long,))
            runtime._send(
                button, "performSelectorOnMainThread:withObject:waitUntilDone:", runtime._objc.sel_registerName(b"performClick:"),
                None, False, argtypes=(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool),
            )

        timer = threading.Timer(0.5, click)
        timer.daemon = True
        timer.start()
        try:
            with source, window:
                summary = run_tracking(source, FakeDetector(counts=[2]), tracker, window)
                self.assertIs(window.close_button_enabled, True)
        finally:
            timer.cancel()
        self.assertEqual(summary.reason, REASON_WINDOW_CLOSED)
        self.assertLess(summary.frames_processed, 3000)
        self.assertEqual(capture.release_calls, 1)

    def test_max_frames_run_with_real_window_draws_tracks_without_error(self) -> None:
        frame = make_frame(0, 320, 240)
        capture = FakeCapture([frame] * 30, props={cv2.CAP_PROP_FPS: 30.0})
        source = FrameSource(SourceConfig.camera(width=None, height=None), capture_factory=FakeFactory(capture))
        with source, PreviewWindow(WINDOW_NAME) as window:
            summary = run_tracking(source, FakeDetector(counts=[0, 1, 3]), FakeTracker(lost=1), window, max_frames=30)
        self.assertEqual((summary.reason, summary.frames_processed), (REASON_MAX_FRAMES, 30))


if __name__ == "__main__":
    unittest.main()
