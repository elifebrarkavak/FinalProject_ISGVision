"""Aşama 3 / Grup 3: kamera/video hata yönetimi testleri (gerçek kamera açılmaz, izin istenmez)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from camera_test_support import FakeCapture, FakeFactory, make_frame, write_synthetic_video  # noqa: E402
from isg_vision.camera import (  # noqa: E402
    FrameReadError,
    FrameSource,
    SourceConfig,
    SourceConfigError,
    SourceDisconnectedError,
    VideoFileError,
)

FPS, COUNT = cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT


def camera(script, *, sleeps=None, max_failures=5, delay=0.05, **cap_kwargs):
    cap = FakeCapture(script, **cap_kwargs)
    source = FrameSource(
        SourceConfig.camera(width=None, height=None),
        capture_factory=FakeFactory(cap),
        max_read_failures=max_failures,
        retry_delay_s=delay,
        sleep=(sleeps.append if sleeps is not None else (lambda s: None)),
    )
    return source, cap


class CameraReadFailureTest(unittest.TestCase):
    def test_transient_failures_are_retried_without_consuming_indices(self) -> None:
        sleeps: list[float] = []
        source, _ = camera([make_frame(0), None, None, make_frame(1)], sleeps=sleeps)
        with source:
            first, second = source.read(), source.read()
        self.assertEqual((first.frame_index, second.frame_index), (0, 1))  # sıra ardışık kalır
        self.assertEqual(sleeps, [0.05, 0.05])  # iki başarısız okuma arasında bekleme

    def test_failure_counter_resets_after_a_success(self) -> None:
        # max_read_failures=5: art arda 4 hata + başarı + 4 hata + başarı hata fırlatmamalı
        script = [None] * 4 + [make_frame(0)] + [None] * 4 + [make_frame(1)]
        source, _ = camera(script, max_failures=5)
        with source:
            self.assertEqual(source.read().frame_index, 0)
            self.assertEqual(source.read().frame_index, 1)

    def test_no_frames_ever_gives_read_error_with_hint_after_bounded_attempts(self) -> None:
        sleeps: list[float] = []
        source, cap = camera([], sleeps=sleeps, max_failures=5)
        with source:
            with self.assertRaises(FrameReadError) as ctx:
                source.read()
        self.assertNotIsInstance(ctx.exception, SourceDisconnectedError)
        self.assertEqual(cap.read_calls, 5)  # sonsuz döngü yok: tam 5 deneme
        self.assertEqual(len(sleeps), 4)  # son denemeden sonra bekleme yok
        self.assertIn("izni", str(ctx.exception))
        self.assertIn("5 deneme", str(ctx.exception))

    def test_disconnect_after_streaming_is_reported_as_disconnect(self) -> None:
        source, _ = camera([make_frame(0), make_frame(1)], max_failures=3)
        with source:
            source.read()
            source.read()
            with self.assertRaises(SourceDisconnectedError) as ctx:
                source.read()
        self.assertIsInstance(ctx.exception, FrameReadError)  # genel yakalayıcılar çalışmaya devam eder
        self.assertIn("bağlantısı kesildi", str(ctx.exception))
        self.assertIn("2 kare", str(ctx.exception))

    def test_closed_device_is_not_retried(self) -> None:
        source, cap = camera([make_frame(0)], max_failures=50)
        with source:
            source.read()
            cap._opened = False  # sürücü cihazın kapandığını bildiriyor
            with self.assertRaises(SourceDisconnectedError):
                source.read()
        self.assertEqual(cap.read_calls, 2)  # tek başarısız okumadan sonra pes edilir

    def test_cv2_error_in_read_is_treated_as_failure_and_reported(self) -> None:
        source, _ = camera([cv2.error("sürücü hatası"), make_frame(0)])
        with source:
            self.assertEqual(source.read().frame_index, 0)  # geçici hata sonrası toparlandı
        source, _ = camera([cv2.error("kalıcı sürücü hatası")] * 5, max_failures=5)
        with source:
            with self.assertRaisesRegex(FrameReadError, "kalıcı sürücü hatası"):
                source.read()

    def test_resources_released_after_read_failure_inside_with(self) -> None:
        source, cap = camera([], max_failures=2)
        with self.assertRaises(FrameReadError):
            with source:
                source.read()
        self.assertEqual(cap.release_calls, 1)
        self.assertFalse(source.is_open)
        source.close()  # tekrar temizleme güvenli
        self.assertEqual(cap.release_calls, 1)

    def test_invalid_retry_settings_rejected(self) -> None:
        cfg = SourceConfig.camera()
        for kwargs in ({"max_read_failures": 0}, {"max_read_failures": -1}, {"max_read_failures": 1.5},
                       {"max_read_failures": True}, {"retry_delay_s": -0.1}):
            with self.subTest(**kwargs):
                with self.assertRaises(SourceConfigError):
                    FrameSource(cfg, **kwargs)


class VideoEndingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.video = self.dir / "klip.mp4"
        self.video.write_bytes(b"x")

    def fake_video(self, n_frames: int, declared: float) -> FrameSource:
        cap = FakeCapture([make_frame(i) for i in range(n_frames)], props={COUNT: declared, FPS: 25.0})
        return FrameSource(SourceConfig.video(self.video), capture_factory=FakeFactory(cap))

    def test_complete_video_is_not_premature(self) -> None:
        with self.fake_video(100, 100.0) as source:
            self.assertFalse(source.premature_end)  # henüz bitmedi
            self.assertEqual(len(list(source.frames())), 100)
            self.assertFalse(source.premature_end)

    def test_small_count_deviation_is_tolerated(self) -> None:
        with self.fake_video(99, 100.0) as source:  # OpenCV kare sayısı yaklaşık olabilir
            list(source.frames())
            self.assertFalse(source.premature_end)

    def test_video_ending_much_earlier_than_declared_is_flagged(self) -> None:
        with self.fake_video(40, 100.0) as source:
            self.assertEqual(len(list(source.frames())), 40)
            self.assertTrue(source.premature_end)
            self.assertIsNone(source.read())  # yine de kontrollü biçimde biter, çökmez

    def test_unknown_frame_count_is_never_premature(self) -> None:
        with self.fake_video(5, 0.0) as source:
            list(source.frames())
            self.assertIsNone(source.frame_count)
            self.assertFalse(source.premature_end)

    def test_camera_is_never_premature(self) -> None:
        source, _ = camera([make_frame(0)])
        with source:
            source.read()
            self.assertFalse(source.premature_end)

    def test_cv2_error_while_decoding_video_becomes_frame_read_error(self) -> None:
        cap = FakeCapture([make_frame(0), cv2.error("kod çözme hatası")], props={COUNT: 10.0})
        source = FrameSource(SourceConfig.video(self.video), capture_factory=FakeFactory(cap))
        with source:
            source.read()
            with self.assertRaisesRegex(FrameReadError, "kod çözme hatası"):
                source.read()
        self.assertEqual(cap.release_calls, 1)

    def test_real_truncated_avi_is_detected_as_premature_end(self) -> None:
        full = write_synthetic_video(self.dir / "tam.avi", n_frames=100, size=(160, 120), fourcc="MJPG")
        cut = self.dir / "kesik.avi"
        data = full.read_bytes()
        cut.write_bytes(data[: int(len(data) * 0.6)])
        with FrameSource(SourceConfig.video(cut)) as source:
            frames = list(source.frames())
            self.assertEqual(source.frame_count, 100)
        self.assertTrue(0 < len(frames) < 98)
        self.assertTrue(source.premature_end)
        self.assertEqual([p.frame_index for p in frames], list(range(len(frames))))

    def test_real_truncated_mp4_fails_cleanly_instead_of_crashing(self) -> None:
        full = write_synthetic_video(self.dir / "tam.mp4", n_frames=100, size=(160, 120))
        cut = self.dir / "kesik.mp4"
        data = full.read_bytes()
        cut.write_bytes(data[: int(len(data) * 0.6)])  # moov atom kaybı: dosya açılamaz
        with self.assertRaises(VideoFileError):
            with FrameSource(SourceConfig.video(cut)) as source:
                list(source.frames())


if __name__ == "__main__":
    unittest.main()
