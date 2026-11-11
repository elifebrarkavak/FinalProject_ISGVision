"""Aşama 3 / Grup 1: kare okuma altyapısı (yapılandırma, FramePacket, FrameSource) testleri.

Gerçek kamera açılmaz; kullanıcı izni istenmez; ağ kullanılmaz. Video testleri geçici sentetik dosyalarla yapılır.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from camera_test_support import FakeCapture, FakeFactory, make_frame, write_synthetic_video  # noqa: E402
from isg_vision.camera import (  # noqa: E402
    CameraOpenError,
    FramePacket,
    FrameReadError,
    FrameSource,
    SourceConfig,
    SourceConfigError,
    SourceError,
    VideoFileError,
)
from isg_vision.camera.capture import resolve_backend  # noqa: E402
from isg_vision.camera.config import MAX_CAMERA_INDEX, validate_video_path  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SourceConfigTest(unittest.TestCase):
    def test_camera_defaults(self) -> None:
        cfg = SourceConfig.camera()
        self.assertEqual(cfg.kind, "camera")
        self.assertEqual(cfg.camera_index, 0)
        self.assertEqual(cfg.backend, "auto")
        self.assertEqual(cfg.label, "camera:0")

    def test_camera_custom_index(self) -> None:
        self.assertEqual(SourceConfig.camera(2).camera_index, 2)
        self.assertEqual(SourceConfig.camera(MAX_CAMERA_INDEX).camera_index, MAX_CAMERA_INDEX)

    def test_invalid_camera_index_rejected(self) -> None:
        for bad in (-1, MAX_CAMERA_INDEX + 1, "0", 1.5, None, True):
            with self.subTest(index=bad):
                with self.assertRaises(SourceConfigError):
                    SourceConfig.camera(bad)  # type: ignore[arg-type]

    def test_invalid_backend_rejected(self) -> None:
        with self.assertRaises(SourceConfigError):
            SourceConfig.camera(0, backend="directshow")

    def test_invalid_kind_rejected(self) -> None:
        with self.assertRaises(SourceConfigError):
            SourceConfig(kind="rtsp")

    def test_video_requires_path(self) -> None:
        for bad in ("", "   ", None):
            with self.subTest(path=bad):
                with self.assertRaises(SourceConfigError):
                    SourceConfig.video(bad)  # type: ignore[arg-type]

    def test_camera_with_video_path_rejected(self) -> None:
        with self.assertRaises(SourceConfigError):
            SourceConfig(kind="camera", video_path=Path("x.mp4"))

    def test_video_path_is_converted_to_path_and_label(self) -> None:
        cfg = SourceConfig.video("klip.mp4")
        self.assertIsInstance(cfg.video_path, Path)
        self.assertEqual(cfg.kind, "video")
        self.assertEqual(cfg.label, "video:klip.mp4")

    def test_config_is_immutable(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            SourceConfig.camera().camera_index = 3  # type: ignore[misc]

    def test_config_creation_does_not_touch_filesystem_or_camera(self) -> None:
        # Var olmayan bir video yolu yapılandırma aşamasında hata vermez; denetim open() sırasındadır.
        SourceConfig.video("/yok/boyle/bir/dosya.mp4")


class ValidateVideoPathTest(unittest.TestCase):
    def test_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(VideoFileError, "bulunamadı"):
                validate_video_path(Path(tmp) / "yok.mp4")

    def test_directory_is_not_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(VideoFileError, "dosya değil"):
                validate_video_path(Path(tmp))

    def test_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "bos.mp4"
            empty.write_bytes(b"")
            with self.assertRaisesRegex(VideoFileError, "boş"):
                validate_video_path(empty)

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root izin kısıtlamasına takılmaz")
    def test_unreadable_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            locked = Path(tmp) / "kilitli.mp4"
            locked.write_bytes(b"x")
            locked.chmod(0o000)
            try:
                with self.assertRaisesRegex(VideoFileError, "okunamıyor"):
                    validate_video_path(locked)
            finally:
                locked.chmod(0o600)

    def test_valid_file_returned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ok = Path(tmp) / "var.mp4"
            ok.write_bytes(b"x")
            self.assertEqual(validate_video_path(ok), ok)


class FramePacketTest(unittest.TestCase):
    def make(self, **overrides) -> FramePacket:
        kwargs = dict(frame=make_frame(1, 64, 48), frame_index=0, timestamp=1.0, wall_time=2.0,
                      source="camera:0", source_kind="camera")
        kwargs.update(overrides)
        return FramePacket(**kwargs)

    def test_metadata_and_real_size(self) -> None:
        p = self.make(frame=make_frame(1, width=320, height=240), frame_index=7)
        self.assertEqual((p.width, p.height), (320, 240))
        self.assertEqual(p.frame_index, 7)
        self.assertEqual(p.source, "camera:0")
        self.assertEqual(p.source_kind, "camera")
        self.assertEqual(p.timestamp, 1.0)
        self.assertEqual(p.wall_time, 2.0)

    def test_invalid_frames_rejected(self) -> None:
        bad_frames = (
            np.zeros((48, 64), np.uint8),          # gri (2B)
            np.zeros((48, 64, 4), np.uint8),       # 4 kanal
            np.zeros((0, 64, 3), np.uint8),        # boş
            [[0, 0, 0]],                           # NumPy değil
        )
        for bad in bad_frames:
            with self.subTest(frame=type(bad).__name__):
                with self.assertRaises(ValueError):
                    self.make(frame=bad)

    def test_invalid_index_and_kind_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.make(frame_index=-1)
        with self.assertRaises(ValueError):
            self.make(source_kind="rtsp")

    def test_packet_is_immutable(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.make().frame_index = 5  # type: ignore[misc]


class ResolveBackendTest(unittest.TestCase):
    def test_auto_uses_avfoundation_on_macos_only(self) -> None:
        self.assertEqual(resolve_backend("auto", platform="darwin"), cv2.CAP_AVFOUNDATION)
        self.assertEqual(resolve_backend("auto", platform="linux"), cv2.CAP_ANY)

    def test_explicit_choices(self) -> None:
        self.assertEqual(resolve_backend("avfoundation", platform="linux"), cv2.CAP_AVFOUNDATION)
        self.assertEqual(resolve_backend("any", platform="darwin"), cv2.CAP_ANY)


class FrameSourceCameraTest(unittest.TestCase):
    def test_open_passes_index_and_backend_to_capture(self) -> None:
        factory = FakeFactory(FakeCapture([make_frame(0)]))
        with FrameSource(SourceConfig.camera(3, backend="any"), capture_factory=factory):
            pass
        self.assertEqual(factory.calls, [(3, cv2.CAP_ANY)])

    def test_camera_not_opened_raises_and_releases(self) -> None:
        cap = FakeCapture(opened=False)
        source = FrameSource(SourceConfig.camera(), capture_factory=FakeFactory(cap))
        with self.assertRaises(CameraOpenError) as ctx:
            source.open()
        self.assertFalse(source.is_open)
        self.assertEqual(cap.release_calls, 1)
        self.assertIn("izni", str(ctx.exception))  # kullanıcıya olası nedeni söyler
        self.assertIn("başka bir uygulama", str(ctx.exception))

    def test_capture_factory_exception_becomes_camera_open_error(self) -> None:
        source = FrameSource(SourceConfig.camera(), capture_factory=FakeFactory(cv2.error("sürücü hatası")))
        with self.assertRaises(CameraOpenError):
            source.open()
        self.assertFalse(source.is_open)

    def test_frame_indices_are_sequential_and_metadata_correct(self) -> None:
        ticks = iter([10.0, 10.5, 11.0, 11.5])
        cap = FakeCapture([make_frame(i, 80, 60) for i in range(4)])
        source = FrameSource(SourceConfig.camera(1), capture_factory=FakeFactory(cap), clock=lambda: next(ticks))
        with source:
            packets = [source.read() for _ in range(4)]
        self.assertEqual([p.frame_index for p in packets], [0, 1, 2, 3])
        self.assertEqual([p.timestamp for p in packets], [10.0, 10.5, 11.0, 11.5])
        for p in packets:
            self.assertEqual(p.source, "camera:1")
            self.assertEqual(p.source_kind, "camera")
            self.assertEqual((p.width, p.height), (80, 60))
            self.assertGreater(p.wall_time, 1e9)  # Unix zamanı
        self.assertEqual(source.frames_read, 4)

    def test_failed_camera_read_raises(self) -> None:
        source = FrameSource(SourceConfig.camera(), capture_factory=FakeFactory(FakeCapture([None])), sleep=lambda s: None)
        with source:
            with self.assertRaises(FrameReadError):
                source.read()

    def test_read_before_open_and_double_open_rejected(self) -> None:
        source = FrameSource(SourceConfig.camera(), capture_factory=FakeFactory(FakeCapture([make_frame(0)])))
        with self.assertRaises(SourceError):
            source.read()
        source.open()
        try:
            with self.assertRaises(SourceError):
                source.open()
        finally:
            source.close()


class FrameSourceReleaseTest(unittest.TestCase):
    def test_close_releases_once_and_is_idempotent(self) -> None:
        cap = FakeCapture([make_frame(0)])
        source = FrameSource(SourceConfig.camera(), capture_factory=FakeFactory(cap))
        source.open()
        source.close()
        source.close()
        source.close()
        self.assertEqual(cap.release_calls, 1)
        self.assertFalse(source.is_open)

    def test_close_without_open_is_safe(self) -> None:
        FrameSource(SourceConfig.camera(), capture_factory=FakeFactory(FakeCapture())).close()

    def test_context_manager_releases_on_exception_in_body(self) -> None:
        cap = FakeCapture([make_frame(0)])
        with self.assertRaises(RuntimeError):
            with FrameSource(SourceConfig.camera(), capture_factory=FakeFactory(cap)):
                raise RuntimeError("beklenmeyen hata")
        self.assertEqual(cap.release_calls, 1)

    def test_context_manager_releases_on_keyboard_interrupt(self) -> None:
        cap = FakeCapture([make_frame(0)])
        with self.assertRaises(KeyboardInterrupt):
            with FrameSource(SourceConfig.camera(), capture_factory=FakeFactory(cap)) as source:
                source.read()
                raise KeyboardInterrupt
        self.assertEqual(cap.release_calls, 1)

    def test_reopen_after_close_restarts_index(self) -> None:
        caps = [FakeCapture([make_frame(0), make_frame(1)]), FakeCapture([make_frame(0)])]
        source = FrameSource(SourceConfig.camera(), capture_factory=lambda *a: caps.pop(0))
        source.open()
        source.read()
        source.read()
        source.close()
        source.open()
        self.assertEqual(source.read().frame_index, 0)
        source.close()


class FrameSourceVideoFakeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.video = Path(self._tmp.name) / "klip.mp4"
        self.video.write_bytes(b"gercek-icerik-onemli-degil")  # FakeCapture dosyayı okumaz; yalnızca varlık denetimi

    def test_reads_all_frames_then_none_and_stays_ended(self) -> None:
        cap = FakeCapture([make_frame(i) for i in range(3)])
        source = FrameSource(SourceConfig.video(self.video), capture_factory=FakeFactory(cap))
        with source:
            packets = list(source.frames())
            reads_at_end = cap.read_calls
            self.assertIsNone(source.read())  # bittikten sonra tekrar okuma
            self.assertEqual(cap.read_calls, reads_at_end)  # alt kaynağa yeniden başvurulmaz
        self.assertEqual([p.frame_index for p in packets], [0, 1, 2])
        self.assertTrue(all(p.source_kind == "video" for p in packets))
        self.assertEqual(packets[0].source, f"video:{self.video}")

    def test_video_open_receives_string_path_without_backend(self) -> None:
        factory = FakeFactory(FakeCapture([make_frame(0)]))
        with FrameSource(SourceConfig.video(self.video), capture_factory=factory):
            pass
        self.assertEqual(factory.calls, [(str(self.video),)])

    def test_missing_video_raises_before_opening_capture(self) -> None:
        factory = FakeFactory(FakeCapture())
        source = FrameSource(SourceConfig.video(self.video.with_name("yok.mp4")), capture_factory=factory)
        with self.assertRaisesRegex(VideoFileError, "bulunamadı"):
            source.open()
        self.assertEqual(factory.calls, [])  # kaynak hiç açılmadı

    def test_unopenable_video_raises_and_releases(self) -> None:
        cap = FakeCapture(opened=False)
        source = FrameSource(SourceConfig.video(self.video), capture_factory=FakeFactory(cap))
        with self.assertRaisesRegex(VideoFileError, "açılamadı"):
            source.open()
        self.assertEqual(cap.release_calls, 1)
        self.assertFalse(source.is_open)

    def test_video_with_no_readable_frames_raises(self) -> None:
        source = FrameSource(SourceConfig.video(self.video), capture_factory=FakeFactory(FakeCapture([])))
        with source:
            with self.assertRaisesRegex(VideoFileError, "hiç kare okunamadı"):
                source.read()


class FrameSourceRealVideoTest(unittest.TestCase):
    """Gerçek ``cv2.VideoCapture`` ile geçici sentetik videolar (kamera açılmaz)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_mp4_mov_avi_are_read_in_order(self) -> None:
        for name, fourcc in (("a.mp4", "mp4v"), ("a.mov", "mp4v"), ("a.avi", "MJPG")):
            with self.subTest(file=name):
                path = write_synthetic_video(self.dir / name, n_frames=12, size=(96, 64), fourcc=fourcc)
                before = sha256(path)
                with FrameSource(SourceConfig.video(path)) as source:
                    packets = list(source.frames())
                self.assertEqual(len(packets), 12)
                self.assertEqual([p.frame_index for p in packets], list(range(12)))
                self.assertTrue(all((p.width, p.height) == (96, 64) for p in packets))
                means = [float(p.frame.mean()) for p in packets]
                self.assertEqual(means, sorted(means), "kareler içerik sırasıyla (artan) okunmalı")
                self.assertGreater(means[-1] - means[0], 50)
                timestamps = [p.timestamp for p in packets]
                self.assertEqual(timestamps, sorted(timestamps))
                self.assertEqual(sha256(path), before, "kaynak video dosyası değiştirilmemeli")

    def test_corrupt_file_raises_video_file_error(self) -> None:
        bad = self.dir / "bozuk.mp4"
        bad.write_bytes(os.urandom(4096))
        with self.assertRaises(VideoFileError):
            with FrameSource(SourceConfig.video(bad)) as source:
                source.read()

    def test_real_missing_file_and_directory(self) -> None:
        with self.assertRaises(VideoFileError):
            FrameSource(SourceConfig.video(self.dir / "yok.mp4")).open()
        with self.assertRaises(VideoFileError):
            FrameSource(SourceConfig.video(self.dir)).open()


if __name__ == "__main__":
    unittest.main()
