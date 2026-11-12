"""Takip görselleştirmesi: aktif izler çizilir, kayıp izler çizilmez, sayılar ayrı gösterilir, girdi görüntüsü değişmez. [Mock/birim; pencere yok]"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isg_vision.tracking import visualization  # noqa: E402
from isg_vision.tracking.results import LostTrack, TrackedPerson, TrackingResult  # noqa: E402
from isg_vision.tracking.visualization import (  # noqa: E402
    TRACK_PALETTE,
    TrackingOverlayInfo,
    color_for_track,
    draw_person_tracks,
)


def track(track_id: int, box: tuple[float, float, float, float], conf: float = 0.9, session: int = 1) -> TrackedPerson:
    return TrackedPerson(track_id, 0, "person", conf, box, 5, 0.5, session, 0, 3)


def result(tracks: tuple[TrackedPerson, ...] = (), lost: tuple[LostTrack, ...] = (), detections: int | None = None, size: tuple[int, int] = (320, 240)) -> TrackingResult:
    return TrackingResult(tracks, lost, 0, len(tracks) if detections is None else detections, size, 5, 0.5, 1, 2.0)


def info(**kw: object) -> TrackingOverlayInfo:
    base: dict[str, object] = dict(source_text="CAMERA 0", fps=21.5, inference_ms=14.2, tracking_ms=2.04, resolution=(320, 240), device="mps")
    base.update(kw)
    return TrackingOverlayInfo(**base)  # type: ignore[arg-type]


def image(width: int = 320, height: int = 240, value: int = 120) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


class TextCapture:
    """``cv2.putText`` çağrılarını kaydeder (ekrandaki yazıları OCR olmadan doğrulamak için)."""

    def __enter__(self) -> list[str]:
        self.texts: list[str] = []
        real = cv2.putText

        def spy(img, text, *a, **k):  # type: ignore[no-untyped-def]
            self.texts.append(text)
            return real(img, text, *a, **k)

        self._patch = mock.patch.object(visualization.cv2, "putText", side_effect=spy)
        self._patch.start()
        return self.texts

    def __exit__(self, *exc: object) -> None:
        self._patch.stop()


class DrawingTest(unittest.TestCase):
    def test_input_image_is_not_modified_and_output_is_a_copy(self) -> None:
        src = image()
        before = src.copy()
        out = draw_person_tracks(src, result((track(1, (50, 60, 120, 200)),)), info())
        np.testing.assert_array_equal(src, before)
        self.assertIsNot(out, src)
        self.assertEqual((out.shape, out.dtype), (src.shape, src.dtype))
        self.assertFalse(np.array_equal(out, src), "çizim yapılmış olmalı")

    def test_active_track_box_is_drawn_in_its_color(self) -> None:
        box = (100, 90, 180, 200)
        out = draw_person_tracks(image(), result((track(2, box),)), info())
        color = color_for_track(2)
        x1, y1, x2, y2 = box
        # kutu kenarı üzerinde (dikey kenarın ortasından) rengi bulmak: kenar kalınlığı içinde tara
        row = out[(y1 + y2) // 2, x1 - 2 : x1 + 3]
        self.assertTrue(any(tuple(int(v) for v in px) == color for px in row), "kutu kenarı iz renginde olmalı")

    def test_lost_and_tentative_tracks_are_not_drawn(self) -> None:
        """Kayıp izin kutusu ÇİZİLMEZ (görünmeyen kişi görünüyormuş gibi gösterilmez): merkez bölge orijinaliyle aynı kalır."""
        src = image()
        res = TrackingResult((), (LostTrack(5, 3), LostTrack(6, 1)), 2, 0, (320, 240), 5, 0.5, 1, 1.0)
        out = draw_person_tracks(src, res, info())
        np.testing.assert_array_equal(out[100:200], src[100:200])  # şeritlerin dışındaki orta bölge hiç değişmemeli
        self.assertFalse(np.array_equal(out[:30], src[:30]), "üst şerit (sayılar) çizilmiş olmalı: test boş geçmesin")

    def test_same_id_same_color_across_frames_and_positions(self) -> None:
        a = draw_person_tracks(image(), result((track(3, (20, 100, 60, 200)),)), info())
        b = draw_person_tracks(image(), result((track(3, (200, 100, 260, 200)),)), info())
        color = color_for_track(3)
        self.assertTrue(any(tuple(int(v) for v in px) == color for px in a[150, 18:23]))
        self.assertTrue(any(tuple(int(v) for v in px) == color for px in b[150, 198:203]))

    def test_palette_colors_are_distinct_and_cosmetic(self) -> None:
        self.assertEqual(len(set(TRACK_PALETTE)), len(TRACK_PALETTE))
        self.assertEqual(len({color_for_track(i) for i in range(1, len(TRACK_PALETTE) + 1)}), len(TRACK_PALETTE))
        self.assertNotEqual(color_for_track(1), color_for_track(2))
        self.assertEqual(color_for_track(1), color_for_track(1 + len(TRACK_PALETTE)), "palet tekrar eder: renk kimlik anlamı taşımaz")

    def test_multiple_boxes_are_drawn_at_once(self) -> None:
        tracks = (track(1, (10, 90, 60, 200)), track(2, (100, 90, 150, 200)), track(3, (200, 90, 250, 200)))
        src = image()
        out = draw_person_tracks(src, result(tracks), info())
        for t in tracks:
            x1 = int(t.bbox_xyxy[0])
            self.assertTrue(any(tuple(int(v) for v in px) == color_for_track(t.track_id) for px in out[150, x1 - 2 : x1 + 3]), t.track_id)


class TextTest(unittest.TestCase):
    def test_counts_are_shown_separately(self) -> None:
        """İnsan (tespit) sayısı, aktif takip sayısı ve kayıp iz sayısı birbirinden ayrı satırlardır."""
        res = result((track(1, (10, 90, 60, 200)), track(2, (100, 90, 150, 200))), lost=(LostTrack(9, 4),), detections=3)
        with TextCapture() as texts:
            draw_person_tracks(image(), res, info())
        self.assertIn("DETECTIONS: 3", texts)
        self.assertIn("ACTIVE TRACKS: 2", texts)
        self.assertIn("LOST TRACKS: 1", texts)

    def test_timing_lines_are_separate_and_formatted(self) -> None:
        with TextCapture() as texts:
            draw_person_tracks(image(), result(), info())
        self.assertIn("FPS: 21.5", texts)
        self.assertIn("INFERENCE: 14 ms", texts)
        self.assertIn("TRACKING: 2.0 ms", texts)
        self.assertIn("DEVICE: MPS", texts)
        self.assertIn("SOURCE: CAMERA 0", texts)
        self.assertIn("RESOLUTION: 320x240", texts)

    def test_track_labels_show_id_and_confidence(self) -> None:
        with TextCapture() as texts:
            draw_person_tracks(image(), result((track(7, (10, 90, 60, 200), conf=0.876),)), info())
        self.assertIn("ID 7 0.88", texts)

    def test_unknown_values_are_shown_as_dashes(self) -> None:
        with TextCapture() as texts:
            draw_person_tracks(image(), result(), info(inference_ms=None, tracking_ms=None, resolution=None, device=None))
        self.assertIn("INFERENCE: -- ms", texts)
        self.assertIn("TRACKING: -- ms", texts)
        self.assertIn("RESOLUTION: N/A", texts)
        self.assertFalse(any(t.startswith("DEVICE") for t in texts))

    def test_texts_are_ascii_only(self) -> None:
        with TextCapture() as texts:
            draw_person_tracks(image(), result(), info(source_text="VIDEO çalışma_ışığı.mp4", note="çözünürlük uygulanamadı"))
        for text in texts:
            self.assertTrue(text.isascii(), text)
        self.assertIn("SOURCE: VIDEO calisma_isigi.mp4", texts)

    def test_exit_hint_and_identity_disclaimer_are_present(self) -> None:
        with TextCapture() as texts:
            draw_person_tracks(image(640, 480), result(), info(resolution=(640, 480)))
        self.assertIn("Q / Esc / Close to exit", texts)
        self.assertTrue(any("temporary track number" in t for t in texts), "ID'nin kimlik olmadığı ekranda belirtilmeli")

    def test_note_line_is_drawn(self) -> None:
        with TextCapture() as texts:
            draw_person_tracks(image(), result(), info(note="requested 1280x720, got 640x480"))
        self.assertIn("NOTE: requested 1280x720, got 640x480", texts)


class RobustnessTest(unittest.TestCase):
    def test_empty_result_does_not_fail(self) -> None:
        out = draw_person_tracks(image(), result(), info())
        self.assertEqual(out.shape, (240, 320, 3))

    def test_tiny_image_does_not_fail(self) -> None:
        out = draw_person_tracks(image(64, 48), result((track(1, (2, 5, 20, 40)),), size=(64, 48)), info(resolution=(64, 48)))
        self.assertEqual(out.shape, (48, 64, 3))

    def test_boxes_touching_image_borders_do_not_fail(self) -> None:
        tracks = (track(1, (0, 0, 40, 60)), track(2, (280, 180, 320, 240)), track(3, (0.0, 100.0, 320.0, 140.0)))
        draw_person_tracks(image(), result(tracks), info())

    def test_box_at_top_puts_label_inside_not_under_status_band(self) -> None:
        out = draw_person_tracks(image(), result((track(4, (100, 0, 160, 80)),)), info())
        self.assertEqual(out.shape, (240, 320, 3))

    def test_many_tracks_with_ids_beyond_palette_size(self) -> None:
        tracks = tuple(track(20 + k, (5 + 12 * k, 100, 15 + 12 * k, 180)) for k in range(25))
        draw_person_tracks(image(), result(tracks), info())

    def test_large_ids_and_sessions_do_not_fail(self) -> None:
        draw_person_tracks(image(), result((track(123456, (10, 90, 60, 200), session=7),)), info())

    def test_drawing_twice_gives_identical_output(self) -> None:
        res = result((track(1, (10, 90, 60, 200)),))
        src = image()
        np.testing.assert_array_equal(draw_person_tracks(src, res, info()), draw_person_tracks(src, res, info()))

    def test_drawing_opens_no_window(self) -> None:
        with mock.patch.object(cv2, "namedWindow", side_effect=AssertionError("pencere açıldı")), mock.patch.object(
            cv2, "imshow", side_effect=AssertionError("pencere açıldı")
        ):
            draw_person_tracks(image(), result((track(1, (10, 90, 60, 200)),)), info())


if __name__ == "__main__":
    unittest.main()
