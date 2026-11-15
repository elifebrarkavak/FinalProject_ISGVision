"""Yüz görselleştirme testleri: çizim girdiyi değiştirmez, her boyutta çalışır, uyarı metni ve etiketler çizilir (pencere açılmaz)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isg_vision.face_detection import visualization as viz  # noqa: E402
from isg_vision.face_detection.results import FaceDetectionResult, build_faces  # noqa: E402
from isg_vision.face_detection.visualization import NOTICE, WINDOW_NAME, FaceOverlayInfo, draw_face_detections  # noqa: E402

INFO = FaceOverlayInfo("CAMERA 0", 29.7, 11.2, (640, 480), "cpu")


def result_for(size: tuple[int, int], boxes: list[tuple[float, float, float, float]], scores: list[float] | None = None) -> FaceDetectionResult:
    faces, dropped, _ = build_faces(boxes, scores or [0.9 - 0.01 * i for i in range(len(boxes))], image_size=size)
    return FaceDetectionResult(faces, size, 0, 0.0, 5.0, "cpu", dropped)


def textured(width: int = 640, height: int = 480) -> np.ndarray:
    return np.random.default_rng(3).integers(0, 255, (height, width, 3), dtype=np.uint8)


class CapturedText:
    """``cv2.putText`` çağrılarındaki metinleri toplar (gerçek çizim de yapılır)."""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self._real = cv2.putText

    def __call__(self, img, text, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.texts.append(text)
        return self._real(img, text, *args, **kwargs)


def draw_capturing(image, result, info=INFO):  # type: ignore[no-untyped-def]
    spy = CapturedText()
    with mock.patch.object(viz.cv2, "putText", spy):
        out = draw_face_detections(image, result, info)
    return out, spy.texts


class DrawBasicsTest(unittest.TestCase):
    def test_returns_a_copy_and_never_modifies_the_input(self) -> None:
        image = textured()
        before = image.copy()
        out = draw_face_detections(image, result_for((640, 480), [(100, 100, 200, 220)]), INFO)
        np.testing.assert_array_equal(image, before)
        self.assertIsNot(out, image)
        self.assertEqual((out.shape, out.dtype), (image.shape, image.dtype))
        self.assertFalse(np.array_equal(out, image), "çizim bir şey değiştirmeli")

    def test_readonly_input_is_accepted(self) -> None:
        image = textured()
        image.flags.writeable = False
        out = draw_face_detections(image, result_for((640, 480), [(10, 10, 60, 70)]), INFO)
        self.assertTrue(out.flags.writeable)

    def test_empty_result_still_draws_the_status_bands(self) -> None:
        image = np.full((480, 640, 3), 128, np.uint8)
        out, texts = draw_capturing(image, FaceDetectionResult((), (640, 480)))
        self.assertIn("FACES: 0", texts)
        self.assertFalse(np.array_equal(out, image))

    def test_overlay_texts_show_count_fps_inference_device_resolution(self) -> None:
        _, texts = draw_capturing(textured(), result_for((640, 480), [(100, 100, 200, 220), (300, 100, 400, 220)]))
        for expected in ("FACES: 2", "SOURCE: CAMERA 0", "RESOLUTION: 640x480", "FPS: 29.7", "INFERENCE: 11 ms", "DEVICE: CPU"):
            self.assertIn(expected, texts)

    def test_the_not_identity_notice_and_exit_hint_are_always_drawn(self) -> None:
        for result in (FaceDetectionResult((), (640, 480)), result_for((640, 480), [(100, 100, 200, 220)])):
            _, texts = draw_capturing(textured(), result)
            self.assertIn(NOTICE, texts)
            self.assertIn(viz.EXIT_HINT, texts)
        self.assertIn("not identity", NOTICE)
        self.assertIn("not liveness", NOTICE)

    def test_each_face_gets_a_confidence_label_and_edge_marker(self) -> None:
        result = result_for((640, 480), [(100, 100, 200, 220), (0, 100, 80, 200)], [0.93, 0.81])
        _, texts = draw_capturing(textured(), result)
        self.assertIn("FACE 0.93", texts)
        self.assertIn("FACE 0.81 EDGE", texts)

    def test_no_label_says_person_track_or_identity(self) -> None:
        _, texts = draw_capturing(textured(), result_for((640, 480), [(100, 100, 200, 220)]))
        for text in texts:
            for word in ("ID ", "TRACK", "PERSON", "NAME", "EMPLOYEE"):
                self.assertNotIn(word, text.upper().replace("NOT IDENTITY", ""), text)

    def test_missing_optional_info_does_not_fail(self) -> None:
        out, texts = draw_capturing(textured(), FaceDetectionResult((), (640, 480)), FaceOverlayInfo("VIDEO a.mp4", 0.0, None, None, None, "requested 1280x720, got 640x480"))
        self.assertIn("INFERENCE: -- ms", texts)
        self.assertIn("RESOLUTION: N/A", texts)
        self.assertIn("NOTE: requested 1280x720, got 640x480", texts)
        self.assertNotIn("DEVICE", " ".join(texts))
        self.assertEqual(out.shape, (480, 640, 3))

    def test_non_ascii_source_text_is_converted_for_opencv_fonts(self) -> None:
        _, texts = draw_capturing(textured(), FaceDetectionResult((), (640, 480)), FaceOverlayInfo("VIDEO çalışan_şef.mp4", 1.0, 1.0, (640, 480)))
        source = next(t for t in texts if t.startswith("SOURCE"))
        self.assertTrue(source.isascii(), source)

    def test_window_title_is_face_specific(self) -> None:
        self.assertEqual(WINDOW_NAME, "ISG Vision - Face Detection")

    def test_face_boxes_use_a_different_colour_than_person_boxes(self) -> None:
        from isg_vision.human_detection import visualization as person_viz

        self.assertNotEqual(viz._CYAN, person_viz._GREEN)


class RobustnessTest(unittest.TestCase):
    def test_many_faces(self) -> None:
        boxes = [(20 + 55 * (i % 10), 60 + 70 * (i // 10), 60 + 55 * (i % 10), 110 + 70 * (i // 10)) for i in range(50)]
        out = draw_face_detections(textured(640, 480), result_for((640, 480), boxes), INFO)
        self.assertEqual(out.shape, (480, 640, 3))

    def test_faces_on_every_border_and_full_frame(self) -> None:
        boxes = [(0, 200, 50, 300), (590, 200, 640, 300), (300, 0, 360, 50), (300, 430, 360, 480), (0, 0, 640, 480)]
        out = draw_face_detections(textured(640, 480), result_for((640, 480), boxes), INFO)
        self.assertEqual(out.shape, (480, 640, 3))

    def test_face_at_the_top_edge_keeps_its_label_inside_the_frame(self) -> None:
        image = np.zeros((480, 640, 3), np.uint8)
        out = draw_face_detections(image, result_for((640, 480), [(300, 0, 360, 60)]), INFO)
        self.assertEqual(out.shape, image.shape)

    def test_various_image_sizes_including_tiny_and_large(self) -> None:
        for w, h in ((16, 16), (32, 24), (64, 48), (160, 120), (320, 240), (1280, 720), (1920, 1080), (640, 40), (40, 640)):
            with self.subTest(size=(w, h)):
                box = (max(0, w // 4), max(0, h // 4), max(2, w // 2), max(2, h // 2))
                result = result_for((w, h), [box]) if box[2] > box[0] and box[3] > box[1] else FaceDetectionResult((), (w, h))
                out = draw_face_detections(np.zeros((h, w, 3), np.uint8), result, FaceOverlayInfo("CAMERA 0", 1.0, 1.0, (w, h), "cpu"))
                self.assertEqual(out.shape, (h, w, 3))

    def test_drawing_is_deterministic(self) -> None:
        image = textured()
        result = result_for((640, 480), [(100, 100, 200, 220)])
        np.testing.assert_array_equal(draw_face_detections(image, result, INFO), draw_face_detections(image, result, INFO))

    def test_importing_the_module_opens_no_window(self) -> None:
        with mock.patch.object(cv2, "imshow", side_effect=AssertionError("pencere açılmamalı")), mock.patch.object(cv2, "namedWindow", side_effect=AssertionError("pencere açılmamalı")):
            draw_face_detections(textured(), result_for((640, 480), [(100, 100, 200, 220)]), INFO)


if __name__ == "__main__":
    unittest.main()
