"""İnsan kutusu/durum şeridi çizimi testleri (gerçek pencere açılmaz; yalnızca piksel ve metin denetimi)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from isg_vision.human_detection.results import DetectionResult, PersonDetection  # noqa: E402
from isg_vision.human_detection.visualization import (  # noqa: E402
    EXIT_HINT,
    PersonOverlayInfo,
    draw_person_detections,
)

GREEN = (80, 255, 80)
W, H = 640, 480


def person(box, conf=0.9) -> PersonDetection:  # type: ignore[no-untyped-def]
    return PersonDetection(0, "person", conf, tuple(box))


def result_of(*detections: PersonDetection, size=(W, H)) -> DetectionResult:  # type: ignore[no-untyped-def]
    return DetectionResult(tuple(detections), size, inference_ms=12.0, device="cpu")


def info(**overrides):  # type: ignore[no-untyped-def]
    values = dict(source_text="CAMERA 0", fps=22.5, inference_ms=35.4, resolution=(W, H), device="mps")
    values.update(overrides)
    return PersonOverlayInfo(**values)


def gray(width=W, height=H, value=128) -> np.ndarray:  # type: ignore[no-untyped-def]
    return np.full((height, width, 3), value, dtype=np.uint8)


def drawn_texts(image, res, inf):  # type: ignore[no-untyped-def]
    """``draw_person_detections`` sırasında ``cv2.putText`` ile yazılan tüm metinleri döndürür."""
    texts: list[str] = []
    real = cv2.putText

    def spy(img, text, *args, **kwargs):  # type: ignore[no-untyped-def]
        texts.append(text)
        return real(img, text, *args, **kwargs)

    with mock.patch.object(cv2, "putText", side_effect=spy):
        out = draw_person_detections(image, res, inf)
    return out, texts


def green_pixels(image: np.ndarray, rows: slice, cols: slice) -> int:
    region = image[rows, cols]
    return int(np.all(region == GREEN, axis=2).sum())


class InputImmutabilityTest(unittest.TestCase):
    def test_input_image_is_not_modified_and_output_is_a_new_array(self) -> None:
        image = np.random.default_rng(1).integers(0, 255, (H, W, 3), dtype=np.uint8)
        before = image.copy()
        out = draw_person_detections(image, result_of(person((100, 150, 300, 400)), person((350, 160, 500, 420))), info())
        self.assertTrue(np.array_equal(image, before), "girdi görüntüsü değiştirilmemeli")
        self.assertIsNot(out, image)
        self.assertFalse(np.shares_memory(out, image))
        self.assertEqual((out.shape, out.dtype), (image.shape, image.dtype))
        self.assertFalse(np.array_equal(out, image), "çıktıda çizim olmalı")

    def test_input_unchanged_even_with_no_detections(self) -> None:
        image = gray()
        draw_person_detections(image, result_of(), info())
        self.assertTrue(np.all(image == 128))

    def test_read_only_input_is_supported(self) -> None:
        image = gray()
        image.flags.writeable = False
        draw_person_detections(image, result_of(person((100, 150, 300, 400))), info())


class BoxDrawingTest(unittest.TestCase):
    def test_box_outline_is_drawn_at_the_given_coordinates(self) -> None:
        out = draw_person_detections(gray(), result_of(person((100, 200, 300, 400))), info())
        # sol kenar çizgisi: x≈100, y 250..350 aralığında yeşil piksel; kutunun içi (merkez) ve dışı değişmemiş olmalı
        self.assertGreater(green_pixels(out, slice(250, 350), slice(97, 104)), 100)
        self.assertGreater(green_pixels(out, slice(397, 404), slice(150, 250)), 100)  # alt kenar
        self.assertEqual(tuple(out[300, 200]), (128, 128, 128), "kutunun içi boyanmamalı")
        self.assertEqual(tuple(out[300, 50]), (128, 128, 128), "kutunun dışı boyanmamalı")

    def test_multiple_people_are_drawn_independently(self) -> None:
        boxes = [(30, 150, 130, 420), (250, 160, 350, 430), (480, 170, 600, 440)]
        out = draw_person_detections(gray(), result_of(*(person(b) for b in boxes)), info())
        for x1, y1, x2, y2 in boxes:
            self.assertGreater(green_pixels(out, slice(y1 + 40, y2 - 40), slice(x1 - 3, x1 + 4)), 100, f"kutu {x1},{y1} çizilmemiş")
        between = green_pixels(out, slice(250, 350), slice(160, 240))  # kutular arası boşluk
        self.assertEqual(between, 0)

    def test_no_detections_draws_no_boxes_below_the_status_band(self) -> None:
        out = draw_person_detections(gray(), result_of(), info())
        self.assertEqual(green_pixels(out, slice(150, 420), slice(0, W)), 0)

    def test_confidence_label_is_drawn_for_each_person(self) -> None:
        _, texts = drawn_texts(gray(), result_of(person((100, 200, 300, 400), 0.94), person((350, 200, 500, 400), 0.871)), info())
        self.assertIn("PERSON 0.94", texts)
        self.assertIn("PERSON 0.87", texts)

    def test_label_background_is_filled_above_the_box(self) -> None:
        out = draw_person_detections(gray(), result_of(person((200, 250, 400, 450), 0.9)), info())
        self.assertGreater(green_pixels(out, slice(228, 250), slice(200, 215)), 50, "etiket arka planı kutunun üstünde olmalı")

    def test_box_at_top_edge_puts_label_inside_box_without_error(self) -> None:
        out = draw_person_detections(gray(), result_of(person((100, 0, 300, 200), 0.8)), info())
        self.assertEqual(out.shape, (H, W, 3))

    def test_edge_and_oversized_boxes_do_not_raise(self) -> None:
        cases = [(0, 0, W, H), (W - 5, H - 5, W, H), (0, 0, 2, 2), (600, 300, 3000, 3000)]  # son kutu görüntü dışına taşıyor
        for box in cases:
            with self.subTest(box=box):
                draw_person_detections(gray(), result_of(person(box)), info())

    def test_tiny_images_do_not_raise(self) -> None:
        for size in ((64, 48), (32, 32), (16, 16), (1, 1)):
            with self.subTest(size=size):
                draw_person_detections(gray(*size), result_of(person((1, 1, 10, 10)), size=size), info(resolution=size))

    def test_many_people(self) -> None:
        people = [person((10 + 30 * i, 100, 35 + 30 * i, 300), 0.5 + i / 100) for i in range(20)]
        _, texts = drawn_texts(gray(W, H), result_of(*people), info())
        self.assertIn("PERSONS: 20", texts)
        self.assertEqual(sum(t.startswith("PERSON ") for t in texts), 20)


class StatusTextTest(unittest.TestCase):
    def test_person_count_fps_inference_source_and_resolution_are_shown(self) -> None:
        _, texts = drawn_texts(gray(), result_of(person((10, 200, 100, 400)), person((200, 200, 300, 400)), person((400, 200, 500, 400))), info())
        self.assertIn("PERSONS: 3", texts)
        self.assertIn("FPS: 22.5", texts)
        self.assertIn("INFERENCE: 35 ms", texts)
        self.assertIn("SOURCE: CAMERA 0", texts)
        self.assertIn("RESOLUTION: 640x480", texts)
        self.assertIn("DEVICE: MPS", texts)
        self.assertIn(EXIT_HINT, texts)
        self.assertEqual(EXIT_HINT, "Q / Esc / Close to exit")

    def test_zero_persons_shows_persons_zero(self) -> None:
        _, texts = drawn_texts(gray(), result_of(), info())
        self.assertIn("PERSONS: 0", texts)

    def test_missing_inference_and_resolution_show_placeholders(self) -> None:
        _, texts = drawn_texts(gray(), result_of(), info(inference_ms=None, resolution=None, device=None))
        self.assertIn("INFERENCE: -- ms", texts)
        self.assertIn("RESOLUTION: N/A", texts)
        self.assertFalse([t for t in texts if t.startswith("DEVICE")])

    def test_note_is_shown_when_given(self) -> None:
        _, texts = drawn_texts(gray(), result_of(), info(note="requested 1920x1080, got 1280x720"))
        self.assertIn("NOTE: requested 1920x1080, got 1280x720", texts)

    def test_texts_are_ascii_only_even_for_turkish_input(self) -> None:
        _, texts = drawn_texts(gray(), result_of(), info(source_text="VIDEO çalışan_şantiye_ığü.mp4", note="çözünürlük uygulanamadı"))
        for text in texts:
            self.assertTrue(text.isascii(), f"ASCII olmayan metin: {text!r}")
        self.assertIn("SOURCE: VIDEO calisan_santiye_igu.mp4", texts)

    def test_status_band_is_darkened_but_center_untouched(self) -> None:
        out = draw_person_detections(gray(), result_of(), info())
        self.assertLess(int(out[2, W // 2, 0]), 100, "üst şerit koyulaştırılmalı")
        self.assertEqual(tuple(out[H // 2, W // 2]), (128, 128, 128))


if __name__ == "__main__":
    unittest.main()
