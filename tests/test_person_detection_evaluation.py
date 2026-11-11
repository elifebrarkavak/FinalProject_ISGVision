"""Değerlendirme altyapısı testleri: Precision/Recall/F1/AP/mAP matematiği, bilinen sonuçları olan SENTETİK örneklerle doğrulanır.

Bu testler, hesaplamanın DOĞRU olduğunu sınar; hiçbir gerçek model doğruluğu ölçmez ve etiketli bir veri kümesi kullanmaz.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isg_vision.human_detection.config import DEFAULT_MODEL_PATH  # noqa: E402
from isg_vision.human_detection.evaluation import (  # noqa: E402
    IOU_THRESHOLDS_COCO,
    EvalSample,
    average_precision,
    evaluate_detections,
    iou_matrix,
    load_yolo_label_file,
    match_detections,
)

SCRIPT = ROOT / "scripts" / "evaluate_person_detection.py"
BOX = (0, 0, 100, 100)


def sample(gt, det, scores):  # type: ignore[no-untyped-def]
    return EvalSample(np.array(gt, dtype=float).reshape(-1, 4), np.array(det, dtype=float).reshape(-1, 4), np.array(scores, dtype=float))


class IouTest(unittest.TestCase):
    def test_identical_disjoint_and_partial(self) -> None:
        self.assertEqual(iou_matrix(np.array([BOX]), np.array([BOX]))[0, 0], 1.0)
        self.assertEqual(iou_matrix(np.array([BOX]), np.array([(200, 200, 300, 300)]))[0, 0], 0.0)
        self.assertAlmostEqual(iou_matrix(np.array([(0, 0, 10, 10)]), np.array([(5, 0, 15, 10)]))[0, 0], 1 / 3)

    def test_touching_edges_have_zero_iou(self) -> None:
        self.assertEqual(iou_matrix(np.array([(0, 0, 10, 10)]), np.array([(10, 0, 20, 10)]))[0, 0], 0.0)

    def test_contained_box(self) -> None:
        self.assertAlmostEqual(iou_matrix(np.array([(0, 0, 100, 100)]), np.array([(0, 0, 50, 100)]))[0, 0], 0.5)

    def test_shapes_and_empty_inputs(self) -> None:
        self.assertEqual(iou_matrix(np.zeros((3, 4)), np.zeros((2, 4))).shape, (3, 2))
        self.assertEqual(iou_matrix(np.zeros((0, 4)), np.array([BOX])).shape, (0, 1))
        self.assertEqual(iou_matrix(np.array([BOX]), np.zeros((0, 4))).shape, (1, 0))

    def test_degenerate_box_gives_zero_not_nan(self) -> None:
        result = iou_matrix(np.array([(5, 5, 5, 5)]), np.array([(5, 5, 5, 5)]))
        self.assertEqual(result[0, 0], 0.0)
        self.assertFalse(np.isnan(result).any())


class MatchingTest(unittest.TestCase):
    def test_higher_score_gets_the_true_positive_and_duplicate_is_false_positive(self) -> None:
        s = sample([BOX], [BOX, (2, 2, 100, 100)], [0.6, 0.9])
        scores, tp = match_detections(s, 0.5)
        self.assertEqual(list(scores), [0.9, 0.6])
        self.assertEqual(list(tp), [True, False])

    def test_each_ground_truth_matches_only_one_detection(self) -> None:
        s = sample([BOX, (200, 0, 300, 100)], [BOX, BOX, (200, 0, 300, 100)], [0.9, 0.8, 0.7])
        _, tp = match_detections(s, 0.5)
        self.assertEqual(list(tp), [True, False, True])

    def test_detection_prefers_the_best_unmatched_ground_truth(self) -> None:
        s = sample([(0, 0, 100, 100), (30, 0, 130, 100)], [(25, 0, 125, 100)], [0.9])
        _, tp = match_detections(s, 0.5)
        self.assertEqual(list(tp), [True])

    def test_threshold_is_inclusive(self) -> None:
        s = sample([(0, 0, 10, 10)], [(0, 0, 5, 10)], [0.9])  # IoU tam 0.5
        self.assertEqual(list(match_detections(s, 0.5)[1]), [True])
        self.assertEqual(list(match_detections(s, 0.51)[1]), [False])

    def test_min_score_filters_detections(self) -> None:
        s = sample([BOX], [BOX, (300, 300, 400, 400)], [0.3, 0.1])
        scores, tp = match_detections(s, 0.5, min_score=0.25)
        self.assertEqual((list(scores), list(tp)), ([0.3], [True]))

    def test_no_ground_truth_means_all_false_positive(self) -> None:
        _, tp = match_detections(sample([], [BOX, (200, 200, 300, 300)], [0.9, 0.8]), 0.5)
        self.assertEqual(list(tp), [False, False])

    def test_no_detections(self) -> None:
        scores, tp = match_detections(sample([BOX], [], []), 0.5)
        self.assertEqual((len(scores), len(tp)), (0, 0))


class AveragePrecisionTest(unittest.TestCase):
    def ap(self, tp: list[bool], scores: list[float] | None, n_gt: int) -> float:
        scores_arr = np.array(scores if scores is not None else [1.0 - 0.01 * i for i in range(len(tp))])
        return average_precision(scores_arr, np.array(tp, dtype=bool), n_gt)

    def test_perfect_detector(self) -> None:
        self.assertAlmostEqual(self.ap([True, True, True], None, 3), 1.0)

    def test_no_detections_is_zero_and_no_ground_truth_is_undefined(self) -> None:
        self.assertEqual(self.ap([], None, 3), 0.0)
        self.assertTrue(math.isnan(self.ap([False], None, 0)))

    def test_all_false_positives_is_zero(self) -> None:
        self.assertEqual(self.ap([False, False], None, 2), 0.0)

    def test_half_recall_with_perfect_precision(self) -> None:
        # recall 0.5'e kadar precision 1.0: 101 noktadan 51'i (r=0.00...0.50) 1.0, kalanı 0
        self.assertAlmostEqual(self.ap([True], None, 2), 51 / 101)

    def test_false_positive_in_the_middle_lowers_ap_via_envelope(self) -> None:
        # TP, FP, TP ve 2 etiket: precision [1, .5, .667]; zarf [1, .667, .667]; recall [.5, .5, 1]
        self.assertAlmostEqual(self.ap([True, False, True], None, 2), (51 * 1.0 + 50 * 2 / 3) / 101)

    def test_ranking_matters_confident_false_positive_hurts_more(self) -> None:
        good = self.ap([True, True, False], [0.9, 0.8, 0.7], 2)
        bad = self.ap([False, True, True], [0.9, 0.8, 0.7], 2)
        self.assertGreater(good, bad)

    def test_unsorted_input_is_sorted_by_score(self) -> None:
        a = average_precision(np.array([0.7, 0.9, 0.8]), np.array([False, True, True]), 2)
        b = average_precision(np.array([0.9, 0.8, 0.7]), np.array([True, True, False]), 2)
        self.assertAlmostEqual(a, b)


class EvaluateTest(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        r = evaluate_detections([sample([BOX, (200, 0, 300, 100)], [BOX, (200, 0, 300, 100)], [0.9, 0.8])])
        self.assertEqual((r.true_positives, r.false_positives, r.false_negatives), (2, 0, 0))
        self.assertEqual((r.precision, r.recall, r.f1), (1.0, 1.0, 1.0))
        self.assertAlmostEqual(r.map50, 1.0)
        self.assertAlmostEqual(r.map50_95, 1.0)
        self.assertEqual((r.num_images, r.num_gt), (1, 2))

    def test_shifted_boxes_pass_loose_but_not_strict_iou_thresholds(self) -> None:
        # IoU = 0.818: 0.50...0.80 eşiklerinde (7 eşik) doğru, 0.85...0.95'te (3 eşik) yanlış
        r = evaluate_detections([sample([BOX], [(10, 0, 110, 100)], [0.9])])
        self.assertAlmostEqual(r.map50, 1.0)
        self.assertAlmostEqual(r.map50_95, 0.7)
        self.assertEqual(sorted(r.ap_per_iou), list(IOU_THRESHOLDS_COCO))
        self.assertAlmostEqual(r.ap_per_iou[0.8], 1.0)
        self.assertAlmostEqual(r.ap_per_iou[0.85], 0.0)

    def test_precision_recall_f1_counts_across_images(self) -> None:
        samples = [
            sample([BOX, (200, 0, 300, 100)], [BOX, (500, 500, 600, 600)], [0.9, 0.8]),  # 1 TP, 1 FP, 1 FN
            sample([BOX], [BOX], [0.7]),  # 1 TP
        ]
        r = evaluate_detections(samples)
        self.assertEqual((r.true_positives, r.false_positives, r.false_negatives), (2, 1, 1))
        self.assertAlmostEqual(r.precision, 2 / 3)
        self.assertAlmostEqual(r.recall, 2 / 3)
        self.assertAlmostEqual(r.f1, 2 / 3)

    def test_confidence_threshold_affects_pr_but_not_ap(self) -> None:
        s = [sample([BOX], [BOX, (500, 500, 600, 600)], [0.9, 0.2])]
        low = evaluate_detections(s, conf_threshold=0.1)
        high = evaluate_detections(s, conf_threshold=0.5)
        self.assertEqual((low.false_positives, high.false_positives), (1, 0))
        self.assertAlmostEqual(low.precision, 0.5)
        self.assertAlmostEqual(high.precision, 1.0)
        self.assertAlmostEqual(low.map50, high.map50)
        self.assertEqual((low.num_detections_at_conf, high.num_detections_at_conf), (2, 1))

    def test_no_detections_with_ground_truth(self) -> None:
        r = evaluate_detections([sample([BOX], [], [])])
        self.assertEqual((r.true_positives, r.false_negatives, r.recall, r.f1, r.map50), (0, 1, 0.0, 0.0, 0.0))
        self.assertTrue(math.isnan(r.precision), "tespit yoksa precision tanımsızdır")

    def test_false_positives_only_no_ground_truth(self) -> None:
        r = evaluate_detections([sample([], [BOX], [0.9])])
        self.assertEqual((r.true_positives, r.false_positives, r.precision), (0, 1, 0.0))
        self.assertTrue(math.isnan(r.recall) and math.isnan(r.map50) and math.isnan(r.f1))

    def test_empty_scene_without_anything_is_all_undefined(self) -> None:
        r = evaluate_detections([sample([], [], [])])
        self.assertTrue(math.isnan(r.precision) and math.isnan(r.recall) and math.isnan(r.map50))

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_detections([])
        with self.assertRaises(ValueError):
            evaluate_detections([sample([BOX], [BOX], [0.9])], conf_threshold=1.5)
        with self.assertRaises(ValueError):
            evaluate_detections([sample([BOX], [BOX], [0.9])], iou_threshold=0.0)
        with self.assertRaises(ValueError):
            EvalSample(np.zeros((0, 4)), np.zeros((2, 4)), np.zeros(1))


class LabelFileTest(unittest.TestCase):
    def label(self, text: str, size=(200, 100), cls: int = 0) -> np.ndarray:  # type: ignore[no-untyped-def]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.txt"
            path.write_text(text, encoding="utf-8")
            return load_yolo_label_file(path, size, cls)

    def test_conversion_to_pixel_xyxy(self) -> None:
        boxes = self.label("0 0.5 0.5 0.5 0.5\n")
        np.testing.assert_allclose(boxes, [[50.0, 25.0, 150.0, 75.0]])

    def test_only_person_class_and_blank_lines(self) -> None:
        boxes = self.label("\n1 0.5 0.5 0.2 0.2\n0 0.25 0.5 0.1 0.2\n\n")
        self.assertEqual(boxes.shape, (1, 4))
        np.testing.assert_allclose(boxes, [[40.0, 40.0, 60.0, 60.0]])

    def test_custom_person_class_id(self) -> None:
        self.assertEqual(self.label("3 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n", cls=3).shape, (1, 4))

    def test_missing_file_means_no_people(self) -> None:
        self.assertEqual(load_yolo_label_file(Path("/olmayan/etiket.txt"), (100, 100)).shape, (0, 4))

    def test_malformed_or_out_of_range_lines_raise(self) -> None:
        for bad in ("0 0.5 0.5 0.5\n", "0 a b c d\n", "0 1.5 0.5 0.5 0.5\n", "0 0.5 0.5 -0.1 0.5\n", "0 0.5 0.5 0.5 0.5 0.9\n"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.label(bad)


class ScriptTest(unittest.TestCase):
    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=300)

    def test_missing_folders_are_usage_errors(self) -> None:
        proc = self.run_script("--images", "/olmayan", "--labels", "/olmayan")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("klasör", proc.stderr)

    def test_help_states_it_is_infrastructure_without_a_dataset(self) -> None:
        proc = self.run_script("--help")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ALTYAPIDIR", proc.stdout)

    @unittest.skipUnless(DEFAULT_MODEL_PATH.is_file(), "ağırlık yok")
    def test_wiring_with_real_model_on_a_trivially_self_labelled_image(self) -> None:
        """Yalnızca BORU HATTI testi: etiketler modelin kendi kutularından türetilir (döngüsel); doğruluk ölçümü DEĞİLDİR."""
        import importlib.util

        from isg_vision.human_detection import DetectionConfig, PersonDetector

        spec = importlib.util.find_spec("ultralytics")
        assert spec is not None and spec.origin is not None
        bus_path = Path(spec.origin).parent / "assets" / "bus.jpg"
        image = cv2.imread(str(bus_path))
        height, width = image.shape[:2]
        reference = PersonDetector(DetectionConfig(device="cpu", conf=0.5)).detect(image)
        with tempfile.TemporaryDirectory() as tmp:
            images, labels, out = Path(tmp, "img"), Path(tmp, "lab"), Path(tmp, "out")
            images.mkdir(), labels.mkdir()
            cv2.imwrite(str(images / "bus.jpg"), image)
            lines = [f"0 {(d.x1 + d.x2) / 2 / width:.6f} {(d.y1 + d.y2) / 2 / height:.6f} {d.width / width:.6f} {d.height / height:.6f}"
                     for d in reference.detections]
            (labels / "bus.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            proc = self.run_script("--images", str(images), "--labels", str(labels), "--output-dir", str(out), "--conf", "0.5")
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            data = json.loads(next(out.glob("evaluation_*.json")).read_text(encoding="utf-8"))
        self.assertEqual((data["num_images"], data["num_gt"]), (1, len(reference.detections)))
        self.assertEqual((data["precision"], data["recall"]), (1.0, 1.0))
        self.assertGreater(data["map50"], 0.99)
        self.assertIn("Güven skoru doğruluk değildir", proc.stdout)


if __name__ == "__main__":
    unittest.main()
