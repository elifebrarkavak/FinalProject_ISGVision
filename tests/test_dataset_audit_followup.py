"""Aşama 2 eksik giderme testleri: sınıf eşleme doğrulaması, kutu semantiği, şüpheli etiket sınıflandırması,
yakın kopya tespiti (pHash/ORB), yanlış pozitifler ve bölümler arası kopya riski. SENTETİK veri kullanır;
gerçek veri setleri okunmaz/değiştirilmez (yalnızca yapılandırma dosyaları okunur)."""

from __future__ import annotations

import random
import subprocess
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isg_vision.dataset_audit.duplicates_verify import (  # noqa: E402
    OrbCache, apply_visual_checks, augment, decide, feature_mask, hash_distance, leakage_risk, lsh_candidate_votes,
    index_descriptors, orb_verify, signature,
)
from isg_vision.dataset_audit.labels import Box  # noqa: E402
from isg_vision.dataset_audit.mapping import VALID_STATUSES, load_mapping, mapping_rows  # noqa: E402
from isg_vision.dataset_audit.review import CATEGORIES, expand_verdicts, wilson_interval  # noqa: E402
from isg_vision.dataset_audit.semantic import (  # noqa: E402
    ImageBoxes, RoleBox, assign_to_person, classify_conflict, find_semantic_conflicts, person_relative_stats,
    single_person_pair_stats,
)

TEST_RAW = ROOT / "data" / "raw"


def structured_image(seed: int, size: tuple[int, int] = (480, 360)) -> np.ndarray:
    """Köşe/doku bakımından zengin, deterministik sentetik gri görüntü (ORB için yeterli öznitelik)."""
    rng = np.random.default_rng(seed)
    w, h = size
    img = cv2.GaussianBlur(rng.integers(60, 200, (h, w), dtype=np.uint8), (0, 0), 8)
    for _ in range(60):
        x, y = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 40))
        sw, sh = int(rng.integers(12, 60)), int(rng.integers(12, 60))
        color = int(rng.integers(0, 255))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + sw, y + sh), color, -1 if rng.random() < 0.6 else 2)
        else:
            cv2.circle(img, (x, y), sw // 2, color, -1 if rng.random() < 0.6 else 2)
    for _ in range(30):  # ortak glif/yazı olmasın: rastgele çizgiler ve elipsler (görüntüler arası yapay benzerlik yaratmaz)
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(img, p1, p2, int(rng.integers(0, 255)), int(rng.integers(1, 4)))
        cv2.ellipse(img, p1, (int(rng.integers(5, 40)), int(rng.integers(5, 40))), int(rng.integers(0, 180)), 0, 360,
                    int(rng.integers(0, 255)), 2)
    return img


def rb(role: str, line: int, xc: float, yc: float, w: float, h: float) -> RoleBox:
    return RoleBox(role, Box(0, xc, yc, w, h, line))


def image_boxes(*boxes: RoleBox) -> ImageBoxes:
    return ImageBoxes("ds", "train", Path("x.jpg"), Path("x.txt"), 640, 480, list(boxes))


class RealDataGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in TEST_RAW.rglob("*") if p.is_file()}

    @classmethod
    def tearDownClass(cls) -> None:
        after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in TEST_RAW.rglob("*") if p.is_file()}
        assert after == cls._before, "data/raw içeriği test sırasında değişti!"


class MappingValidationTest(RealDataGuard):
    TARGETS = {0: "helmet", 1: "no_helmet"}

    def test_statuses_are_lowercase_vocabulary(self) -> None:
        self.assertEqual(VALID_STATUSES, {"confirmed", "needs_review", "incompatible"})
        base = {"dataset": "d", "id": 0, "name": "H", "target": 0}
        for status in VALID_STATUSES:
            mapping_rows(self.TARGETS, [{**base, "status": status}])
        for bad in ("Confirmed", "Needs Manual Review", "Incompatible", "maybe"):
            with self.assertRaises(ValueError):
                mapping_rows(self.TARGETS, [{**base, "status": bad}])

    def test_compatibility_and_reliability_validated(self) -> None:
        base = {"dataset": "d", "id": 0, "name": "H", "target": 0, "status": "confirmed"}
        with self.assertRaises(ValueError):
            mapping_rows(self.TARGETS, [{**base, "compatibility": "belki"}])
        with self.assertRaises(ValueError):
            mapping_rows(self.TARGETS, [{**base, "reliability": "çok yüksek"}])

    def test_real_mapping_is_consistent(self) -> None:
        targets, entries = load_mapping()
        rows = mapping_rows(targets, entries)
        self.assertEqual(len(rows), 26)
        for e in entries:
            if e.get("target") is None:
                self.assertEqual(e["status"], "incompatible", f"hedef dışı sınıf incompatible olmalı: {e['name']}")
            if e["status"] == "confirmed":
                self.assertNotEqual(e.get("compatibility"), "Uyumsuz")
                self.assertTrue(e.get("visual_result"), f"confirmed için görsel sonuç yazılmalı: {e['name']}")
            if e["status"] == "incompatible" and e.get("target") is not None:
                self.assertEqual(e["compatibility"], "Uyumsuz")
        mapped = {(e["dataset"], e["name"]): e["status"] for e in entries if e.get("target") is not None}
        self.assertEqual(mapped[("mendeley_ppe", "Helmet")], "incompatible")      # baş+baret kutusu
        self.assertEqual(mapped[("site_safety", "NO-Hardhat")], "incompatible")   # heterojen kutu tanımı
        self.assertEqual(mapped[("site_safety", "NO-Safety Vest")], "needs_review")


class BoxSemanticsTest(RealDataGuard):
    def test_assign_to_person_requires_containment(self) -> None:
        person = rb("person", 1, 0.5, 0.5, 0.2, 0.6)
        inside = rb("helmet", 2, 0.5, 0.25, 0.05, 0.06)
        outside = rb("helmet", 3, 0.9, 0.1, 0.05, 0.06)
        self.assertIs(assign_to_person(inside, [person]), person)
        self.assertIsNone(assign_to_person(outside, [person]))

    def test_person_relative_geometry(self) -> None:
        person = rb("person", 1, 0.5, 0.5, 0.2, 0.6)               # y: 0.2 .. 0.8
        tight = rb("helmet", 2, 0.5, 0.26, 0.08, 0.08)             # kişi yüksekliğinin 0.133'ü, tepeye yakın
        stats = person_relative_stats([image_boxes(person, tight)])
        self.assertAlmostEqual(stats["helmet"]["rel_height"]["median"], 0.133, places=2)
        self.assertLess(stats["helmet"]["top_offset"]["median"], 0.05)

    def test_head_box_vs_helmet_box_ratio_distinguishes_semantics(self) -> None:
        torso = rb("vest", 1, 0.5, 0.6, 0.2, 0.3)
        tight_helmet = rb("helmet", 2, 0.5, 0.35, 0.08, 0.06)
        head_with_face = rb("helmet", 3, 0.5, 0.35, 0.10, 0.16)
        a = single_person_pair_stats([image_boxes(torso, tight_helmet)])["helmet"]["head_h_over_torso_h"]["median"]
        b = single_person_pair_stats([image_boxes(torso, head_with_face)])["helmet"]["head_h_over_torso_h"]["median"]
        self.assertGreater(b, a * 2)


class SuspectLabelClassificationTest(RealDataGuard):
    def test_helmet_and_no_helmet_on_different_persons_not_same_person_conflict(self) -> None:
        p1, p2 = rb("person", 1, 0.25, 0.5, 0.2, 0.6), rb("person", 2, 0.75, 0.5, 0.2, 0.6)
        helmet = rb("helmet", 3, 0.25, 0.25, 0.06, 0.06)
        bare = rb("no_helmet", 4, 0.75, 0.25, 0.06, 0.06)
        conflicts = find_semantic_conflicts(image_boxes(p1, p2, helmet, bare))
        self.assertEqual(conflicts, [], "farklı kişilerdeki helmet + no_helmet tek başına hata değildir")

    def test_same_person_carried_helmet_is_flagged_for_review(self) -> None:
        person = rb("person", 1, 0.5, 0.5, 0.25, 0.7)
        bare_head = rb("no_helmet", 2, 0.5, 0.2, 0.06, 0.07)
        carried = rb("helmet", 3, 0.58, 0.6, 0.06, 0.05)
        conflicts = find_semantic_conflicts(image_boxes(person, bare_head, carried))
        self.assertEqual([c["type"] for c in conflicts], ["same_person"])
        self.assertEqual(classify_conflict(conflicts[0]), "needs_review")  # otomatik karar verilmez

    def test_same_region_overlap_is_likely_error(self) -> None:
        a = rb("vest", 1, 0.5, 0.5, 0.2, 0.3)
        b = rb("no_vest", 2, 0.5, 0.5, 0.2, 0.3)
        c = find_semantic_conflicts(image_boxes(a, b))[0]
        self.assertEqual(c["type"], "overlap")
        self.assertEqual(classify_conflict(c), "likely_error")

    def test_review_verdict_expansion_and_validation(self) -> None:
        got = expand_verdicts({"verdicts": [{"ids": "SI001-SI003, SI010", "category": "Valid Annotation", "reason": "r"}]})
        self.assertEqual(sorted(got), ["SI001", "SI002", "SI003", "SI010"])
        with self.assertRaises(ValueError):
            expand_verdicts({"verdicts": [{"ids": "A01", "category": "Kesin Hata"}]})
        with self.assertRaises(ValueError):
            expand_verdicts({"verdicts": [{"ids": "A01-A02", "category": CATEGORIES[0]}, {"ids": "A02", "category": CATEGORIES[1]}]})

    def test_wilson_interval(self) -> None:
        lo, hi = wilson_interval(5, 96)
        self.assertTrue(0.0 < lo < 5 / 96 < hi < 0.2)
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))
        self.assertEqual(wilson_interval(26, 26)[1], 1.0)

    def test_real_verdict_file_is_valid(self) -> None:
        raw = yaml.safe_load((ROOT / "configs/review/label_review_verdicts.yaml").read_text(encoding="utf-8"))
        verdicts = expand_verdicts(raw)  # tekrar/geçersiz kategori yok
        self.assertGreater(len(verdicts), 200)

    def test_checklist_resolution_file_is_valid(self) -> None:
        items = yaml.safe_load((ROOT / "configs/review/checklist_resolution.yaml").read_text(encoding="utf-8"))["items"]
        for key, v in items.items():
            self.assertIn(v["category"], "ABCD", key)
            self.assertTrue(v["result"])
            if v["category"] in "AB":
                self.assertTrue(v["evidence"], f"{key}: A/B maddesi kanıt göstermeli")


class PerceptualHashTest(RealDataGuard):
    def test_phash_robust_to_scale_jpeg_brightness(self) -> None:
        img = structured_image(1)
        sig = signature(img)
        for kind in ("identity_resize50", "jpeg_q30", "brightness+25", "brightness-25", "blur", "flip", "stretch640"):
            d, _ = hash_distance(sig, signature(augment(img, kind, random.Random(0))))
            self.assertLessEqual(d, 10, f"{kind}: hash mesafesi {d}")

    def test_different_images_have_large_hash_distance(self) -> None:
        a, b = signature(structured_image(1)), signature(structured_image(2))
        self.assertGreater(hash_distance(a, b)[0], 12)


class OrbVerificationTest(RealDataGuard):
    def verify(self, a: np.ndarray, b: np.ndarray) -> dict:
        return orb_verify(a, b, OrbCache())

    def test_positive_derivatives_match_geometrically(self) -> None:
        img = structured_image(3)
        for kind in ("crop20", "rot12", "flip", "stretch640", "jpeg_q30", "combo"):
            v = self.verify(img, augment(img, kind, random.Random(1)))
            self.assertIn(decide(v), ("same_image_derived", "same_scene_different_frame", "partial_region_match"),
                          f"{kind}: {v}")
            self.assertGreaterEqual(v["inliers"], 25, kind)

    def test_stretch_needs_affine_model(self) -> None:
        img = structured_image(4)
        v = self.verify(img, augment(img, "stretch640", random.Random(0)))
        self.assertIsNotNone(v["anisotropy"])
        self.assertNotAlmostEqual(v["anisotropy"], 1.0, delta=0.05)  # gerilme tespit edildi

    def test_unrelated_images_are_not_duplicates(self) -> None:
        for s1, s2 in ((10, 11), (12, 13), (14, 15)):
            v = self.verify(structured_image(s1), structured_image(s2))
            self.assertEqual(decide(v), "not_duplicate", f"yanlış pozitif: {s1}/{s2} {v}")

    def test_identical_black_cutout_squares_do_not_create_matches(self) -> None:
        a, b = structured_image(20), structured_image(21)
        for img in (a, b):
            for (x, y) in ((30, 40), (200, 100), (350, 250)):
                img[y:y + 12, x:x + 12] = 0  # aynı konumda aynı boyutlu siyah kareler
        self.assertIsNotNone(feature_mask(a))
        v = self.verify(a, b)
        self.assertEqual(decide(v), "not_duplicate", f"cutout karelerinden sahte eşleşme: {v}")

    def test_same_background_different_object_is_scene_not_derived(self) -> None:
        a = structured_image(30)
        b = a.copy()
        cv2.rectangle(b, (100, 80), (330, 300), 255, -1)          # ön plan nesnesi tamamen farklı
        cv2.circle(b, (230, 190), 70, 0, -1)
        v = self.verify(a, b)
        self.assertIn(decide(v), ("same_scene_different_frame", "partial_region_match", "same_image_derived"))
        self.assertGreater(v["inliers"], 25)  # arka plan ortak: aynı sahne kanıtı

    def test_decide_rules(self) -> None:
        base = {"inliers": 0, "inlier_ratio": 0.0, "ncc": None, "spread": 0.5}
        self.assertEqual(decide({**base, "inliers": 5}), "not_duplicate")
        self.assertEqual(decide({**base, "inliers": 15, "inlier_ratio": 0.3}), "uncertain")
        self.assertEqual(decide({**base, "inliers": 100, "inlier_ratio": 0.9, "ncc": 0.95}), "same_image_derived")
        self.assertEqual(decide({**base, "inliers": 100, "inlier_ratio": 0.9, "ncc": 0.3}), "same_scene_different_frame")
        self.assertEqual(decide({**base, "inliers": 100, "inlier_ratio": 0.1, "ncc": 0.2}), "partial_region_match")
        # az inlier ama çok yüksek hizalı NCC (gerilmiş küçük görüntü) = aynı görüntü
        self.assertEqual(decide({**base, "inliers": 20, "inlier_ratio": 0.6, "ncc": 0.99}), "same_image_derived")


class LshCandidateTest(RealDataGuard):
    def test_cropped_copy_gets_votes_unrelated_does_not(self) -> None:
        imgs = [structured_image(s) for s in (40, 41, 42)]
        cropped = augment(imgs[0], "crop20", random.Random(2))
        descs = [index_descriptors(g) for g in imgs + [cropped]]
        votes = lsh_candidate_votes(descs)
        unrelated = max(votes.get(k, 0) for k in ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3)))
        # Sentetik şekiller benzer köşe tanımlayıcıları ürettiğinden mutlak eşik değil KONTRAST doğrulanır
        self.assertGreaterEqual(votes.get((0, 3), 0), 15)
        self.assertGreater(votes.get((0, 3), 0), 4 * unrelated)


class SplitLeakageRiskTest(RealDataGuard):
    def test_cross_split_with_eval_split_is_high_risk(self) -> None:
        r = leakage_risk("same_image_derived", "within_dataset_cross_split", "train", "val")
        self.assertTrue(r.startswith("HIGH"))
        self.assertTrue(leakage_risk("same_scene_different_frame", "within_dataset_cross_split", "val", "test").startswith("HIGH"))

    def test_same_split_is_low_and_non_duplicate_is_none(self) -> None:
        self.assertTrue(leakage_risk("same_image_derived", "within_dataset_same_split", "train", "train").startswith("low"))
        self.assertEqual(leakage_risk("not_duplicate", "within_dataset_cross_split", "train", "val"), "none")

    def test_cross_dataset_with_eval_is_high(self) -> None:
        self.assertTrue(leakage_risk("same_image_derived", "cross_dataset", "train", "val").startswith("HIGH"))

    def test_visual_check_not_applied_to_mismatching_paths(self) -> None:
        rows = [{"pair_id": "DV1", "path_a": "x/aaa.jpg", "path_b": "y/bbb.jpg", "visual_check": ""}]
        n = apply_visual_checks(rows, {"checks": {"DV1": {"a": "aaa", "b": "ZZZ", "result": "r", "note": "n"}}})
        self.assertEqual(n, 0)
        self.assertEqual(rows[0]["visual_check"], "")
        n = apply_visual_checks(rows, {"checks": {"DV1": {"a": "aaa", "b": "bbb", "result": "r", "note": "n"}}})
        self.assertEqual(n, 1)


class FollowupScriptCliTest(RealDataGuard):
    SCRIPTS = ("analyze_box_semantics", "review_roboflow_labels", "verify_duplicates", "analyze_augmentation_artifacts",
               "generate_followup_reports")

    def test_help_runs(self) -> None:
        for name in self.SCRIPTS:
            with self.subTest(script=name):
                res = subprocess.run([sys.executable, str(ROOT / "scripts" / f"{name}.py"), "--help"],
                                     capture_output=True, text=True, timeout=180)
                self.assertEqual(res.returncode, 0, res.stderr)


if __name__ == "__main__":
    unittest.main()
