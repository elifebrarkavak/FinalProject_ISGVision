"""Aşama 2B testleri: veri kullanım politikası, karantina kararları, kaynak grupları, kaynak/karar kayıtlarının bütünlüğü.

Yalnızca sentetik veri ve yapılandırma/rapor dosyaları kullanılır; ham veri setleri DEĞİŞTİRİLMEZ (veri varsa salt-okunur kontrol).
"""

from __future__ import annotations

import copy
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from isg_vision.dataset_audit.box_filters import evaluate_no_hardhat, evaluate_no_vest, hivis_fraction  # noqa: E402
from isg_vision.dataset_audit.common import load_registry  # noqa: E402
from isg_vision.dataset_audit.duplicates_verify import OrbCache, tile_level_verify, tile_match  # noqa: E402
from isg_vision.dataset_audit.labels import Box  # noqa: E402
from isg_vision.dataset_audit.mapping import load_mapping  # noqa: E402
from isg_vision.dataset_audit.semantic import ImageBoxes, RoleBox  # noqa: E402
from isg_vision.dataset_audit.source_groups import assign_unit_splits, build_groups  # noqa: E402
from isg_vision.dataset_audit.stage2b_reports import EXTERNAL_PENDING_EXCLUDED, ITEM_OVERRIDES, STATUSES  # noqa: E402
from isg_vision.dataset_audit.usage_policy import decide_conditional, iter_image_records, load_policy, validate_policy  # noqa: E402
from test_dataset_audit_followup import structured_image  # noqa: E402

AUDIT = ROOT / "reports" / "dataset_audit"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def classes_from_registry() -> dict[str, dict[int, str]]:
    reg = load_registry()
    return {n: {i: c for i, c in enumerate(r["expected_class_names"])} for n, r in reg.items()}


def box(cid: int, line: int, xc: float, yc: float, w: float, h: float) -> Box:
    return Box(cid, xc, yc, w, h, line)


def rec(boxes: list[Box], height: int = 640, incomplete: int = 0, readable: bool = True) -> dict:
    return {"image_path": Path("x.jpg"), "split": "train", "height": height, "readable": readable, "boxes": boxes,
            "n_lines": len(boxes), "incomplete_persons": incomplete, "has_label": True}


class PolicyFileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_policy()
        cls.registry = load_registry()
        cls.targets, cls.entries = load_mapping()
        cls.classes = classes_from_registry()

    def test_real_policy_is_valid(self) -> None:
        self.assertEqual(validate_policy(self.policy, self.registry, self.entries, self.classes), [])

    def test_only_license_verified_source_is_approved_scope(self) -> None:
        scoped = [n for n, s in self.policy["sources"].items() if s["source_usage"] == "approved_scope"]
        self.assertEqual(scoped, ["construction_ppe"])
        for n, s in self.policy["sources"].items():
            if s["license_status"] != "verified_official":
                self.assertEqual(s["source_usage"], "pending_license_verification", n)

    def test_unverified_license_cannot_be_approved(self) -> None:
        for status in ("partially_verified", "verified_local_only", "not_verified"):
            bad = copy.deepcopy(self.policy)
            bad["sources"]["site_safety"]["source_usage"] = "approved_scope"
            bad["sources"]["site_safety"]["license_status"] = status
            problems = validate_policy(bad, self.registry, self.entries, self.classes)
            self.assertTrue(any("onaylı kapsamda olamaz" in p for p in problems), status)

    def test_rights_unverified_cannot_be_approved(self) -> None:
        bad = copy.deepcopy(self.policy)
        bad["sources"]["mendeley_ppe"]["source_usage"] = "approved_scope"
        bad["sources"]["mendeley_ppe"]["license_status"] = "verified_official"
        self.assertTrue(any("hakları doğrulanmamış" in p for p in validate_policy(bad, self.registry, self.entries, self.classes)))

    def test_excluded_class_cannot_be_mapped(self) -> None:
        bad = copy.deepcopy(self.policy)
        bad["sources"]["mendeley_ppe"]["conditional_class_map"][0] = 0  # E05: Helmet dışlanmış olmalı
        problems = validate_policy(bad, self.registry, self.entries, self.classes)
        self.assertTrue(any("hem eşlenen hem dışlanan" in p for p in problems))

    def test_only_confirmed_mappings_can_be_used(self) -> None:
        bad = copy.deepcopy(self.policy)
        bad["sources"]["site_safety"]["conditional_class_map"][4] = 3  # NO-Safety Vest needs_review
        del bad["sources"]["site_safety"]["excluded_classes"][4]
        problems = validate_policy(bad, self.registry, self.entries, self.classes)
        self.assertTrue(any("yalnızca confirmed" in p for p in problems))

    def test_wrong_target_and_missing_class_detected(self) -> None:
        bad = copy.deepcopy(self.policy)
        bad["sources"]["construction_ppe"]["class_map"][0] = 1  # helmet -> no_helmet (eşleme dosyasıyla çelişir)
        self.assertTrue(any("uyuşmuyor" in p for p in validate_policy(bad, self.registry, self.entries, self.classes)))
        bad = copy.deepcopy(self.policy)
        bad["sources"]["construction_ppe"]["ignored_classes"].remove(10)
        self.assertTrue(any("karşılığı olmayan sınıflar" in p for p in validate_policy(bad, self.registry, self.entries, self.classes)))

    def test_leakage_policy_requires_unit_integrity(self) -> None:
        bad = copy.deepcopy(self.policy)
        bad["leakage"]["never_split_a_unit"] = False
        self.assertTrue(any("never_split_a_unit" in p for p in validate_policy(bad, self.registry, self.entries, self.classes)))
        bad = copy.deepcopy(self.policy)
        bad["leakage"]["ratios"] = {"train": 0.5, "val": 0.2, "test": 0.2}
        self.assertTrue(any("1'e toplanmıyor" in p for p in validate_policy(bad, self.registry, self.entries, self.classes)))
        self.assertEqual(self.policy["leakage"]["random_image_level_split"], "forbidden")

    def test_rules_reference_only_defined_rules(self) -> None:
        for n, s in self.policy["sources"].items():
            for r in s.get("conditional_rules", []):
                self.assertIn(r, self.policy["rules"], f"{n}: {r}")


class ConditionalDecisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_policy()
        cls.rf = cls.policy["sources"]["site_safety"]
        cls.md = cls.policy["sources"]["mendeley_ppe"]

    def rf_decide(self, boxes, path="a.jpg", incomplete=0, review=None, negatives=frozenset(), overrides=None, height=640):
        return decide_conditional("site_safety", rec(boxes, height, incomplete), self.rf, self.policy, review or {}, set(negatives), overrides or {}, path)

    def md_decide(self, boxes):
        return decide_conditional("mendeley_ppe", rec(boxes), self.md, self.policy, {}, set(), {}, "m.jpg")

    def test_clean_roboflow_image_is_approved_only_conditionally(self) -> None:
        status, reasons = self.rf_decide([box(0, 1, 0.5, 0.3, 0.1, 0.1), box(7, 2, 0.5, 0.6, 0.2, 0.3)])
        self.assertEqual((status, reasons), ("approved", []))

    def test_excluded_class_image_needs_reannotation_not_box_deletion(self) -> None:
        status, reasons = self.rf_decide([box(0, 1, 0.5, 0.3, 0.1, 0.1), box(2, 2, 0.2, 0.2, 0.1, 0.1)])
        self.assertEqual(status, "needs_reannotation")
        self.assertIn("RF-EXCLUDED-CLASS-IMAGE", reasons)
        status, _ = self.rf_decide([box(4, 1, 0.5, 0.6, 0.2, 0.3), box(0, 2, 0.5, 0.3, 0.1, 0.1)])
        self.assertEqual(status, "needs_reannotation")

    def test_tiny_target_box_quarantines_image(self) -> None:
        status, reasons = self.rf_decide([box(0, 1, 0.5, 0.5, 0.1, 0.01)])  # 6.4 px
        self.assertEqual(status, "quarantine")
        self.assertIn("RF-TINY-TARGET", reasons)

    def test_missing_annotation_risk_quarantines(self) -> None:
        status, reasons = self.rf_decide([box(0, 1, 0.5, 0.3, 0.1, 0.1)], incomplete=1)
        self.assertEqual(status, "quarantine")
        self.assertIn("RF-PERSON-INCOMPLETE", reasons)

    def test_unverified_negative_vs_verified_negative(self) -> None:
        self.assertEqual(self.rf_decide([box(8, 1, 0.5, 0.5, 0.3, 0.3)])[0], "quarantine")  # yalnızca machinery: doğrulanmamış negatif
        self.assertEqual(self.rf_decide([], path="neg.jpg", negatives={"neg.jpg"})[0], "approved")
        self.assertEqual(self.rf_decide([], path="other.jpg", negatives={"neg.jpg"})[0], "quarantine")

    def test_unresolved_conflict_is_quarantined_not_silently_accepted(self) -> None:
        review = {"a.jpg": [{"item_id": "SI999", "source": "semantic_conflict:vest:overlap", "category": "Needs Manual Review"}]}
        status, reasons = self.rf_decide([box(7, 1, 0.5, 0.5, 0.2, 0.3)], review=review)
        self.assertEqual(status, "quarantine")
        self.assertTrue(any(r.startswith("RF-CONFLICT-UNRESOLVED") for r in reasons))

    def test_explicit_conflict_overrides(self) -> None:
        review = {"a.jpg": [{"item_id": "SI057", "source": "semantic_conflict:vest:overlap", "category": "Needs Manual Review"},
                            {"item_id": "SI081", "source": "semantic_conflict:vest:overlap", "category": "Needs Manual Review"}]}
        ov = {"SI057": {"decision": "quarantine"}, "SI081": {"decision": "needs_reannotation"}}
        self.assertEqual(self.rf_decide([box(7, 1, 0.5, 0.5, 0.2, 0.3)], review={"a.jpg": review["a.jpg"][:1]}, overrides=ov)[0], "quarantine")
        self.assertEqual(self.rf_decide([box(7, 1, 0.5, 0.5, 0.2, 0.3)], review={"a.jpg": review["a.jpg"][1:]}, overrides=ov)[0], "needs_reannotation")

    def test_likely_error_quarantines_but_duplicate_rows_do_not(self) -> None:
        likely = {"a.jpg": [{"item_id": "TB01", "source": "tiny_box", "category": "Likely Error"}]}
        self.assertEqual(self.rf_decide([box(7, 1, 0.5, 0.5, 0.2, 0.3)], review=likely)[0], "quarantine")
        dup = {"a.jpg": [{"item_id": "DB01", "source": "duplicate_box", "category": "Confirmed Error"}]}
        self.assertEqual(self.rf_decide([box(7, 1, 0.5, 0.5, 0.2, 0.3)], review=dup)[0], "approved")  # türetmede tekilleştirilir

    def test_unreadable_image_never_approved(self) -> None:
        status, reasons = decide_conditional("site_safety", rec([box(0, 1, .5, .3, .1, .1)], readable=False), self.rf, self.policy, {}, set(), {}, "a.jpg")
        self.assertEqual(status, "quarantine")

    def test_mendeley_helmet_image_needs_reannotation(self) -> None:
        status, reasons = self.md_decide([box(0, 1, 0.5, 0.3, 0.1, 0.15), box(3, 2, 0.5, 0.6, 0.2, 0.3)])
        self.assertEqual((status, reasons), ("needs_reannotation", ["MD-HELMET-IMAGE"]))

    def test_mendeley_crowd_quarantined_and_simple_image_approved_if_rights(self) -> None:
        crowd = [box(1, i, 0.1 * i, 0.3, 0.05, 0.08) for i in range(1, 4)]
        self.assertEqual(self.md_decide(crowd)[0], "quarantine")
        self.assertEqual(self.md_decide([box(1, 1, 0.5, 0.3, 0.1, 0.15), box(3, 2, 0.5, 0.6, 0.2, 0.3)])[0], "approved")
        # Helmet + kalabalık: quarantine öncelikli
        self.assertEqual(self.md_decide(crowd + [box(0, 9, 0.5, 0.1, 0.1, 0.1)])[0], "quarantine")


class MissingAnnotationRiskTest(unittest.TestCase):
    def test_person_without_head_or_torso_label_counted(self) -> None:
        from isg_vision.dataset_audit.layout import discover_dataset
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "ds"
            (ds / "train" / "images").mkdir(parents=True)
            (ds / "train" / "labels").mkdir(parents=True)
            (ds / "data.yaml").write_text("names: [helmet, no_helmet, vest, no_vest, Person]\n", encoding="utf-8")
            cv2.imwrite(str(ds / "train" / "images" / "a.jpg"), np.full((400, 200, 3), 127, np.uint8))
            cv2.imwrite(str(ds / "train" / "images" / "b.jpg"), np.full((400, 200, 3), 127, np.uint8))
            # a: kişi + baş + gövde etiketli (eksiksiz); b: kişi var ama baş/gövde etiketi yok
            (ds / "train" / "labels" / "a.txt").write_text("4 0.5 0.5 0.4 0.8\n0 0.5 0.2 0.1 0.1\n2 0.5 0.55 0.3 0.3\n", encoding="utf-8")
            (ds / "train" / "labels" / "b.txt").write_text("4 0.5 0.5 0.4 0.8\n", encoding="utf-8")
            entry = {"roles": {"helmet": [0], "no_helmet": [1], "vest": [2], "no_vest": [3], "person": [4]}}
            got = {r["image_path"].name: r["incomplete_persons"] for r in iter_image_records(discover_dataset("ds", ds), entry)}
        self.assertEqual(got, {"a.jpg": 0, "b.jpg": 1})


class BoxFilterTest(unittest.TestCase):
    def image(self, *boxes: RoleBox) -> ImageBoxes:
        return ImageBoxes("d", "train", Path("x.jpg"), Path("x.txt"), 640, 640, list(boxes))

    def rb(self, role: str, line: int, xc: float, yc: float, w: float, h: float) -> RoleBox:
        return RoleBox(role, Box(0, xc, yc, w, h, line))

    def test_no_hardhat_filter(self) -> None:
        person = self.rb("person", 1, 0.5, 0.5, 0.5, 0.9)
        whole_head = self.rb("no_helmet", 2, 0.5, 0.2, 0.12, 0.15)   # 96 px, h/w=1.25
        strip = self.rb("no_helmet", 3, 0.5, 0.2, 0.2, 0.06)         # h/w=0.3 şerit
        tiny = self.rb("no_helmet", 4, 0.5, 0.2, 0.02, 0.03)         # <32 px
        edge = self.rb("no_helmet", 5, 0.05, 0.2, 0.12, 0.15)        # kenara değiyor (x1<0)
        im = self.image(person, whole_head, strip, tiny, edge)
        self.assertTrue(evaluate_no_hardhat(im, whole_head).passed)
        self.assertFalse(evaluate_no_hardhat(im, strip).passed)
        self.assertFalse(evaluate_no_hardhat(im, tiny).passed)
        self.assertFalse(evaluate_no_hardhat(im, edge).passed)

    def test_no_hardhat_overlapping_hardhat_fails(self) -> None:
        person = self.rb("person", 1, 0.5, 0.5, 0.5, 0.9)
        head = self.rb("no_helmet", 2, 0.5, 0.2, 0.12, 0.15)
        hat = self.rb("helmet", 3, 0.5, 0.2, 0.1, 0.1)
        res = evaluate_no_hardhat(self.image(person, head, hat), head)
        self.assertFalse(res.passed)
        self.assertTrue(any("Hardhat" in r for r in res.reasons))

    def test_no_vest_filter_and_hivis_color(self) -> None:
        person = self.rb("person", 1, 0.5, 0.5, 0.4, 0.9)
        torso = self.rb("no_vest", 2, 0.5, 0.55, 0.3, 0.35)
        img_dark = np.full((640, 640, 3), 40, np.uint8)
        img_lime = np.full((640, 640, 3), (0, 255, 180), np.uint8)   # BGR: parlak sarı-yeşil
        im = self.image(person, torso)
        ok, hv = evaluate_no_vest(im, torso, img_dark)
        self.assertTrue(ok.passed and not hv)
        bad, hv2 = evaluate_no_vest(im, torso, img_lime)
        self.assertFalse(bad.passed)
        self.assertTrue(hv2)
        self.assertGreater(hivis_fraction(img_lime, torso), 0.9)
        small = self.rb("no_vest", 3, 0.5, 0.55, 0.05, 0.05)
        self.assertFalse(evaluate_no_vest(self.image(person, small), small, img_dark)[0].passed)

    def test_no_vest_overlapping_vest_fails(self) -> None:
        person = self.rb("person", 1, 0.5, 0.5, 0.4, 0.9)
        torso = self.rb("no_vest", 2, 0.5, 0.55, 0.3, 0.35)
        vest = self.rb("vest", 3, 0.5, 0.55, 0.3, 0.35)
        res, _ = evaluate_no_vest(self.image(person, torso, vest), torso, None)
        self.assertFalse(res.passed)


class SourceGroupTest(unittest.TestCase):
    def test_hard_and_soft_edges(self) -> None:
        imgs = ["a", "b", "c", "d", "e", "f"]
        g = build_groups(imgs, hard_edges=[("a", "b"), ("b", "c")], soft_edges=[("c", "d")])
        self.assertEqual(g["a"]["hard_group"], g["c"]["hard_group"])
        self.assertNotEqual(g["a"]["hard_group"], g["d"]["hard_group"])         # yumuşak kenar sert grubu birleştirmez
        self.assertEqual(g["a"]["split_unit"], g["d"]["split_unit"])            # ama bölme birimini birleştirir
        self.assertEqual(g["a"]["unit_size"], 4)
        self.assertEqual(g["e"]["unit_size"], 1)                                # bağlantısız = singleton

    def test_unit_is_never_split(self) -> None:
        imgs = [f"i{k}" for k in range(60)]
        edges = [(imgs[k], imgs[k + 1]) for k in range(0, 40)]                  # 41'lik zincir + 19 tekil
        g = build_groups(imgs, edges, [])
        units: dict[str, int] = {}
        for v in g.values():
            units[v["split_unit"]] = v["unit_size"]
        split = assign_unit_splits(units, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)
        per_image = {p: split[v["split_unit"]] for p, v in g.items()}
        self.assertEqual(len({per_image[f"i{k}"] for k in range(41)}), 1)       # zincir tek bölümde
        self.assertEqual(set(per_image.values()) <= {"train", "val", "test"}, True)

    def test_assignment_deterministic_and_balanced(self) -> None:
        units = {f"u{k}": 1 + (k % 3) for k in range(200)}
        a = assign_unit_splits(units, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
        b = assign_unit_splits(units, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
        self.assertEqual(a, b)
        total = sum(units.values())
        got = {s: sum(units[u] for u, sp in a.items() if sp == s) / total for s in ("train", "val", "test")}
        self.assertAlmostEqual(got["train"], 0.70, delta=0.03)
        self.assertAlmostEqual(got["val"], 0.15, delta=0.03)

    def test_tile_level_verify_finds_collage_tile_not_unrelated(self) -> None:
        tiles = [structured_image(s, (320, 240)) for s in (101, 102, 103, 104)]
        top = np.hstack(tiles[:2])
        bottom = np.hstack(tiles[2:])
        collage = np.vstack([top, bottom])                                       # 640x480 kolaj
        single_from_tile = cv2.resize(tiles[3], (400, 300))                     # kolajın sağ alt karesinin ölçeklenmiş hali
        unrelated = structured_image(777, (400, 300))
        hit = tile_level_verify(collage, single_from_tile, OrbCache(), key="hit")
        miss = tile_level_verify(collage, unrelated, OrbCache(), key="miss")
        self.assertTrue(tile_match(hit), hit)
        self.assertFalse(tile_match(miss), miss)


class RecordIntegrityTest(unittest.TestCase):
    """Üretilmiş rapor/karar dosyalarının bütünlüğü (dosyalar varsa)."""

    @classmethod
    def setUpClass(cls) -> None:
        needed = ["image_usage_decisions.csv", "source_group_manifest.csv", "quarantine_candidates.csv", "label_review.csv"]
        if not all((AUDIT / n).is_file() for n in needed):
            raise unittest.SkipTest("Aşama 2B çıktıları üretilmemiş")
        cls.usage = read_rows(AUDIT / "image_usage_decisions.csv")
        cls.manifest = {r["image_path"]: r for r in read_rows(AUDIT / "source_group_manifest.csv")}
        cls.quarantine = read_rows(AUDIT / "quarantine_candidates.csv")
        cls.policy = load_policy()

    def test_every_image_has_exactly_one_decision_and_group(self) -> None:
        paths = [r["image_path"] for r in self.usage]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(set(paths), set(self.manifest))
        self.assertEqual(len(paths), 6085)

    def test_no_unverified_source_image_is_approved_now(self) -> None:
        for r in self.usage:
            scope = self.policy["sources"][r["dataset"]]["source_usage"]
            if scope != "approved_scope":
                self.assertEqual(r["status_now"], "pending_license_verification", r["image_path"])
            else:  # onaylı kaynak: approved ya da politikada açıkça kayıtlı görüntü kararı (2C image_overrides)
                src = self.policy["sources"][r["dataset"]]
                ov = (src.get("image_overrides") or {}).get(r["image_path"])
                rv = ROOT / src.get("user_visual_review", {}).get("file", "-")
                if ov is None and rv.is_file():  # 2D: kullanıcı görsel inceleme kararı
                    from isg_vision.dataset_audit.visual_review import read_review_csv, user_overrides
                    ov = user_overrides(read_review_csv(rv)).get(r["image_path"])
                self.assertEqual(r["status_now"], ov["status"] if ov else "approved", r["image_path"])

    def test_suspected_source_groups_not_split_in_preview(self) -> None:
        by_unit: dict[tuple[str, str], set[str]] = {}
        for r in self.usage:
            if r["proposed_split_preview"]:
                by_unit.setdefault((r["preview_scope"], r["split_unit"]), set()).add(r["proposed_split_preview"])
        self.assertTrue(by_unit)
        for key, splits in by_unit.items():
            self.assertEqual(len(splits), 1, f"bölme birimi bölündü: {key} -> {splits}")

    def test_excluded_classes_never_approved_conditionally(self) -> None:
        # E05/E10/E12: dışlanan sınıfı içeren görüntü hak doğrulansa bile 'approved' olamaz
        for r in self.usage:
            if int(r["n_excluded_class_boxes"]) > 0:
                self.assertNotEqual(r["status_if_rights_verified"], "approved", r["image_path"])

    def test_quarantine_list_matches_decisions(self) -> None:
        flagged = {r["image_path"] for r in self.usage if r["status_if_rights_verified"] in ("quarantine", "needs_reannotation")}
        self.assertEqual(flagged, {r["image_path"] for r in self.quarantine})
        for r in self.quarantine:
            self.assertTrue(r["reason_codes"], r["image_path"])
            self.assertIn(r["status"], ("quarantine", "needs_reannotation"))

    def test_decision_ids_exist_in_review_table(self) -> None:
        decisions = yaml.safe_load((ROOT / "configs/review/stage2b_decisions.yaml").read_text(encoding="utf-8"))
        ids = {r["item_id"] for r in read_rows(AUDIT / "label_review.csv")}
        for item in decisions["conflict_decisions"]:
            self.assertIn(item, ids)
        for d in decisions["source_relation_decisions"]:
            self.assertIn(d["decision"], ("Confirmed Same Source", "Likely Same Source", "Different Source", "Inconclusive"))

    def test_class_decisions_consistent_with_mapping(self) -> None:
        _, entries = load_mapping()
        status = {(e["dataset"], e["name"]): e["status"] for e in entries}
        decisions = yaml.safe_load((ROOT / "configs/review/stage2b_decisions.yaml").read_text(encoding="utf-8"))["class_decisions"]
        self.assertEqual(status[("mendeley_ppe", "Helmet")], "incompatible")
        self.assertEqual(status[("site_safety", "NO-Hardhat")], "incompatible")
        self.assertEqual(status[("site_safety", "NO-Safety Vest")], "needs_review")
        for v in decisions.values():
            self.assertEqual(v["decision"], "excluded_from_merged_training_candidates")


class ChecklistStatusTest(unittest.TestCase):
    def test_override_statuses_valid(self) -> None:
        for item, (status, note) in ITEM_OVERRIDES.items():
            self.assertIn(status, STATUSES, item)
            self.assertTrue(note)

    def test_license_items_stay_pending_and_excluded_classes_are_excluded(self) -> None:
        # 2C: lisans doğrulanamadı -> doğrulanmış SAYILMAZ; kaynak dışlandığı için 'external_verification_pending_source_excluded'
        self.assertEqual(ITEM_OVERRIDES["F01"][0], EXTERNAL_PENDING_EXCLUDED)
        self.assertEqual(ITEM_OVERRIDES["F02"][0], EXTERNAL_PENDING_EXCLUDED)
        for k in ("E05", "E10", "E12"):
            self.assertEqual(ITEM_OVERRIDES[k][0], "resolved_excluded")
        self.assertNotEqual(ITEM_OVERRIDES["F01"][0], "resolved_verified")

    def test_checklist_file_has_59_items_and_no_unresolved(self) -> None:
        path = AUDIT / "manual_review_checklist.md"
        if not path.is_file():
            self.skipTest("kontrol listesi üretilmemiş")
        text = path.read_text(encoding="utf-8")
        self.assertIn("| unresolved | 0 |", text)
        self.assertIn("| **Toplam** | 59 |", text)

    def test_attribution_file_does_not_attribute_unverified_license(self) -> None:
        text = (ROOT / "docs" / "dataset_attributions.md").read_text(encoding="utf-8")
        self.assertIn("Ultralytics Construction-PPE Dataset, Ultralytics team, November 2026, CC BY 4.0.", text)
        self.assertIn("pending_license_verification", text)
        self.assertIn("BAĞIMSIZ DOĞRULANAMADI", text)
        self.assertIn("rights_unverified", text)
        self.assertIn("YAZILMAMIŞTIR", text)


if __name__ == "__main__":
    unittest.main()
