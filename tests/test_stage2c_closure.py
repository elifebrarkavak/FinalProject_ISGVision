"""Aşama 2C testleri: nihai bölme (birim bütünlüğü), eksik etiket risk işaretleri, politika bekleme gerekçeleri,
F01/F02 durumu, envanter/ZIP bütünlük denetimleri ve üretilmiş 2C çıktılarının tutarlılığı.

Sentetik veriler geçici klasörlerde üretilir; ham veri setleri yalnızca OKUNUR (değiştirilmez).
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isg_vision.dataset_audit.common import load_registry  # noqa: E402
from isg_vision.dataset_audit.labels import Box  # noqa: E402
from isg_vision.dataset_audit.mapping import load_mapping  # noqa: E402
from isg_vision.dataset_audit.semantic import RoleBox  # noqa: E402
from isg_vision.dataset_audit.stage2b_reports import (  # noqa: E402
    EXTERNAL_PENDING_EXCLUDED, ITEM_OVERRIDES, OPEN_STATUSES, STATUSES, item_statuses,
)
from isg_vision.dataset_audit.stage2c import verify_against_zip, verify_publisher_manifest  # noqa: E402
from isg_vision.dataset_audit.visual_review import read_review_csv, user_overrides  # noqa: E402
from isg_vision.dataset_audit.usage_policy import (  # noqa: E402
    final_split_assignment, load_policy, person_attribute_issues, unit_integrity_violations, validate_policy,
)

AUDIT = ROOT / "reports" / "dataset_audit"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def rb(role: str, xc: float, yc: float, w: float, h: float) -> RoleBox:
    return RoleBox(role, Box(0, xc, yc, w, h, 1))


class FinalSplitTest(unittest.TestCase):
    def test_cross_split_unit_moves_to_majority(self) -> None:
        imgs = [("a", "train", "U1"), ("b", "train", "U1"), ("c", "val", "U1"), ("d", "val", "U2"), ("e", "train", "U3")]
        final, fixes = final_split_assignment(imgs)
        self.assertEqual(final, {"a": "train", "b": "train", "c": "train", "d": "val", "e": "train"})
        self.assertEqual(len(fixes), 1)
        self.assertEqual(fixes[0]["moved"], ["c"])

    def test_tie_goes_to_train_and_majority_val_respected(self) -> None:
        final, _ = final_split_assignment([("a", "train", "U"), ("b", "val", "U")])
        self.assertEqual(set(final.values()), {"train"})
        final, _ = final_split_assignment([("a", "train", "U"), ("b", "val", "U"), ("c", "val", "U")])
        self.assertEqual(set(final.values()), {"val"})

    def test_excluded_images_get_no_split_and_do_not_vote(self) -> None:
        final, fixes = final_split_assignment([("a", "train", "U"), ("b", "val", "U"), ("c", "val", "U")], excluded={"b"})
        self.assertEqual(final["b"], "")
        self.assertEqual(final["a"], "train")   # train 1 - val 1 eşitlik -> train
        self.assertEqual(final["c"], "train")
        self.assertTrue(fixes)

    def test_uncontested_original_split_is_preserved(self) -> None:
        imgs = [(f"i{k}", "val" if k % 5 == 0 else "train", f"U{k}") for k in range(50)]
        final, fixes = final_split_assignment(imgs)
        self.assertEqual(fixes, [])
        self.assertEqual(final, {p: s for p, s, _ in imgs})

    def test_unit_integrity_checker(self) -> None:
        ok = [{"final_split": "train", "split_unit": "U"}, {"final_split": "train", "split_unit": "U"}, {"final_split": "", "split_unit": "U"}]
        self.assertEqual(unit_integrity_violations(ok), {})
        bad = ok + [{"final_split": "val", "split_unit": "U"}]
        self.assertEqual(unit_integrity_violations(bad), {"U": {"train", "val"}})


class PersonAttributeIssueTest(unittest.TestCase):
    def test_complete_person_not_flagged(self) -> None:
        boxes = [rb("person", 0.5, 0.5, 0.3, 0.8), rb("helmet", 0.5, 0.15, 0.08, 0.08), rb("vest", 0.5, 0.4, 0.2, 0.25)]
        self.assertEqual(person_attribute_issues(boxes, 1000), [])

    def test_small_person_ignored(self) -> None:
        self.assertEqual(person_attribute_issues([rb("person", 0.5, 0.5, 0.02, 0.05)], 1000), [])  # 50 px < 60

    def test_reason_codes(self) -> None:
        top_cut = [rb("person", 0.5, 0.35, 0.3, 0.7), rb("vest", 0.5, 0.3, 0.2, 0.25)]           # y1 = 0.0
        self.assertEqual(person_attribute_issues(top_cut, 1000), [["head_out_of_frame_top"]])
        side_cut = [rb("person", 0.05, 0.5, 0.1, 0.6)]                                           # x1 = 0.0
        self.assertEqual(person_attribute_issues(side_cut, 1000), [["person_truncated_at_edge"]])
        unexplained = [rb("person", 0.5, 0.5, 0.3, 0.6), rb("vest", 0.5, 0.45, 0.2, 0.2)]
        self.assertEqual(person_attribute_issues(unexplained, 1000), [["head_unexplained"]])
        # baş kutusu kişi kutusunun biraz üstünde: etiket var, atanamadı
        above = [rb("person", 0.5, 0.55, 0.3, 0.6), rb("no_helmet", 0.5, 0.24, 0.08, 0.06), rb("vest", 0.5, 0.45, 0.2, 0.2)]
        res = person_attribute_issues(above, 1000)
        self.assertTrue(res == [] or res == [["head_label_present_unassigned"]], res)


class PolicyHoldReasonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_policy()
        cls.registry = load_registry()
        cls.targets, cls.entries = load_mapping()
        cls.classes = {n: {i: c for i, c in enumerate(r["expected_class_names"])} for n, r in cls.registry.items()}

    def problems(self, pol) -> list[str]:
        return validate_policy(pol, self.registry, self.entries, self.classes)

    def test_required_hold_reasons_present(self) -> None:
        md = set(self.policy["sources"]["mendeley_ppe"]["hold_reasons"])
        rf = set(self.policy["sources"]["site_safety"]["hold_reasons"])
        self.assertTrue({"rights_unverified", "annotation_quality"} <= md, md)
        self.assertTrue({"license_unverified", "label_noise", "split_leakage"} <= rf, rf)
        for d in ("mendeley_ppe", "site_safety"):
            self.assertEqual(self.policy["sources"][d]["source_usage"], "pending_license_verification")
            self.assertEqual(self.policy["sources"][d]["raw_data_retention"], "keep")

    def test_missing_or_invalid_hold_reason_rejected(self) -> None:
        bad = copy.deepcopy(self.policy)
        del bad["sources"]["mendeley_ppe"]["hold_reasons"]
        self.assertTrue(any("hold_reasons zorunlu" in p for p in self.problems(bad)))
        bad = copy.deepcopy(self.policy)
        bad["sources"]["site_safety"]["hold_reasons"] = ["looks_fine"]
        self.assertTrue(any("geçersiz hold_reason" in p for p in self.problems(bad)))

    def test_pending_source_must_keep_raw_data_and_not_have_class_map(self) -> None:
        bad = copy.deepcopy(self.policy)
        bad["sources"]["site_safety"]["raw_data_retention"] = "delete"
        self.assertTrue(any("korunmalı" in p for p in self.problems(bad)))
        bad = copy.deepcopy(self.policy)
        bad["sources"]["mendeley_ppe"]["class_map"] = {1: 1}
        self.assertTrue(any("koşulsuz class_map" in p for p in self.problems(bad)))

    def test_pending_source_cannot_be_promoted_with_hold_reasons_or_unverified_rights(self) -> None:
        bad = copy.deepcopy(self.policy)
        bad["sources"]["site_safety"]["source_usage"] = "approved_scope"
        problems = self.problems(bad)
        self.assertTrue(any("onaylı kapsamda olamaz" in p for p in problems))
        self.assertTrue(any("bekleme gerekçesi" in p for p in problems))

    def test_image_override_validation(self) -> None:
        bad = copy.deepcopy(self.policy)
        ov = bad["sources"]["construction_ppe"]["image_overrides"]
        key = next(iter(ov))
        ov[key]["status"] = "approved"
        self.assertTrue(any("geçersiz override" in p for p in self.problems(bad)))
        bad = copy.deepcopy(self.policy)
        bad["sources"]["construction_ppe"]["image_overrides"]["data/raw/site_safety/x.jpg"] = {"status": "quarantine", "rule": "RF-LIKELY-ERROR"}
        self.assertTrue(any("başka kaynağın" in p for p in self.problems(bad)))
        bad = copy.deepcopy(self.policy)
        del bad["sources"]["construction_ppe"]["split_policy"]
        self.assertTrue(any("split_policy" in p for p in self.problems(bad)))


class LicenseItemStatusTest(unittest.TestCase):
    def test_new_status_in_schema_and_counted_as_open(self) -> None:
        self.assertIn(EXTERNAL_PENDING_EXCLUDED, STATUSES)
        self.assertIn(EXTERNAL_PENDING_EXCLUDED, OPEN_STATUSES)
        self.assertNotIn("resolved_verified", OPEN_STATUSES)

    def test_f01_f02_closed_with_excluded_status_not_verified(self) -> None:
        for f in ("F01", "F02"):
            status, note = ITEM_OVERRIDES[f]
            self.assertEqual(status, EXTERNAL_PENDING_EXCLUDED)
            self.assertIn("ÇÖZÜLMEDİ", note)
        items = item_statuses([{"id": "F01", "category": "C"}, {"id": "F02", "category": "C"}, {"id": "A01", "category": "B"}])
        self.assertEqual([i["status"] for i in items], [EXTERNAL_PENDING_EXCLUDED, EXTERNAL_PENDING_EXCLUDED, "resolved_verified"])
        self.assertEqual(items[0]["training_risk"], "contained_by_exclusion")

    def test_policy_checklist_items_match_overrides(self) -> None:
        pol = load_policy()
        self.assertEqual(pol["sources"]["site_safety"]["checklist_items"]["F01"], ITEM_OVERRIDES["F01"][0])
        self.assertEqual(pol["sources"]["mendeley_ppe"]["checklist_items"]["F02"], ITEM_OVERRIDES["F02"][0])


class IntegrityToolTest(unittest.TestCase):
    def make_ds(self, root: Path) -> Path:
        ds = root / "ds"
        for split, names in (("train", ["a", "b"]), ("val", ["c"])):
            (ds / "images" / split).mkdir(parents=True)
            (ds / "labels" / split).mkdir(parents=True)
            for n in names:
                (ds / "images" / split / f"{n}.jpg").write_bytes(n.encode() * 10)
                (ds / "labels" / split / f"{n}.txt").write_text("0 0.5 0.5 0.1 0.1\n2 0.5 0.6 0.2 0.2\n", encoding="utf-8")
        lines = []
        for split, names in (("train", ["a", "b"]), ("val", ["c"])):
            for n in names:
                sha = hashlib.sha256(n.encode() * 10).hexdigest()
                lines.append(json.dumps({"file": f"{n}.jpg", "split": split, "sha256": sha, "annotations": 2}))
        (ds / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return ds

    def test_publisher_manifest_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ds = self.make_ds(Path(tmp))
            self.assertTrue(verify_publisher_manifest(ds)["ok"])
            (ds / "images" / "train" / "a.jpg").write_bytes(b"changed")
            (ds / "labels" / "val" / "c.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            res = verify_publisher_manifest(ds)
            self.assertFalse(res["ok"])
            self.assertEqual(res["sha256_mismatch"], ["a.jpg"])
            self.assertEqual(res["annotation_count_mismatch"], ["c.jpg"])

    def test_zip_comparison_detects_change_and_extra_but_lists_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ds = self.make_ds(Path(tmp))
            zp = Path(tmp) / "orig.zip"
            with zipfile.ZipFile(zp, "w") as zf:
                for p in sorted(ds.rglob("*")):
                    if p.is_file():
                        zf.write(p, "wrap/" + p.relative_to(ds).as_posix())
            (ds / ".gitkeep").write_bytes(b"")
            res = verify_against_zip(zp, ds, "wrap")
            self.assertTrue(res["ok"], res)
            self.assertEqual(res["placeholders_ignored"], [".gitkeep"])
            (ds / "labels" / "train" / "b.txt").write_text("0 0.5 0.5 0.1 0.1\n2 0.5 0.6 0.2 0.3\n", encoding="utf-8")  # aynı boyut, farklı içerik
            (ds / "extra.txt").write_text("x", encoding="utf-8")
            res = verify_against_zip(zp, ds, "wrap")
            self.assertFalse(res["ok"])
            self.assertEqual(res["content_mismatch"], ["labels/train/b.txt"])
            self.assertEqual(res["extra_on_disk"], ["extra.txt"])


class Stage2COutputTest(unittest.TestCase):
    """Üretilmiş 2C çıktılarının tutarlılığı (çıktılar yoksa atlanır)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_policy()
        path = ROOT / cls.policy["sources"]["construction_ppe"]["split_policy"]["manifest"]
        if not path.is_file() or not (AUDIT / "image_usage_decisions.csv").is_file():
            raise unittest.SkipTest("2C çıktıları üretilmemiş")
        cls.split = rows(path)
        cls.usage = rows(AUDIT / "image_usage_decisions.csv")
        rv = ROOT / cls.policy["sources"]["construction_ppe"]["user_visual_review"]["file"]
        # 2D: kullanıcının görsel incelemede dışladığı görüntüler (karar verilmemişse boş)
        cls.user_excluded = user_overrides(read_review_csv(rv)) if rv.is_file() else {}

    def test_split_manifest_covers_all_construction_images_once(self) -> None:
        paths = [r["image_path"] for r in self.split]
        self.assertEqual(len(paths), 1000)
        self.assertEqual(len(set(paths)), 1000)
        self.assertTrue(all(p.startswith("data/raw/construction_ppe/") for p in paths))
        self.assertTrue(all(len(r["publisher_sha256"]) == 64 for r in self.split))

    def test_final_counts_and_known_decisions(self) -> None:
        usable = [r for r in self.split if r["final_split"]]
        ex = {p: next(r["original_split"] for r in self.split if r["image_path"] == p) for p in self.user_excluded}
        self.assertEqual(len(usable), 999 - len(ex))
        self.assertEqual(sum(r["final_split"] == "train" for r in usable), 800 - sum(1 for s in ex.values() if s == "train"))
        self.assertEqual(sum(r["final_split"] == "val" for r in usable), 199 - sum(1 for s in ex.values() if s == "val"))
        by = {r["file"]: r for r in self.split}
        moved = by["source-9e10b0409f922546c35b276eb5a5756e.jpg"]
        self.assertEqual((moved["original_split"], moved["final_split"], moved["split_changed"]), ("val", "train", "yes"))
        ex = by["ultralytics-ppe-nov2026-690.jpg"]
        self.assertEqual((ex["final_split"], ex["training_status"]), ("", "needs_reannotation"))
        # bilinen 3 bağlantılı görüntü aynı bölümde
        linked = ["source-3978abe9cfed5d8a5faa5a98dfe4b00a.jpg", "source-efc23273c04e007083114c0f217335f0.jpg",
                  "source-9e10b0409f922546c35b276eb5a5756e.jpg"]
        self.assertEqual({by[f]["final_split"] for f in linked}, {"train"})
        self.assertEqual(len({by[f]["split_unit"] for f in linked}), 1)
        self.assertEqual(unit_integrity_violations(self.split), {})

    def test_val_is_subset_of_original_val(self) -> None:
        for r in self.split:
            if r["final_split"] == "val":
                self.assertEqual(r["original_split"], "val", r["image_path"])

    def test_no_pending_source_image_in_approved_training_scope(self) -> None:
        for r in self.usage:
            if r["dataset"] in ("mendeley_ppe", "site_safety"):
                self.assertEqual(r["status_now"], "pending_license_verification", r["image_path"])
                self.assertEqual(r["final_split"], "", r["image_path"])
        self.assertFalse(any(r["preview_scope"] == "approved_now" for r in self.usage))  # 2B önizlemesi geçersiz

    def test_annotation_risk_flags_present(self) -> None:
        flagged = [r for r in self.split if r["annotation_risk"]]
        self.assertEqual(len(flagged), 63)
        self.assertTrue(all(r["annotation_risk"].startswith("CP-PERSON-ATTR-INCOMPLETE:") for r in flagged))
        self.assertTrue(all(int(r["n_flagged_persons"]) >= 1 for r in flagged))

    def test_integrity_check_output_all_passed(self) -> None:
        path = AUDIT / "stage2c_integrity_checks.json"
        if not path.is_file():
            self.skipTest("run_stage2c_checks.py çalıştırılmamış")
        d = json.loads(path.read_text(encoding="utf-8"))
        failed = [c["id"] for c in d["checks"] if not c["passed"]]
        self.assertEqual(failed, [])
        self.assertEqual(d["total"], 12)

    def test_stage2c_decision_record_consistent_with_policy(self) -> None:
        d = yaml.safe_load((ROOT / "configs/review/stage2c_decisions.yaml").read_text(encoding="utf-8"))
        ov = self.policy["sources"]["construction_ppe"]["image_overrides"]
        for p, dec in d["construction_ppe_annotation_risk"]["decisions"].items():
            self.assertEqual(ov[p]["status"], dec["decision"])
            self.assertEqual(ov[p]["rule"], dec["rule"])
        for f in ("F01", "F02"):
            self.assertEqual(d["license_items"][f]["status"], EXTERNAL_PENDING_EXCLUDED)


class DocumentationTest(unittest.TestCase):
    def test_evaluation_plan_covers_required_topics_and_claims_no_results(self) -> None:
        text = (ROOT / "docs" / "evaluation_plan.md").read_text(encoding="utf-8")
        for needle in ("Train / validation / test", "sızıntı", "saha görüntülerinin toplanması", "Birden fazla insan",
                       "Baret ve yelek", "Precision", "Recall", "mAP@0.5", "Hatalı tespitlerin", "Gerçek kamera testlerinin"):
            self.assertIn(needle, text)
        self.assertIn("hiçbir performans sonucu yoktur", text)
        self.assertIn("Ölçülmedi (model yok)", text)
        self.assertIn("en az **%85 başarı**", text)
        self.assertIn("`not_measured`", text)

    def test_attribution_has_ready_text_and_change_procedure(self) -> None:
        text = (ROOT / "docs" / "dataset_attributions.md").read_text(encoding="utf-8")
        self.assertIn("Bitirme raporu için hazır atıf metinleri", text)
        self.assertIn("Gelecekteki değişikliklerin kayıt prosedürü", text)
        self.assertIn(EXTERNAL_PENDING_EXCLUDED, text)
        self.assertIn("e5817aa9", text)


if __name__ == "__main__":
    unittest.main()
