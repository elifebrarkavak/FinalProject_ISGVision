"""Aşama 2D testleri: görsel inceleme paketi, kullanıcı karar CSV'sinin doğrulanması/uygulanması ve aşama numarası tutarlılığı.

Sentetik veriler geçici klasörlerde üretilir; ham veri setleri yalnızca OKUNUR.
"""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isg_vision.dataset_audit.stage2c import pdf_page_count, stage_reference_problems  # noqa: E402
from isg_vision.dataset_audit.usage_policy import load_policy  # noqa: E402
from isg_vision.dataset_audit.visual_review import (  # noqa: E402
    MODEL_DECISIONS, PAGE_H, PAGE_W, REVIEW_FIELDS, USER_DECISIONS, collect_items, read_review_csv, render_page, review_state,
    user_overrides, validate_review_rows, write_review_csv,
)

REVIEW_DIR = ROOT / "reports" / "dataset_audit" / "final_visual_review"
STAGES = yaml.safe_load((ROOT / "configs" / "project_stages.yaml").read_text(encoding="utf-8"))["stages"]


def row(rid: str, path: str, model: str = "approved_candidate", user: str = "", note: str = "") -> dict[str, str]:
    r = {k: "" for k in REVIEW_FIELDS}
    r.update({"review_id": rid, "image_path": path, "split": "train", "model_decision": model, "model_rationale": "x",
              "user_decision": user, "user_note": note})
    return r


class ReviewCsvValidationTest(unittest.TestCase):
    paths = {"a.jpg", "b.jpg", "c.jpg"}

    def test_pending_decisions_are_valid_but_not_user_verified(self) -> None:
        rows = [row("R01", "a.jpg"), row("R02", "b.jpg", "needs_reannotation")]
        self.assertEqual(validate_review_rows(rows, self.paths), [])
        st = review_state(rows)
        self.assertEqual((st["user_decided"], st["pending_user"], st["complete"]), (0, 2, False))
        self.assertEqual(user_overrides(rows), {})          # model önerisi tek başına hiçbir şeyi uygulamaz

    def test_invalid_values_rejected(self) -> None:
        bad = [row("R01", "a.jpg", user="approved_candidate"),         # model değeri kullanıcı kararı olamaz
               row("R02", "b.jpg", user="inconclusive"),
               row("R03", "zzz.jpg"),                                   # manifestte olmayan görüntü
               row("R03", "c.jpg", model="maybe")]                      # yinelenen id + geçersiz model değeri
        problems = " | ".join(validate_review_rows(bad, self.paths))
        for needle in ("approved_candidate", "inconclusive", "manifestte yok", "yinelenen", "geçersiz model_decision"):
            self.assertIn(needle, problems)

    def test_exclusion_requires_note(self) -> None:
        self.assertTrue(validate_review_rows([row("R01", "a.jpg", user="quarantine")], self.paths))
        self.assertEqual(validate_review_rows([row("R01", "a.jpg", user="quarantine", note="baş görünür")], self.paths), [])

    def test_missing_column_and_expected_set(self) -> None:
        r = row("R01", "a.jpg")
        del r["user_note"]
        self.assertTrue(any("eksik sütun" in p for p in validate_review_rows([r], self.paths)))
        self.assertTrue(any("beklenen" in p for p in validate_review_rows([row("R01", "a.jpg")], self.paths, {"R01", "R02"})))

    def test_user_decisions_map_to_overrides(self) -> None:
        rows = [row("R01", "a.jpg", user="approved"), row("R02", "b.jpg", user="needs_reannotation", note="no_helmet eksik"),
                row("R03", "c.jpg", model="inconclusive", user="quarantine", note="belirsiz")]
        ov = user_overrides(rows)
        self.assertEqual(set(ov), {"b.jpg", "c.jpg"})
        self.assertEqual(ov["b.jpg"]["status"], "needs_reannotation")
        self.assertEqual(ov["c.jpg"]["rule"], "USER-VISUAL-REVIEW")
        self.assertTrue(review_state(rows)["complete"])

    def test_csv_roundtrip_keeps_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.csv"
            write_review_csv(p, [row("R01", "a.jpg", user="approved", note="ok")])
            got = read_review_csv(p)
        self.assertEqual(list(got[0]), REVIEW_FIELDS)
        self.assertEqual((got[0]["user_decision"], got[0]["user_note"]), ("approved", "ok"))

    def test_decision_vocabularies(self) -> None:
        self.assertEqual(MODEL_DECISIONS, ("approved_candidate", "needs_reannotation", "quarantine", "inconclusive"))
        self.assertEqual(USER_DECISIONS, ("approved", "needs_reannotation", "quarantine"))


class RenderTest(unittest.TestCase):
    def test_collect_and_render_synthetic_item(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ds/images/train").mkdir(parents=True)
            (root / "ds/labels/train").mkdir(parents=True)
            Image.new("RGB", (640, 480), (120, 120, 120)).save(root / "ds/images/train/x.jpg")
            # kişi (sınıf 5) + yalnızca gövde etiketi (vest=2): baş açıklanamayan eksik
            (root / "ds/labels/train/x.txt").write_text("5 0.5 0.55 0.2 0.6\n2 0.5 0.5 0.15 0.2\n", encoding="utf-8")
            entry = {"expected_class_names": ["helmet", "gloves", "vest", "boots", "goggles", "Person", "no_helmet",
                                              "no_goggle", "no_gloves", "no_boots", "forklift", "no_vest"],
                     "roles": {"helmet": [0], "no_helmet": [6], "vest": [2], "no_vest": [11], "person": [5]}}
            manifest = [{"image_path": "ds/images/train/x.jpg", "final_split": "train", "original_split": "train",
                         "annotation_risk": "CP-PERSON-ATTR-INCOMPLETE:head_unexplained"}]
            items = collect_items(manifest, entry, root)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].codes, ["head_unexplained"])
            self.assertEqual(items[0].nearby[0]["class"], "vest")
            self.assertEqual(items[0].nearby[0]["assigned_person"], 1)
            page = render_page(items[0], "R01", "approved_candidate", "test")
            self.assertEqual(page.size, (PAGE_W, PAGE_H))
            pdf = root / "p.pdf"
            page.save(pdf, save_all=True, append_images=[page], resolution=300)
            self.assertEqual(pdf_page_count(pdf), 2)


class StageReferenceTest(unittest.TestCase):
    def make_project(self, root: Path, extra_doc: str) -> None:
        titles = [STAGES[i]["title"] for i in range(1, 13)]
        (root / "docs").mkdir()
        listing = "\n".join(f"{i}. {t}" for i, t in enumerate(titles, 1))
        (root / "docs/project_plan.md").write_text(listing, encoding="utf-8")
        (root / "README.md").write_text("# x\n" + listing, encoding="utf-8")
        (root / "docs/other.md").write_text(extra_doc, encoding="utf-8")

    def test_real_repository_is_consistent(self) -> None:
        self.assertEqual(stage_reference_problems(ROOT, STAGES), [])

    def test_detects_wrong_stage_for_training(self) -> None:
        for bad in ("| Aşama 3: model eğitimi (kullanıcı onayından sonra) |", "Türetilmiş eğitim klasörü (Aşama 3) üretilir.",
                    "Aşama 13'te rapor yazılır."):
            with tempfile.TemporaryDirectory() as tmp:
                self.make_project(Path(tmp), bad)
                self.assertTrue(stage_reference_problems(Path(tmp), STAGES), bad)

    def test_negated_or_correct_mentions_allowed(self) -> None:
        ok = ("Aşama 3 kamera altyapısıdır; YOLO eğitimi yok.\nİlk YOLO eğitimi Aşama 8'dedir.\n"
              "Aşama 3'te YOLO eğitimi yapılmayacaktır.")
        with tempfile.TemporaryDirectory() as tmp:
            self.make_project(Path(tmp), ok)
            self.assertEqual(stage_reference_problems(Path(tmp), STAGES), [])

    def test_order_and_completeness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_project(root, "")
            titles = [STAGES[i]["title"] for i in range(1, 13)]
            swapped = titles[:2] + [titles[3], titles[2]] + titles[4:]
            (root / "README.md").write_text("\n".join(swapped), encoding="utf-8")
            self.assertTrue(any("sırasında değil" in p for p in stage_reference_problems(root, STAGES)))
            (root / "README.md").write_text("\n".join(titles[:11]), encoding="utf-8")
            self.assertTrue(any("eksik" in p for p in stage_reference_problems(root, STAGES)))

    def test_stage_plan_content(self) -> None:
        self.assertEqual(STAGES[3]["title"], "Canlı kamera altyapısı")
        self.assertEqual(STAGES[8]["title"], "İlk YOLO İSG modelinin eğitilmesi")
        self.assertEqual(STAGES[9]["title"], "Veri setlerinin birleştirilmesi")
        ev = (ROOT / "docs/evaluation_plan.md").read_text(encoding="utf-8")
        self.assertNotIn("Aşama 3: model eğitimi", ev)
        self.assertIn("Aşama 8 (ilk YOLO İSG modelinin eğitilmesi) hazırlığında", ev)
        att = (ROOT / "docs/dataset_attributions.md").read_text(encoding="utf-8")
        self.assertIn("(Aşama 8: ilk YOLO İSG modelinin eğitim hazırlığı", att)


class ReviewPackageOutputTest(unittest.TestCase):
    """Üretilmiş 2D paketinin eksiksizliği (dosyalar yoksa atlanır)."""

    @classmethod
    def setUpClass(cls) -> None:
        if not (REVIEW_DIR / "review_decisions.csv").is_file():
            raise unittest.SkipTest("2D inceleme paketi üretilmemiş")
        cls.rows = read_review_csv(REVIEW_DIR / "review_decisions.csv")
        pol = load_policy()
        with (ROOT / pol["sources"]["construction_ppe"]["split_policy"]["manifest"]).open(encoding="utf-8") as fh:
            cls.manifest = list(csv.DictReader(fh))

    def test_exact_22_images_18_train_4_val(self) -> None:
        expected = {r["image_path"] for r in self.manifest if "unexplained" in r["annotation_risk"]}
        self.assertEqual({r["image_path"] for r in self.rows}, expected)
        self.assertEqual(len(self.rows), 22)
        self.assertEqual(sum(r["split"] == "train" for r in self.rows), 18)
        self.assertEqual(sum(r["split"] == "val" for r in self.rows), 4)

    def test_every_row_has_page_model_decision_and_valid_user_field(self) -> None:
        self.assertEqual(validate_review_rows(self.rows, {r["image_path"] for r in self.manifest}), [])
        for r in self.rows:
            self.assertIn(r["model_decision"], MODEL_DECISIONS)
            self.assertTrue(r["model_rationale"].strip())
            self.assertTrue(r["risk_code"].startswith("CP-PERSON-ATTR-INCOMPLETE:"))
            self.assertTrue((REVIEW_DIR / r["contact_sheet_page"].split(" ")[0]).is_file(), r["review_id"])
        self.assertEqual(pdf_page_count(REVIEW_DIR / "contact_sheet.pdf"), 22)
        self.assertTrue((REVIEW_DIR / "review_guide.md").is_file())

    def test_model_decisions_match_record_and_are_not_user_decisions(self) -> None:
        rec = yaml.safe_load((ROOT / "configs/review/stage2d_visual_review.yaml").read_text(encoding="utf-8"))["reviews"]
        by = {d["image_path"]: d["model_decision"] for d in rec}
        for r in self.rows:
            self.assertEqual(r["model_decision"], by[r["image_path"]])
            self.assertNotIn(r["user_decision"].strip(), ("approved_candidate", "inconclusive"))  # yalnızca kullanıcı değerleri

    def test_manifest_carries_review_tag_for_all_22(self) -> None:
        tagged = [r for r in self.manifest if r.get("visual_review")]
        self.assertEqual(len(tagged), 22)
        for r in tagged:
            user = r["visual_review"].split("user=")[1]
            self.assertIn(user, ("pending",) + USER_DECISIONS)
            if user in ("needs_reannotation", "quarantine"):
                self.assertEqual(r["final_split"], "", r["image_path"])

    def test_final_user_decisions_are_complete(self) -> None:
        counts = {decision: sum(r["user_decision"] == decision for r in self.rows) for decision in USER_DECISIONS}
        self.assertEqual(counts, {"approved": 13, "needs_reannotation": 4, "quarantine": 5})
        self.assertTrue(all(r["user_decision"] for r in self.rows))
        self.assertTrue(all(r["user_note"].strip() for r in self.rows))


if __name__ == "__main__":
    unittest.main()
