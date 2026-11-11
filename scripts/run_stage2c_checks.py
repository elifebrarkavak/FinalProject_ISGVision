#!/usr/bin/env python
"""Aşama 2C nihai bütünlük denetimlerini çalıştırır ve sonucu yazar (reports/dataset_audit/stage2c_integrity_checks.json).

Salt okunur: ham veri, etiket ve bölünmeler DEĞİŞTİRİLMEZ; özgün ZIP'ler yalnızca okunur (CRC32 karşılaştırması).
Önce çalıştırılmış olmalı: build_source_groups.py, apply_usage_policy.py, generate_stage2b_reports.py.
Başarısız denetim varsa çıkış kodu 1'dir (hiçbir sonuç gizlenmez).

Örnek:
    python scripts/run_stage2c_checks.py
    python scripts/run_stage2c_checks.py --skip-zip     # özgün ZIP'ler yoksa / hızlı çalıştırma
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.dataset_audit.common import dataset_raw_dir, ensure_audit_dir, load_registry, read_csv, read_json, write_json  # noqa: E402
from isg_vision.dataset_audit.layout import discover_dataset  # noqa: E402
from isg_vision.dataset_audit.mapping import load_mapping  # noqa: E402
from isg_vision.dataset_audit.stage2b_reports import EXTERNAL_PENDING_EXCLUDED, ITEM_OVERRIDES  # noqa: E402
from isg_vision.dataset_audit.stage2c import (  # noqa: E402
    check, count_target_boxes, pdf_page_count, stage_reference_problems, verify_against_zip, verify_publisher_manifest,
    writable_files,
)
from isg_vision.dataset_audit.visual_review import (  # noqa: E402
    MODEL_DECISIONS, read_review_csv, review_state, validate_review_rows,
)
from isg_vision.dataset_audit.usage_policy import (  # noqa: E402
    LICENSE_STATUSES_APPROVABLE, RIGHTS_STATUSES_APPROVABLE, iter_image_records, load_policy, unit_integrity_violations, validate_policy,
)
from isg_vision.utils.paths import PROJECT_ROOT  # noqa: E402

REQUIRED_HOLD = {"mendeley_ppe": {"rights_unverified", "annotation_quality"},
                 "site_safety": {"license_unverified", "label_noise", "split_leakage"}}
ANALYSIS_RECORDS = ["dataset_inventory.csv", "label_review.csv", "quarantine_candidates.csv", "needs_reannotation_list.csv",
                    "duplicate_verification.csv", "source_group_manifest.csv", "source_group_edges.csv", "license_verification_report.md",
                    "roboflow_label_review_report.md", "semantic_validation_report.md", "final_dataset_decisions.md"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--skip-zip", action="store_true")
    args = ap.parse_args()
    out = ensure_audit_dir(args.out_dir)
    registry, policy = load_registry(), load_policy()
    targets, entries = load_mapping()
    layouts = {n: discover_dataset(n, dataset_raw_dir(registry, n)) for n in registry}
    usage = read_csv(out / "image_usage_decisions.csv")
    summary = read_json(out / "usage_policy_summary.json")
    edges = read_csv(out / "source_group_edges.csv")
    approved_src = [d for d, s in policy["sources"].items() if s["source_usage"] == "approved_scope"]
    cp = "construction_ppe"
    cp_src = policy["sources"][cp]
    split_rows = read_csv(PROJECT_ROOT / cp_src["split_policy"]["manifest"])
    d2c = yaml.safe_load((PROJECT_ROOT / "configs/review/stage2c_decisions.yaml").read_text(encoding="utf-8"))
    raw_cp = dataset_raw_dir(registry, cp)
    checks = []

    # --- Görev 1: Construction-PPE envanter ve sınıf sayıları
    inv = verify_publisher_manifest(raw_cp)
    checks.append(check("T1.1", "Construction-PPE 1000 görüntü yayıncı envanteriyle (manifest.jsonl: ad, bölüm, SHA-256, anotasyon sayısı) eşleşiyor mu?",
                        inv["ok"] and inv["manifest_entries"] == 1000 == registry[cp]["published"]["images"],
                        {k: (v if not isinstance(v, list) else len(v)) for k, v in inv.items()}))
    cmap = {int(k): int(v) for k, v in cp_src["class_map"].items()}
    all_counts = count_target_boxes(raw_cp, cmap, targets)
    total = Counter()
    for c in all_counts.values():
        total.update(c)
    expected_2b = {"helmet": 1576, "no_helmet": 813, "vest": 1816, "no_vest": 567}
    usable = {r["file"] for r in split_rows if r["final_split"]}
    usable_counts = Counter()
    for c in count_target_boxes(raw_cp, cmap, targets, usable).values():
        usable_counts.update(c)
    checks.append(check("T1.2", "Dört hedef sınıfın kutu sayıları etiket dosyalarından yeniden sayıldığında önceki raporla ve politika özetiyle tutarlı mı?",
                        dict(total) == expected_2b and dict(usable_counts) == summary["approved_scope_target_boxes"],
                        {"all_1000_by_original_split": all_counts, "all_1000_total": dict(total), "expected_stage2b": expected_2b,
                         "usable_images_total": dict(usable_counts), "usage_policy_summary": summary["approved_scope_target_boxes"]}))

    # --- Görev 3 / 6: onaylı kapsam, politika tutarlılığı
    classes = {n: lay.classes.names for n, lay in layouts.items()}
    problems = validate_policy(policy, registry, entries, classes)
    hold_ok = all(REQUIRED_HOLD[d] <= set(policy["sources"][d].get("hold_reasons", [])) for d in REQUIRED_HOLD)
    checks.append(check("T6.2", "Veri kullanım politikası iç tutarlı ve kararlarla uyumlu mu (Construction onaylı; Mendeley hak+anotasyon, Roboflow lisans+etiket+sızıntı nedeniyle beklemede)?",
                        not problems and hold_ok and approved_src == [cp]
                        and all(policy["sources"][d]["source_usage"] == "pending_license_verification" for d in REQUIRED_HOLD),
                        {"validate_policy_problems": problems, "approved_scope_sources": approved_src,
                         "hold_reasons": {d: policy["sources"][d].get("hold_reasons") for d in REQUIRED_HOLD}}))
    approved_rows = [r for r in usage if r["status_now"] == "approved"]
    foreign_in_scope = [r["image_path"] for r in usage if r["dataset"] != cp and (r["status_now"] == "approved" or r["final_split"])]
    foreign_in_manifest = [r["image_path"] for r in split_rows if not r["image_path"].startswith(f"data/raw/{cp}/")]
    usable_rows = [r for r in split_rows if r["final_split"]]
    checks.append(check("T6.1", "Onaylı veri kapsamı doğru mu (yalnızca Construction-PPE; Mendeley/Roboflow'dan hiçbir görüntü onaylı/bölmeli değil; kullanım dışı görüntü bölmede yok)?",
                        not foreign_in_scope and not foreign_in_manifest and len(split_rows) == 1000
                        and all(r["training_status"] == "approved" for r in usable_rows)
                        and {r["image_path"] for r in approved_rows} == {r["image_path"] for r in usable_rows},
                        {"approved_images": len(approved_rows), "by_final_split": dict(Counter(r["final_split"] for r in usable_rows)),
                         "excluded_in_scope": {r["image_path"]: r["decision_rule"] for r in split_rows if not r["final_split"]},
                         "mendeley_or_roboflow_in_approved_or_split": len(foreign_in_scope) + len(foreign_in_manifest),
                         "pending_by_dataset": dict(Counter(r["dataset"] for r in usage if r["status_now"] == "pending_license_verification"))}))

    # --- sızıntı kararı ve birim bütünlüğü
    cross_units = {}
    for r in split_rows:
        cross_units.setdefault(r["split_unit"], set()).add(r["original_split"])
    cross_units = {u for u, s in cross_units.items() if len(s) > 1}
    fixes = {f["split_unit"] for f in summary.get("approved_scope_unit_fixes", {}).get(cp, [])}
    decided = {m for u in d2c["construction_ppe_linked_units"] for m in u["split_unit_members"]}
    cross_members = {r["image_path"] for r in split_rows if r["split_unit"] in cross_units}
    checks.append(check("T6.3", "Construction-PPE train/val sızıntısı için karar tanımlanmış mı (her bölümler arası birim için kayıtlı düzeltme + görsel karar)?",
                        bool(cross_units) and cross_units <= fixes and cross_members <= decided,
                        {"cross_split_units_original": sorted(cross_units), "unit_fixes": summary.get("approved_scope_unit_fixes", {}).get(cp),
                         "visual_decisions": [u["decision"] for u in d2c["construction_ppe_linked_units"]]}))
    viol = unit_integrity_violations(split_rows)
    final_of = {r["image_path"]: r["final_split"] for r in usage if r["final_split"]}
    edge_viol = [(e["path_a"], e["path_b"]) for e in edges if e["path_a"] in final_of and e["path_b"] in final_of
                 and final_of[e["path_a"]] != final_of[e["path_b"]]]
    prev_viol = {}
    for r in usage:
        if r["proposed_split_preview"]:
            prev_viol.setdefault((r["preview_scope"], r["split_unit"]), set()).add(r["proposed_split_preview"])
    prev_viol = {f"{k[0]}/{k[1]}": sorted(v) for k, v in prev_viol.items() if len(v) > 1}
    checks.append(check("T6.4", "Kaynak gruplarının ayrılmama kuralı korunuyor mu (nihai bölme, tek tek kenarlar ve beklemedeki önizleme)?",
                        not viol and not edge_viol and not prev_viol and policy["leakage"]["never_split_a_unit"]
                        and policy["leakage"]["random_image_level_split"] == "forbidden",
                        {"final_split_unit_violations": {k: sorted(v) for k, v in viol.items()}, "edge_violations": edge_viol,
                         "preview_violations": prev_viol, "group_integrity_rules": [g["id"] for g in policy["leakage"]["group_integrity_rules"]]}))

    # --- lisans
    bad_lic = [r["image_path"] for r in usage if r["status_now"] == "approved" and (
        policy["sources"][r["dataset"]]["license_status"] not in LICENSE_STATUSES_APPROVABLE
        or policy["sources"][r["dataset"]]["rights_status"] not in RIGHTS_STATUSES_APPROVABLE)]
    checklist = (out / "manual_review_checklist.md").read_text(encoding="utf-8")
    att = (PROJECT_ROOT / "docs" / "dataset_attributions.md").read_text(encoding="utf-8")
    f_ok = all(ITEM_OVERRIDES[f][0] == EXTERNAL_PENDING_EXCLUDED for f in ("F01", "F02")) and \
        all(f"| {f} |" in checklist and EXTERNAL_PENDING_EXCLUDED in checklist for f in ("F01", "F02"))
    checks.append(check("T6.5", "Lisansı/hakkı doğrulanmamış veri yanlışlıkla onaylanıyor mu? (geçmesi için: HAYIR)",
                        not bad_lic and f_ok and "BAĞIMSIZ DOĞRULANAMADI" in att and "YAZILMAMIŞTIR" in att,
                        {"approved_images_from_unverified_sources": len(bad_lic),
                         "F01": ITEM_OVERRIDES["F01"][0], "F02": ITEM_OVERRIDES["F02"][0],
                         "license_status": {d: s["license_status"] for d, s in policy["sources"].items()}}))

    # --- eksik anotasyon riski
    recomputed = {}
    for rec in iter_image_records(layouts[cp], registry[cp]):
        if rec["person_issues"]:
            recomputed[str(rec["image_path"].relative_to(PROJECT_ROOT))] = len(rec["person_issues"])
    flagged = {r["image_path"]: int(r["n_flagged_persons"]) for r in split_rows if r["annotation_risk"]}
    ov = cp_src.get("image_overrides") or {}
    ov_ok = all(any(r["image_path"] == p and not r["final_split"] and r["training_status"] == o["status"] for r in split_rows) for p, o in ov.items())
    pend_flag = [r for r in usage if r["dataset"] == "site_safety" and "RF-PERSON-INCOMPLETE" in r["reason_codes"]
                 and r["status_if_rights_verified"] == "approved"]
    checks.append(check("T6.6", "Eksik anotasyon riski taşıyan görüntüler doğru işaretleniyor mu (bağımsız yeniden hesapla birebir; görsel olarak doğrulanan eksik etiket kullanım dışı)?",
                        recomputed == flagged and ov_ok and not pend_flag and bool(flagged),
                        {"flagged_images": len(flagged), "flagged_persons": sum(flagged.values()),
                         "recomputed_equal": recomputed == flagged, "overrides": {p: o["status"] for p, o in ov.items()},
                         "flag_by_final_split": dict(Counter(r["final_split"] or "excluded" for r in split_rows if r["annotation_risk"])),
                         "roboflow_person_incomplete_marked_approved": len(pend_flag)}))

    # --- önceki raporlarla çelişki
    final_md = (out / "final_dataset_decisions.md").read_text(encoding="utf-8")
    remaining = (out / "remaining_manual_reviews.md").read_text(encoding="utf-8")
    contradictions = []
    if any(r["preview_scope"] == "approved_now" for r in usage):
        contradictions.append("image_usage_decisions.csv hâlâ Construction-PPE için 2B %70/15/15 önizlemesi içeriyor")
    if "Seçilen durum: 2" in final_md:
        contradictions.append("final_dataset_decisions.md eski kapanış ifadesini içeriyor")
    if "construction_ppe_split_manifest.csv" not in final_md:
        contradictions.append("final_dataset_decisions.md nihai bölme manifestine atıf yapmıyor")
    if EXTERNAL_PENDING_EXCLUDED not in remaining:
        contradictions.append("remaining_manual_reviews.md F01/F02 için yeni durumu içermiyor")
    if f"'approved': {len(approved_rows)}" not in final_md:
        contradictions.append("final_dataset_decisions.md onaylı görüntü sayısı güncel değil")
    if "## 18. Aşama 2C" not in (out / "dataset_audit_report.md").read_text(encoding="utf-8"):
        contradictions.append("dataset_audit_report.md 2C özetini içermiyor (2B sayıları güncel sanılabilir)")
    if not (PROJECT_ROOT / "docs" / "evaluation_plan.md").is_file():
        contradictions.append("docs/evaluation_plan.md yok")
    if policy["policy_version"] < 2:
        contradictions.append("politika sürümü 2C kararlarını içermiyor")
    for d in REQUIRED_HOLD:
        if dataset_raw_dir(registry, d).is_dir() is False:
            contradictions.append(f"{d} ham verisi silinmiş")
    missing_records = [n for n in ANALYSIS_RECORDS if not (out / n).is_file()]
    checks.append(check("T6.7", "Önceki raporlarla güncel kararlar çelişiyor mu ve beklemedeki kaynakların analiz kayıtları korunuyor mu? (geçmesi için: çelişki yok, kayıt eksik yok)",
                        not contradictions and not missing_records,
                        {"contradictions": contradictions, "missing_analysis_records": missing_records,
                         "pending_raw_dirs_present": {d: dataset_raw_dir(registry, d).is_dir() for d in REQUIRED_HOLD}}))

    # --- ham veri korunuyor mu
    log = read_json(out / "extraction_log.json")
    zips = {}
    for run in log["runs"]:
        for e in run["entries"]:
            zips[e["dataset"]] = (Path(e["zip_path"]).expanduser(), e.get("stripped_prefix", ""))
    zres = {}
    if not args.skip_zip:
        for d, (zp, prefix) in zips.items():
            r = verify_against_zip(zp, dataset_raw_dir(registry, d), prefix)
            zres[d] = {k: (v if not isinstance(v, list) or k == "placeholders_ignored" else len(v)) for k, v in r.items()}
    writable = {d: len(writable_files(dataset_raw_dir(registry, d))) for d in registry}
    zip_ok = all(v.get("ok") is not False for v in zres.values())
    checks.append(check("T6.8", "Ham veri dosyaları korunuyor mu (Construction SHA-256 = yayıncı manifesti; tüm ham dosyalar özgün ZIP'lerle boyut+CRC32 eşleşmesi; ek/eksik dosya yok)?",
                        inv["ok"] and not inv["sha256_mismatch"] and zip_ok,
                        {"construction_sha256_mismatch": len(inv["sha256_mismatch"]), "zip_comparison": zres or "atlandı (--skip-zip)",
                         "owner_writable_files": writable,
                         "note": "Yalnızca okunarak karşılaştırıldı. owner_writable_files bilgi amaçlıdır (dosya izni), içerik değişikliği göstermez."}))

    # --- Aşama 2D: görsel inceleme paketi eksiksiz mi?
    uvr = cp_src["user_visual_review"]
    rdir = (PROJECT_ROOT / uvr["file"]).parent
    expected = {r["image_path"] for r in split_rows if "unexplained" in r["annotation_risk"]}
    rrows = read_review_csv(PROJECT_ROOT / uvr["file"]) if (PROJECT_ROOT / uvr["file"]).is_file() else []
    rprob = validate_review_rows(rrows, {r["image_path"] for r in split_rows}) if rrows else ["review_decisions.csv yok"]
    missing_pages = [r["review_id"] for r in rrows if not (rdir / r["contact_sheet_page"].split(" ")[0]).is_file()]
    no_model = [r["review_id"] for r in rrows if r["model_decision"] not in MODEL_DECISIONS or not r["model_rationale"].strip()]
    pdf = rdir / "contact_sheet.pdf"
    n_pdf = pdf_page_count(pdf) if pdf.is_file() else 0
    by_split = dict(Counter(r["split"] for r in rrows))
    checks.append(check("T2D.1", "22 görüntülük görsel inceleme paketi eksiksiz mi (CSV şeması, beklenen görüntü kümesi, her görüntü için sayfa, 22 sayfalık PDF, model önerisi + gerekçe, rehber)?",
                        not rprob and {r["image_path"] for r in rrows} == expected and len(rrows) == 22 and by_split == {"train": 18, "val": 4}
                        and not missing_pages and not no_model and n_pdf == 22 and (rdir / "review_guide.md").is_file(),
                        {"rows": len(rrows), "by_split": by_split, "csv_problems": rprob, "missing_pages": missing_pages,
                         "missing_model_decision": no_model, "pdf_pages": n_pdf,
                         "model_decisions": dict(Counter(r["model_decision"] for r in rrows)),
                         "user_review_state": review_state(rrows) if rrows else None,
                         "note": "Kullanıcı kararı bekleyen satırlar bu denetimi başarısız yapmaz; kullanıcı doğrulaması SAYILMAZ (Aşama 8 öncesi kapı: check_visual_review.py --require-complete)."}))
    # --- Aşama 2D: aşama numaraları 12 aşamalı planla tutarlı mı?
    stages = yaml.safe_load((PROJECT_ROOT / "configs/project_stages.yaml").read_text(encoding="utf-8"))["stages"]
    sprob = stage_reference_problems(PROJECT_ROOT, stages)
    checks.append(check("T2D.2", "Belgelerdeki aşama numaraları 12 aşamalı planla tutarlı mı (Aşama 3 = kamera altyapısı; eğitim Aşama 8, birleştirme Aşama 9)?",
                        not sprob, {"problems": sprob, "reference": "configs/project_stages.yaml, docs/project_plan.md"}))

    passed = sum(c["passed"] for c in checks)
    result = {"date": date.today().isoformat(), "total": len(checks), "passed": passed, "all_passed": passed == len(checks), "checks": checks}
    write_json(out / "stage2c_integrity_checks.json", result)
    for c in checks:
        print(f"[{'GEÇTİ' if c['passed'] else 'BAŞARISIZ'}] {c['id']} {c['question']}")
    print(f"Toplam: {passed}/{len(checks)} denetim geçti.")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
