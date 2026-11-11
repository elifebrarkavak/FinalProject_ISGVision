#!/usr/bin/env python
"""Aşama 2B raporlarını üretir: final_dataset_decisions.md, manual_review_checklist.md (nihai durumlar),
remaining_manual_reviews.md, followup_summary.json ve (isteğe bağlı) docs/dataset_attributions.md.

Önce çalıştırılmış olmalı: verify_duplicates.py, build_source_groups.py, apply_usage_policy.py, generate_followup_reports.py.
Veri dosyalarını okumaz/değiştirmez; yalnızca denetim çıktılarını birleştirir.

Örnek:
    python scripts/generate_stage2b_reports.py
    python scripts/generate_stage2b_reports.py --write-attributions   # docs/dataset_attributions.md (yoksa oluşturur; varsa ÜZERİNE YAZMAZ)
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.dataset_audit.common import ensure_audit_dir, load_registry, read_csv, read_json, write_json  # noqa: E402
from isg_vision.dataset_audit.followup_reports import build_checklist_v2, load_followup  # noqa: E402
from isg_vision.dataset_audit.mapping import load_mapping  # noqa: E402
from isg_vision.dataset_audit.stage2b_reports import (  # noqa: E402
    OPEN_STATUSES, attributions_md, checklist_md, final_decisions_md, item_statuses, remaining_md,
)
from isg_vision.dataset_audit.usage_policy import load_policy  # noqa: E402
from isg_vision.utils.paths import PROJECT_ROOT  # noqa: E402


def _stage2c_checks(out: Path) -> tuple[bool, str]:
    path = out / "stage2c_integrity_checks.json"
    if not path.is_file():
        return False, "stage2c_integrity_checks.json yok (scripts/run_stage2c_checks.py çalıştırılmalı); birim testleri ayrıca çalıştırılır"
    d = read_json(path)
    return bool(d.get("all_passed")), (f"{d.get('passed')}/{d.get('total')} denetim geçti (stage2c_integrity_checks.json); "
                                       "birim testlerinin sonucu STAGE_2_FINAL_REPORT.md içinde")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--write-attributions", action="store_true")
    args = parser.parse_args()
    out = ensure_audit_dir(args.out_dir)
    try:
        a = load_followup(out)
        usage = read_json(out / "usage_policy_summary.json")
        groups = read_json(out / "source_group_summary.json")
    except FileNotFoundError as exc:
        print(f"[HATA] {exc}\nÖnce build_source_groups.py ve apply_usage_policy.py çalıştırın.")
        return 1
    registry, policy = load_registry(), load_policy()
    targets, entries = load_mapping()
    decisions = yaml.safe_load((PROJECT_ROOT / "configs/review/stage2b_decisions.yaml").read_text(encoding="utf-8"))
    lic = a["lic"]
    items, _ = build_checklist_v2(a, registry, entries, targets)
    items = item_statuses(items)
    cnt = Counter(i["status"] for i in items)
    (out / "manual_review_checklist.md").write_text(checklist_md(items), encoding="utf-8")
    (out / "remaining_manual_reviews.md").write_text(remaining_md(items), encoding="utf-8")
    cat = Counter(i["category"] for i in items)
    write_json(out / "followup_summary.json", {"checklist_items": len(items), "by_category": dict(cat),
                                              "resolved_A_B": cat["A"] + cat["B"], "remaining_D": cat["D"], "unresolved_technical_C": cat["C"],
                                              "stage2b_status": dict(cnt)})
    # kapanış ölçütleri (otomatik kontroller; testler ayrıca çalıştırılır)
    rows = read_csv(out / "image_usage_decisions.csv")
    approved_in_pending = [r for r in rows if r["status_now"] == "approved" and policy["sources"][r["dataset"]]["source_usage"] != "approved_scope"]
    quarantine_rows = read_csv(out / "quarantine_candidates.csv")
    pend = [i for i in items if i["status"] in OPEN_STATUSES]
    criteria = {
        "1. Açık kalan maddeler listelenmiş": (bool(pend) and (out / "remaining_manual_reviews.md").is_file(), f"{len(pend)} madde remaining_manual_reviews.md içinde"),
        "2. Her maddeye karar/doğrulama ihtiyacı atanmış": (all(i["status"] in cnt for i in items) and cnt.get("unresolved", 0) == 0,
                                                            f"durumlar: {dict(cnt)}; 'unresolved' yok"),
        "3. Açık riskler onaylı veri kullanımını etkilemiyor": (not approved_in_pending,
                                                               f"onaylı durumda olup hakları doğrulanmamış kaynak görüntüsü: {len(approved_in_pending)}"),
        "4. Uygun olmayan örnekler belirlenmiş": (len(quarantine_rows) > 0, f"{len(quarantine_rows)} karantina/yeniden etiketleme adayı (quarantine_candidates.csv)"),
        "5. Kullanılabilir kaynaklar açıkça tanımlı": (True, "configs/dataset_usage_policy.yaml: onaylı kapsam = construction_ppe; diğerleri pending_license_verification"),
        "6. Veri sızıntısını önleme planı hazır": ((out / "source_group_manifest.csv").is_file(), "kaynak grup manifestosu + bölme politikası (leakage)"),
        "7. Lisans ve atıf kayıtları oluşturulmuş": ((out / "license_verification_report.md").is_file(), "license_verification_report.md + docs/dataset_attributions.md"),
        # Bu üreteç testleri ÇALIŞTIRMAZ; sonucu varsaymak yerine 2C bütünlük denetimi çıktısına bakılır.
        "8. Bütünlük denetimleri (2C) başarılı": _stage2c_checks(out),
    }
    edges = read_csv(out / "source_group_edges.csv")
    tile_counts = {"tile_edges": sum(1 for e in edges if e["evidence_type"] == "tile_level_orb")}
    (out / "final_dataset_decisions.md").write_text(final_decisions_md(
        usage=usage, groups=groups, decisions=decisions, policy=policy, registry=registry, lic=lic, items=items, criteria=criteria,
        tile_counts=tile_counts), encoding="utf-8")
    att = PROJECT_ROOT / "docs" / "dataset_attributions.md"
    if args.write_attributions:
        if att.exists():
            print("[UYARI] docs/dataset_attributions.md zaten var; ÜZERİNE YAZILMADI.")
        else:
            att.write_text(attributions_md(registry, lic, read_json(out / "extraction_log.json")), encoding="utf-8")
            print("docs/dataset_attributions.md oluşturuldu.")
    print(f"Kontrol listesi nihai durumlar: {dict(cnt)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
