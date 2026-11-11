#!/usr/bin/env python
"""Aşama 2 eksik giderme raporlarını üretir (Türkçe).

Girdiler: reports/dataset_audit/*.csv|json (analyze_box_semantics, review_roboflow_labels, verify_duplicates,
analyze_augmentation_artifacts çıktıları) ve configs/review/*.yaml (görsel inceleme kayıtları, lisans kanıtı,
kontrol listesi çözümleri), configs/candidate_class_mapping.yaml.
Çıktılar: semantic_validation_report.md, roboflow_label_review_report.md, roboflow_cleanup_suggestions.md,
roboflow_cleanup_candidates.csv, duplicate_verification_report.md, license_verification_report.md,
(kontrol listesi/kalan incelemeler: generate_stage2b_reports.py)
Bu betik veri dosyalarını OKUMAZ/DEĞİŞTİRMEZ; yalnızca denetim çıktılarını birleştirir.

Örnek:
    python scripts/generate_followup_reports.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.dataset_audit.common import ensure_audit_dir, load_registry, write_csv, write_json  # noqa: E402
from isg_vision.dataset_audit.followup_reports import (  # noqa: E402
    build_checklist_v2, cleanup_candidate_rows, cleanup_suggestions, duplicate_report, license_report, load_followup,
    remaining_reviews, roboflow_report, semantic_report,
)
from isg_vision.dataset_audit.mapping import load_mapping  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    out = ensure_audit_dir(args.out_dir)
    try:
        a = load_followup(out)
    except FileNotFoundError as exc:
        print(f"[HATA] {exc}\nÖnce analiz betiklerini çalıştırın (README: Aşama 2 eksik giderme komutları).")
        return 1
    registry = load_registry()
    targets, entries = load_mapping()
    (out / "semantic_validation_report.md").write_text(semantic_report(a, entries, targets), encoding="utf-8")
    (out / "roboflow_label_review_report.md").write_text(roboflow_report(a), encoding="utf-8")
    (out / "roboflow_cleanup_suggestions.md").write_text(cleanup_suggestions(a), encoding="utf-8")
    write_csv(out / "roboflow_cleanup_candidates.csv",
              ["item_id", "category", "image_path", "label_path", "lines", "label_text", "rationale", "suggested_fix"],
              cleanup_candidate_rows(a))
    (out / "duplicate_verification_report.md").write_text(duplicate_report(a), encoding="utf-8")
    (out / "license_verification_report.md").write_text(license_report(a, registry), encoding="utf-8")
    # Kontrol listesi, kalan incelemeler ve followup_summary.json artık generate_stage2b_reports.py tarafından (nihai durumlarla) üretilir.
    items, _ = build_checklist_v2(a, registry, entries, targets)
    print(f"Kontrol listesi maddesi (nihai durumlar için generate_stage2b_reports.py çalıştırın): {len(items)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
