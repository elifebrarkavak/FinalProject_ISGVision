#!/usr/bin/env python
"""Aşama 2 denetim raporunu (Türkçe), aday sınıf eşleme CSV'sini ve manuel inceleme listesini üretir.

Girdi: diğer betiklerin ürettiği reports/dataset_audit/*.csv|json dosyaları (önce onları çalıştırın:
extract_datasets, inspect_datasets, validate_labels, analyze_class_distribution, detect_duplicates).
Çıktılar: dataset_audit_report.md, candidate_class_mapping.csv (kontrol listesi artık generate_followup_reports.py tarafından üretilir)
Bu betik hiçbir veri dosyasını okumaz veya değiştirmez; yalnızca denetim çıktılarını birleştirir.

Örnek:
    python scripts/generate_dataset_report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.dataset_audit.common import ensure_audit_dir, load_registry, write_csv  # noqa: E402
from isg_vision.dataset_audit.mapping import (  # noqa: E402
    MAPPING_FIELDS, check_completeness, load_mapping, mapping_rows,
)
from isg_vision.dataset_audit.report import build_report, load_artifacts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, help="Denetim klasörü (varsayılan: reports/dataset_audit)")
    args = parser.parse_args()

    out_dir = ensure_audit_dir(args.out_dir)
    try:
        artifacts = load_artifacts(out_dir)
    except FileNotFoundError as exc:
        print(f"[HATA] {exc}")
        return 1
    registry = load_registry()
    targets, entries = load_mapping()
    classes = {n: {int(k): v for k, v in d["classes"].items()} for n, d in artifacts["inventory"].items()}
    problems = check_completeness(entries, classes)
    for p in problems:
        print(f"[UYARI] {p}")

    n = write_csv(out_dir / "candidate_class_mapping.csv", MAPPING_FIELDS, mapping_rows(targets, entries))
    text = build_report(artifacts, registry, targets, entries, problems)
    summary_file = out_dir / "followup_summary.json"
    if summary_file.is_file():  # eksik giderme çıktıları varsa özet bölüm eklenir
        from isg_vision.dataset_audit.common import read_json
        from isg_vision.dataset_audit.followup_reports import load_followup, main_report_addendum
        fa = load_followup(out_dir)
        text += main_report_addendum(fa, entries, read_json(summary_file))
        usage_file = out_dir / "usage_policy_summary.json"
        if usage_file.is_file():  # Aşama 2B özeti
            from isg_vision.dataset_audit.followup_reports import build_checklist_v2
            from isg_vision.dataset_audit.stage2b_reports import item_statuses, stage2b_addendum
            items, _ = build_checklist_v2(fa, registry, entries, targets)
            text += stage2b_addendum(read_json(usage_file), item_statuses(items))
    (out_dir / "dataset_audit_report.md").write_text(text, encoding="utf-8")
    print(f"Yazıldı: candidate_class_mapping.csv ({n} satır), dataset_audit_report.md -> {out_dir}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
