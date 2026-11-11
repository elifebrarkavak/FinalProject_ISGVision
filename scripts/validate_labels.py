#!/usr/bin/env python
"""YOLO detection etiketlerini doğrular ve şüpheli durumları CSV'ye yazar.

Kontroller: etiket yok/boş, okunamayan dosya, sütun sayısı, sayısal değer, sınıf ID'si,
normalize aralık, pozitif kutu boyutu, görüntü sınırı, tekrarlı kutular, yetim etiket, bozuk görüntü,
desteklenmeyen biçim (segmentation/OBB), ve birbirini dışlaması gereken sınıfların (ör. Hardhat/NO-Hardhat)
aynı bölgede örtüşmesi (configs/datasets.yaml: contradiction_pairs).
Hiçbir dosya silinmez veya düzeltilmez. Boş/eksik etiket "info" düzeyindedir (negatif örnek olabilir).

Örnek:
    python scripts/validate_labels.py
    python scripts/validate_labels.py --datasets construction_ppe
Çıktılar: annotation_issues.csv, label_validation_summary.json
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.dataset_audit.common import (  # noqa: E402
    dataset_raw_dir, ensure_audit_dir, load_registry, rel, select_datasets, write_csv, write_json,
)
from isg_vision.dataset_audit.layout import discover_dataset  # noqa: E402
from isg_vision.dataset_audit.scan import scan_dataset  # noqa: E402

ISSUE_FIELDS = ["dataset", "split", "severity", "issue_code", "image_path", "label_path", "line_no", "detail"]
MAX_ROWS_PER_CODE_INFO = 100000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--full-decode", action="store_true", help="Görüntüleri tamamen çöz (yavaş, bozukları yakalar)")
    args = parser.parse_args()

    registry = load_registry()
    out_dir = ensure_audit_dir(args.out_dir)
    issue_rows: list[dict] = []
    summary: dict[str, dict] = {}

    for name in select_datasets(registry, args.datasets):
        layout = discover_dataset(name, dataset_raw_dir(registry, name))
        if not layout.found:
            print(f"[UYARI] {name}: veri bulunamadı; atlandı. Önce scripts/extract_datasets.py çalıştırın.")
            continue
        print(f"[{name}] doğrulanıyor ...")
        counts: Counter[str] = Counter()
        sev_counts: Counter[str] = Counter()
        files_with_error: set[str] = set()
        stats = defaultdict(int)
        pairs = [tuple(p) for p in registry[name].get("contradiction_pairs", [])]
        for rec in scan_dataset(layout, full_decode=args.full_decode, conflict_pairs=pairs):
            stats["samples"] += 1
            if rec.label is not None:
                stats["label_files_read"] += 1
                stats["boxes_parsed"] += len(rec.label.boxes)
            has_error = False
            for issue in rec.issues:
                counts[issue.code] += 1
                sev_counts[issue.severity] += 1
                if issue.severity == "error":
                    has_error = True
                issue_rows.append({
                    "dataset": name, "split": rec.sample.split, "severity": issue.severity,
                    "issue_code": issue.code,
                    "image_path": rel(rec.sample.image_path) if rec.sample.image_path else "",
                    "label_path": rel(rec.sample.label_path) if rec.sample.label_path else "",
                    "line_no": issue.line_no if issue.line_no is not None else "", "detail": issue.detail,
                })
            if has_error:
                files_with_error.add(str(rec.sample.label_path or rec.sample.image_path))
        summary[name] = {
            "samples": stats["samples"], "label_files_read": stats["label_files_read"],
            "boxes_parsed": stats["boxes_parsed"], "samples_with_error_level_issue": len(files_with_error),
            "issues_by_code": dict(sorted(counts.items())), "issues_by_severity": dict(sev_counts),
            "class_warnings": layout.classes.warnings,
        }
        print(f"   örnek={stats['samples']} kutu={stats['boxes_parsed']} "
              f"hata={sev_counts['error']} uyarı={sev_counts['warning']} bilgi={sev_counts['info']}")
        for code, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"     {code}: {n}")

    order = {"error": 0, "warning": 1, "info": 2}
    issue_rows.sort(key=lambda r: (order[r["severity"]], r["dataset"], r["issue_code"], r["image_path"]))
    n = write_csv(out_dir / "annotation_issues.csv", ISSUE_FIELDS, issue_rows)
    write_json(out_dir / "label_validation_summary.json", summary)
    print(f"\n{n} satır yazıldı: {out_dir / 'annotation_issues.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
