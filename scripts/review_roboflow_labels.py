#!/usr/bin/env python
"""Roboflow (site_safety) şüpheli etiketlerini inceleme sayfalarına döker ve inceleme kararlarını tek tabloda birleştirir.

--render: küçük kutular, boş etiketler, kopya kutular ve rastgele kalite örneklemi için numaralı sayfalar üretir
          (reports/dataset_audit/roboflow_review/, Git'e eklenmez). Etiket/görüntü dosyalarına DOKUNULMAZ.
Kararlar (Confirmed Error / Likely Error / Valid Annotation / Needs Manual Review) elle yapılan görsel incelemenin
kaydıdır: configs/review/label_review_verdicts.yaml. Bu betik onları rapora (label_review.csv) birleştirir;
karar eksik olan her kayıt "Needs Manual Review (karar kaydı yok)" olarak işaretlenir.

Örnek:
    python scripts/review_roboflow_labels.py --render
    python scripts/review_roboflow_labels.py
Çıktılar: label_review.csv, label_review_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.dataset_audit.common import (  # noqa: E402
    dataset_raw_dir, ensure_audit_dir, load_registry, read_csv, rel, write_csv, write_json,
)
from isg_vision.dataset_audit.labels import Box, parse_label_file  # noqa: E402
from isg_vision.dataset_audit.layout import discover_dataset  # noqa: E402
from isg_vision.dataset_audit.review import (  # noqa: E402
    CATEGORIES, REVIEW_FIELDS, build_review_rows, expand_verdicts, wilson_interval,
)
from isg_vision.dataset_audit.semantic import RoleBox, class_roles, iter_images  # noqa: E402

VERDICT_FILE = Path(__file__).resolve().parents[1] / "configs" / "review" / "label_review_verdicts.yaml"
DATASET = "site_safety"


def render(audit_dir: Path, registry: dict) -> None:
    from isg_vision.dataset_audit.semantic_visual import render_labeled_boxes, render_role_audit_sheet
    layout = discover_dataset(DATASET, dataset_raw_dir(registry, DATASET))
    base = audit_dir / "roboflow_review"
    issues = [r for r in read_csv(audit_dir / "annotation_issues.csv") if r["dataset"] == DATASET]
    root = Path(__file__).resolve().parents[1]

    def entries(code: str, focus: bool) -> list[dict]:
        out = []
        for i, r in enumerate([x for x in issues if x["issue_code"] == code], start=1):
            boxes = parse_label_file(root / r["label_path"]).boxes
            line = int(r["line_no"]) if r["line_no"] else None
            sel = [b for b in boxes if b.line_no == line] if focus and line else []
            out.append({"image_path": root / r["image_path"], "note": f"{code[:3]}{i:02d} {r['split']} L{line or '-'}",
                        "focus": [RoleBox("other", b) for b in sel] or None})
        return out

    render_labeled_boxes(entries("BOX_VERY_SMALL", True), base / "tiny_boxes.jpg", tile=220, cols=7)
    empties = entries("LABEL_EMPTY", False)
    render_labeled_boxes(empties[:12], base / "empty_labels_01.jpg", tile=300, cols=4)
    render_labeled_boxes(empties[12:], base / "empty_labels_02.jpg", tile=300, cols=4)
    dups = entries("DUPLICATE_BOX", True) + entries("NEAR_DUPLICATE_BOX", True)
    render_labeled_boxes(dups, base / "duplicate_boxes.jpg", tile=260, cols=5)
    roles = class_roles(registry[DATASET])
    images = list(iter_images(layout, roles))
    listing = []
    for role in ("helmet", "no_helmet", "vest", "no_vest"):
        listing += render_role_audit_sheet(images, role, base / f"audit_{role}.jpg", n=24, seed=7)
    write_csv(audit_dir / "roboflow_audit_sample.csv", ["tile", "dataset", "split", "role", "image_path", "line_no"],
              [{**r, "image_path": rel(r["image_path"])} for r in listing])
    print(f"Sayfalar yazıldı: {base}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    audit_dir = ensure_audit_dir(args.out_dir)
    registry = load_registry()
    if args.render:
        render(audit_dir, registry)
    verdicts = yaml.safe_load(VERDICT_FILE.read_text(encoding="utf-8")) if VERDICT_FILE.is_file() else {}
    rows = build_review_rows(audit_dir, expand_verdicts(verdicts))
    write_csv(audit_dir / "label_review.csv", REVIEW_FIELDS, rows)
    counts = Counter(r["category"] for r in rows)
    by_source = {s: dict(Counter(r["category"] for r in rows if r["source"] == s)) for s in sorted({r["source"] for r in rows})}
    audit = [r for r in rows if r["source"].startswith("audit_")]
    n_err = sum(1 for r in audit if r["category"] in ("Confirmed Error", "Likely Error"))
    lo, hi = wilson_interval(n_err, len(audit)) if audit else (None, None)
    by_dataset = {d: dict(Counter(r["category"] for r in rows if r["dataset"] == d)) for d in sorted({r["dataset"] for r in rows})}
    summary = {"total_items": len(rows), "by_category": dict(counts), "by_dataset": by_dataset, "by_source": by_source,
               "audit_sample": {"n": len(audit), "error_like": n_err, "wilson95": [lo, hi]}}
    write_json(audit_dir / "label_review_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
