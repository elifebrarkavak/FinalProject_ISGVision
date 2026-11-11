#!/usr/bin/env python
"""Tekrarlanan ve yakın kopya görüntüleri tespit eder (veri sızıntısı riski için).

1) Birebir aynı dosyalar: SHA-256.
2) Yakın kopyalar: 64-bit dHash (Hamming mesafesi <= --threshold); yatay çevrilmiş kopyalar da aranır.
Yakın kopya eşleşmeleri KESİN aynı görüntü değildir, yalnızca inceleme adayıdır.
Aynı veri seti içi, bölümler arası (train/val/test) ve veri setleri arası ilişkiler ayrı raporlanır.
Hiçbir görüntü silinmez veya değiştirilmez.

Örnek:
    python scripts/detect_duplicates.py
    python scripts/detect_duplicates.py --threshold 8
Çıktılar: duplicate_images.csv, duplicate_summary.json, source_name_overlap.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.dataset_audit.common import (  # noqa: E402
    dataset_raw_dir, ensure_audit_dir, load_registry, rel, select_datasets, write_csv, write_json,
)
from isg_vision.dataset_audit.duplicates import (  # noqa: E402
    DUP_FIELDS, find_exact, find_name_overlaps, find_near, index_dataset, summarize,
)
from isg_vision.dataset_audit.layout import discover_dataset  # noqa: E402


def verify_manifest(layout, entries) -> dict | None:
    """Ultralytics manifest.jsonl varsa SHA-256, bölüm ve kutu sayısını yerel dosyalarla karşılaştırır."""
    manifest = layout.root / "manifest.jsonl" if layout.root else None
    if manifest is None or not manifest.is_file():
        return None
    by_name = {Path(e.path).name: e for e in entries}
    result = {"entries": 0, "sha_mismatch": 0, "split_mismatch": 0, "missing_locally": 0,
              "box_count_mismatch": 0, "label_unchecked": 0}
    from isg_vision.dataset_audit.labels import parse_label_file
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        result["entries"] += 1
        entry = by_name.get(item["file"])
        if entry is None:
            result["missing_locally"] += 1
            continue
        if entry.sha256 != item.get("sha256"):
            result["sha_mismatch"] += 1
        if entry.split != item.get("split"):
            result["split_mismatch"] += 1
        label = layout.splits[entry.split].labels_dir / (Path(item["file"]).stem + ".txt")
        if label.is_file() and "annotations" in item:
            if len(parse_label_file(label).boxes) != item["annotations"]:
                result["box_count_mismatch"] += 1
        else:
            result["label_unchecked"] += 1
    result["only_in_local_not_in_manifest"] = len(entries) - result["entries"] + result["missing_locally"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--threshold", type=int, default=5, help="dHash Hamming eşiği (varsayılan 5/64)")
    parser.add_argument("--max-near-pairs", type=int, default=200_000, help="CSV'ye yazılacak en fazla yakın çift")
    args = parser.parse_args()

    registry = load_registry()
    out_dir = ensure_audit_dir(args.out_dir)
    entries, unreadable, layouts, manifest_checks = [], [], {}, {}

    for name in select_datasets(registry, args.datasets):
        layout = discover_dataset(name, dataset_raw_dir(registry, name))
        if not layout.found:
            print(f"[UYARI] {name}: veri bulunamadı; atlandı. Önce scripts/extract_datasets.py çalıştırın.")
            continue
        print(f"[{name}] hash'leniyor ...")
        found, bad = index_dataset(layout, rel)
        entries.extend(found)
        unreadable.extend(bad)
        layouts[name] = layout
        check = verify_manifest(layout, found)
        if check:
            manifest_checks[name] = check
            print(f"   manifest.jsonl kontrolü: {check}")

    print(f"{len(entries)} görüntü hash'lendi; çiftler aranıyor ...")
    exact = find_exact(entries)
    scan_bits = max(args.threshold, 12)  # duyarlılık analizi için daha geniş eşikle tara
    near_all, _ = find_near(entries, scan_bits, 10**9)
    sensitivity = {f"<={t}": sum(1 for r in near_all if r["distance"] <= t) for t in (0, 2, 5, 8, 10, 12) if t <= scan_bits}
    near_all = [r for r in near_all if r["distance"] <= args.threshold]
    truncated = len(near_all) > args.max_near_pairs
    near = near_all[:args.max_near_pairs]
    rows = exact + near
    write_csv(out_dir / "duplicate_images.csv", DUP_FIELDS, rows)
    name_rows, name_stats = find_name_overlaps(entries)
    write_csv(out_dir / "source_name_overlap.csv", ["base_name", "kind", "n_images", "datasets_splits", "paths"], name_rows)
    summary = {"images_hashed": len(entries), "unreadable_images": unreadable,
               "near_threshold_bits_of_64": args.threshold,
               "near_pair_count_by_threshold": sensitivity, "near_pairs_truncated": truncated,
               "per_dataset_images": {n: sum(1 for e in entries if e.dataset == n) for n in layouts},
               "source_name_overlap_groups": name_stats,
               "matches": summarize(rows), "manifest_checks": manifest_checks}
    write_json(out_dir / "duplicate_summary.json", summary)

    print(f"\nBirebir (SHA-256) çift: {len(exact)}   yakın/çevrilmiş yakın çift: {len(near)}"
          f"{' (KESİLDİ)' if truncated else ''}")
    for mt, info in summary["matches"].items():
        print(f"  {mt}: {info['total_pairs']} çift, {info['distinct_images_involved']} görüntü")
        for key, n in info["pairs"].items():
            print(f"     {key}: {n}")
    print(f"Aynı kaynak adını paylaşan gruplar (ipucu): {name_stats}")
    print(f"\nYazıldı: {out_dir / 'duplicate_images.csv'}, duplicate_summary.json, source_name_overlap.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
