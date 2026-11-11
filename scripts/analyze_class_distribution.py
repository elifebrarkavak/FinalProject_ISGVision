#!/usr/bin/env python
"""Sınıf dağılımını (kutu sayısı, görüntü sayısı, bölümlere göre) ve kutu geometrisini analiz eder.

Her veri setinin KENDİ sınıf adları ve numaraları korunur (yeniden eşleme yapılmaz).
Oversampling/undersampling/augmentation uygulanmaz.

Örnek:
    python scripts/analyze_class_distribution.py
    python scripts/analyze_class_distribution.py --datasets mendeley_ppe --no-plots
Çıktılar: class_distribution.csv, class_box_stats.csv, class_distribution_summary.json, charts/*.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.dataset_audit.common import (  # noqa: E402
    dataset_raw_dir, ensure_audit_dir, load_registry, select_datasets, write_csv, write_json,
)
from isg_vision.dataset_audit.distribution import (  # noqa: E402
    BOXSTAT_FIELDS, DIST_FIELDS, compute_distribution, plot_distribution,
)
from isg_vision.dataset_audit.layout import discover_dataset  # noqa: E402
from isg_vision.dataset_audit.scan import scan_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true", help="Grafik üretme")
    args = parser.parse_args()

    registry = load_registry()
    out_dir = ensure_audit_dir(args.out_dir)
    all_rows, all_stats, summary = [], [], {}

    for name in select_datasets(registry, args.datasets):
        layout = discover_dataset(name, dataset_raw_dir(registry, name))
        if not layout.found:
            print(f"[UYARI] {name}: veri bulunamadı; atlandı. Önce scripts/extract_datasets.py çalıştırın.")
            continue
        rows, stats, excluded = compute_distribution(layout, scan_dataset(layout, full_decode=False))
        all_rows.extend(rows)
        all_stats.extend(stats)
        summary[name] = {"excluded": excluded}
        print(f"\n[{name}]  hariç tutulan: {excluded}")
        print(f"  {'ID':>3} {'sınıf':<16} {'kutu':>7} {'görüntü':>8}  " + "  ".join(
            f"{s:>5}" for s in layout.splits))
        for r in (x for x in rows if x["split"] == "ALL"):
            per = "  ".join(f"{next(y['boxes'] for y in rows if y['class_id'] == r['class_id'] and y['split'] == s):>5}"
                            for s in layout.splits)
            print(f"  {r['class_id']:>3} {r['class_name']:<16} {r['boxes']:>7} {r['images_with_class']:>8}  {per}")
        if not args.no_plots:
            plot_distribution(name, rows, out_dir / "charts" / f"class_distribution_{name}.png")

    write_csv(out_dir / "class_distribution.csv", DIST_FIELDS, all_rows)
    write_csv(out_dir / "class_box_stats.csv", BOXSTAT_FIELDS, all_stats)
    write_json(out_dir / "class_distribution_summary.json", summary)
    print(f"\nYazıldı: {out_dir / 'class_distribution.csv'}, class_box_stats.csv" +
          ("" if args.no_plots else ", charts/*.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
