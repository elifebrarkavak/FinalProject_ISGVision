#!/usr/bin/env python
"""Veri setlerinin klasör yapısını, YAML tanımlarını ve sayısal envanterini çıkarır.

Ham veriye yalnızca OKUMA yapar. Görüntüler tamamen çözülerek bozuk dosyalar tespit edilir.
Gerçek dosyalar sayılır; yayımlanan sayılar (configs/datasets.yaml) yalnızca karşılaştırma içindir.

Örnek:
    python scripts/inspect_datasets.py
    python scripts/inspect_datasets.py --datasets site_safety --out-dir /tmp/audit
Çıktılar: dataset_inventory.csv, dataset_inventory.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.dataset_audit.common import (  # noqa: E402
    dataset_raw_dir, ensure_audit_dir, human_size, load_registry, select_datasets, write_csv, write_json,
)
from isg_vision.dataset_audit.inventory import INVENTORY_FIELDS, build_inventory  # noqa: E402
from isg_vision.dataset_audit.layout import discover_dataset  # noqa: E402
from isg_vision.dataset_audit.scan import scan_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="*", help="Yalnızca bu veri setleri (varsayılan: hepsi)")
    parser.add_argument("--out-dir", type=Path, help="Çıktı klasörü (varsayılan: reports/dataset_audit)")
    parser.add_argument("--header-only", action="store_true",
                        help="Görüntüleri tamamen çözme, yalnızca başlık oku (hızlı; bozuk dosyaları kaçırabilir)")
    args = parser.parse_args()

    registry = load_registry()
    out_dir = ensure_audit_dir(args.out_dir)
    all_rows, details, missing = [], [], []

    for name in select_datasets(registry, args.datasets):
        raw = dataset_raw_dir(registry, name)
        layout = discover_dataset(name, raw)
        if not layout.found:
            print(f"[UYARI] {name}: veri bulunamadı ({'; '.join(layout.notes)}). Önce extract_datasets.py çalıştırın "
                  f"veya ZIP içeriğini {registry[name]['raw_dir']}/ altına koyun.")
            missing.append(name)
            continue
        print(f"[{name}] taranıyor ... ({', '.join(layout.splits)})")
        rows, detail = build_inventory(
            layout, scan_dataset(layout, full_decode=not args.header_only, read_labels=True,
                                 progress=lambda n, c: print(f"   {n}: {c} örnek")))
        detail["published"] = registry[name].get("published")
        pub = registry[name].get("published") or {}
        total = rows[-1]
        diffs = []
        if pub.get("images") is not None:
            diffs.append(f"görüntü: gerçek {total['images']} / yayımlanan {pub['images']}")
        if pub.get("boxes") is not None:
            diffs.append(f"kutu: gerçek {total['boxes']} / yayımlanan {pub['boxes']}")
        detail["published_vs_actual"] = diffs
        all_rows.extend(rows)
        details.append(detail)
        print(f"   görüntü={total['images']} etiket={total['label_files']} kutu={total['boxes']} "
              f"sınıf={total['classes_defined']} bozuk_görüntü={total['unreadable_images']} "
              f"boyut={human_size(detail['total_size_bytes_on_disk'])}")
        for d in diffs:
            print(f"   {d}")

    write_csv(out_dir / "dataset_inventory.csv", INVENTORY_FIELDS, all_rows)
    write_json(out_dir / "dataset_inventory.json", details)
    print(f"\nYazıldı: {out_dir / 'dataset_inventory.csv'} ve dataset_inventory.json")
    if missing:
        print(f"Eksik veri setleri: {missing}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
