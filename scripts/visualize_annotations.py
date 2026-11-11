#!/usr/bin/env python
"""Etiketleri gerçek görüntüler üzerinde çizerek manuel (görsel) doğrulama örnekleri üretir.

Her veri setinden sınıfları ve bölümleri temsil eden örnek görüntüler seçilir (sabit seed ile),
kutular ve GERÇEK sınıf adları çizilir. Çıktılar:
  reports/dataset_audit/visual_samples/<veri_seti>/images/*.jpg      tek tek örnekler
  reports/dataset_audit/visual_samples/<veri_seti>/contact_sheet_*.jpg  contact sheet
  reports/dataset_audit/visual_samples/<veri_seti>/crops/class_*.jpg     sınıf başına kutu kırpmaları
  reports/dataset_audit/visual_samples/<veri_seti>/selection.csv          seçilen dosyalar
Orijinal görüntüler değiştirilmez. Çıktılar Git'e eklenmez (.gitignore + pre-commit hook).

Örnek:
    python scripts/visualize_annotations.py
    python scripts/visualize_annotations.py --datasets site_safety --num-samples 50 --seed 1
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.dataset_audit.common import (  # noqa: E402
    dataset_raw_dir, ensure_audit_dir, load_registry, select_datasets, write_csv,
)
from isg_vision.dataset_audit.layout import discover_dataset  # noqa: E402
from isg_vision.dataset_audit.visualize import render_class_crops, render_dataset_samples  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--out-dir", type=Path, help="Denetim çıktı klasörü (varsayılan: reports/dataset_audit)")
    parser.add_argument("--num-samples", type=int, default=40, help="Veri seti başına görüntü sayısı (30-50 önerilir)")
    parser.add_argument("--crops-per-class", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    registry = load_registry()
    base = ensure_audit_dir(args.out_dir) / "visual_samples"
    for name in select_datasets(registry, args.datasets):
        layout = discover_dataset(name, dataset_raw_dir(registry, name))
        if not layout.found:
            print(f"[UYARI] {name}: veri bulunamadı; atlandı. Önce scripts/extract_datasets.py çalıştırın.")
            continue
        out = base / name
        if out.exists():  # yalnızca BU betiğin ürettiği çıktı klasörü yenilenir (ham veriye dokunulmaz)
            shutil.rmtree(out)
        print(f"[{name}] örnekler çiziliyor (seed={args.seed}) ...")
        records = render_dataset_samples(layout, out, n=args.num_samples, seed=args.seed)
        counts = render_class_crops(layout, out, per_class=args.crops_per_class, seed=args.seed)
        write_csv(out / "selection.csv", ["split", "file", "reason", "status", "n_boxes", "classes"], records)
        ok = sum(1 for r in records if r["status"] == "ok")
        print(f"   {ok}/{len(records)} görüntü çizildi; sınıf kırpma sayfaları: {len(counts)}  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
