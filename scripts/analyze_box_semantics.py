#!/usr/bin/env python
"""Baret/yelek kutularının anlamını (neyi kapsadığını) ölçer ve çelişkili etiketleri listeler.

Ölçümler (configs/datasets.yaml ``roles`` alanındaki sınıf rolleriyle):
- kişi kutusuna göre göreli yükseklik/genişlik/dikey konum (kişi sınıfı olan veri setlerinde),
- tam bir baş + tam bir gövde kutusu olan görüntülerde baş/gövde oranları (tüm veri setleri için ortak ölçü),
- helmet/no_helmet ve vest/no_vest çiftlerinin aynı bölgede / aynı kişide çelişkili kullanımı.
Rol başına karşılaştırma sayfaları (--render) reports/dataset_audit/semantic_review/ altına yazılır (Git'e eklenmez).
Etiket dosyaları ve görüntüler DEĞİŞTİRİLMEZ.

Örnek:
    python scripts/analyze_box_semantics.py --render
Çıktılar: semantic_metrics.json, semantic_conflicts.csv, semantic_review/*
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.dataset_audit.common import (  # noqa: E402
    dataset_raw_dir, ensure_audit_dir, load_registry, rel, select_datasets, write_csv, write_json,
)
from isg_vision.dataset_audit.layout import discover_dataset  # noqa: E402
from isg_vision.dataset_audit.semantic import (  # noqa: E402
    box_aspect_stats, class_roles, classify_conflict, find_semantic_conflicts, iter_images,
    person_relative_stats, single_person_pair_stats,
)

CONFLICT_FIELDS = ["conflict_id", "dataset", "split", "attribute", "type", "pos_line", "neg_line", "person_line",
                   "iou", "center_inside", "auto_class", "image_path", "label_path"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--render", action="store_true", help="Karşılaştırma ve çelişki sayfalarını üret")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    registry = load_registry()
    out_dir = ensure_audit_dir(args.out_dir)
    metrics, all_conflicts, loaded, by_path = {}, [], {}, {}
    for name in select_datasets(registry, args.datasets):
        layout = discover_dataset(name, dataset_raw_dir(registry, name))
        if not layout.found:
            print(f"[UYARI] {name}: veri bulunamadı; atlandı.")
            continue
        roles = class_roles(registry[name])
        images = list(iter_images(layout, roles))
        loaded[name] = images
        metrics[name] = {"roles": {r: [c for c, rr in roles.items() if rr == r] for r in set(roles.values())},
                         "aspect_and_size": box_aspect_stats(images),
                         "person_relative": person_relative_stats(images),
                         "single_head_torso_pairs": single_person_pair_stats(images)}
        count = 0
        for im in images:
            by_path[rel(im.image_path)] = im
            for c in find_semantic_conflicts(im):
                count += 1
                all_conflicts.append({
                    "conflict_id": f"{name[:2].upper()}{len(all_conflicts) + 1:03d}", "dataset": name, "split": im.split,
                    "image_path": rel(im.image_path), "label_path": rel(im.label_path),
                    "auto_class": classify_conflict(c), **c})
        print(f"[{name}] rol kutusu olan görüntü: {len(images)}, çelişki adayı: {count} "
              f"{dict(Counter((c['attribute'], c['type']) for c in all_conflicts if c['dataset'] == name))}")

    write_json(out_dir / "semantic_metrics.json", metrics)
    write_csv(out_dir / "semantic_conflicts.csv", CONFLICT_FIELDS, all_conflicts)
    print(f"\nYazıldı: semantic_metrics.json, semantic_conflicts.csv ({len(all_conflicts)} satır)")

    if args.render:
        from isg_vision.dataset_audit.semantic_visual import render_conflict_sheets, render_role_comparison
        base = out_dir / "semantic_review"
        comp = {}
        for role in ("helmet", "no_helmet", "vest", "no_vest"):
            comp[role] = render_role_comparison(loaded, role, base / f"compare_{role}.jpg", seed=args.seed)
        write_json(out_dir / "semantic_comparison_sampling.json", comp)
        for name in loaded:
            items = [c for c in all_conflicts if c["dataset"] == name]
            if items:
                paths = render_conflict_sheets(items, by_path, base / f"conflicts_{name}")
                print(f"  {name}: {len(items)} çelişki -> {len(paths)} sayfa")
        print(f"Karşılaştırma sayfaları: {base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
