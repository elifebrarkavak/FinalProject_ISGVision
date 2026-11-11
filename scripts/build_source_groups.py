#!/usr/bin/env python
"""Kaynak grupları (veri sızıntısı grafiği) ve bölme birimlerini çıkarır; mevcut bölünmeleri DEĞİŞTİRMEZ.

1) duplicate_verification.csv kararlarından sert (kanıtlı) ve yumuşak (sınırda) kenarlar çıkarılır.
2) Kolaj görüntüleri için başarısız/sınırdaki çiftlere karo düzeyinde ORB doğrulaması uygulanır (tile_match).
3) configs/review/stage2b_decisions.yaml içindeki görsel kaynak-ilişkisi kararları (Confirmed/Likely Same Source) eklenir.
4) Bağlantılı bileşenler: sert grup (hard_group) ve bölme birimi (split_unit = sert + yumuşak).
Çıktılar: source_group_edges.csv, source_group_manifest.csv, source_group_summary.json
Dosya adı öneklerinden kesin kaynak eşleşmesi üretilmez.

Örnek:
    python scripts/build_source_groups.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.dataset_audit.common import (  # noqa: E402
    dataset_raw_dir, ensure_audit_dir, load_registry, read_csv, rel, write_csv, write_json,
)
from isg_vision.dataset_audit.duplicates import base_name  # noqa: E402
from isg_vision.dataset_audit.duplicates_verify import OrbCache, load_gray, tile_level_verify, tile_match  # noqa: E402
from isg_vision.dataset_audit.layout import SPLIT_ORDER, discover_dataset, iter_samples  # noqa: E402
from isg_vision.dataset_audit.source_groups import (  # noqa: E402
    SOFT_DECISIONS, STRONG_DECISIONS, build_groups,
)
from isg_vision.utils.paths import PROJECT_ROOT  # noqa: E402

EDGE_FIELDS = ["path_a", "path_b", "dataset_a", "split_a", "dataset_b", "split_b", "edge", "evidence_type", "detail"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--no-tile-level", action="store_true", help="Karo düzeyinde doğrulamayı atla (hızlı)")
    args = parser.parse_args()
    audit = ensure_audit_dir(args.out_dir)
    registry = load_registry()
    decisions = yaml.safe_load((PROJECT_ROOT / "configs/review/stage2b_decisions.yaml").read_text(encoding="utf-8"))

    meta: dict[str, dict] = {}
    for name in registry:
        layout = discover_dataset(name, dataset_raw_dir(registry, name))
        for split in SPLIT_ORDER:
            sd = layout.splits.get(split)
            if sd is None:
                continue
            for s in iter_samples(sd):
                if s.image_path is not None:
                    meta[rel(s.image_path)] = {"dataset": name, "split": split}
    dv = read_csv(audit / "duplicate_verification.csv")
    edges: dict[tuple[str, str], dict] = {}

    def add(a: str, b: str, kind: str, etype: str, detail: str) -> None:
        key = (min(a, b), max(a, b))
        cur = edges.get(key)
        if cur is None or (kind == "hard" and cur["edge"] == "soft"):
            edges[key] = {"path_a": key[0], "path_b": key[1], "dataset_a": meta[key[0]]["dataset"], "split_a": meta[key[0]]["split"],
                          "dataset_b": meta[key[1]]["dataset"], "split_b": meta[key[1]]["split"], "edge": kind,
                          "evidence_type": etype, "detail": detail}

    retry = []
    for r in dv:
        a, b = r["path_a"], r["path_b"]
        if r["decision"] in STRONG_DECISIONS:
            add(a, b, "hard", f"orb:{r['decision']}", f"inlier={r['orb_inliers']} ncc={r['aligned_ncc']}")
        elif r["decision"] in SOFT_DECISIONS:
            add(a, b, "soft", "orb:uncertain", f"inlier={r['orb_inliers']}")
            retry.append(r)
        else:
            retry.append(r)
    # karo düzeyinde doğrulama: bölümler/veri setleri arası, hash adayı olup geometrik karar vermeyen çiftler
    n_tile = 0
    if not args.no_tile_level:
        cache = OrbCache()
        for r in retry:
            if r["relation"] == "within_dataset_same_split":
                continue
            ga, gb = load_gray(PROJECT_ROOT / r["path_a"], 640), load_gray(PROJECT_ROOT / r["path_b"], 640)
            if ga is None or gb is None:
                continue
            res = {}
            for swap in (False, True):  # her iki yönde kolaj aranır
                coll, single = (gb, ga) if swap else (ga, gb)
                v = tile_level_verify(coll, single, cache, key=("t", r["pair_id"], swap))
                if not res or v["inliers"] > res["inliers"]:
                    res = v
            cache.store.clear()
            n_tile += 1
            if tile_match(res):
                add(r["path_a"], r["path_b"], "hard", "tile_level_orb", f"pencere={res['window']} inlier={res['inliers']} oran={res['inlier_ratio']} ncc={res['ncc']}")
    # görsel kaynak-ilişkisi kararları (D07/D08): yalnızca bölümler arası, aynı kaynak adı grubundaki çiftler
    by_base: dict[str, list[str]] = {}
    for p in meta:
        by_base.setdefault(base_name(p), []).append(p)
    for d in decisions["source_relation_decisions"]:
        if d["edge"] == "none":
            continue
        members = by_base.get(d["group"], [])
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if meta[a]["dataset"] == meta[b]["dataset"] and meta[a]["split"] != meta[b]["split"]:
                    add(a, b, d["edge"], f"visual_decision:{d['decision']}", d["group"])

    hard = [(k[0], k[1]) for k, e in edges.items() if e["edge"] == "hard"]
    soft = [(k[0], k[1]) for k, e in edges.items() if e["edge"] == "soft"]
    groups = build_groups(meta, hard, soft)
    cross_hard, cross_unit = Counter(), Counter()
    splits_by_hard: dict[str, set] = {}
    splits_by_unit: dict[str, set] = {}
    for p, g in groups.items():
        splits_by_hard.setdefault(g["hard_group"], set()).add((meta[p]["dataset"], meta[p]["split"]))
        splits_by_unit.setdefault(g["split_unit"], set()).add((meta[p]["dataset"], meta[p]["split"]))
    rows = []
    for p in sorted(meta):
        g = groups[p]
        n_hard = sum(1 for k, e in edges.items() if p in k and e["edge"] == "hard")
        n_soft = sum(1 for k, e in edges.items() if p in k and e["edge"] == "soft")
        hard_sp, unit_sp = splits_by_hard[g["hard_group"]], splits_by_unit[g["split_unit"]]
        rows.append({"image_path": p, "dataset": meta[p]["dataset"], "current_split": meta[p]["split"],
                     "hard_group": g["hard_group"], "hard_group_size": g["hard_size"], "split_unit": g["split_unit"],
                     "split_unit_size": g["unit_size"], "n_hard_links": n_hard, "n_soft_links": n_soft,
                     "group_spans_splits": "yes" if len({s for _, s in hard_sp}) > 1 else "no",
                     "unit_spans_splits": "yes" if len({s for _, s in unit_sp}) > 1 else "no",
                     "unit_spans_datasets": "yes" if len({d for d, _ in unit_sp}) > 1 else "no",
                     "relation_status": ("linked_hard" if n_hard else ("linked_soft_uncertain" if n_soft else "no_known_link")),
                     "note": "kaynak ilişkisi yalnızca bilinen kenarlardan (alt sınır); bağlantı yok = ilişkisiz DEĞİL, bilinmiyor"})
    write_csv(audit / "source_group_manifest.csv", list(rows[0].keys()), rows)
    write_csv(audit / "source_group_edges.csv", EDGE_FIELDS, sorted(edges.values(), key=lambda e: (e["path_a"], e["path_b"])))
    by_ds = {}
    for ds in registry:
        sel = [r for r in rows if r["dataset"] == ds]
        by_ds[ds] = {"images": len(sel), "linked_hard": sum(r["relation_status"] == "linked_hard" for r in sel),
                     "linked_soft_only": sum(r["relation_status"] == "linked_soft_uncertain" for r in sel),
                     "images_in_cross_split_hard_groups": sum(r["group_spans_splits"] == "yes" for r in sel),
                     "images_in_cross_split_units": sum(r["unit_spans_splits"] == "yes" for r in sel),
                     "images_in_cross_dataset_units": sum(r["unit_spans_datasets"] == "yes" for r in sel),
                     "by_split_in_cross_split_hard_groups": dict(Counter(r["current_split"] for r in sel if r["group_spans_splits"] == "yes")),
                     "largest_unit": max((r["split_unit_size"] for r in sel), default=0)}
    sizes = Counter(g["unit_size"] for g in {v["split_unit"]: v for v in groups.values()}.values())
    write_json(audit / "source_group_summary.json", {
        "images": len(meta), "hard_edges": len(hard), "soft_edges": len(soft), "tile_level_pairs_checked": n_tile,
        "hard_groups": len({g["hard_group"] for g in groups.values()}), "split_units": len({g["split_unit"] for g in groups.values()}),
        "unit_size_histogram": dict(sorted(sizes.items())), "by_dataset": by_ds,
        "caveat": "Kenarlar aday üretimi + ORB doğrulaması ile bulunan ALT SINIRDIR; bağlantısız görüntüler ilişkisiz kanıtlı DEĞİLDİR."})
    print({k: v for k, v in by_ds.items()})
    print("hard edges", len(hard), "soft", len(soft), "tile-level pairs", n_tile, "units", len({g['split_unit'] for g in groups.values()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
