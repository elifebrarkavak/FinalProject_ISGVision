#!/usr/bin/env python
"""Yakın kopya adaylarını çok yöntemli (SHA-256, dHash, çok pencereli pHash, ORB-LSH, ORB+RANSAC, hizalı NCC) doğrular.

Aşama 1: aday üretimi (dHash/pHash, kırpma/dönme/çevirmeye dayanıklı ORB-LSH oyları, aynı kaynak adı).
Aşama 2: her aday çifti geometrik olarak doğrulanır (afin RANSAC inlier'ları + hizalanmış NCC); karar yalnızca hash
         mesafesine dayanmaz. Eşikler gerçek görüntülerden üretilen bozulmuş kopyalar ve rastgele çiftlerle denenir.
Hiçbir görüntü silinmez/değiştirilmez; bölünmeler değişmez. Yan yana karşılaştırma görüntüleri
reports/dataset_audit/duplicate_verification/ altına yazılır (Git'e eklenmez).

Örnek:
    python scripts/verify_duplicates.py
    python scripts/verify_duplicates.py --no-render --min-votes 12
Çıktılar: duplicate_verification.csv, duplicate_verification_summary.json, duplicate_method_validation.json
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.dataset_audit.common import (  # noqa: E402
    dataset_raw_dir, ensure_audit_dir, load_registry, rel, write_csv, write_json,
)
from isg_vision.dataset_audit.duplicates import base_name  # noqa: E402
from isg_vision.dataset_audit.duplicates_verify import (  # noqa: E402
    DEFAULT_THRESHOLDS, OrbCache, apply_visual_checks, candidate_pairs, decide, hash_distance, index_descriptors, leakage_risk,
    load_gray, lsh_votes_with_flip, orb_verify, signature, validate_method,
)
from isg_vision.dataset_audit.layout import SPLIT_ORDER, discover_dataset, iter_samples  # noqa: E402

FIELDS = ["pair_id", "dataset_a", "split_a", "path_a", "dataset_b", "split_b", "path_b", "relation", "methods_flagged",
          "dhash_bits", "phash_bits", "lsh_votes", "lsh_votes_flipped", "orb_good_matches", "orb_inliers",
          "orb_inlier_ratio", "aligned_ncc", "scale", "rotation_deg", "anisotropy", "flipped", "decision", "leakage_risk",
          "name_overlap", "visual_check"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while b := fh.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def relation(a: dict, b: dict) -> str:
    if a["dataset"] != b["dataset"]:
        return "cross_dataset"
    return "within_dataset_same_split" if a["split"] == b["split"] else "within_dataset_cross_split"


def side_by_side(a: Path, b: Path, out: Path, caption: str) -> None:
    ia, ib = cv2.imread(str(a)), cv2.imread(str(b))
    if ia is None or ib is None:
        return
    h = 360
    ia = cv2.resize(ia, (int(ia.shape[1] * h / ia.shape[0]), h))
    ib = cv2.resize(ib, (int(ib.shape[1] * h / ib.shape[0]), h))
    bar = np.full((24, ia.shape[1] + ib.shape[1] + 6, 3), 40, np.uint8)
    cv2.putText(bar, caption[:110], (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([bar, np.hstack([ia, np.full((h, 6, 3), 255, np.uint8), ib])]),
                [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--min-votes", type=int, default=15, help="ORB-LSH aday oyu eşiği")
    parser.add_argument("--hash-bits", type=int, default=DEFAULT_THRESHOLDS["candidate_hash_bits"])
    parser.add_argument("--sibling-sample", type=int, default=400, help="Aynı bölüm içi aynı-kaynak-adı çiftlerinden örnek sayısı")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    audit = ensure_audit_dir(args.out_dir)
    registry = load_registry()
    items: list[dict] = []
    for name in registry:
        layout = discover_dataset(name, dataset_raw_dir(registry, name))
        for split in SPLIT_ORDER:
            sd = layout.splits.get(split)
            if sd is None:
                continue
            for s in iter_samples(sd):
                if s.image_path is not None:
                    items.append({"dataset": name, "split": split, "path": s.image_path, "rel": rel(s.image_path)})
    n = len(items)
    print(f"{n} görüntü yükleniyor ...")
    t0 = time.time()
    grays = [load_gray(it["path"]) for it in items]
    bad = [items[i]["rel"] for i, g in enumerate(grays) if g is None]
    if bad:
        print(f"[UYARI] okunamayan görüntü: {len(bad)} (atlanıyor)")
    sigs = [signature(g) if g is not None else None for g in grays]
    print(f"  imzalar hazır ({time.time() - t0:.0f} sn); ORB-LSH indeksi kuruluyor ...")
    descs = [index_descriptors(g) if g is not None else None for g in grays]
    descs_f = [index_descriptors(np.ascontiguousarray(g[:, ::-1])) if g is not None else None for g in grays]
    votes = lsh_votes_with_flip(descs, descs_f)
    print(f"  LSH oy alan çift: {len(votes)}")

    # ---- aday birleştirme
    cand: dict[tuple[int, int], dict] = {}

    def add(i: int, j: int, **kw) -> None:
        key = (min(i, j), max(i, j))
        c = cand.setdefault(key, {"methods": set()})
        c["methods"].add(kw.pop("method"))
        c.update(kw)

    valid = [i for i in range(n) if sigs[i] is not None]
    vs = [sigs[i] for i in valid]
    for a, b, d, how in candidate_pairs(vs, args.hash_bits):
        add(valid[a], valid[b], method="phash_dhash_multicrop", hash_bits=d)
    for (i, j), (v, vf) in votes.items():
        if max(v, vf) >= args.min_votes:
            add(i, j, method="orb_lsh_votes")
    groups: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        groups[base_name(it["rel"])].append(i)
    rng = random.Random(args.seed)
    same_split_siblings = []
    for base, members in groups.items():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                i, j = members[x], members[y]
                if relation(items[i], items[j]) == "within_dataset_same_split":
                    same_split_siblings.append((i, j))
                else:
                    add(i, j, method="name_overlap")
    sib_sample = rng.sample(same_split_siblings, min(args.sibling_sample, len(same_split_siblings)))
    sib_set = set()
    for i, j in sib_sample:
        key = (min(i, j), max(i, j))
        sib_set.add(key)
        add(i, j, method="name_overlap_sibling_sample")
    # SHA-256 yalnızca adaylar için hesaplanır (tüm veri için hash'leme önceki betikte yapıldı: 0 birebir kopya)
    print(f"  doğrulanacak aday çift: {len(cand)}")

    # ---- geometrik doğrulama
    cache = OrbCache()
    rows = []
    for k, ((i, j), c) in enumerate(sorted(cand.items()), start=1):
        a, b = items[i], items[j]
        if grays[i] is None or grays[j] is None:
            continue
        v = orb_verify(grays[i], grays[j], cache, i, j)
        d_hash, _ = hash_distance(sigs[i], sigs[j])
        sha_equal = sha256_file(a["path"]) == sha256_file(b["path"]) if "orb_lsh_votes" in c["methods"] or d_hash == 0 else False
        decision = "exact_duplicate" if sha_equal else decide(v)
        rel_ = relation(a, b)
        vv = votes.get((i, j), (0, 0))
        rows.append({
            "pair_id": f"DV{k:04d}", "dataset_a": a["dataset"], "split_a": a["split"], "path_a": a["rel"],
            "dataset_b": b["dataset"], "split_b": b["split"], "path_b": b["rel"], "relation": rel_,
            "methods_flagged": "+".join(sorted(c["methods"])) + ("+orb_ransac_affine+aligned_ncc"),
            "dhash_bits": min(bin(sigs[i]["d"] ^ sigs[j]["d"]).count("1"), bin(sigs[i]["d"] ^ sigs[j]["dflip"]).count("1")),
            "phash_bits": d_hash, "lsh_votes": vv[0], "lsh_votes_flipped": vv[1],
            "orb_good_matches": v["good_matches"], "orb_inliers": v["inliers"], "orb_inlier_ratio": v["inlier_ratio"],
            "aligned_ncc": "" if v["ncc"] is None else v["ncc"], "scale": "" if v["scale"] is None else v["scale"],
            "rotation_deg": "" if v["rotation_deg"] is None else v["rotation_deg"], "anisotropy": "" if v["anisotropy"] is None else v["anisotropy"], "flipped": v["flipped"],
            "decision": decision, "leakage_risk": leakage_risk(decision, rel_, a["split"], b["split"]),
            "name_overlap": "yes" if base_name(a["rel"]) == base_name(b["rel"]) else "no", "visual_check": "",
            "_sibling": (min(i, j), max(i, j)) in sib_set,
        })
        if k % 200 == 0:
            print(f"   {k}/{len(cand)} doğrulandı")
            cache.store.clear()  # bellek

    # ---- sibling (gerçek augmentation) üzerinde aday üretici geri çağırma ölçümü
    sib_rows = [r for r in rows if r["_sibling"]]
    sib_eval = {"n": len(sib_rows),
                "decided_geometric_match": sum(r["decision"] in ("same_image_derived", "same_scene_different_frame") for r in sib_rows),
                "dhash_le5": sum(int(r["dhash_bits"]) <= 5 for r in sib_rows),
                "phash_multicrop_le10": sum(int(r["phash_bits"]) <= args.hash_bits for r in sib_rows),
                "lsh_votes_ge_min": sum(max(int(r["lsh_votes"]), int(r["lsh_votes_flipped"])) >= args.min_votes for r in sib_rows)}
    for r in rows:
        r.pop("_sibling", None)

    import yaml
    checks_file = Path(__file__).resolve().parents[1] / "configs" / "review" / "duplicate_visual_checks.yaml"
    n_applied = apply_visual_checks(rows, yaml.safe_load(checks_file.read_text(encoding="utf-8"))) if checks_file.is_file() else 0
    print(f"Görsel inceleme kaydı uygulanan çift: {n_applied}")
    write_csv(audit / "duplicate_verification.csv", FIELDS, rows)
    dec = Counter(r["decision"] for r in rows)
    by_rel = {rl: dict(Counter(r["decision"] for r in rows if r["relation"] == rl)) for rl in sorted({r["relation"] for r in rows})}
    risk = Counter(r["leakage_risk"].split(" ")[0] for r in rows if r["decision"] != "not_duplicate")
    # ---- yöntem doğrulama deneyi (gerçek görüntülerden üretilen bozulmuşlar + rastgele çiftler)
    sample_idx = random.Random(args.seed).sample(valid, 90)
    method_validation = validate_method([grays[i] for i in sample_idx], seed=args.seed)
    # ORB-LSH aday üreticisinin geri çağırması: bozulmuş kopyalar 6085 gerçek görüntünün arasına katılır
    from isg_vision.dataset_audit.duplicates_verify import AUGMENTATIONS, augment
    pool_g, pool_meta = [], []
    for kind in AUGMENTATIONS:
        r_aug = random.Random(args.seed)
        for bi in sample_idx:
            pool_g.append(augment(grays[bi], kind, r_aug))
            pool_meta.append((kind, bi))
    ex_desc = [index_descriptors(g) for g in pool_g]
    ex_desc_f = [index_descriptors(np.ascontiguousarray(g[:, ::-1])) for g in pool_g]
    big_votes = lsh_votes_with_flip(descs + ex_desc, descs_f + ex_desc_f)
    recall: dict[str, list[bool]] = defaultdict(list)
    for k, (kind, bi) in enumerate(pool_meta):
        key = (min(bi, n + k), max(bi, n + k))
        v_n, v_f = big_votes.get(key, (0, 0))
        recall[kind].append(max(v_n, v_f) >= args.min_votes)
    method_validation["lsh_candidate_recall"] = {"min_votes": args.min_votes, "distractor_images": n,
                                                 "by_augmentation": {k: round(sum(v) / len(v), 3) for k, v in recall.items()}}
    for fp in method_validation["negatives"]["geometric_matches_among_random_pairs"]:  # indeksleri yola çevir
        fp["path_a"], fp["path_b"] = items[sample_idx[fp.pop("i")]]["rel"], items[sample_idx[fp.pop("j")]]["rel"]
    write_json(audit / "duplicate_method_validation.json", method_validation)
    summary = {"images": n, "unreadable": bad, "candidate_pairs": len(cand), "min_votes": args.min_votes,
               "hash_bits": args.hash_bits, "thresholds": DEFAULT_THRESHOLDS, "decisions": dict(dec),
               "decisions_by_relation": by_rel, "risk_counts_excluding_not_duplicate": dict(risk),
               "sibling_candidate_recall_eval": sib_eval}
    write_json(audit / "duplicate_verification_summary.json", summary)
    print("\nKararlar:", dict(dec))
    for rl, d in by_rel.items():
        print(f"  {rl}: {d}")
    print("Sibling değerlendirme:", sib_eval)

    if not args.no_render:
        base = audit / "duplicate_verification"
        prio = [r for r in rows if r["decision"] != "not_duplicate" and (
            r["relation"] == "cross_dataset" or (r["relation"] == "within_dataset_cross_split"))]
        prio.sort(key=lambda r: (r["relation"] != "cross_dataset", -int(r["orb_inliers"])))
        for r in prio[:40]:
            side_by_side(Path(__file__).resolve().parents[1] / r["path_a"], Path(__file__).resolve().parents[1] / r["path_b"],
                         base / f"{r['pair_id']}.jpg", f"{r['pair_id']} {r['decision']} inl={r['orb_inliers']} ncc={r['aligned_ncc']} "
                         f"{r['split_a']}|{r['split_b']} {r['dataset_a'][:4]}|{r['dataset_b'][:4]}")
        print(f"Yan yana görüntüler: {base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
