#!/usr/bin/env python
"""Veri kullanım politikasını (configs/dataset_usage_policy.yaml) DEĞERLENDİRİR ve karar tablolarını yazar.

Veriyi değiştirmez, filtrelemez, birleştirmez; yalnızca okur. Çıktılar (reports/dataset_audit/):
  image_usage_decisions.csv       her görüntü: şu anki durum + hak doğrulanırsa alacağı durum + nedenler + kaynak grubu
  quarantine_candidates.csv       karantina / yeniden etiketleme adayı görüntüler (gerekçe ve kanıtla)
  needs_reannotation_list.csv     yeniden etiketlenmesi gereken görüntüler ve sınıflar (gelecekteki manuel iş listesi)
  usage_policy_summary.json       sayımlar (kaynak, durum, neden, onaylı kapsamdaki sınıf kutuları)
  construction_ppe_split_manifest.csv  (2C) onaylı kaynak için nihai bölme + görüntü kararları + eksik etiket risk işaretleri
Önce build_source_groups.py (kaynak grupları) çalıştırılmış olmalıdır.

Örnek:
    python scripts/apply_usage_policy.py
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
from isg_vision.dataset_audit.layout import discover_dataset  # noqa: E402
from isg_vision.dataset_audit.mapping import load_mapping  # noqa: E402
from isg_vision.dataset_audit.source_groups import assign_unit_splits  # noqa: E402
from isg_vision.dataset_audit.usage_policy import (  # noqa: E402
    decide_conditional, final_split_assignment, iter_image_records, load_policy, unit_integrity_violations, validate_policy,
)
from isg_vision.dataset_audit.visual_review import (  # noqa: E402
    read_review_csv, review_state, user_overrides, validate_review_rows,
)
from isg_vision.utils.paths import PROJECT_ROOT  # noqa: E402

FIELDS = ["image_path", "dataset", "current_split", "status_now", "status_if_rights_verified", "reason_codes", "n_target_boxes",
          "n_excluded_class_boxes", "hard_group", "split_unit", "relation_status", "unit_spans_splits", "unit_spans_datasets",
          "proposed_split_preview", "preview_scope", "final_split", "annotation_risk", "note"]
SPLIT_FIELDS = ["image_path", "file", "publisher_sha256", "original_split", "final_split", "split_changed", "training_status",
                "decision_rule", "split_unit", "unit_size", "relation_status", "annotation_risk", "n_flagged_persons", "visual_review",
                "helmet", "no_helmet", "vest", "no_vest"]
Q_FIELDS = ["image_path", "dataset", "current_split", "status", "reason_codes", "source_license_status", "evidence", "next_step"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    audit = ensure_audit_dir(args.out_dir)
    registry = load_registry()
    policy = load_policy()
    targets, entries = load_mapping()
    layouts = {n: discover_dataset(n, dataset_raw_dir(registry, n)) for n in registry}
    classes = {n: lay.classes.names for n, lay in layouts.items()}
    problems = validate_policy(policy, registry, entries, classes)
    if problems:
        print("[HATA] Politika geçersiz:\n  - " + "\n  - ".join(problems))
        return 1
    decisions = yaml.safe_load((PROJECT_ROOT / "configs/review/stage2b_decisions.yaml").read_text(encoding="utf-8"))
    overrides = decisions["conflict_decisions"]
    manifest = {r["image_path"]: r for r in read_csv(audit / "source_group_manifest.csv")}
    review = read_csv(audit / "label_review.csv")
    by_image: dict[str, list[dict]] = {}
    for r in review:
        by_image.setdefault(r["image_path"], []).append(r)
    verified_neg = {r["image_path"] for r in review if r["source"] == "empty_label" and r["category"] == "Valid Annotation"}

    # (2D) kullanıcı görsel inceleme kararları: yalnızca user_decision uygulanır
    review_rows: dict[str, dict[str, str]] = {}
    user_ov: dict[str, dict[str, dict[str, str]]] = {}
    review_summary: dict[str, dict] = {}
    for ds, src in policy["sources"].items():
        uvr = src.get("user_visual_review")
        if src["source_usage"] != "approved_scope" or not uvr:
            continue
        rpath = PROJECT_ROOT / uvr["file"]
        if not rpath.is_file():
            continue
        rrows = read_review_csv(rpath)
        paths = {rel(p) for p in (dataset_raw_dir(registry, ds) / "images").glob("*/*")}
        problems = validate_review_rows(rrows, paths)
        if problems:
            print("[HATA] Görsel inceleme karar dosyası geçersiz (" + uvr["file"] + "):\n  - " + "\n  - ".join(problems))
            return 1
        review_rows.update({r["image_path"]: r for r in rrows})
        user_ov[ds] = user_overrides(rrows)
        review_summary[ds] = review_state(rrows)

    rows, qrows, na_rows = [], [], []
    for ds, src in policy["sources"].items():
        cmap = {int(k) for k in (src.get("class_map") or src.get("conditional_class_map") or {})}
        excluded = {int(k) for k in (src.get("excluded_classes") or {})}
        for rec in iter_image_records(layouts[ds], registry[ds]):
            p = rel(rec["image_path"])
            m = manifest[p]
            target_n = sum(1 for b in rec["boxes"] if b.class_id in cmap)
            excl_n = sum(1 for b in rec["boxes"] if b.class_id in excluded)
            risk = ""
            if src["source_usage"] == "approved_scope":
                ov = (src.get("image_overrides") or {}).get(p) or user_ov.get(ds, {}).get(p)
                if not rec["readable"]:
                    status_now, reasons = "quarantine", ["IMAGE-UNREADABLE"]
                elif ov is not None:
                    status_now, reasons = ov["status"], [ov["rule"]]
                else:
                    status_now, reasons = "approved", []
                cond = status_now
                if "CP-PERSON-ATTR-INCOMPLETE" in (src.get("annotation_risk_rules") or []) and rec["person_issues"]:
                    risk = "CP-PERSON-ATTR-INCOMPLETE:" + "|".join(sorted({c for codes in rec["person_issues"] for c in codes}))
                note = "; ".join(src.get("notes", [])[:1])
            else:
                status_now = "pending_license_verification"
                cond, reasons = decide_conditional(ds, rec, src, policy, by_image, verified_neg, overrides, p)
                note = "hak doğrulanana kadar onaylı kümeye ALINMAZ"
            elig_now = status_now == "approved"
            elig_cond = cond == "approved"
            rows.append({"image_path": p, "dataset": ds, "current_split": rec["split"], "status_now": status_now,
                         "status_if_rights_verified": cond, "reason_codes": ";".join(reasons), "n_target_boxes": target_n,
                         "n_excluded_class_boxes": excl_n, "hard_group": m["hard_group"], "split_unit": m["split_unit"],
                         "relation_status": m["relation_status"], "unit_spans_splits": m["unit_spans_splits"],
                         "unit_spans_datasets": m["unit_spans_datasets"], "proposed_split_preview": "", "preview_scope": "",
                         "final_split": "", "annotation_risk": risk,
                         "note": note, "_elig_now": elig_now, "_elig_cond": elig_cond,
                         "_cls": Counter(b.class_id for b in rec["boxes"]), "_n_flag": len(rec["person_issues"])})
            if cond in ("quarantine", "needs_reannotation") or status_now in ("quarantine", "needs_reannotation"):
                ev = {"USER-VISUAL-REVIEW": "2D kullanıcı görsel inceleme kararı (review_decisions.csv)",
                      "CP-VISIBLE-TARGET-UNLABELED": "2C görsel inceleme: görünür hedef nesne etiketsiz (stage2c_decisions.yaml)",
                      "MD-HELMET-IMAGE": "E05: Helmet kutusu baş+baret; Helmet satırları silinirse etiketsiz baret kalır",
                      "MD-CROWD-INCOMPLETE": "Kalabalık sahnelerde etiket eksikliği (görsel inceleme)",
                      "RF-EXCLUDED-CLASS-IMAGE": "E10/E12: NO-Hardhat/NO-Safety Vest sınıfları güvenilir değil",
                      "RF-TINY-TARGET": "Hedef kutu <12 px", "RF-PERSON-INCOMPLETE": "Kişi başına baş/gövde etiketi eksik (alt sınır kanıt: %47 tam)",
                      "RF-UNVERIFIED-NEGATIVE": "Hedefsiz görüntü doğrulanmış negatif değil"}
                first = reasons[0].split(":")[0] if reasons else ""
                qrows.append({"image_path": p, "dataset": ds, "current_split": rec["split"],
                              "status": cond if status_now == "pending_license_verification" else status_now,
                              "reason_codes": ";".join(reasons), "source_license_status": src["license_status"],
                              "evidence": ev.get(first, "label_review.csv / stage2b_decisions.yaml"),
                              "next_step": ("yeniden etiketle" if cond == "needs_reannotation" else "hak ve etiket kalitesi doğrulanana kadar kullanma")})
                if cond == "needs_reannotation":
                    cls = Counter(b.class_id for b in rec["boxes"] if b.class_id in excluded)
                    na_rows.append({"image_path": p, "dataset": ds, "current_split": rec["split"], "reason_codes": ";".join(reasons),
                                    "classes_to_reannotate": ";".join(f"{registry[ds]['expected_class_names'][c] if c < len(registry[ds]['expected_class_names']) else c}:{n}" for c, n in sorted(cls.items())),
                                    "boxes_to_reannotate": sum(cls.values())})
                    if any(r_.startswith(("CP-", "USER-")) for r_ in reasons):  # 2C/2D: onaylı kaynakta eksik hedef etiketi
                        na_rows[-1]["classes_to_reannotate"] = "eksik hedef etiket eklenecek (configs/review/stage2c_decisions.yaml)"
    # (2C) onaylı kaynakların NİHAİ bölmesi: yayıncı bölünmesi + kaynak birimi düzeltmesi (split_policy)
    unit_fixes: dict[str, list] = {}
    split_rows: list[dict] = []
    for ds, src in policy["sources"].items():
        if src["source_usage"] != "approved_scope":
            continue
        sel = [r for r in rows if r["dataset"] == ds]
        excluded = {r["image_path"] for r in sel if r["status_now"] != "approved"}
        final, fixes = final_split_assignment([(r["image_path"], r["current_split"], r["split_unit"]) for r in sel], excluded)
        unit_fixes[ds] = fixes
        pub = _publisher_manifest(dataset_raw_dir(registry, ds))
        names = registry[ds]["expected_class_names"]
        cmap = {int(k): int(v) for k, v in src["class_map"].items()}
        for r in sel:
            r["final_split"] = final[r["image_path"]]
            cls = r["_cls"]
            split_rows.append({"image_path": r["image_path"], "file": Path(r["image_path"]).name,
                               "publisher_sha256": pub.get(Path(r["image_path"]).name, {}).get("sha256", ""),
                               "original_split": r["current_split"], "final_split": r["final_split"],
                               "split_changed": "yes" if r["final_split"] and r["final_split"] != r["current_split"] else "no",
                               "training_status": r["status_now"],
                               "decision_rule": r["reason_codes"] or ("UNIT-FIX:majority_split" if r["final_split"] and r["final_split"] != r["current_split"] else ""),
                               "split_unit": r["split_unit"], "unit_size": manifest[r["image_path"]]["split_unit_size"],
                               "relation_status": r["relation_status"], "annotation_risk": r["annotation_risk"], "n_flagged_persons": r["_n_flag"],
                               "visual_review": _review_tag(review_rows.get(r["image_path"])),
                               **{targets[t]: sum(n for c, n in cls.items() if cmap.get(c) == t) for t in sorted(targets)}})
        bad = unit_integrity_violations([r for r in rows if r["dataset"] == ds])
        if bad:
            print(f"[HATA] {ds}: bölme birimi birden çok bölümde: {bad}")
            return 1
        out_path = PROJECT_ROOT / src["split_policy"]["manifest"]
        write_csv(out_path, SPLIT_FIELDS, split_rows)
    # bölme önizlemesi (UYGULANMAMIŞ öneri) yalnızca BEKLEMEDEKİ kaynaklar için: bölme birimi ASLA bölünmez
    ratios = policy["leakage"]["ratios"]
    cond_sel = [r for r in rows if r["_elig_cond"] and r["status_now"] == "pending_license_verification"]
    units = Counter(r["split_unit"] for r in cond_sel)
    if units:
        split_of = assign_unit_splits(dict(units), ratios, policy["leakage"]["seed"])
        for r in cond_sel:
            r["proposed_split_preview"], r["preview_scope"] = split_of[r["split_unit"]], "if_rights_verified"
    for r in rows:
        for k in ("_elig_now", "_elig_cond", "_cls", "_n_flag"):
            r.pop(k)
    write_csv(audit / "image_usage_decisions.csv", FIELDS, rows)
    write_csv(audit / "quarantine_candidates.csv", Q_FIELDS, qrows)
    write_csv(audit / "needs_reannotation_list.csv", ["image_path", "dataset", "current_split", "reason_codes", "classes_to_reannotate", "boxes_to_reannotate"], na_rows)

    # özet
    summary: dict = {"policy_version": policy["policy_version"], "by_dataset": {}}
    for ds in registry:
        sel = [r for r in rows if r["dataset"] == ds]
        reasons = Counter(c.split(":")[0] for r in sel for c in r["reason_codes"].split(";") if c)
        summary["by_dataset"][ds] = {"images": len(sel), "status_now": dict(Counter(r["status_now"] for r in sel)),
                                     "status_if_rights_verified": dict(Counter(r["status_if_rights_verified"] for r in sel)),
                                     "reason_counts": dict(reasons)}
    # onaylı kapsamdaki sınıf kutuları
    cls_counts: Counter = Counter()
    approved_paths = {r["image_path"] for r in rows if r["status_now"] == "approved"}
    for ds, src in policy["sources"].items():
        if src["source_usage"] != "approved_scope":
            continue
        cmap = {int(k): int(v) for k, v in src["class_map"].items()}
        for rec in iter_image_records(layouts[ds], registry[ds]):
            if rel(rec["image_path"]) in approved_paths:
                for b in rec["boxes"]:
                    if b.class_id in cmap:
                        cls_counts[targets[cmap[b.class_id]]] += 1
    summary["approved_scope_target_boxes"] = dict(cls_counts)
    fs = [r for r in rows if r["final_split"]]
    summary["approved_scope_final_split_images"] = dict(sorted(Counter(r["final_split"] for r in fs).items()))
    by_split: dict = {}
    for r in split_rows:
        if r["final_split"]:
            d = by_split.setdefault(r["final_split"], Counter())
            for t in targets.values():
                d[t] += int(r[t])
    summary["approved_scope_target_boxes_by_final_split"] = {k: dict(v) for k, v in sorted(by_split.items())}
    summary["approved_scope_excluded_images"] = {r["image_path"]: r["reason_codes"] for r in rows
                                                 if policy["sources"][r["dataset"]]["source_usage"] == "approved_scope" and r["status_now"] != "approved"}
    summary["approved_scope_unit_fixes"] = unit_fixes
    summary["approved_scope_user_visual_review"] = review_summary
    risk_rows = [r for r in split_rows if r["annotation_risk"]]
    summary["approved_scope_annotation_risk"] = {
        "flagged_images": len(risk_rows), "flagged_persons": sum(int(r["n_flagged_persons"]) for r in risk_rows),
        "flagged_images_with_unexplained": sum(1 for r in risk_rows if "unexplained" in r["annotation_risk"]),
        "by_final_split": dict(sorted(Counter(r["final_split"] or "excluded" for r in risk_rows).items()))}
    preview = Counter((r["preview_scope"], r["proposed_split_preview"]) for r in rows if r["proposed_split_preview"])
    summary["split_preview_image_counts"] = {f"{a}/{b}": n for (a, b), n in sorted(preview.items())}
    write_json(audit / "usage_policy_summary.json", summary)
    for ds, v in summary["by_dataset"].items():
        print(ds, v["status_now"], "| hak doğrulanırsa:", v["status_if_rights_verified"])
    print("onaylı kapsam hedef kutuları:", summary["approved_scope_target_boxes"])
    print("bölme önizlemesi:", summary["split_preview_image_counts"])
    return 0


def _review_tag(r: dict[str, str] | None) -> str:
    """Manifest için görsel inceleme etiketi: 'R05:model=quarantine;user=pending|approved|...'."""
    if r is None:
        return ""
    return f"{r['review_id']}:model={r['model_decision']};user={r['user_decision'].strip() or 'pending'}"


def _publisher_manifest(raw_dir: Path) -> dict[str, dict]:
    """Yayıncının manifest.jsonl dosyası (varsa): dosya adı -> {split, sha256, annotations}."""
    path = raw_dir / "manifest.jsonl"
    if not path.is_file():
        return {}
    return {d["file"]: d for d in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


if __name__ == "__main__":
    sys.exit(main())
