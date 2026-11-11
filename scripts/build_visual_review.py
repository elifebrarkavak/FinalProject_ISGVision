#!/usr/bin/env python
"""Aşama 2D: insan doğrulaması için görsel inceleme paketini üretir (salt okunur; ham veri DEĞİŞTİRİLMEZ).

Çıktılar (reports/dataset_audit/final_visual_review/):
  pages/RNN_<dosya>.jpg   her görüntü için A4 yatay (300 dpi) sayfa: orijinal, tüm kutular, şüpheli kişi, baş/gövde yakınlaştırması
  contact_sheet.pdf       tüm sayfalar tek PDF'te
  review_decisions.csv    model önerisi + gerekçe + BOŞ kullanıcı kararı/notu

Model önerileri configs/review/stage2d_visual_review.yaml dosyasından okunur. CSV yeniden üretilirken mevcut
'user_decision' ve 'user_note' değerleri KORUNUR (kullanıcı girdisi silinmez).

Örnek:
    python scripts/build_visual_review.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.dataset_audit.common import load_registry, read_csv  # noqa: E402
from isg_vision.dataset_audit.usage_policy import load_policy  # noqa: E402
from isg_vision.dataset_audit.visual_review import (  # noqa: E402
    MODEL_DECISIONS, USER_FIELDS, collect_items, read_review_csv, render_page, write_review_csv,
)
from isg_vision.utils.paths import PROJECT_ROOT  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "dataset_audit" / "final_visual_review"
DECISIONS = PROJECT_ROOT / "configs" / "review" / "stage2d_visual_review.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    policy, registry = load_policy(), load_registry()
    manifest = read_csv(PROJECT_ROOT / policy["sources"]["construction_ppe"]["split_policy"]["manifest"])
    items = collect_items(manifest, registry["construction_ppe"], PROJECT_ROOT)
    model = {}
    if DECISIONS.is_file():
        model = {d["image_path"]: d for d in yaml.safe_load(DECISIONS.read_text(encoding="utf-8"))["reviews"]}
    csv_path = args.out_dir / "review_decisions.csv"
    previous = {r["image_path"]: r for r in read_review_csv(csv_path)} if csv_path.is_file() else {}
    pages_dir = args.out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    for old in pages_dir.glob("R*.jpg"):
        old.unlink()
    rows, pages = [], []
    for i, it in enumerate(items, 1):
        rid = f"R{i:02d}"
        m = model.get(it.rel_path, {})
        dec = m.get("model_decision", "")
        if dec and dec not in MODEL_DECISIONS:
            print(f"[HATA] {rid}: geçersiz model_decision {dec!r}")
            return 1
        page = render_page(it, rid, dec, m.get("model_rationale", ""))
        name = f"{rid}_{it.image_path.stem}.jpg"
        page.save(pages_dir / name, quality=90, dpi=(300, 300))
        pages.append(page)
        missing = [c for c in it.codes if c.startswith("head")] and "baş etiketi (helmet/no_helmet)" or ""
        if any(c.startswith("torso") for c in it.codes):
            missing = (missing + " + " if missing else "") + "gövde etiketi (vest/no_vest)"
        near = "; ".join(f"{n['class']}#L{n['line']}->{'P' + str(n['assigned_person']) if n['assigned_person'] else 'atanmamış'}"
                         for n in it.nearby)
        prev = previous.get(it.rel_path, {})
        rows.append({"review_id": rid, "image_path": it.rel_path, "split": it.split,
                     "risk_code": f"CP-PERSON-ATTR-INCOMPLETE:{it.risk_code}",
                     "suspect_person_box_px": f"P{it.suspect_index}: {int(it.suspect.box.w * it.width)}x{int(it.suspect.box.h * it.height)}",
                     "suspect_label": f"P{it.suspect_index} için atanmış {missing} yok",
                     "nearby_labels": near, "model_decision": dec, "model_rationale": m.get("model_rationale", ""),
                     "contact_sheet_page": f"pages/{name} (PDF sayfa {i})",
                     **{k: prev.get(k, "") for k in USER_FIELDS}})
    if pages:
        pages[0].save(args.out_dir / "contact_sheet.pdf", save_all=True, append_images=pages[1:], resolution=300)
    write_review_csv(csv_path, rows)
    kept = sum(1 for r in rows if r["user_decision"] or r["user_note"])
    print(f"{len(rows)} görüntü; sayfalar: {pages_dir}; PDF: contact_sheet.pdf; CSV: review_decisions.csv "
          f"(korunan kullanıcı girdisi: {kept}); model kararı eksik: {sum(1 for r in rows if not r['model_decision'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
