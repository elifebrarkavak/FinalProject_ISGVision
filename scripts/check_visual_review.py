#!/usr/bin/env python
"""Aşama 2D: kullanıcı görsel inceleme kararlarını (review_decisions.csv) DOĞRULAR ve uygulanınca ne olacağını gösterir.

Ham veriyi veya etiketleri DEĞİŞTİRMEZ. Kararlar, `python scripts/apply_usage_policy.py` çalıştırıldığında nihai bölme
manifestine işlenir (needs_reannotation / quarantine -> görüntü kullanım dışı; approved -> kullanıcı doğrulamalı).

Geçerli user_decision değerleri: approved | needs_reannotation | quarantine | (boş = karar bekliyor).
needs_reannotation ve quarantine için user_note (kısa gerekçe) zorunludur.

Örnek:
    python scripts/check_visual_review.py                 # doğrula + özet
    python scripts/check_visual_review.py --require-complete   # boş karar varsa çıkış kodu 1 (Aşama 8 öncesi kapı)
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.dataset_audit.common import read_csv  # noqa: E402
from isg_vision.dataset_audit.usage_policy import load_policy  # noqa: E402
from isg_vision.dataset_audit.visual_review import (  # noqa: E402
    read_review_csv, review_state, user_overrides, validate_review_rows,
)
from isg_vision.utils.paths import PROJECT_ROOT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--require-complete", action="store_true", help="karar bekleyen satır varsa başarısız say")
    args = ap.parse_args()
    src = load_policy()["sources"]["construction_ppe"]
    path = PROJECT_ROOT / src["user_visual_review"]["file"]
    manifest = read_csv(PROJECT_ROOT / src["split_policy"]["manifest"])
    rows = read_review_csv(path)
    expected = {r["image_path"] for r in manifest if "unexplained" in r["annotation_risk"]}
    problems = validate_review_rows(rows, {r["image_path"] for r in manifest})
    if {r["image_path"] for r in rows} != expected:
        problems.append("CSV'deki görüntüler, manifestte 'unexplained' kodlu görüntülerle birebir aynı değil")
    if problems:
        print("[HATA] review_decisions.csv geçersiz:\n  - " + "\n  - ".join(problems))
        return 1
    st = review_state(rows)
    print(f"CSV geçerli: {st['total']} satır; kullanıcı kararı verilen {st['user_decided']}, bekleyen {st['pending_user']}.")
    print("Model önerileri (kullanıcı kararı DEĞİL):", dict(Counter(r["model_decision"] for r in rows)))
    print("Kullanıcı kararları:", st["by_user_decision"])
    diff = [r for r in rows if r["user_decision"].strip() and
            {"approved_candidate": "approved"}.get(r["model_decision"], r["model_decision"]) != r["user_decision"].strip()]
    if diff:
        print("Model önerisinden farklı kullanıcı kararları:")
        for r in diff:
            print(f"  {r['review_id']}: model={r['model_decision']} kullanıcı={r['user_decision']} ({r['user_note']})")
    ov = user_overrides(rows)
    print(f"Uygulanınca kullanım dışı kalacak görüntü: {len(ov)}")
    for p, o in sorted(ov.items()):
        print(f"  {o['status']}: {p}")
    if st["pending_user"]:
        print("Karar bekleyen satırlar kullanıcı doğrulaması SAYILMAZ (Aşama 8 öncesi tamamlanmalı).")
    print("Uygulamak için: python scripts/apply_usage_policy.py && python scripts/run_stage2c_checks.py")
    return 1 if args.require_complete and st["pending_user"] else 0


if __name__ == "__main__":
    sys.exit(main())
