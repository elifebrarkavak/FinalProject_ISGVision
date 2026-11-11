#!/usr/bin/env python
"""İndirilen veri seti ZIP'lerini bulur ve data/raw/<veri_seti>/ altına GÜVENLİ biçimde çıkarır.

- Orijinal ZIP dosyalarına dokunmaz (yalnızca okur); yeniden indirme yapmaz.
- Var olan dosyaların üzerine yazmaz; hedef klasör doluysa o dosyaları atlar ve raporlar.
- Zip-slip, sembolik bağ ve zip bombası denetimi yapar; çıkarılan dosyaları çalıştırmaz.
- Çıkarılan dosyalar salt-okunur yapılır (--writable ile kapatılabilir).

Örnek:
    python scripts/extract_datasets.py --dry-run
    python scripts/extract_datasets.py
    python scripts/extract_datasets.py --downloads-dir ~/Downloads --datasets site_safety
Çıktı günlüğü: reports/dataset_audit/extraction_log.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.dataset_audit.common import (  # noqa: E402
    dataset_raw_dir, ensure_audit_dir, load_registry, read_json, rel, select_datasets, write_json,
)
from isg_vision.dataset_audit.extract import extract_zip_safely, find_zip_candidates  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--downloads-dir", type=Path, default=Path.home() / "Downloads",
                        help="ZIP dosyalarının aranacağı klasör (varsayılan: ~/Downloads)")
    parser.add_argument("--datasets", nargs="*", help="Yalnızca bu veri setleri (varsayılan: hepsi)")
    parser.add_argument("--dry-run", action="store_true", help="Hiçbir şey yazmadan ne yapılacağını göster")
    parser.add_argument("--keep-wrapper", action="store_true",
                        help="Arşivdeki tek üst düzey klasörü sıyırma (varsayılan: sıyrılır)")
    parser.add_argument("--writable", action="store_true", help="Çıkarılan dosyaları salt-okunur yapma")
    args = parser.parse_args()

    registry = load_registry()
    log = []
    problems = 0
    for name in select_datasets(registry, args.datasets):
        print(f"\n== {name} ==")
        try:
            candidates = find_zip_candidates(args.downloads_dir.expanduser(), registry[name]["zip_patterns"])
        except FileNotFoundError as exc:
            print(f"[HATA] {exc}")
            print("       Klasöre erişim izni yoksa: Sistem Ayarları > Gizlilik ve Güvenlik > "
                  "Dosyalar ve Klasörler bölümünden terminal/uygulamaya İndirilenler izni verin.")
            problems += 1
            continue
        if len(candidates) != 1:
            print(f"[HATA] {len(candidates)} aday ZIP bulundu ({[c.name for c in candidates]}); tam 1 olmalı.")
            print(f"       ZIP'i elle şuraya koyun/yolunu belirtin: {registry[name]['raw_dir']}/ (ya da --downloads-dir)")
            problems += 1
            continue
        zip_path = candidates[0]
        dest = dataset_raw_dir(registry, name)
        dest.mkdir(parents=True, exist_ok=True)
        print(f"ZIP   : {zip_path}")
        print(f"Hedef : {rel(dest)}{'  (dry-run)' if args.dry_run else ''}")
        res = extract_zip_safely(zip_path, dest, name, strip_wrapper=not args.keep_wrapper,
                                 read_only=not args.writable, dry_run=args.dry_run)
        print(f"Girdi: {res.entries_total}  çıkarılan: {res.extracted}  atlanan(var olan): {len(res.skipped_existing)}"
              f"  reddedilen: {len(res.rejected)}  hata: {len(res.errors)}  sıyrılan üst klasör: '{res.stripped_prefix}'")
        for line in (res.rejected + res.errors)[:10]:
            print(f"  ! {line}")
        problems += len(res.errors)
        entry = res.as_dict()
        entry["zip_path"] = str(zip_path).replace(str(Path.home()), "~")  # günlükte kullanıcı adı sızmasın
        entry["dest_dir"] = rel(dest)
        entry["skipped_existing_count"] = len(entry["skipped_existing"])
        entry["skipped_existing"] = entry["skipped_existing"][:20]  # günlük şişmesin: yalnızca örnek
        log.append(entry)

    if not args.dry_run and log:
        from datetime import datetime, timezone
        log_path = ensure_audit_dir() / "extraction_log.json"
        history = []
        if log_path.is_file():  # önceki çalıştırma kayıtları korunur (üzerine yazılmaz)
            previous = read_json(log_path)
            history = previous.get("runs", []) if isinstance(previous, dict) else [{"entries": previous}]
        history.append({"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "entries": log})
        write_json(log_path, {"runs": history})
        print("\nGünlük (eklendi): reports/dataset_audit/extraction_log.json")
    print("\nÖzet:", "SORUN VAR" if problems else "tamam")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
