#!/usr/bin/env python
"""Resmî Ultralytics YOLO ağırlığını açıkça indirir (Aşama 4A insan tespiti modeli).

Bu betik, insan tespiti için ağ kullanan TEK adımdır; ``run_person_detection.py`` hiçbir zaman indirme yapmaz.
Yalnızca izin listesindeki dosyalar indirilir (``isg_vision.human_detection.weights.OFFICIAL_MODELS``): kaynak
Ultralytics'in resmî GitHub release sayfasıdır. İndirilen dosya ``models/human_detection/`` altına yazılır
(Git tarafından yok sayılır) ve SHA-256 özeti sabitlenmiş değerle doğrulanır.

Kullanım:
    .venv/bin/python scripts/download_person_model.py                 # yolo11n.pt
    .venv/bin/python scripts/download_person_model.py --verify-only   # indirmeden mevcut dosyanın özetini yazdır
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.human_detection.config import DEFAULT_MODEL_NAME  # noqa: E402
from isg_vision.human_detection.errors import WeightsDownloadError  # noqa: E402
from isg_vision.human_detection.weights import (  # noqa: E402
    OFFICIAL_MODELS,
    default_weights_dir,
    download_official_weights,
    sha256_file,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, choices=sorted(OFFICIAL_MODELS),
                        help=f"İndirilecek resmî ağırlık (varsayılan: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--force", action="store_true", help="Var olan dosyanın üzerine yaz")
    parser.add_argument("--verify-only", action="store_true", help="İndirme yapma; mevcut dosyanın boyut ve SHA-256 özetini yazdır")
    args = parser.parse_args(argv)

    spec = OFFICIAL_MODELS[args.model]
    target = default_weights_dir() / spec.name
    if args.verify_only:
        if not target.is_file():
            print(f"[HATA] Dosya yok: {target}", file=sys.stderr)
            return 1
        print(f"[BİLGİ] {target}\n[BİLGİ] boyut: {target.stat().st_size} bayt\n[BİLGİ] sha256: {sha256_file(target)}")
        return 0

    print(f"[BİLGİ] Kaynak (resmî): {spec.url}")
    print(f"[BİLGİ] Hedef: {target}")
    try:
        result = download_official_weights(args.model, force=args.force)
    except WeightsDownloadError as exc:
        print(f"[HATA] {exc}", file=sys.stderr)
        return 1
    print(f"[BİLGİ] {'İndirildi' if result.downloaded else 'Dosya zaten vardı (indirme yapılmadı)'}: {result.path}")
    print(f"[BİLGİ] boyut: {result.size_bytes} bayt | sha256: {result.sha256}")
    print("[BİLGİ] SHA-256 sabitlenmiş değerle doğrulandı." if result.pinned
          else "[UYARI] SHA-256 henüz sabitlenmemiş; özeti resmî kaynakla karşılaştırıp weights.py'ye sabitleyin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
