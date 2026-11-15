#!/usr/bin/env python
"""Resmî YuNet yüz tespit modelini açıkça indirir (Aşama 5A çoklu yüz tespiti).

Bu betik, yüz tespiti için ağ kullanan TEK adımdır; ``run_face_detection.py`` hiçbir zaman indirme yapmaz. Yalnızca izin listesindeki
dosya indirilir (``isg_vision.face_detection.weights.OFFICIAL_MODELS``): kaynak OpenCV Zoo deposunun SABİT bir commit'idir
(``main`` dalı değil). Dosya ``models/face_detection/`` altına yazılır (Git tarafından yok sayılır); boyutu ve SHA-256 özeti
yayıncının kaydıyla sabitlenmiştir, uyuşmazlıkta dosya reddedilir. Lisans: MIT.

Kullanım:
    .venv/bin/python scripts/download_face_model.py                 # face_detection_yunet_2026may.onnx (~230 KB)
    .venv/bin/python scripts/download_face_model.py --verify-only   # indirmeden mevcut dosyanın özetini doğrula
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.face_detection.config import DEFAULT_MODEL_NAME  # noqa: E402
from isg_vision.face_detection.errors import FaceWeightsDownloadError  # noqa: E402
from isg_vision.face_detection.weights import (  # noqa: E402
    OFFICIAL_MODELS,
    default_weights_dir,
    download_official_model,
    sha256_file,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, choices=sorted(OFFICIAL_MODELS),
                        help=f"İndirilecek resmî model (varsayılan: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--force", action="store_true", help="Var olan dosyanın üzerine yaz")
    parser.add_argument("--verify-only", action="store_true", help="İndirme yapma; mevcut dosyanın boyut ve SHA-256 özetini beklenenle karşılaştır")
    args = parser.parse_args(argv)

    spec = OFFICIAL_MODELS[args.model]
    target = default_weights_dir() / spec.name
    if args.verify_only:
        if not target.is_file():
            print(f"[HATA] Dosya yok: {target}", file=sys.stderr)
            return 1
        size, digest = target.stat().st_size, sha256_file(target)
        ok = size == spec.size_bytes and digest == spec.sha256
        print(f"[BİLGİ] {target}\n[BİLGİ] boyut: {size} bayt (beklenen {spec.size_bytes})\n[BİLGİ] sha256: {digest}")
        print("[BİLGİ] Bütünlük doğrulandı." if ok else "[HATA] Bütünlük doğrulanamadı: beklenen değerlerle uyuşmuyor.")
        return 0 if ok else 1

    print(f"[BİLGİ] Kaynak (resmî, sabit commit): {spec.url}")
    print(f"[BİLGİ] Lisans: {spec.license_name} | beklenen boyut: {spec.size_bytes} bayt")
    print(f"[BİLGİ] Hedef: {target}")
    try:
        result = download_official_model(args.model, force=args.force)
    except FaceWeightsDownloadError as exc:
        print(f"[HATA] {exc}", file=sys.stderr)
        return 1
    print(f"[BİLGİ] {'İndirildi' if result.downloaded else 'Dosya zaten vardı (indirme yapılmadı)'}: {result.path}")
    print(f"[BİLGİ] boyut: {result.size_bytes} bayt | sha256: {result.sha256}")
    print("[BİLGİ] Boyut ve SHA-256 yayıncının kaydıyla doğrulandı.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
