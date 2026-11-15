#!/usr/bin/env python
"""Yüz tespiti performans ölçümü (Aşama 5A): model yükleme, çıkarım süresi, yüz sayısı/çözünürlük/tespit-genişliği etkisi, bellek, GUI maliyeti.

Kamera KULLANMAZ. Ölçümler Ultralytics paketinin örnek fotoğraflarından üretilen SENTETİK montajlar üzerindedir (hız ve entegrasyon içindir,
doğruluk kanıtı değildir). Aşama 4B/4A sonuçları 5A sonucu gibi sunulmaz. Ölçülmeyen değerler yazılmaz.

Kullanım:
    .venv/bin/python scripts/benchmark_face_detection.py
    .venv/bin/python scripts/benchmark_face_detection.py --repeats 200 --gui          # gerçek pencere açar (grafik oturumu gerekir)
    .venv/bin/python scripts/benchmark_face_detection.py --import-ultralytics-first   # ultralytics içe aktarılmış süreci ölçer (OpenCV iş parçacığı etkisi)

Çıktı: results/face_detection/benchmark_<tarih>.json (Git dışı) ve konsolda Markdown tabloları.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sizes(text: str) -> tuple[tuple[int, int], ...]:
    out = []
    for part in text.split(","):
        w, _, h = part.strip().lower().partition("x")
        out.append((int(w), int(h)))
    return tuple(out)


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(p) for p in text.split(",") if p.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=100, help="Her ölçümde çıkarım tekrarı (varsayılan 100)")
    parser.add_argument("--loads", type=int, default=5, help="Model yükleme tekrarı (varsayılan 5)")
    parser.add_argument("--face-counts", type=_ints, default=(0, 1, 2, 4, 8, 16), metavar="N,N,...", help="Montajdaki yüz sayıları")
    parser.add_argument("--face-px", type=int, default=96, help="Montajdaki yüz yüksekliği (px, varsayılan 96)")
    parser.add_argument("--sizes", type=_sizes, default=_sizes("640x360,960x540,1280x720,1920x1080"), metavar="WxH,...", help="Çözünürlük taraması")
    parser.add_argument("--detect-widths", type=_ints, default=(960, 640), metavar="PX,PX", help="Denenecek --detect-width değerleri")
    parser.add_argument("--gui", action="store_true", help="Çizim ve gerçek pencere show maliyetini de ölç (grafik oturumu gerekir)")
    parser.add_argument("--import-ultralytics-first", action="store_true",
                        help="Ölçümden önce ultralytics'i içe aktar (OpenCV'nin iş parçacığı sayısı düşer; 5B'deki ortak süreç durumunu ölçmek için)")
    parser.add_argument("--output", default=None, metavar="DOSYA.json", help="JSON çıktı yolu (varsayılan: results/face_detection/benchmark_<tarih>.json)")
    args = parser.parse_args(argv)

    if args.import_ultralytics_first:
        import ultralytics  # noqa: F401, PLC0415 (bilinçli: etkiyi ölçmek için)

    from isg_vision.camera.display import PreviewWindow
    from isg_vision.face_detection.benchmark import run_benchmark, write_json
    from isg_vision.face_detection.config import FaceDetectionConfig
    from isg_vision.face_detection.errors import FaceDetectionError
    from isg_vision.face_detection.visualization import WINDOW_NAME

    try:
        report = run_benchmark(
            FaceDetectionConfig(), repeats=args.repeats, face_counts=args.face_counts, face_px=args.face_px, sizes=args.sizes,
            detect_widths=args.detect_widths, loads=args.loads, gui=args.gui,
            window_factory=(lambda: PreviewWindow(WINDOW_NAME)) if args.gui else None,
        )
    except FaceDetectionError as exc:
        print(f"[HATA] {exc}", file=sys.stderr)
        return 1

    out = Path(args.output) if args.output else PROJECT_ROOT / "results" / "face_detection" / f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.json"
    write_json(out, report)
    print(report["tables_markdown"])
    print(f"[BİLGİ] Ortam: {report['environment']}")
    print(f"[BİLGİ] Model yükleme (medyan): {report['model_load']['load_ms_median']:.2f} ms | ısınma 1280x720: {report['warmup_seconds_1280x720']:.3f} sn")
    print(f"[BİLGİ] Tepe bellek (ru_maxrss, MiB): {report['peak_rss_mb']}")
    if report.get("gui"):
        g = report["gui"]
        print(f"[BİLGİ] GUI: çizim medyan {g['draw_median_ms']:.2f} ms | show (imshow+waitKey(1)) medyan {g['show']['median_ms']:.2f} ms")
    print(f"[BİLGİ] {report['disclaimer']}")
    print(f"[BİLGİ] JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
