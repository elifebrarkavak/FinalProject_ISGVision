#!/usr/bin/env python
"""CPU ve MPS için insan tespiti hız/gecikme karşılaştırması (doğruluk ölçümü DEĞİLDİR).

Aynı video, aynı model ve aynı eşiklerle cihazları dönüşümlü geçişlerle karşılaştırır; yükleme, ısınma ve burn-in süreleri sürekli
çıkarımdan ayrı raporlanır. Sonuçlar ``results/person_detection/`` altına JSON olarak yazılır (Git dışı; görüntü içermez).

Örnekler:
    .venv/bin/python scripts/benchmark_person_detection.py --make-sample-video results/person_detection/sentetik.mp4
    .venv/bin/python scripts/benchmark_person_detection.py --video-path results/person_detection/sentetik.mp4
    .venv/bin/python scripts/benchmark_person_detection.py --video-path /yol/video.mp4 --devices cpu,mps --passes 5 --max-frames 200

Not: ``--make-sample-video`` SENTETİK bir montaj üretir (Ultralytics örnek görüntüleri); yalnızca hız karşılaştırması içindir.
Canlı kamera hızı için: ``run_person_detection.py --source camera --no-display --max-frames 300 --device cpu|mps``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.human_detection.benchmark import (  # noqa: E402
    SYNTHETIC_DISCLAIMER,
    make_synthetic_video,
    report_to_dict,
    report_to_markdown,
    run_benchmark,
)
from isg_vision.camera.errors import SourceError  # noqa: E402
from isg_vision.human_detection.config import (  # noqa: E402
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_IOU,
    DEFAULT_MODEL_PATH,
    DetectionConfig,
)
from isg_vision.human_detection.errors import DetectionConfigError, DetectionError  # noqa: E402
from isg_vision.utils.paths import project_path  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-path", help="Karşılaştırmada kullanılacak video dosyası")
    parser.add_argument("--make-sample-video", metavar="DOSYA.mp4", help="SENTETİK örnek video üretip çıkar (indirme yok)")
    parser.add_argument("--devices", default="cpu,mps", help="Virgülle ayrılmış cihazlar (varsayılan: cpu,mps). Kullanılamayan cihaz atlanır ve nedeni yazılır.")
    parser.add_argument("--passes", type=int, default=3, help="Ölçülen geçiş sayısı (varsayılan: 3)")
    parser.add_argument("--burn-in", type=int, default=10, help="Ölçüme alınmayan ilk kare sayısı (varsayılan: 10)")
    parser.add_argument("--max-frames", type=int, default=None, help="Geçiş başına en fazla kare")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--output-dir", default=str(project_path("results", "person_detection")), help="JSON çıktı klasörü (varsayılan: results/person_detection)")
    args = parser.parse_args(argv)

    if args.make_sample_video:
        path = Path(args.make_sample_video)
        path.parent.mkdir(parents=True, exist_ok=True)
        make_synthetic_video(path)
        print(f"[BİLGİ] Sentetik video yazıldı: {path}\n[UYARI] {SYNTHETIC_DISCLAIMER}")
        return 0
    if not args.video_path:
        parser.error("--video-path veya --make-sample-video gerekir.")

    try:
        config = DetectionConfig(model_path=args.model, conf=args.conf, iou=args.iou, imgsz=args.imgsz)
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
        report = run_benchmark(Path(args.video_path), devices, config, passes=args.passes, burn_in=args.burn_in, max_frames=args.max_frames)
    except (DetectionConfigError, ValueError) as exc:
        parser.error(str(exc))
    except (DetectionError, SourceError) as exc:  # model/ağırlık hatası veya video açılamadı: benchmark durur
        print(f"[HATA] {exc}", file=sys.stderr)
        return 1

    import cv2
    import torch
    import ultralytics

    report.environment = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "machine": platform.machine(), "platform": platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__, "torch_threads": torch.get_num_threads(),
    }
    print(report_to_markdown(report))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report_to_dict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[BİLGİ] JSON: {out}")
    print("[UYARI] Güç/ısıl durum ve arka plan süreçleri kontrol edilmedi; sonuçlar bu makine ve oturum için gözlemdir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
