#!/usr/bin/env python
"""Takip (ByteTrack) ek yükü ölçümü: yalnızca-tespit ↔ tespit+takip, CPU ↔ MPS (doğruluk ölçümü DEĞİLDİR).

Aynı video, aynı model ve aynı eşiklerle cihazları ve modları (yalnızca tespit / tespit+takip / tespit+takip+çizim) dönüşümlü geçişlerle
karşılaştırır; model/takipçi yükleme, ısınma ve burn-in süreleri sürekli performanstan ayrı raporlanır. ``--scaling`` YOLO olmadan ByteTrack
süresinin kişi sayısıyla ölçeklenmesini ölçer. Sonuçlar ``results/person_tracking/`` altına JSON olarak yazılır (Git dışı; görüntü içermez).

Örnekler:
    .venv/bin/python scripts/benchmark_person_tracking.py --video-path results/person_detection/sentetik_cok_kisili_1280x720.mp4
    .venv/bin/python scripts/benchmark_person_tracking.py --video-path /yol/video.mp4 --devices cpu,mps --passes 5 --max-frames 200 --scaling
    .venv/bin/python scripts/benchmark_person_tracking.py --scaling-only

Not: 4A'nın SENTETİK montaj videosu (``benchmark_person_detection.py --make-sample-video``) yalnızca hız/entegrasyon içindir; ondan takip
DOĞRULUĞU (IDF1/HOTA) çıkarılamaz. Canlı kamera hızı için: ``run_person_tracking.py --source camera --no-display --max-frames 300 --device cpu|mps``.
Pencere (imshow) maliyeti yalnızca gerçek pencerede ölçülebilir: ``run_person_tracking.py --source camera --device mps`` (özet satırı "Pencereye gösterme süresi").
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.camera.errors import SourceError  # noqa: E402
from isg_vision.human_detection.config import DEFAULT_CONF, DEFAULT_IMGSZ, DEFAULT_IOU, DEFAULT_MODEL_PATH, DetectionConfig  # noqa: E402
from isg_vision.human_detection.errors import DetectionConfigError, DetectionError  # noqa: E402
from isg_vision.tracking.benchmark import (  # noqa: E402
    TrackingBenchmarkReport,
    measure_tracker_scaling,
    report_to_dict,
    report_to_markdown,
    run_tracking_benchmark,
)
from isg_vision.tracking.config import TrackingConfig  # noqa: E402
from isg_vision.tracking.errors import TrackingConfigError, TrackingError  # noqa: E402
from isg_vision.utils.paths import project_path  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-path", help="Karşılaştırmada kullanılacak video dosyası")
    parser.add_argument("--devices", default="cpu,mps", help="Virgülle ayrılmış cihazlar (varsayılan: cpu,mps). Kullanılamayan cihaz atlanır ve nedeni yazılır.")
    parser.add_argument("--passes", type=int, default=3, help="Ölçülen geçiş sayısı (varsayılan: 3)")
    parser.add_argument("--burn-in", type=int, default=10, help="Ölçüme alınmayan ilk kare sayısı (varsayılan: 10)")
    parser.add_argument("--max-frames", type=int, default=None, help="Geçiş başına en fazla kare")
    parser.add_argument("--scaling", action="store_true", help="Ayrıca ByteTrack ölçeklenmesini (YOLO yok) ölç")
    parser.add_argument("--scaling-only", action="store_true", help="Yalnızca ByteTrack ölçeklenmesini ölç (video/model gerekmez)")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--output-dir", default=str(project_path("results", "person_tracking")), help="JSON çıktı klasörü (varsayılan: results/person_tracking)")
    args = parser.parse_args(argv)

    if not args.video_path and not args.scaling_only:
        parser.error("--video-path veya --scaling-only gerekir.")

    report: TrackingBenchmarkReport
    try:
        tracking_config = TrackingConfig()  # varsayılan eşikler (4A conf'u ve ByteTrack varsayılanları)
        if args.scaling_only:
            report = TrackingBenchmarkReport(video="", video_size=None, video_frames=None, frames_per_pass=0, config={"tracking": tracking_config.bytetrack_args()},
                                             devices=[], pass_order=[], mode_order=[])
        else:
            config = DetectionConfig(model_path=args.model, conf=args.conf, iou=args.iou, imgsz=args.imgsz)
            devices = [d.strip() for d in args.devices.split(",") if d.strip()]
            report = run_tracking_benchmark(Path(args.video_path), devices, config, tracking_config, passes=args.passes, burn_in=args.burn_in,
                                            max_frames=args.max_frames)
        if args.scaling or args.scaling_only:
            report.scaling = measure_tracker_scaling(tracking_config=tracking_config)
    except (DetectionConfigError, TrackingConfigError, ValueError) as exc:
        parser.error(str(exc))
    except (DetectionError, TrackingError, SourceError) as exc:  # model/bağımlılık hatası veya video açılamadı: benchmark durur
        print(f"[HATA] {exc}", file=sys.stderr)
        return 1

    import cv2
    import numpy
    import torch
    import ultralytics

    report.environment = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "machine": platform.machine(), "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "ultralytics": ultralytics.__version__, "opencv": cv2.__version__, "numpy": numpy.__version__,
        "torch_threads": torch.get_num_threads(),
    }
    print(report_to_markdown(report))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report_to_dict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[BİLGİ] JSON: {out}")
    print("[UYARI] Güç/ısıl durum ve arka plan süreçleri kontrol edilmedi; sonuçlar bu makine ve oturum için gözlemdir. Doğruluk ölçümü DEĞİLDİR.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
