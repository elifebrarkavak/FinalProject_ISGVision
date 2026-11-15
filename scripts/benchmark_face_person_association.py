#!/usr/bin/env python
"""Birleşik hattın (YOLO11n + ByteTrack + YuNet + ilişkilendirme) performansını ve OpenCV iş parçacığı etkisini ölçer (Aşama 5B).

Üst süreç her varyantı YENİ bir alt süreçte çalıştırır (``ultralytics`` içe aktarımı OpenCV iş parçacığı durumunu süreç genelinde değiştirir). Ölçüm KAYITLI
SENTETİK montaj videosundadır: gerçek kamera ölçümü DEĞİLDİR ve canlı FPS gibi sunulmamalıdır. Hiçbir ağ kullanılmaz, hiçbir paket kurulmaz; görüntü yalnızca
geçici videoda tutulur (ölçüm sonunda silinir).

Örnekler:
    .venv/bin/python scripts/benchmark_face_person_association.py                      # tam matris: {mps,cpu} x {keep,restore} x 3 tekrar + iş parçacığı sondaları
    .venv/bin/python scripts/benchmark_face_person_association.py --repeats 1 --max-frames 120 --devices cpu
    .venv/bin/python scripts/benchmark_face_person_association.py --video /yol/video.mp4 --out sonuc.json
    .venv/bin/python scripts/benchmark_face_person_association.py --worker probe --set-threads -1      # (iç kullanım) tek sonda
    .venv/bin/python scripts/benchmark_face_person_association.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.face_association import benchmark  # noqa: E402
from isg_vision.face_association.threads import threads_argument  # noqa: E402
from isg_vision.utils.paths import project_path  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker", choices=("probe", "pipeline"), default=None, help="(iç kullanım) tek bir varyantı bu süreçte çalıştırıp JSON yazdırır")
    p.add_argument("--video", type=Path, default=None, help="Ölçüm videosu (varsayılan: geçici sentetik montaj üretilir)")
    p.add_argument("--frames", type=int, default=benchmark.DEFAULT_FRAMES, help=f"Üretilen montaj kare sayısı (varsayılan: {benchmark.DEFAULT_FRAMES})")
    p.add_argument("--max-frames", type=int, default=None, help="Her varyantta işlenecek en çok kare")
    p.add_argument("--devices", nargs="+", default=["mps", "cpu"], choices=("mps", "cpu"), help="YOLO cihazları (YuNet/ByteTrack her zaman CPU)")
    p.add_argument("--threads", nargs="+", default=["keep", "restore"], type=threads_argument, help="Karşılaştırılacak --opencv-threads değerleri")
    p.add_argument("--repeats", type=int, default=3, help="Varyant başına tekrar (varsayılan: 3; sıra dönüşümlüdür)")
    p.add_argument("--probe-threads", nargs="*", type=int, default=[0, 1, 2, 4, 8, -1], help="cv2.setNumThreads sondası değerleri (boş = sonda yok)")
    p.add_argument("--out", type=Path, default=None, help="JSON çıktı yolu (varsayılan: results/face_association/benchmark_<tarih>.json)")
    p.add_argument("--device", default="cpu", choices=("mps", "cpu"), help="(işçi) YOLO cihazı")
    p.add_argument("--opencv-threads", type=threads_argument, default="keep", help="(işçi) iş parçacığı ayarı")
    p.add_argument("--set-threads", type=int, default=None, help="(işçi/probe) üçüncü aşamada cv2.setNumThreads değeri")
    p.add_argument("--no-yolo", action="store_true", help="(işçi/probe) YOLO süresini ölçme (yalnızca YuNet)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker == "probe":
        print(json.dumps(benchmark.run_probe_worker(set_threads=args.set_threads, device=args.device, with_yolo=not args.no_yolo)))
        return 0
    if args.worker == "pipeline":
        if args.video is None:
            print("[HATA] --worker pipeline için --video gerekir.", file=sys.stderr)
            return 2
        print(json.dumps(benchmark.run_pipeline_worker(video=args.video, device=args.device, opencv_threads=args.opencv_threads, max_frames=args.max_frames)))
        return 0

    script = Path(__file__).resolve()
    out = args.out or project_path("results", "face_association", f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with tempfile.TemporaryDirectory() as tmp:
        video = args.video or benchmark.make_benchmark_video(Path(tmp) / "montaj.mp4", args.frames)
        print(f"[BİLGİ] {benchmark.SYNTHETIC_DISCLAIMER}")
        print(f"[BİLGİ] Video: {video.name} | cihazlar: {args.devices} | iş parçacığı ayarları: {args.threads} | tekrar: {args.repeats}")
        report = benchmark.run_matrix(
            benchmark.subprocess_runner(script), video, devices=args.devices, threads=args.threads, repeats=args.repeats, max_frames=args.max_frames,
            probe_thread_values=args.probe_threads,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(benchmark.summary_to_markdown(report["summary"]))
    for probe in report["probes"]:
        stages = "; ".join(f"{s['stage']}: cv2={s['cv2_threads']} YuNet {s['yunet']['median_ms']:.1f} ms" + (f" YOLO {s['yolo']['median_ms']:.1f} ms" if s.get("yolo") else "")
                           for s in probe["stages"])
        print(f"[SONDA set={probe['set_threads']}] {stages}")
    print(f"[BİLGİ] JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
