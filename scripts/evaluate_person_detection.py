#!/usr/bin/env python
"""Etiketli bir görüntü kümesinde insan tespiti doğruluğunu ölçer (Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95).

ÖNEMLİ: Bu betik bir ALTYAPIDIR. Anlamlı bir sonuç için **bağımsız, etiketli, temsili** bir insan test kümesi gerekir (bkz. docs/person_detection.md,
"Değerlendirme verisi"). Proje şu anda böyle bir kümeye sahip değildir; bu yüzden raporlarda hiçbir doğruluk sonucu yoktur. Küçük veya
temsili olmayan bir kümenin sonucu genellenemez; çıktıdaki görüntü ve etiket sayısı her zaman birlikte yorumlanmalıdır.

Etiket biçimi: görüntü ile aynı adlı ``.txt`` dosyası, satır başına ``sınıf cx cy w h`` (YOLO, normalize). Yalnızca ``--person-class-id`` sınıfı
(varsayılan 0) kullanılır. Etiket dosyası olmayan görüntü "insanı yok" sayılır (bilinçli bir veri kararı olmalıdır).

Kullanım:
    .venv/bin/python scripts/evaluate_person_detection.py --images /yol/goruntuler --labels /yol/etiketler --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.human_detection.config import DEFAULT_MODEL_PATH, DetectionConfig  # noqa: E402
from isg_vision.human_detection.errors import DetectionConfigError, DetectionError  # noqa: E402
from isg_vision.human_detection.evaluation import EvalSample, evaluate_detections, load_yolo_label_file  # noqa: E402
from isg_vision.human_detection.person_detector import PersonDetector  # noqa: E402
from isg_vision.utils.paths import project_path  # noqa: E402

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
COLLECT_CONF = 0.001  # AP için tam precision-recall eğrisi gerekir: tespitler düşük eşikle toplanır


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, help="Görüntü klasörü")
    parser.add_argument("--labels", required=True, help="YOLO .txt etiket klasörü (görüntüyle aynı ad)")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "mps"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25, help="Precision/Recall/F1 için güven eşiği (AP bu eşikten bağımsızdır)")
    parser.add_argument("--match-iou", type=float, default=0.5, help="Precision/Recall/F1 için IoU eşiği")
    parser.add_argument("--person-class-id", type=int, default=0, help="Etiket dosyalarındaki insan sınıfı kimliği")
    parser.add_argument("--output-dir", default=str(project_path("results", "person_detection")))
    args = parser.parse_args(argv)

    images_dir, labels_dir = Path(args.images), Path(args.labels)
    for label, folder in (("--images", images_dir), ("--labels", labels_dir)):
        if not folder.is_dir():
            parser.error(f"{label} bir klasör olmalıdır: {folder}")
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        parser.error(f"Klasörde görüntü yok: {images_dir}")

    try:
        detector = PersonDetector(DetectionConfig(model_path=args.model, conf=COLLECT_CONF, imgsz=args.imgsz, device=args.device))
        detector.load()
        samples = []
        for path in images:
            image = cv2.imread(str(path))
            if image is None:
                print(f"[HATA] Görüntü okunamadı: {path}", file=sys.stderr)
                return 1
            result = detector.detect(image)
            samples.append(EvalSample(
                gt_boxes=load_yolo_label_file(labels_dir / f"{path.stem}.txt", (image.shape[1], image.shape[0]), args.person_class_id),
                det_boxes=[d.bbox_xyxy for d in result.detections], det_scores=[d.confidence for d in result.detections],
            ))
        report = evaluate_detections(samples, conf_threshold=args.conf, iou_threshold=args.match_iou)
    except (DetectionConfigError, ValueError) as exc:
        parser.error(str(exc))
    except DetectionError as exc:
        print(f"[HATA] {exc}", file=sys.stderr)
        return 1
    finally:
        if "detector" in locals():
            detector.close()

    print(f"Görüntü: {report.num_images} | etiketli insan: {report.num_gt} | cihaz: {detector.device} | model: {Path(args.model).name}")
    print(f"conf>={report.conf_threshold}, IoU>={report.iou_threshold}: TP {report.true_positives} | FP {report.false_positives} | FN {report.false_negatives}")
    print(f"Precision {report.precision:.4f} | Recall {report.recall:.4f} | F1 {report.f1:.4f}")
    print(f"mAP@0.5 {report.map50:.4f} | mAP@0.5:0.95 {report.map50_95:.4f}")
    print("[UYARI] Sonuçlar yalnızca verilen kümeyi temsil eder; küçük/temsili olmayan kümeler genellenemez. Güven skoru doğruluk değildir.")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"evaluation_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({**report.__dict__, "ap_per_iou": {str(k): v for k, v in report.ap_per_iou.items()}}, indent=2), encoding="utf-8")
    print(f"[BİLGİ] JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
