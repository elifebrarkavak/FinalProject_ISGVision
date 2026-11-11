"""İnsan tespiti için nesnel değerlendirme altyapısı (Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95).

**Bu modül bir ALTYAPIDIR; kendi başına hiçbir doğruluk sonucu üretmez.** Anlamlı bir sonuç için etiketli, bağımsız bir insan
tespit test kümesi gerekir (bkz. docs/person_detection.md, "Değerlendirme verisi"). Böyle bir küme henüz yoktur; bu yüzden proje raporlarında
hiçbir Precision/Recall/F1/mAP değeri yoktur. Burada yazılan matematik, bilinen sonuçları olan sentetik örneklerle birim testlerinde doğrulanır.

Tanımlar (tek sınıf: ``person``)
--------------------------------
* Bir tespit, aynı karedeki henüz eşleşmemiş bir etiketle IoU >= eşik ise **doğru pozitif (TP)** olur. Tespitler güvene göre azalan sırayla
  işlenir ve her biri IoU'su en yüksek eşleşmemiş etikete (açgözlü) bağlanır; bağlanamayan tespit **yanlış pozitif (FP)**, hiçbir
  tespitle eşleşmeyen etiket **kaçırılan (FN)** olur.
* **Precision** = TP/(TP+FP); **Recall** = TP/(TP+FN); **F1** = 2PR/(P+R). Bunlar tek bir güven eşiğinde (``conf_threshold``) ve IoU 0.5'te hesaplanır.
* **AP**: güven eşiği uygulanmamış (düşük eşikli) tüm tespitlerle, precision-recall eğrisinin COCO tarzı 101 noktalı enterpolasyonlu alanı.
  **mAP@0.5** = IoU 0.5'te AP; **mAP@0.5:0.95** = IoU 0.50, 0.55, ..., 0.95 eşiklerindeki AP ortalaması. Tek sınıf olduğu için mAP = AP'dir.
* **Güven skoru doğruluk değildir**: modelin kutusuna duyduğu kalibre edilmemiş güvendir. Bu modül güveni yalnızca sıralama ve eşikleme için kullanır.

Yalnızca NumPy kullanır (torch/OpenCV/ultralytics import etmez).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IOU_THRESHOLDS_COCO: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))  # 0.50 ... 0.95
RECALL_POINTS = 101


@dataclass(frozen=True)
class EvalSample:
    """Tek görüntünün etiketleri ve tespitleri (hepsi piksel ``xyxy`` koordinatında).

    Attributes:
        gt_boxes: Etiketli insan kutuları, şekil ``(N, 4)``.
        det_boxes: Model tespitleri, şekil ``(M, 4)``.
        det_scores: Tespit güven skorları, şekil ``(M,)``.
    """

    gt_boxes: np.ndarray
    det_boxes: np.ndarray
    det_scores: np.ndarray

    def __post_init__(self) -> None:
        gt = np.asarray(self.gt_boxes, dtype=np.float64).reshape(-1, 4)
        det = np.asarray(self.det_boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(self.det_scores, dtype=np.float64).reshape(-1)
        if len(det) != len(scores):
            raise ValueError(f"det_boxes ({len(det)}) ve det_scores ({len(scores)}) aynı uzunlukta olmalıdır")
        object.__setattr__(self, "gt_boxes", gt)
        object.__setattr__(self, "det_boxes", det)
        object.__setattr__(self, "det_scores", scores)


@dataclass(frozen=True)
class EvaluationReport:
    """Değerlendirme sonucu. ``num_images`` ve ``num_gt`` her zaman raporlanmalıdır: küçük örneklem sonuçları genellenemez."""

    num_images: int
    num_gt: int
    num_detections_at_conf: int
    conf_threshold: float
    iou_threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    map50: float
    map50_95: float
    ap_per_iou: dict[float, float]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a`` (N,4) ve ``b`` (M,4) ``xyxy`` kutuları arasındaki IoU matrisi (N, M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(bottom_right - top_left, 0.0, None).prod(axis=2)
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_detections(sample: EvalSample, iou_threshold: float, min_score: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Bir görüntüdeki tespitleri etiketlerle eşler.

    Returns:
        ``(skorlar, tp)``: ``min_score`` ve üzerindeki tespitlerin güven skorları (azalan sırada) ve her birinin TP olup olmadığı (bool).
    """
    keep = sample.det_scores >= min_score
    order = np.argsort(-sample.det_scores[keep], kind="stable")
    boxes, scores = sample.det_boxes[keep][order], sample.det_scores[keep][order]
    ious = iou_matrix(boxes, sample.gt_boxes)
    taken = np.zeros(len(sample.gt_boxes), dtype=bool)
    tp = np.zeros(len(boxes), dtype=bool)
    for i in range(len(boxes)):
        if ious.shape[1] == 0:
            break
        candidates = np.where(taken, -1.0, ious[i])
        j = int(np.argmax(candidates))
        if candidates[j] >= iou_threshold:
            taken[j] = True
            tp[i] = True
    return scores, tp


def average_precision(scores: np.ndarray, tp: np.ndarray, num_gt: int) -> float:
    """COCO tarzı (101 noktalı enterpolasyon) AP. Etiket yoksa ``nan`` (tanımsız)."""
    if num_gt == 0:
        return math.nan
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp_sorted = tp[order].astype(np.float64)
    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(1.0 - tp_sorted)
    recall = cum_tp / num_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]  # her recall için o noktadan sonraki en yüksek precision
    points = np.linspace(0.0, 1.0, RECALL_POINTS)
    index = np.searchsorted(recall, points, side="left")
    values = np.where(index < len(envelope), envelope[np.minimum(index, len(envelope) - 1)], 0.0)
    return float(values.mean())


def evaluate_detections(
    samples: Sequence[EvalSample],
    *,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    iou_thresholds: Sequence[float] = IOU_THRESHOLDS_COCO,
) -> EvaluationReport:
    """Veri kümesi düzeyinde Precision/Recall/F1 (``conf_threshold``, ``iou_threshold``'da) ile mAP@0.5 ve mAP@0.5:0.95 hesaplar.

    ``samples`` içindeki tespitler **düşük güven eşiğiyle** (ör. 0.001) alınmış olmalıdır; AP, tam precision-recall eğrisini gerektirir.
    Precision/Recall/F1 ise yalnızca ``conf_threshold`` ve üzerindeki tespitlerle hesaplanır.
    """
    if not samples:
        raise ValueError("En az bir görüntü gerekir.")
    if not 0.0 <= conf_threshold <= 1.0 or not 0.0 < iou_threshold <= 1.0:
        raise ValueError("conf_threshold [0,1], iou_threshold (0,1] aralığında olmalıdır.")
    num_gt = int(sum(len(s.gt_boxes) for s in samples))

    tp_total = fp_total = at_conf = 0
    for sample in samples:
        _, tp = match_detections(sample, iou_threshold, conf_threshold)
        tp_total += int(tp.sum())
        fp_total += int((~tp).sum())
        at_conf += len(tp)
    fn_total = num_gt - tp_total
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else math.nan
    recall = tp_total / num_gt if num_gt else math.nan
    if tp_total == 0:
        f1 = 0.0 if num_gt > 0 else math.nan  # hiç doğru pozitif yoksa F1 = 0 (etiket varsa); etiket yoksa tanımsız
    else:
        f1 = 2 * precision * recall / (precision + recall)

    ap_per_iou: dict[float, float] = {}
    for threshold in iou_thresholds:
        scores_parts, tp_parts = [], []
        for sample in samples:
            scores, tp = match_detections(sample, threshold)
            scores_parts.append(scores)
            tp_parts.append(tp)
        ap_per_iou[float(threshold)] = average_precision(np.concatenate(scores_parts), np.concatenate(tp_parts), num_gt)
    ap50 = ap_per_iou.get(0.5, math.nan)
    return EvaluationReport(
        num_images=len(samples), num_gt=num_gt, num_detections_at_conf=at_conf, conf_threshold=conf_threshold,
        iou_threshold=iou_threshold, true_positives=tp_total, false_positives=fp_total, false_negatives=fn_total,
        precision=precision, recall=recall, f1=f1, map50=ap50,
        map50_95=float(np.mean(list(ap_per_iou.values()))) if ap_per_iou else math.nan, ap_per_iou=ap_per_iou,
    )


def load_yolo_label_file(path: Path, image_size: tuple[int, int], person_class_id: int = 0) -> np.ndarray:
    """YOLO biçimli etiket dosyasını (``sınıf cx cy w h``, normalize) piksel ``xyxy`` kutularına çevirir; yalnızca ``person_class_id``.

    Dosya yoksa boş dizi döner (etiketsiz = insan yok kabul edilir; çağıran bunun bilinçli bir veri kararı olduğundan emin olmalıdır).
    Biçimi bozuk satırlar ``ValueError`` verir (sessizce atlanmaz).
    """
    width, height = image_size
    path = Path(path)
    if not path.is_file():
        return np.zeros((0, 4))
    boxes: list[list[float]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{path}:{number}: 5 alan beklenir (sınıf cx cy w h): {line!r}")
        cls, cx, cy, w, h = int(float(parts[0])), *(float(v) for v in parts[1:])
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
            raise ValueError(f"{path}:{number}: koordinatlar 0..1 aralığında olmalıdır: {line!r}")
        if cls == person_class_id:
            boxes.append([(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height])
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
