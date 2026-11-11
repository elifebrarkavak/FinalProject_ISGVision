"""Roboflow `NO-Hardhat` ve `NO-Safety Vest` kutuları için AÇIK, test edilebilir kalite ölçütleri.

Ölçütler yalnızca KUTU/GÖRÜNTÜ özelliklerine bakar; hiçbir etiket dosyası değiştirilmez. Görüntü seviyesinde karar
(approved / quarantine / needs_reannotation) usage_policy.py içinde verilir: etiketsiz hedef nesne bırakmamak için
başarısız kutu tek başına SİLİNMEZ, kutuyu içeren görüntü kullanım dışı bırakılır.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from isg_vision.dataset_audit.semantic import ImageBoxes, RoleBox, assign_to_person, center_inside, iou

# NO-Hardhat (tam baş kutusu ile uyumlu alt küme)
NH_MIN_HEIGHT_PX = 32
NH_ASPECT_RANGE = (0.9, 1.8)          # h/w: şerit (<0.9) ve aşırı dar kırpıkları dışlar
NH_BORDER_MARGIN_PX = 2
# NO-Safety Vest (gövde kutusu ile uyumlu alt küme)
NV_MIN_HEIGHT_PX = 48
NV_ASPECT_RANGE = (0.8, 3.0)
NV_MIN_REL_PERSON_HEIGHT = 0.25
NV_HIVIS_FRACTION_MAX = 0.12
OVERLAP_IOU_MAX = 0.05


@dataclass
class FilterResult:
    passed: bool
    reasons: list[str]
    metrics: dict[str, Any]


def _touches_border(rb: RoleBox, im: ImageBoxes, margin: int) -> bool:
    return (rb.x1 * im.width < margin or rb.y1 * im.height < margin
            or rb.x2 * im.width > im.width - margin or rb.y2 * im.height > im.height - margin)


def hivis_fraction(image_bgr: np.ndarray, rb: RoleBox) -> float:
    """Kutu içinde floresan sarı-yeşil / turuncu (yüksek görünürlüklü) piksel oranı (HSV eşikleri)."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = max(0, int(rb.x1 * w)), max(0, int(rb.y1 * h)), min(w, int(rb.x2 * w)), min(h, int(rb.y2 * h))
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lime = (hh >= 25) & (hh <= 45) & (ss >= 100) & (vv >= 140)
    orange = (hh >= 5) & (hh <= 18) & (ss >= 160) & (vv >= 180)
    return float((lime | orange).mean())


def evaluate_no_hardhat(im: ImageBoxes, rb: RoleBox) -> FilterResult:
    reasons: list[str] = []
    h_px, w_px = rb.box.h * im.height, rb.box.w * im.width
    aspect = h_px / w_px if w_px else 0.0
    if h_px < NH_MIN_HEIGHT_PX:
        reasons.append(f"yükseklik<{NH_MIN_HEIGHT_PX}px")
    if not (NH_ASPECT_RANGE[0] <= aspect <= NH_ASPECT_RANGE[1]):
        reasons.append(f"h/w={aspect:.2f} aralık dışı (şerit/kırpık)")
    if _touches_border(rb, im, NH_BORDER_MARGIN_PX):
        reasons.append("görüntü kenarına değiyor (kırpık)")
    hats = [b for b in im.boxes if b.role == "helmet"]
    if any(iou(rb, b) > OVERLAP_IOU_MAX or center_inside(b, rb) or center_inside(rb, b) for b in hats):
        reasons.append("Hardhat kutusuyla örtüşüyor")
    persons = [b for b in im.boxes if b.role == "person"]
    if persons and assign_to_person(rb, persons) is None:
        reasons.append("kişi kutusuna atanamadı")
    return FilterResult(not reasons, reasons, {"height_px": round(h_px, 1), "aspect_h_over_w": round(aspect, 2)})


def evaluate_no_vest(im: ImageBoxes, rb: RoleBox, image_bgr: np.ndarray | None = None) -> tuple[FilterResult, bool]:
    """(sonuç, yüksek_görünürlük_şüphesi). Yüksek görünürlük şüphesi yeniden etiketleme gerektirir (needs_reannotation)."""
    reasons: list[str] = []
    h_px, w_px = rb.box.h * im.height, rb.box.w * im.width
    aspect = h_px / w_px if w_px else 0.0
    if h_px < NV_MIN_HEIGHT_PX:
        reasons.append(f"yükseklik<{NV_MIN_HEIGHT_PX}px")
    if not (NV_ASPECT_RANGE[0] <= aspect <= NV_ASPECT_RANGE[1]):
        reasons.append(f"h/w={aspect:.2f} aralık dışı")
    if _touches_border(rb, im, NH_BORDER_MARGIN_PX):
        reasons.append("görüntü kenarına değiyor (kırpık)")
    persons = [b for b in im.boxes if b.role == "person"]
    p = assign_to_person(rb, persons) if persons else None
    if persons and p is None:
        reasons.append("kişi kutusuna atanamadı")
    if p is not None and rb.box.h / p.box.h < NV_MIN_REL_PERSON_HEIGHT:
        reasons.append("kişi yüksekliğine göre çok küçük")
    vests = [b for b in im.boxes if b.role == "vest"]
    if any(iou(rb, b) > OVERLAP_IOU_MAX or center_inside(b, rb) or center_inside(rb, b) for b in vests):
        reasons.append("Safety Vest kutusuyla örtüşüyor")
    hat = [b for b in im.boxes if b.role == "helmet"]
    if any(center_inside(b, rb) for b in hat):
        reasons.append("içinde Hardhat kutusu var (baş bölgesi)")
    hivis = False
    frac = None
    if image_bgr is not None:
        frac = hivis_fraction(image_bgr, rb)
        if frac > NV_HIVIS_FRACTION_MAX:
            hivis = True
            reasons.append(f"yüksek görünürlüklü renk oranı {frac:.2f}")
    return FilterResult(not reasons, reasons, {"height_px": round(h_px, 1), "aspect_h_over_w": round(aspect, 2),
                                               "hivis_fraction": None if frac is None else round(frac, 3)}), hivis
