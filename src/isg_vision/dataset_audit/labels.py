"""YOLO detection etiket dosyalarını ayrıştırma ve doğrulama (bellekte yalnızca tek dosya tutulur).

Standart satır biçimi:  class_id x_center y_center width height   (5 sütun, 0-1 normalize)
5'ten fazla sütun segmentation/OBB/güven değeri olabilir; bunlar detection etiketi gibi İŞLENMEZ,
"UNSUPPORTED_FORMAT" olarak işaretlenir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

EPS = 1e-4          # normalize aralığı / görüntü sınırı toleransı
MIN_BOX_PIXELS = 2.0
NEAR_DUP_IOU = 0.95
CONFLICT_IOU = 0.3
MAX_BOXES_FOR_PAIRWISE = 300

# Şiddet: error = eğitime zarar verebilir, warning = şüpheli, info = bilgi (hata sayılmaz)
SEVERITY = {
    "LABEL_UNREADABLE": "error",
    "UNSUPPORTED_FORMAT": "error",
    "BAD_COLUMN_COUNT": "error",
    "NON_NUMERIC": "error",
    "CLASS_ID_INVALID": "error",
    "CLASS_ID_UNDEFINED": "error",
    "COORD_OUT_OF_RANGE": "error",
    "BOX_NONPOSITIVE_SIZE": "error",
    "BOX_OUTSIDE_IMAGE": "warning",
    "BOX_VERY_SMALL": "warning",
    "DUPLICATE_BOX": "warning",
    "NEAR_DUPLICATE_BOX": "warning",
    "CONFLICTING_OVERLAP": "warning",
    "ORPHAN_LABEL": "warning",
    "IMAGE_UNREADABLE": "error",
    "STEM_COLLISION": "warning",
    "EXIF_ORIENTATION": "info",
    "LABEL_MISSING": "info",
    "LABEL_EMPTY": "info",
}


@dataclass
class Box:
    class_id: int
    xc: float
    yc: float
    w: float
    h: float
    line_no: int


@dataclass
class Issue:
    code: str
    line_no: int | None = None
    detail: str = ""

    @property
    def severity(self) -> str:
        return SEVERITY.get(self.code, "warning")


@dataclass
class LabelResult:
    boxes: list[Box] = field(default_factory=list)       # geçerli biçimdeki (5 sütunlu, sayısal) kutular
    issues: list[Issue] = field(default_factory=list)
    n_lines: int = 0
    unsupported_rows: int = 0
    empty: bool = False
    readable: bool = True


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.xc - a.w / 2, a.yc - a.h / 2, a.xc + a.w / 2, a.yc + a.h / 2
    bx1, by1, bx2, by2 = b.xc - b.w / 2, b.yc - b.h / 2, b.xc + b.w / 2, b.yc + b.h / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def parse_label_file(
    path: Path,
    valid_class_ids: set[int] | None = None,
    image_size: tuple[int, int] | None = None,
) -> LabelResult:
    """Tek bir etiket dosyasını okur ve bulunan sorunları listeler. Dosyayı DEĞİŞTİRMEZ.

    valid_class_ids: YAML'dan gelen tanımlı sınıf ID'leri (None ise sınıf kontrolü yapılmaz).
    image_size: (genişlik, yükseklik) piksel; verilirse çok küçük kutu denetimi yapılır.
    """
    result = LabelResult()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        result.readable = False
        result.issues.append(Issue("LABEL_UNREADABLE", None, f"{type(exc).__name__}: {exc}"))
        return result

    lines = [(i, ln.strip()) for i, ln in enumerate(text.splitlines(), start=1)]
    lines = [(i, ln) for i, ln in lines if ln]
    result.n_lines = len(lines)
    if not lines:
        result.empty = True
        result.issues.append(Issue("LABEL_EMPTY", None, "Etiket dosyası boş (negatif örnek olabilir)"))
        return result

    seen: dict[tuple[int, float, float, float, float], int] = {}
    for line_no, line in lines:
        cols = line.split()
        if len(cols) > 5:
            result.unsupported_rows += 1
            kind = "segmentation/OBB poligonu olabilir" if len(cols) >= 7 and len(cols) % 2 == 1 \
                else "fazladan sütun (güven değeri olabilir)"
            result.issues.append(Issue("UNSUPPORTED_FORMAT", line_no, f"{len(cols)} sütun: {kind}"))
            continue
        if len(cols) < 5:
            result.issues.append(Issue("BAD_COLUMN_COUNT", line_no, f"{len(cols)} sütun (5 bekleniyor)"))
            continue
        try:
            nums = [float(c) for c in cols]
        except ValueError:
            result.issues.append(Issue("NON_NUMERIC", line_no, f"Sayısal olmayan değer: {line[:60]}"))
            continue
        if any(v != v or v in (float("inf"), float("-inf")) for v in nums):
            result.issues.append(Issue("NON_NUMERIC", line_no, "NaN/Inf değeri"))
            continue
        raw_id = nums[0]
        if raw_id != int(raw_id) or raw_id < 0:
            result.issues.append(Issue("CLASS_ID_INVALID", line_no, f"Sınıf numarası tam sayı/pozitif değil: {cols[0]}"))
            continue
        class_id = int(raw_id)
        xc, yc, w, h = nums[1:]
        box = Box(class_id, xc, yc, w, h, line_no)
        result.boxes.append(box)

        if valid_class_ids is not None and class_id not in valid_class_ids:
            result.issues.append(Issue("CLASS_ID_UNDEFINED", line_no, f"Sınıf {class_id} YAML'da tanımlı değil"))
        if any(v < -EPS or v > 1 + EPS for v in (xc, yc, w, h)):
            result.issues.append(Issue("COORD_OUT_OF_RANGE", line_no,
                                       f"xc={xc:.5f} yc={yc:.5f} w={w:.5f} h={h:.5f} (0-1 dışı)"))
        if w <= 0 or h <= 0:
            result.issues.append(Issue("BOX_NONPOSITIVE_SIZE", line_no, f"w={w:.6f} h={h:.6f}"))
            continue
        over = max(0.0, -(xc - w / 2), -(yc - h / 2), (xc + w / 2) - 1, (yc + h / 2) - 1)
        if over > EPS:
            result.issues.append(Issue("BOX_OUTSIDE_IMAGE", line_no,
                                       f"Kutu görüntü sınırını normalize {over:.4f} aşıyor"))
        if image_size:
            if w * image_size[0] < MIN_BOX_PIXELS or h * image_size[1] < MIN_BOX_PIXELS:
                result.issues.append(Issue(
                    "BOX_VERY_SMALL", line_no,
                    f"{w * image_size[0]:.1f}x{h * image_size[1]:.1f} piksel"))
        key = (class_id, round(xc, 6), round(yc, 6), round(w, 6), round(h, 6))
        if key in seen:
            result.issues.append(Issue("DUPLICATE_BOX", line_no, f"Satır {seen[key]} ile birebir aynı kutu"))
        else:
            seen[key] = line_no

    # Aynı sınıfta çok yüksek örtüşen (neredeyse aynı) kutular
    if 1 < len(result.boxes) <= MAX_BOXES_FOR_PAIRWISE:
        exact = {line for line in (i.line_no for i in result.issues if i.code == "DUPLICATE_BOX")}
        boxes = [b for b in result.boxes if b.w > 0 and b.h > 0 and b.line_no not in exact]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if boxes[i].class_id == boxes[j].class_id and _iou(boxes[i], boxes[j]) >= NEAR_DUP_IOU:
                    result.issues.append(Issue(
                        "NEAR_DUPLICATE_BOX", boxes[j].line_no,
                        f"Satır {boxes[i].line_no} ile IoU>={NEAR_DUP_IOU} (aynı sınıf)"))
    return result


def conflicting_overlaps(boxes: list[Box], pairs: list[tuple[int, int]], iou_thr: float = CONFLICT_IOU) -> list[Issue]:
    """Birbirini dışlaması gereken sınıf çiftlerinin (ör. Hardhat / NO-Hardhat) aynı bölgede örtüşmesini bulur.

    Çiftler veri setine özgüdür (configs/datasets.yaml, ``contradiction_pairs``). Örtüşme şüpheli etikettir;
    kesin hata olduğu görsel inceleme ile doğrulanmalıdır.
    """
    issues: list[Issue] = []
    if not pairs or len(boxes) > MAX_BOXES_FOR_PAIRWISE:
        return issues
    for a_id, b_id in pairs:
        group_a = [b for b in boxes if b.class_id == a_id and b.w > 0 and b.h > 0]
        group_b = [b for b in boxes if b.class_id == b_id and b.w > 0 and b.h > 0]
        for a in group_a:
            for b in group_b:
                iou = _iou(a, b)
                if iou >= iou_thr:
                    issues.append(Issue("CONFLICTING_OVERLAP", a.line_no,
                                        f"Sınıf {a_id} (satır {a.line_no}) ile sınıf {b_id} (satır {b.line_no}) "
                                        f"aynı bölgede, IoU={iou:.2f}"))
    return issues
