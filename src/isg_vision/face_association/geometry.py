"""Kutu geometrisi ve (yüz, kişi) çifti için geometrik uygunluk skoru.

Saf fonksiyonlar: ``cv2``/``numpy``/model import edilmez, durum tutulmaz. Koordinat kuralı projenin tamamıyla aynıdır:
``bbox_xyxy = (x1, y1, x2, y2)``, piksel, sol-üst başlangıç, ``x1 < x2`` ve ``y1 < y2``.

Skor neyi ÖLÇER, neyi ÖLÇMEZ
---------------------------
Skor, "bu yüz kutusu bu kişi kutusunun KAFA bölgesine geometrik olarak uyuyor mu" sorusunun 0..1 yanıtıdır. **Kimlik doğruluğu, canlılık
veya yüz tespit güveni DEĞİLDİR** (yüz tespit skoru bu hesaba katılmaz; "Yüz tespit skoru çalışan kimlik doğruluğu olarak yorumlanmamalıdır").
Poster/ekrandaki yüz, bir kişi kutusunun kafa bölgesine denk gelirse yüksek skor alabilir: geometri bunu ayıramaz.

Formül (``F`` yüz kutusu, ``P`` kişi kutusu; ``Pw, Ph`` kişi kutusu genişliği/yüksekliği)
---------------------------------------------------------------------------------------
Önce SERT ELEMELER (herhangi biri sağlanmazsa çift aday DEĞİLDİR):

1. ``containment = alan(F ∩ P) / alan(F) >= min_containment``
2. ``cy = (F.merkez_y - P.y1) / Ph <= max_center_y_ratio``   (yüz kişinin üst bölgesinde)
3. ``min_face_height_ratio <= F.yükseklik / Ph <= max_face_height_ratio``   (makul yüz/kişi boyutu)

Adaylar için skor, dört terimin ağırlıklı ortalamasıdır (hepsi 0..1):

* ``c`` (ağırlık 0,30): ``clamp((containment - 0,5) / 0,5)``  — yüz kutunun içinde ne kadar tam.
* ``v`` (0,35): dikey konum; ``cy <= 0,30`` ise 1, ``cy >= max_center_y_ratio`` ise 0, arası doğrusal — yüz üstte mi.
* ``h`` (0,20): yatay merkezlilik; ``dx = |F.merkez_x - P.merkez_x| / Pw``; ``dx <= 0,20`` ise 1, ``dx >= 0,5`` ise 0, arası doğrusal.
* ``z`` (0,15): boyut tipikliği; ``r = F.yükseklik / Ph``; ``0,07 <= r <= 0,5`` ise 1, uçlara (``min``/``max`` oran) doğru doğrusal azalır.

Kadraj kenarındaki (``at_border``) yüzün skoru ``border_factor`` ile çarpılır (kısmen görünen yüzün kutusu daha az güvenilirdir).

Ağırlıklar ve plato sınırları (0,30 / 0,20 / 0,07 / 0,5 …) gerçek kare ölçümlerinden optimize EDİLMEMİŞ, anatomik ve kamera geometrisi
gerekçeleriyle seçilmiş makul başlangıç değerleridir (``docs/face_association.md``); ayarlanabilir eşikler ``AssociationConfig``'dedir.
"""

from __future__ import annotations

from dataclasses import dataclass

from isg_vision.human_detection.results import BBox

from .config import AssociationConfig

WEIGHT_CONTAINMENT = 0.30
WEIGHT_VERTICAL = 0.35
WEIGHT_HORIZONTAL = 0.20
WEIGHT_SIZE = 0.15

_V_FULL_Y = 0.30  # yüz merkezi bu dikey orana kadar tam puan
_H_FULL_DX = 0.20  # yatay sapma bu orana kadar tam puan
_Z_PLATEAU = (0.07, 0.5)  # yüz/kişi yükseklik oranı bu aralıkta tam puan


def box_area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def iou(a: BBox, b: BBox) -> float:
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def containment(inner: BBox, outer: BBox) -> float:
    """``inner`` kutusunun alanının ne kadarının ``outer`` içinde olduğu (0..1); ``inner`` dejenere ise 0."""
    area = box_area(inner)
    return intersection_area(inner, outer) / area if area > 0.0 else 0.0


def overlap_fraction(a: BBox, b: BBox) -> float:
    """İki kutunun ortak alanının KÜÇÜK kutunun alanına oranı (0..1). Bir kutu diğerinin içindeyse 1'dir (simetriktir)."""
    smaller = min(box_area(a), box_area(b))
    return intersection_area(a, b) / smaller if smaller > 0.0 else 0.0


def center(box: BBox) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _ramp_down(x: float, full: float, zero: float) -> float:
    """``x <= full`` için 1, ``x >= zero`` için 0, arası doğrusal (``zero > full``)."""
    if x <= full:
        return 1.0
    if x >= zero:
        return 0.0
    return (zero - x) / (zero - full)


@dataclass(frozen=True)
class PairScore:
    """Aday bir (yüz, kişi) çiftinin geometrik ölçüleri ve skoru.

    Attributes:
        score: Nihai geometrik uygunluk skoru, ``0..1`` (``at_border`` çarpanı uygulanmış). Kimlik doğruluğu DEĞİLDİR.
        containment: Yüz alanının kişi kutusu içinde kalan oranı.
        center_y_ratio: Yüz merkezinin kişi kutusundaki dikey konumu (üstten 0).
        center_x_offset: Yüz merkezinin kişi merkezinden yatay sapması (kutu genişliğine oranla, ``0..0,5``).
        height_ratio: Yüz yüksekliği / kişi yüksekliği.
        terms: ``(c, v, h, z)`` terim değerleri (hata ayıklama ve testler için).
    """

    score: float
    containment: float
    center_y_ratio: float
    center_x_offset: float
    height_ratio: float
    terms: tuple[float, float, float, float]


def gate_failure(face: BBox, person: BBox, config: AssociationConfig) -> str | None:
    """Çift SERT elemelerden birine takılıyorsa nedeni (``containment`` / ``vertical`` / ``size``), aksi halde ``None``."""
    if containment(face, person) < config.min_containment:
        return "containment"
    person_h = person[3] - person[1]
    if person_h <= 0.0:
        return "size"
    cy = (center(face)[1] - person[1]) / person_h
    if cy > config.max_center_y_ratio:
        return "vertical"
    ratio = (face[3] - face[1]) / person_h
    if not config.min_face_height_ratio <= ratio <= config.max_face_height_ratio:
        return "size"
    return None


def score_pair(face: BBox, person: BBox, config: AssociationConfig, *, at_border: bool = False) -> PairScore | None:
    """(yüz, kişi) çiftinin geometrik skorunu hesaplar; sert elemelere takılırsa ``None`` döndürür (bkz. modül açıklaması)."""
    if gate_failure(face, person, config) is not None:
        return None
    person_w = person[2] - person[0]
    person_h = person[3] - person[1]
    fx, fy = center(face)
    px, _ = center(person)
    cont = containment(face, person)
    cy = (fy - person[1]) / person_h
    dx = abs(fx - px) / person_w if person_w > 0.0 else 0.5
    ratio = (face[3] - face[1]) / person_h

    c = _clamp01((cont - 0.5) / 0.5)
    v = _ramp_down(cy, _V_FULL_Y, config.max_center_y_ratio)
    h = _ramp_down(dx, _H_FULL_DX, 0.5)
    lo, hi = _Z_PLATEAU
    z = 1.0
    if ratio < lo and lo > config.min_face_height_ratio:
        z = _clamp01((ratio - config.min_face_height_ratio) / (lo - config.min_face_height_ratio))
    elif ratio > hi and config.max_face_height_ratio > hi:
        z = _clamp01((config.max_face_height_ratio - ratio) / (config.max_face_height_ratio - hi))
    score = WEIGHT_CONTAINMENT * c + WEIGHT_VERTICAL * v + WEIGHT_HORIZONTAL * h + WEIGHT_SIZE * z
    if at_border:
        score *= config.border_factor
    return PairScore(_clamp01(score), cont, cy, dx, ratio, (c, v, h, z))
