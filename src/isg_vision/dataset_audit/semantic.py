"""Kutu semantiği analizi: baret/yelek kutularının kişiye göre konumu, boyutu ve çelişkili etiket tespiti.

Sınıf rolleri (helmet / no_helmet / vest / no_vest / person) ``configs/datasets.yaml`` içindeki ``roles``
alanından gelir; ID'ler kodda sabitlenmez. Hiçbir etiket dosyası değiştirilmez.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Iterator

import numpy as np

from isg_vision.dataset_audit.imageio import read_image_info
from isg_vision.dataset_audit.labels import Box, parse_label_file
from isg_vision.dataset_audit.layout import DatasetLayout, SPLIT_ORDER, iter_samples

ROLES = ("helmet", "no_helmet", "vest", "no_vest", "person")
HEAD_ROLES = ("helmet", "no_helmet")
TORSO_ROLES = ("vest", "no_vest")
ATTRIBUTE_PAIRS = (("helmet", "no_helmet"), ("vest", "no_vest"))


@dataclass
class RoleBox:
    role: str
    box: Box

    @property
    def x1(self) -> float: return self.box.xc - self.box.w / 2
    @property
    def x2(self) -> float: return self.box.xc + self.box.w / 2
    @property
    def y1(self) -> float: return self.box.yc - self.box.h / 2
    @property
    def y2(self) -> float: return self.box.yc + self.box.h / 2
    @property
    def area(self) -> float: return self.box.w * self.box.h


@dataclass
class ImageBoxes:
    dataset: str
    split: str
    image_path: Path
    label_path: Path
    width: int
    height: int
    boxes: list[RoleBox] = field(default_factory=list)


def class_roles(dataset_cfg: dict[str, Any]) -> dict[int, str]:
    """class_id -> rol (configs/datasets.yaml ``roles``)."""
    out: dict[int, str] = {}
    for role, ids in (dataset_cfg.get("roles") or {}).items():
        if role not in ROLES:
            raise ValueError(f"Bilinmeyen rol: {role}")
        for cid in ids:
            out[int(cid)] = role
    return out


def iter_images(layout: DatasetLayout, roles: dict[int, str]) -> Iterator[ImageBoxes]:
    """Rolü olan kutuları görüntü bazında üretir (yalnızca etiketi ve görüntüsü olan örnekler)."""
    for split in SPLIT_ORDER:
        sd = layout.splits.get(split)
        if sd is None:
            continue
        for s in iter_samples(sd):
            if s.image_path is None or s.label_path is None:
                continue
            res = parse_label_file(s.label_path)
            rb = [RoleBox(roles[b.class_id], b) for b in res.boxes if b.class_id in roles and b.w > 0 and b.h > 0]
            if not rb:
                continue
            info = read_image_info(s.image_path, full_decode=False)
            if not info.readable:
                continue
            yield ImageBoxes(layout.name, split, s.image_path, s.label_path, info.width, info.height, rb)


def _inter_area(a: RoleBox, b: RoleBox) -> float:
    iw = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    ih = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    return iw * ih


def iou(a: RoleBox, b: RoleBox) -> float:
    inter = _inter_area(a, b)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def center_inside(inner: RoleBox, outer: RoleBox) -> bool:
    return outer.x1 <= inner.box.xc <= outer.x2 and outer.y1 <= inner.box.yc <= outer.y2


def assign_to_person(item: RoleBox, persons: list[RoleBox]) -> RoleBox | None:
    """Kutunun merkezini içeren ve kutunun alanının çoğunu (>=%60) kapsayan en küçük kişi kutusu."""
    best = None
    for p in persons:
        if center_inside(item, p) and _inter_area(item, p) >= 0.6 * item.area:
            if best is None or p.area < best.area:
                best = p
    return best


def person_relative_stats(images: list[ImageBoxes]) -> dict[str, dict[str, Any]]:
    """Kişi kutusu olan görüntülerde baret/yelek kutularının kişiye göre boyut ve konumu."""
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    unassigned: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for im in images:
        persons = [b for b in im.boxes if b.role == "person"]
        if not persons:
            continue
        for rb in im.boxes:
            if rb.role == "person":
                continue
            totals[rb.role] += 1
            p = assign_to_person(rb, persons)
            if p is None:
                unassigned[rb.role] += 1
                continue
            ph, pw = p.box.h, p.box.w
            acc[rb.role]["rel_height"].append(rb.box.h / ph)
            acc[rb.role]["rel_width"].append(rb.box.w / pw)
            acc[rb.role]["top_offset"].append((rb.y1 - p.y1) / ph)
            acc[rb.role]["center_y"].append((rb.box.yc - p.y1) / ph)
            acc[rb.role]["aspect_px"].append((rb.box.h * im.height) / (rb.box.w * im.width))
    out: dict[str, dict[str, Any]] = {}
    for role, metrics in acc.items():
        row: dict[str, Any] = {"n_assigned": len(metrics["rel_height"]), "n_total": totals[role],
                               "n_unassigned": unassigned[role]}
        for name, vals in metrics.items():
            arr = np.array(vals)
            row[name] = {"median": round(float(np.median(arr)), 3), "p25": round(float(np.percentile(arr, 25)), 3),
                         "p75": round(float(np.percentile(arr, 75)), 3)}
        out[role] = row
    return out


def single_person_pair_stats(images: list[ImageBoxes]) -> dict[str, dict[str, Any]]:
    """Kişi sınıfından BAĞIMSIZ karşılaştırma: tam bir baş kutusu ve tam bir gövde kutusu olan görüntülerde
    baş/gövde oranları (veri setleri arası anlam karşılaştırması için ortak ölçü).
    """
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for im in images:
        heads = [b for b in im.boxes if b.role in HEAD_ROLES]
        torsos = [b for b in im.boxes if b.role in TORSO_ROLES]
        if len(heads) != 1 or len(torsos) != 1:
            continue
        h, t = heads[0], torsos[0]
        if abs(h.box.xc - t.box.xc) > max(t.box.w, h.box.w):  # farklı kişilere ait olabilir
            continue
        acc[h.role]["head_h_over_torso_h"].append(h.box.h / t.box.h)
        acc[h.role]["head_w_over_torso_w"].append(h.box.w / t.box.w)
        acc[h.role]["head_center_above_torso_top"].append((t.y1 - h.box.yc) / t.box.h)
        acc[h.role]["aspect_px"].append((h.box.h * im.height) / (h.box.w * im.width))
    out = {}
    for role, metrics in acc.items():
        row: dict[str, Any] = {"n_pairs": len(metrics["aspect_px"])}
        for name, vals in metrics.items():
            arr = np.array(vals)
            row[name] = {"median": round(float(np.median(arr)), 3), "p25": round(float(np.percentile(arr, 25)), 3),
                         "p75": round(float(np.percentile(arr, 75)), 3)}
        out[role] = row
    return out


def box_aspect_stats(images: list[ImageBoxes]) -> dict[str, dict[str, Any]]:
    """Rol başına piksel cinsinden en-boy oranı (h/w) ve görüntüye göre göreli alan."""
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for im in images:
        for rb in im.boxes:
            acc[rb.role]["aspect_px"].append((rb.box.h * im.height) / (rb.box.w * im.width))
            acc[rb.role]["area_frac"].append(rb.area)
            acc[rb.role]["height_px"].append(rb.box.h * im.height)
    out = {}
    for role, m in acc.items():
        out[role] = {"n": len(m["aspect_px"]),
                     "aspect_h_over_w_median": round(float(np.median(m["aspect_px"])), 3),
                     "area_frac_median": round(float(np.median(m["area_frac"])), 5),
                     "height_px_median": round(float(np.median(m["height_px"])), 1),
                     "height_px_p10": round(float(np.percentile(m["height_px"], 10)), 1)}
    return out


def find_semantic_conflicts(im: ImageBoxes, iou_thr: float = 0.1) -> list[dict[str, Any]]:
    """Aynı özniteliğin olumlu ve olumsuz sınıflarının (ör. helmet/no_helmet) çelişkili kullanımını bulur.

    Tür 1 ``overlap``: iki kutu IoU>=iou_thr ile örtüşüyor veya birinin merkezi diğerinin içinde.
    Tür 2 ``same_person``: aynı kişi kutusuna hem olumlu hem olumsuz kutu atanmış.
    Aynı görüntüde farklı kişilere ait helmet + no_helmet çelişki SAYILMAZ.
    """
    found: list[dict[str, Any]] = []
    persons = [b for b in im.boxes if b.role == "person"]
    for pos_role, neg_role in ATTRIBUTE_PAIRS:
        pos = [b for b in im.boxes if b.role == pos_role]
        neg = [b for b in im.boxes if b.role == neg_role]
        for a in pos:
            for b in neg:
                ov = iou(a, b)
                inside = center_inside(a, b) or center_inside(b, a)
                if ov >= iou_thr or inside:
                    found.append({"attribute": pos_role, "type": "overlap", "pos_line": a.box.line_no,
                                  "neg_line": b.box.line_no, "iou": round(ov, 3), "center_inside": inside,
                                  "person_line": ""})
        if persons:
            by_person: dict[int, dict[str, list[RoleBox]]] = defaultdict(lambda: {"pos": [], "neg": []})
            for role, key in ((pos_role, "pos"), (neg_role, "neg")):
                for rb in im.boxes:
                    if rb.role == role:
                        p = assign_to_person(rb, persons)
                        if p is not None:
                            by_person[p.box.line_no][key].append(rb)
            for p_line, d in by_person.items():
                if d["pos"] and d["neg"]:
                    already = {(f["pos_line"], f["neg_line"]) for f in found if f["attribute"] == pos_role}
                    for a in d["pos"]:
                        for b in d["neg"]:
                            if (a.box.line_no, b.box.line_no) not in already:
                                found.append({"attribute": pos_role, "type": "same_person",
                                              "pos_line": a.box.line_no, "neg_line": b.box.line_no,
                                              "iou": round(iou(a, b), 3), "center_inside": False,
                                              "person_line": p_line})
    return found


def classify_conflict(c: dict[str, Any]) -> str:
    """Otomatik ön sınıflandırma (kesin karar değildir; görsel incelemeyle teyit edilir)."""
    if c["type"] == "overlap" and (c["iou"] >= 0.5):
        return "likely_error"
    if c["type"] == "overlap" and c["center_inside"]:
        return "needs_review"
    return "needs_review"


def median_or_none(values: list[float]) -> float | None:
    return median(values) if values else None
