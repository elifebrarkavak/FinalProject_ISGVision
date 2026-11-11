"""Etiketleri görüntü üzerine çizerek görsel doğrulama örnekleri üretir.

Orijinal görüntülere dokunulmaz: çıktılar yeni dosyalar olarak ayrı klasöre yazılır.
Seçim sabit bir tohumla (seed) yapılır, sonuçlar tekrarlanabilir.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from isg_vision.dataset_audit.labels import Box, parse_label_file
from isg_vision.dataset_audit.layout import DatasetLayout, SPLIT_ORDER, iter_samples

MAX_SIDE = 1280
TILE_W = 480


@dataclass(frozen=True)
class Candidate:
    split: str
    image_path: Path
    label_path: Path


def _palette(n: int) -> list[tuple[int, int, int]]:
    rng = random.Random(7)
    base = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
            (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190), (0, 128, 128), (170, 110, 40)]
    while len(base) < n:
        base.append((rng.randint(60, 255), rng.randint(60, 255), rng.randint(60, 255)))
    return [(b, g, r) for r, g, b in base[:n]]  # OpenCV BGR


def index_by_class(layout: DatasetLayout) -> tuple[dict[int, list[Candidate]], list[Candidate]]:
    """class_id -> bu sınıfı içeren görüntüler; ayrıca etiketi olan tüm görüntüler."""
    by_class: dict[int, list[Candidate]] = defaultdict(list)
    everything: list[Candidate] = []
    for split in SPLIT_ORDER:
        sd = layout.splits.get(split)
        if sd is None:
            continue
        for s in iter_samples(sd):
            if s.image_path is None or s.label_path is None:
                continue
            res = parse_label_file(s.label_path)
            cand = Candidate(split, s.image_path, s.label_path)
            everything.append(cand)
            for cid in {b.class_id for b in res.boxes}:
                by_class[cid].append(cand)
    return by_class, everything


def select_samples(by_class: dict[int, list[Candidate]], everything: list[Candidate],
                   n: int, seed: int) -> list[tuple[Candidate, str]]:
    """Her sınıfı (ve bölümü) temsil edecek şekilde en fazla n görüntü seçer. (aday, seçim_nedeni) döner."""
    rng = random.Random(seed)
    chosen: dict[Path, tuple[Candidate, str]] = {}
    classes = sorted(by_class, key=lambda c: len(by_class[c]))  # nadir sınıflar önce
    quota = max(2, min(6, math.ceil(n / max(1, len(classes)))))
    for cid in classes:
        buckets: dict[str, list[Candidate]] = defaultdict(list)
        for cand in by_class[cid]:
            buckets[cand.split].append(cand)
        for b in buckets.values():
            rng.shuffle(b)
        have = sum(1 for cand, _ in chosen.values() if cand in by_class[cid])
        order = [s for s in SPLIT_ORDER if s in buckets]
        i = 0
        while have < quota and len(chosen) < n and any(buckets.values()):
            split = order[i % len(order)]
            i += 1
            while buckets[split]:
                cand = buckets[split].pop()
                if cand.image_path not in chosen:
                    chosen[cand.image_path] = (cand, f"sınıf {cid}")
                    have += 1
                    break
    pool = [c for c in everything if c.image_path not in chosen]
    rng.shuffle(pool)
    for cand in pool:
        if len(chosen) >= n:
            break
        chosen[cand.image_path] = (cand, "rastgele")
    return sorted(chosen.values(), key=lambda t: (t[0].split, t[0].image_path.name))


def _to_xyxy(b: Box, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = int(round((b.xc - b.w / 2) * w)); y1 = int(round((b.yc - b.h / 2) * h))
    x2 = int(round((b.xc + b.w / 2) * w)); y2 = int(round((b.yc + b.h / 2) * h))
    return x1, y1, x2, y2


def annotate_image(image: np.ndarray, boxes: list[Box], names: dict[int, str],
                   colors: list[tuple[int, int, int]]) -> np.ndarray:
    """Kutuları ve gerçek sınıf adlarını bir KOPYA üzerine çizer (orijinal dizi değişmez)."""
    out = image.copy()
    h, w = out.shape[:2]
    thick = max(2, round(max(h, w) / 400))
    scale = max(0.5, max(h, w) / 1400)
    for b in boxes:
        x1, y1, x2, y2 = _to_xyxy(b, w, h)
        color = colors[b.class_id % len(colors)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        label = f"{b.class_id}:{names.get(b.class_id, '?')}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ty = y1 - 4 if y1 - th - 6 > 0 else y1 + th + 6
        cv2.rectangle(out, (x1, ty - th - 3), (x1 + tw + 2, ty + 3), color, -1)
        cv2.putText(out, label, (x1 + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _fit(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(image, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else image


def _caption(tile: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((22, tile.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(bar, text[:70], (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, tile])


def contact_sheet(tiles: list[np.ndarray], cols: int = 4, tile_w: int = TILE_W) -> np.ndarray:
    """Eşit genişlikli karoları ızgarada birleştirir."""
    resized = []
    for t in tiles:
        s = tile_w / t.shape[1]
        resized.append(cv2.resize(t, (tile_w, max(1, int(t.shape[0] * s))), interpolation=cv2.INTER_AREA))
    rows = []
    for i in range(0, len(resized), cols):
        row = resized[i:i + cols]
        height = max(t.shape[0] for t in row)
        padded = [cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
                  for t in row]
        while len(padded) < cols:
            padded.append(np.full((height, tile_w, 3), 20, dtype=np.uint8))
        rows.append(np.hstack(padded))
    return np.vstack(rows)


def render_dataset_samples(layout: DatasetLayout, out_dir: Path, n: int = 40, seed: int = 42,
                           per_sheet: int = 12) -> list[dict[str, Any]]:
    """Seçilen görüntüleri tek tek ve contact sheet olarak kaydeder. Seçim listesini döndürür."""
    names = layout.classes.names
    colors = _palette(max(names, default=0) + 1 if names else 1)
    by_class, everything = index_by_class(layout)
    selection = select_samples(by_class, everything, n, seed)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    tiles, records = [], []
    for cand, reason in selection:
        image = cv2.imread(str(cand.image_path))
        if image is None:
            records.append({"split": cand.split, "file": cand.image_path.name, "reason": reason, "status": "okunamadı"})
            continue
        boxes = parse_label_file(cand.label_path).boxes
        annotated = _fit(annotate_image(image, boxes, names, colors), MAX_SIDE)
        out_name = f"{cand.split}__{cand.image_path.stem[:60]}.jpg"
        cv2.imwrite(str(img_dir / out_name), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        tiles.append(_caption(_fit(annotated, TILE_W * 2), f"{cand.split} | {cand.image_path.stem[:45]}"))
        classes = sorted({b.class_id for b in boxes})
        records.append({"split": cand.split, "file": cand.image_path.name, "reason": reason, "status": "ok",
                        "n_boxes": len(boxes), "classes": " ".join(f"{c}:{names.get(c, '?')}" for c in classes)})
    for i in range(0, len(tiles), per_sheet):
        sheet = contact_sheet(tiles[i:i + per_sheet])
        cv2.imwrite(str(out_dir / f"contact_sheet_{i // per_sheet + 1:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return records


def render_class_crops(layout: DatasetLayout, out_dir: Path, per_class: int = 24, seed: int = 42,
                       margin: float = 0.25, tile: int = 220) -> dict[int, int]:
    """Her sınıf için kutu KIRPMALARINDAN oluşan sayfa üretir: kutu neyi kapsıyor sorusunu görmek için.

    Kırpma kutudan %margin daha geniş alınır; kutunun kendisi çizilir. Sınıf -> kırpma sayısı döner.
    """
    names = layout.classes.names
    colors = _palette(max(names, default=0) + 1 if names else 1)
    rng = random.Random(seed)
    pool: dict[int, list[tuple[str, Path, Box]]] = defaultdict(list)
    for split in SPLIT_ORDER:
        sd = layout.splits.get(split)
        if sd is None:
            continue
        for s in iter_samples(sd):
            if s.image_path is None or s.label_path is None:
                continue
            for b in parse_label_file(s.label_path).boxes:
                if b.w > 0 and b.h > 0:
                    pool[b.class_id].append((split, s.image_path, b))
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[int, int] = {}
    for cid in sorted(pool):
        items = pool[cid]
        rng.shuffle(items)
        # Bölümlere dengeli: önce her bölümden sırayla
        by_split: dict[str, list] = defaultdict(list)
        for it in items:
            by_split[it[0]].append(it)
        picked = []
        while len(picked) < per_class and any(by_split.values()):
            for sp in list(by_split):
                if by_split[sp] and len(picked) < per_class:
                    picked.append(by_split[sp].pop())
        tiles = []
        for split, img_path, b in picked:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            h, w = image.shape[:2]
            x1, y1, x2, y2 = _to_xyxy(b, w, h)
            mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
            cx1, cy1, cx2, cy2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
            crop = image[cy1:cy2, cx1:cx2].copy()
            if crop.size == 0:
                continue
            cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), colors[cid % len(colors)], 2)
            s = tile / max(crop.shape[:2])
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
            canvas = np.full((tile, tile, 3), 20, dtype=np.uint8)
            canvas[:crop.shape[0], :crop.shape[1]] = crop
            tiles.append(_caption(canvas, f"{split} {img_path.stem[:28]}"))
        if tiles:
            sheet = contact_sheet(tiles, cols=6, tile_w=tile)
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in names.get(cid, str(cid)))
            cv2.imwrite(str(crops_dir / f"class_{cid:02d}_{safe}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        counts[cid] = len(tiles)
    return counts
