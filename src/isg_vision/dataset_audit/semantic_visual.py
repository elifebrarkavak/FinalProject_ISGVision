"""Anlamsal doğrulama için numaralı karşılaştırma ve şüpheli etiket sayfaları (yalnızca yeni dosya yazar)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from isg_vision.dataset_audit.semantic import ImageBoxes, RoleBox
from isg_vision.dataset_audit.visualize import _caption, contact_sheet

COLORS = {"helmet": (0, 200, 0), "vest": (0, 200, 0), "no_helmet": (0, 0, 255), "no_vest": (0, 0, 255),
          "person": (255, 255, 255), "other": (0, 255, 255)}


def _px(rb: RoleBox, w: int, h: int) -> tuple[int, int, int, int]:
    return int(rb.x1 * w), int(rb.y1 * h), int(rb.x2 * w), int(rb.y2 * h)


def crop_with_boxes(image: np.ndarray, focus: list[RoleBox], draw: list[RoleBox], margin: float, tile: int,
                    note: str = "") -> np.ndarray:
    """focus kutularının birleşimini margin kadar genişleterek kırpar; draw içindeki kutuları çizer."""
    h, w = image.shape[:2]
    boxes = [_px(b, w, h) for b in focus]
    x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    cx1, cy1, cx2, cy2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
    crop = image[cy1:cy2, cx1:cx2].copy()
    if crop.size == 0:
        return np.full((tile, tile, 3), 30, np.uint8)
    s = tile / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))), interpolation=cv2.INTER_AREA)
    for rb in draw:
        bx1, by1, bx2, by2 = _px(rb, w, h)
        pts = [int((bx1 - cx1) * s), int((by1 - cy1) * s), int((bx2 - cx1) * s), int((by2 - cy1) * s)]
        thick = 1 if rb.role == "person" else 2
        cv2.rectangle(crop, (pts[0], pts[1]), (pts[2], pts[3]), COLORS[rb.role], thick)
    canvas = np.full((tile, tile, 3), 25, np.uint8)
    canvas[:crop.shape[0], :crop.shape[1]] = crop
    return _caption(canvas, note)


def render_role_comparison(datasets: dict[str, list[ImageBoxes]], role: str, out_path: Path, per_row: int = 8,
                           seed: int = 42, min_height_px: float = 24, tile: int = 160, margin: float = 0.6
                           ) -> dict[str, dict[str, Any]]:
    """Her veri seti için bir satır: aynı rolün (ör. helmet) kutuları, aynı çevre payıyla yan yana.

    Yeşil kutu = ilgili rol; beyaz ince kutu = aynı kişiye atanabilen kişi kutusu (varsa).
    Çok küçük kutular (<min_height_px) ayırt edilemediği için örneklemeden çıkarılır; oranı döndürülür.
    """
    rows: list[np.ndarray] = []
    info: dict[str, dict[str, Any]] = {}
    for name, images in datasets.items():
        pool = [(im, rb) for im in images for rb in im.boxes if rb.role == role]
        eligible = [(im, rb) for im, rb in pool if rb.box.h * im.height >= min_height_px]
        info[name] = {"n_total": len(pool), "n_eligible": len(eligible)}
        rng = random.Random(seed)
        rng.shuffle(eligible)
        picked = []
        for split in ("train", "val", "test"):  # bölümlere dengeli
            picked += [e for e in eligible if e[0].split == split][: per_row // 2 + 1]
        picked = (picked + eligible)[:per_row]
        tiles = []
        for im, rb in picked:
            image = cv2.imread(str(im.image_path))
            if image is None:
                continue
            persons = [b for b in im.boxes if b.role == "person" and b.x1 <= rb.box.xc <= b.x2 and b.y1 <= rb.box.yc <= b.y2]
            tiles.append(crop_with_boxes(image, [rb], [rb] + persons[:1], margin, tile, f"{name[:4]} {im.split} {rb.box.h * im.height:.0f}px"))
        info[name]["shown"] = len(tiles)
        label = np.full((24, tile * per_row, 3), 60, np.uint8)
        cv2.putText(label, f"{name} / {role}", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        row = contact_sheet(tiles, cols=per_row, tile_w=tile) if tiles else np.zeros((tile, tile * per_row, 3), np.uint8)
        if row.shape[1] < tile * per_row:
            row = cv2.copyMakeBorder(row, 0, 0, 0, tile * per_row - row.shape[1], cv2.BORDER_CONSTANT, value=(20, 20, 20))
        rows.append(np.vstack([label, row]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return info


def render_conflict_sheets(items: list[dict[str, Any]], images_by_path: dict[str, ImageBoxes], out_dir: Path,
                           per_sheet: int = 16, tile: int = 300) -> list[Path]:
    """Her çelişki için: olumlu (yeşil) / olumsuz (kırmızı) kutular + kişi kutusu (beyaz), numaralı karo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles, paths = [], []
    for it in items:
        im = images_by_path[it["image_path"]]
        by_line = {rb.box.line_no: rb for rb in im.boxes}
        pos, neg = by_line[it["pos_line"]], by_line[it["neg_line"]]
        draw = [pos, neg] + ([by_line[it["person_line"]]] if it["person_line"] != "" and it["person_line"] in by_line else [])
        image = cv2.imread(str(im.image_path))
        note = f"#{it['conflict_id']} {it['attribute']} {it['type']} iou={it['iou']}"
        tiles.append(crop_with_boxes(image, [pos, neg], draw, 0.5, tile, note))
    for i in range(0, len(tiles), per_sheet):
        sheet = contact_sheet(tiles[i:i + per_sheet], cols=4, tile_w=tile)
        p = out_dir / f"conflicts_{i // per_sheet + 1:02d}.jpg"
        cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(p)
    return paths


def render_labeled_boxes(entries: list[dict[str, Any]], out_path: Path, tile: int = 220, cols: int = 6) -> None:
    """Genel amaçlı: entries = {image_path, focus:[RoleBox...]|None, note}. focus yoksa tüm görüntü küçültülür."""
    tiles = []
    for e in entries:
        image = cv2.imread(str(e["image_path"]))
        if image is None:
            continue
        if e.get("focus"):
            tiles.append(crop_with_boxes(image, e["focus"], e["focus"], 3.0, tile, e["note"]))
        else:
            s = tile / max(image.shape[:2])
            small = cv2.resize(image, (int(image.shape[1] * s), int(image.shape[0] * s)))
            canvas = np.full((tile, tile, 3), 25, np.uint8)
            canvas[:small.shape[0], :small.shape[1]] = small
            tiles.append(_caption(canvas, e["note"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), contact_sheet(tiles, cols=cols, tile_w=tile), [cv2.IMWRITE_JPEG_QUALITY, 90])


def render_role_audit_sheet(images: list[ImageBoxes], role: str, out_path: Path, n: int = 24, seed: int = 7,
                            min_height_px: float = 24, tile: int = 200, cols: int = 6,
                            only: list[tuple[str, int]] | None = None) -> list[dict[str, Any]]:
    """Tek veri seti ve rol için numaralı rastgele örnek sayfası (etiket kalitesi örneklemesi).

    only: (image_path, line_no) listesi verilirse rastgele seçim yerine bu kutular çizilir.
    Dönen liste, karo numarası -> dosya/satır eşlemesidir (bulgular yeniden üretilebilir olsun diye CSV'ye yazılır).
    """
    pool = [(im, rb) for im in images for rb in im.boxes if rb.role == role]
    if only is not None:
        wanted = set(only)
        chosen = [(im, rb) for im, rb in pool if (str(im.image_path), rb.box.line_no) in wanted]
    else:
        eligible = [(im, rb) for im, rb in pool if rb.box.h * im.height >= min_height_px]
        random.Random(seed).shuffle(eligible)
        chosen = eligible[:n]
    tiles, listing = [], []
    for i, (im, rb) in enumerate(chosen, start=1):
        image = cv2.imread(str(im.image_path))
        if image is None:
            continue
        persons = [b for b in im.boxes if b.role == "person" and b.x1 <= rb.box.xc <= b.x2 and b.y1 <= rb.box.yc <= b.y2]
        tiles.append(crop_with_boxes(image, [rb], [rb] + persons[:1], 0.6, tile, f"#{i} {im.split} {rb.box.h * im.height:.0f}px"))
        listing.append({"tile": i, "dataset": im.dataset, "split": im.split, "role": role,
                        "image_path": str(im.image_path), "line_no": rb.box.line_no})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), contact_sheet(tiles, cols=cols, tile_w=tile), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return listing
