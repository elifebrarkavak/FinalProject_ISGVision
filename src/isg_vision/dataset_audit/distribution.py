"""Sınıf dağılımı ve kutu geometrisi istatistikleri; grafik üretimi."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from isg_vision.dataset_audit.layout import DatasetLayout, SPLIT_ORDER
from isg_vision.dataset_audit.scan import Record

DIST_FIELDS = ["dataset", "class_id", "class_name", "split", "boxes", "images_with_class",
               "share_of_split_boxes_pct"]
BOXSTAT_FIELDS = ["dataset", "class_id", "class_name", "boxes", "median_width", "median_height",
                  "median_area", "p10_area", "p90_area", "median_aspect_h_over_w", "median_y_center"]
UNDEFINED = "<tanımsız>"


def compute_distribution(layout: DatasetLayout, records: Iterable[Record]
                         ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """(dağılım satırları, kutu istatistikleri, hariç tutulan sayılar) döndürür.

    Yalnızca 5 sütunlu, sayısal ve negatif olmayan tam sayı sınıflı satırlar sayılır;
    desteklenmeyen biçimdeki satırlar (segmentation vb.) ``excluded`` içinde raporlanır.
    """
    boxes = defaultdict(lambda: defaultdict(int))        # split -> class -> kutu
    images = defaultdict(lambda: defaultdict(int))       # split -> class -> görüntü
    geom: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    excluded = {"unsupported_rows": 0, "images_without_labels": 0, "empty_label_images": 0}

    for rec in records:
        split = rec.sample.split
        if rec.label is None:
            excluded["images_without_labels"] += 1
            continue
        excluded["unsupported_rows"] += rec.label.unsupported_rows
        if rec.label.empty:
            excluded["empty_label_images"] += 1
        present = set()
        for b in rec.label.boxes:
            boxes[split][b.class_id] += 1
            present.add(b.class_id)
            if b.w > 0 and b.h > 0:
                geom[b.class_id].append((b.w, b.h, b.yc))
        for cid in present:
            images[split][cid] += 1

    names = dict(layout.classes.names)
    seen_ids = {cid for s in boxes.values() for cid in s}
    all_ids = sorted(set(names) | seen_ids)

    def cname(cid: int) -> str:
        return names.get(cid, UNDEFINED)

    rows: list[dict[str, Any]] = []
    splits = [s for s in SPLIT_ORDER if s in layout.splits]
    split_totals = {s: sum(boxes[s].values()) for s in splits}
    grand = sum(split_totals.values())
    for cid in all_ids:
        for s in splits:
            n = boxes[s].get(cid, 0)
            rows.append({"dataset": layout.name, "class_id": cid, "class_name": cname(cid), "split": s,
                         "boxes": n, "images_with_class": images[s].get(cid, 0),
                         "share_of_split_boxes_pct": round(100 * n / split_totals[s], 2) if split_totals[s] else 0})
        n_all = sum(boxes[s].get(cid, 0) for s in splits)
        rows.append({"dataset": layout.name, "class_id": cid, "class_name": cname(cid), "split": "ALL",
                     "boxes": n_all, "images_with_class": sum(images[s].get(cid, 0) for s in splits),
                     "share_of_split_boxes_pct": round(100 * n_all / grand, 2) if grand else 0})

    stats = []
    for cid in all_ids:
        g = np.array(geom.get(cid, []), dtype=float)
        if len(g) == 0:
            stats.append({"dataset": layout.name, "class_id": cid, "class_name": cname(cid), "boxes": 0})
            continue
        area = g[:, 0] * g[:, 1]
        stats.append({
            "dataset": layout.name, "class_id": cid, "class_name": cname(cid), "boxes": len(g),
            "median_width": round(float(np.median(g[:, 0])), 4), "median_height": round(float(np.median(g[:, 1])), 4),
            "median_area": round(float(np.median(area)), 5), "p10_area": round(float(np.percentile(area, 10)), 5),
            "p90_area": round(float(np.percentile(area, 90)), 5),
            "median_aspect_h_over_w": round(float(np.median(g[:, 1] / g[:, 0])), 3),
            "median_y_center": round(float(np.median(g[:, 2])), 3),
        })
    return rows, stats, excluded


def plot_distribution(dataset: str, rows: list[dict[str, Any]], out_path: Path) -> None:
    """Sınıf başına kutu sayısını bölümlere göre yığılmış yatay çubuk grafiğiyle kaydeder."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_class = [r for r in rows if r["dataset"] == dataset and r["split"] != "ALL"]
    class_labels = []
    for r in per_class:
        label = f"{r['class_id']}: {r['class_name']}"
        if label not in class_labels:
            class_labels.append(label)
    splits = [s for s in SPLIT_ORDER if any(r["split"] == s for r in per_class)]
    colors = {"train": "#4C78A8", "val": "#F58518", "test": "#54A24B"}
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(class_labels) + 2))
    left = np.zeros(len(class_labels))
    for s in splits:
        vals = np.array([next(r["boxes"] for r in per_class
                              if f"{r['class_id']}: {r['class_name']}" == lab and r["split"] == s)
                         for lab in class_labels], dtype=float)
        ax.barh(class_labels, vals, left=left, label=s, color=colors.get(s))
        left += vals
    for y, total in enumerate(left):
        ax.text(total, y, f" {int(total)}", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Kutu sayısı")
    ax.set_title(f"{dataset} - sınıf başına kutu sayısı (bölümlere göre)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
