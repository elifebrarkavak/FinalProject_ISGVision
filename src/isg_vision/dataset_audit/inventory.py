"""Veri seti envanteri: sayımlar, boyutlar, çözünürlükler, YAML bilgisi."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from isg_vision.dataset_audit.layout import DatasetLayout, SPLIT_ORDER, yaml_path_report
from isg_vision.dataset_audit.scan import Record

INVENTORY_FIELDS = [
    "dataset", "split", "images", "label_files", "boxes", "unsupported_rows", "classes_defined",
    "image_formats", "min_resolution", "max_resolution", "distinct_resolutions", "top_resolutions",
    "size_mb", "empty_labels", "missing_labels", "orphan_labels", "unreadable_images", "unreadable_labels",
    "exif_rotated_images",
]


class _Acc:
    def __init__(self) -> None:
        self.images = 0
        self.label_files = 0
        self.boxes = 0
        self.unsupported_rows = 0
        self.empty = 0
        self.missing = 0
        self.orphan = 0
        self.unreadable_img = 0
        self.unreadable_lbl = 0
        self.exif = 0
        self.bytes = 0
        self.formats: Counter[str] = Counter()
        self.resolutions: Counter[tuple[int, int]] = Counter()
        self.min_wh: tuple[int, int] | None = None
        self.max_wh: tuple[int, int] | None = None

    def add(self, rec: Record) -> None:
        s = rec.sample
        if s.image_path is not None:
            self.images += 1
            self.formats[s.image_path.suffix.lower()] += 1
            try:
                self.bytes += s.image_path.stat().st_size
            except OSError:
                pass
            info = rec.image
            if info is not None and not info.readable:
                self.unreadable_img += 1
            elif info is not None:
                wh = (info.width, info.height)
                self.resolutions[wh] += 1
                area = wh[0] * wh[1]
                if self.min_wh is None or area < self.min_wh[0] * self.min_wh[1]:
                    self.min_wh = wh
                if self.max_wh is None or area > self.max_wh[0] * self.max_wh[1]:
                    self.max_wh = wh
                if info.exif_orientation != 1:
                    self.exif += 1
        else:
            self.orphan += 1
        if s.label_path is not None:
            # Aynı etiket birden çok görüntüye (stem çakışması) bağlıysa tek sayılmalı
            if not rec.stem_collision:
                self.label_files += 1
            if rec.label is not None:
                self.boxes += len(rec.label.boxes)
                self.unsupported_rows += rec.label.unsupported_rows
                if rec.label.empty:
                    self.empty += 1
                if not rec.label.readable:
                    self.unreadable_lbl += 1
        elif s.image_path is not None:
            self.missing += 1

    def merge(self, other: "_Acc") -> None:
        for name in ("images", "label_files", "boxes", "unsupported_rows", "empty", "missing", "orphan",
                     "unreadable_img", "unreadable_lbl", "exif", "bytes"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.formats.update(other.formats)
        self.resolutions.update(other.resolutions)
        for wh in (other.min_wh, other.max_wh):
            if wh is None:
                continue
            if self.min_wh is None or wh[0] * wh[1] < self.min_wh[0] * self.min_wh[1]:
                self.min_wh = wh
            if self.max_wh is None or wh[0] * wh[1] > self.max_wh[0] * self.max_wh[1]:
                self.max_wh = wh


def _fmt_wh(wh: tuple[int, int] | None) -> str:
    return f"{wh[0]}x{wh[1]}" if wh else ""


def directory_size(path: Path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for fn in filenames:
            try:
                total += (Path(dirpath) / fn).stat().st_size
            except OSError:
                pass
    return total


def build_inventory(layout: DatasetLayout, records: Iterable[Record]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(CSV satırları, JSON ayrıntısı) döndürür."""
    per_split: dict[str, _Acc] = {s: _Acc() for s in SPLIT_ORDER if s in layout.splits}
    for rec in records:
        per_split[rec.sample.split].add(rec)
    total = _Acc()
    for acc in per_split.values():
        total.merge(acc)

    def row(split: str, acc: _Acc) -> dict[str, Any]:
        return {
            "dataset": layout.name, "split": split, "images": acc.images, "label_files": acc.label_files,
            "boxes": acc.boxes, "unsupported_rows": acc.unsupported_rows,
            "classes_defined": len(layout.classes.names),
            "image_formats": ";".join(f"{k}:{v}" for k, v in sorted(acc.formats.items())),
            "min_resolution": _fmt_wh(acc.min_wh), "max_resolution": _fmt_wh(acc.max_wh),
            "distinct_resolutions": len(acc.resolutions),
            "top_resolutions": ";".join(f"{w}x{h}:{n}" for (w, h), n in acc.resolutions.most_common(5)),
            "size_mb": round(acc.bytes / 1024**2, 1), "empty_labels": acc.empty,
            "missing_labels": acc.missing, "orphan_labels": acc.orphan,
            "unreadable_images": acc.unreadable_img, "unreadable_labels": acc.unreadable_lbl,
            "exif_rotated_images": acc.exif,
        }

    rows = [row(s, a) for s, a in per_split.items()] + [row("ALL", total)]
    detail = {
        "dataset": layout.name,
        "raw_dir": layout.raw_dir.name,
        "dataset_root": layout.root.relative_to(layout.raw_dir).as_posix() if layout.root else None,
        "yaml_file": layout.yaml_path.relative_to(layout.raw_dir).as_posix() if layout.yaml_path else None,
        "yaml_fields": {k: layout.yaml_data.get(k) for k in ("path", "train", "val", "test", "nc")},
        "yaml_paths": yaml_path_report(layout) if layout.yaml_path else None,
        "names_format": layout.classes.names_format,
        "nc_declared": layout.classes.nc_declared,
        "classes": {str(k): v for k, v in sorted(layout.classes.names.items())},
        "class_warnings": layout.classes.warnings,
        "splits": {
            s: {"layout": d.layout, "original_name": d.original_name,
                "images_dir": d.images_dir.relative_to(layout.raw_dir).as_posix(),
                "labels_dir": d.labels_dir.relative_to(layout.raw_dir).as_posix()}
            for s, d in layout.splits.items()},
        "notes": layout.notes,
        "total_size_bytes_on_disk": directory_size(layout.raw_dir),
        "resolutions_top20": [{"size": f"{w}x{h}", "count": n} for (w, h), n in total.resolutions.most_common(20)],
        "totals": rows[-1],
    }
    return rows, detail
