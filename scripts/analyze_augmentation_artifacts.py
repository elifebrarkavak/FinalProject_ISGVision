#!/usr/bin/env python
"""Roboflow tarzı augmentation izlerini sayar: siyah kare (cutout).

Cutout: kenara değmeyen, 6-24 px, tamamen siyah (0) dolu kare bileşenleri (SEZGİSEL sayım). 2x2 kolaj/mozaik yapısı bu
betikle güvenilir biçimde sayılamadı (sabit dikiş yok); kolajlar yalnızca görsel incelemeyle belgelenmiştir. Dosyalar yalnızca okunur.

Örnek:
    python scripts/analyze_augmentation_artifacts.py
Çıktı: augmentation_artifacts.json
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.dataset_audit.common import (  # noqa: E402
    dataset_raw_dir, ensure_audit_dir, load_registry, select_datasets, write_json,
)
from isg_vision.dataset_audit.layout import SPLIT_ORDER, discover_dataset, iter_samples  # noqa: E402


def has_cutout(gray: np.ndarray) -> bool:
    mask = (gray == 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    h, w = gray.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if 6 <= bw <= 24 and 6 <= bh <= 24 and area >= 0.95 * bw * bh and x > 0 and y > 0 and x + bw < w and y + bh < h \
                and abs(bw - bh) <= 4:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    registry = load_registry()
    out: dict = {}
    for name in select_datasets(registry, args.datasets):
        layout = discover_dataset(name, dataset_raw_dir(registry, name))
        counts = defaultdict(lambda: {"images": 0, "cutout": 0})
        for split in SPLIT_ORDER:
            sd = layout.splits.get(split)
            if sd is None:
                continue
            for s in iter_samples(sd):
                if s.image_path is None:
                    continue
                gray = cv2.imread(str(s.image_path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                c = counts[split]
                c["images"] += 1
                c["cutout"] += has_cutout(gray)
        out[name] = {sp: {**v, "cutout_pct": round(100 * v["cutout"] / max(1, v["images"]), 1)} for sp, v in counts.items()}
        print(name, out[name])
    write_json(ensure_audit_dir(args.out_dir) / "augmentation_artifacts.json", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
