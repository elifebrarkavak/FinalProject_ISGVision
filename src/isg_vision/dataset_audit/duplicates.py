"""Birebir (SHA-256) ve yakın kopya (dHash) görüntü tespiti.

Yakın kopya eşleşmeleri KESİN aynı görüntü değildir; yalnızca inceleme adayıdır.
Hiçbir dosya silinmez veya değiştirilmez.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from isg_vision.dataset_audit.layout import DatasetLayout, SPLIT_ORDER, iter_samples

DUP_FIELDS = ["match_type", "distance", "relation", "split_pair", "dataset_a", "split_a", "path_a",
              "dataset_b", "split_b", "path_b", "sha256_a", "sha256_b", "note"]
_POPCOUNT16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)


@dataclass
class ImageEntry:
    dataset: str
    split: str
    path: str          # proje köküne göreli
    sha256: str
    dhash: int
    dhash_flipped: int


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def dhash_pair(path: Path) -> tuple[int, int] | None:
    """(dHash, yatay çevrilmiş görüntünün dHash'i) — 64 bit. Okunamazsa None."""
    try:
        with Image.open(path) as im:
            if im.format == "JPEG":
                im.draft("L", (64, 64))
            small = im.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return None
    arr = np.asarray(small, dtype=np.int16)

    def to_int(a: np.ndarray) -> int:
        bits = (a[:, 1:] > a[:, :-1]).flatten()
        value = 0
        for b in bits:
            value = (value << 1) | int(b)
        return value

    return to_int(arr), to_int(arr[:, ::-1])


def index_dataset(layout: DatasetLayout, rel_fn) -> tuple[list[ImageEntry], list[str]]:
    """Bir veri setindeki tüm görüntüleri hash'ler. (girdiler, okunamayan dosyalar) döndürür."""
    entries: list[ImageEntry] = []
    unreadable: list[str] = []
    for split in SPLIT_ORDER:
        sd = layout.splits.get(split)
        if sd is None:
            continue
        for sample in iter_samples(sd):
            if sample.image_path is None:
                continue
            hashes = dhash_pair(sample.image_path)
            if hashes is None:
                unreadable.append(rel_fn(sample.image_path))
                continue
            entries.append(ImageEntry(layout.name, split, rel_fn(sample.image_path),
                                      sha256_file(sample.image_path), hashes[0], hashes[1]))
    return entries, unreadable


def _relation(a: ImageEntry, b: ImageEntry) -> tuple[str, str]:
    pair = "|".join(sorted((a.split, b.split)))
    if a.dataset != b.dataset:
        return "cross_dataset", pair
    if a.split == b.split:
        return "within_dataset_same_split", pair
    return "within_dataset_cross_split", pair


def _row(match_type: str, distance: int, a: ImageEntry, b: ImageEntry, note: str = "") -> dict[str, Any]:
    relation, pair = _relation(a, b)
    return {"match_type": match_type, "distance": distance, "relation": relation, "split_pair": pair,
            "dataset_a": a.dataset, "split_a": a.split, "path_a": a.path,
            "dataset_b": b.dataset, "split_b": b.split, "path_b": b.path,
            "sha256_a": a.sha256, "sha256_b": b.sha256, "note": note}


def find_exact(entries: list[ImageEntry]) -> list[dict[str, Any]]:
    groups: dict[str, list[ImageEntry]] = defaultdict(list)
    for e in entries:
        groups[e.sha256].append(e)
    rows = []
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda e: e.path)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                rows.append(_row("exact", 0, members[i], members[j]))
    return rows


def find_near(entries: list[ImageEntry], threshold: int = 5, max_pairs: int = 200_000
              ) -> tuple[list[dict[str, Any]], bool]:
    """dHash Hamming mesafesi <= threshold olan, birebir aynı OLMAYAN çiftleri bulur.

    Yatay çevrilmiş kopyalar da aranır (match_type="near_flipped"). (satırlar, kesildi_mi) döndürür.
    """
    if len(entries) < 2:
        return [], False
    h = np.array([e.dhash for e in entries], dtype=np.uint64)
    hf = np.array([e.dhash_flipped for e in entries], dtype=np.uint64)
    shas = np.array([e.sha256 for e in entries])
    found: list[tuple[int, int, int, str]] = []
    n = len(entries)
    for i in range(n - 1):
        rest = slice(i + 1, n)
        for kind, other in (("near", h), ("near_flipped", hf)):
            x = np.bitwise_xor(h[i], other[rest])
            dist = _POPCOUNT16[x.view(np.uint16).reshape(-1, 4)].sum(axis=1)
            idx = np.nonzero((dist <= threshold) & (shas[rest] != shas[i]))[0]
            for k in idx:
                found.append((int(dist[k]), i, i + 1 + int(k), kind))
    # Aynı çift hem normal hem çevrilmiş eşleştiyse en küçük mesafeli olanı tut
    best: dict[tuple[int, int], tuple[int, str]] = {}
    for d, i, j, kind in found:
        key = (i, j)
        if key not in best or d < best[key][0]:
            best[key] = (d, kind)
    ordered = sorted(((d, i, j, kind) for (i, j), (d, kind) in best.items()), key=lambda t: (t[0], t[1], t[2]))
    truncated = len(ordered) > max_pairs
    rows = [_row(kind, d, entries[i], entries[j]) for d, i, j, kind in ordered[:max_pairs]]
    return rows, truncated


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Eşleşmeleri eşleşme türü, ilişki ve bölüm çiftine göre sayar; ilgili benzersiz görüntüleri de verir."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    images: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        key = f"{r['relation']}:{r['split_pair']}" if r["relation"] != "cross_dataset" \
            else f"cross_dataset:{r['dataset_a']}|{r['dataset_b']}"
        counts[r["match_type"]][key] += 1
        images[r["match_type"]].add(r["path_a"])
        images[r["match_type"]].add(r["path_b"])
    return {mt: {"pairs": dict(sorted(c.items())), "total_pairs": sum(c.values()),
                 "distinct_images_involved": len(images[mt])} for mt, c in counts.items()}


def base_name(path: str) -> str:
    """Roboflow dışa aktarım adından ('<ad>_jpg.rf.<hash>.jpg') özgün kaynak adını çıkarır.

    '.rf.' yoksa dosya adının gövdesi döner. Bu YALNIZCA bir ipucudur: farklı kaynaklarda aynı genel ad
    (ör. '00596') tesadüfen bulunabilir; kesin eşleşme için görüntü içeriği karşılaştırılmalıdır.
    """
    stem = Path(path).stem
    if ".rf." in stem:
        stem = stem.split(".rf.")[0]
    for suffix in ("_jpg", "_jpeg", "_png"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def find_name_overlaps(entries: list[ImageEntry]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aynı özgün kaynak adını paylaşan görüntü gruplarını bulur (ipucu düzeyinde)."""
    groups: dict[str, list[ImageEntry]] = defaultdict(list)
    for e in entries:
        groups[base_name(e.path)].append(e)
    rows = []
    stats = defaultdict(int)
    for base, members in groups.items():
        if len(members) < 2:
            continue
        datasets = sorted({m.dataset for m in members})
        splits = sorted({(m.dataset, m.split) for m in members})
        if len(datasets) > 1:
            kind = "cross_dataset"
        elif len({s for _, s in splits}) > 1:
            kind = "within_dataset_cross_split"
        else:
            kind = "within_dataset_same_split"
        stats[kind] += 1
        rows.append({"base_name": base, "kind": kind, "n_images": len(members),
                     "datasets_splits": ";".join(f"{d}/{s}" for d, s in splits),
                     "paths": ";".join(m.path for m in members)})
    rows.sort(key=lambda r: (r["kind"], r["base_name"]))
    return rows, dict(stats)
