"""Veri seti klasör yapısı ve YAML tanımlarını tespit etme.

Desteklenen yerleşimler (iç içe klasörler dahil):
    A) <kök>/images/<split>/  +  <kök>/labels/<split>/
    B) <kök>/<split>/images/  +  <kök>/<split>/labels/
    C) <kök>/<split>/ içinde görüntü ve .txt dosyaları yan yana (düz)
Split adları: train | val / valid / validation | test. Hepsi "train", "val", "test" olarak normalleştirilir;
özgün klasör adı ``original_name`` alanında saklanır.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}
SPLIT_ORDER = ("train", "val", "test")
MAX_DEPTH = 5


@dataclass
class SplitDirs:
    split: str
    original_name: str
    images_dir: Path
    labels_dir: Path
    layout: str  # "A" | "B" | "C"


@dataclass
class ClassInfo:
    """data.yaml içindeki sınıf tanımları (sıra DEĞİŞTİRİLMEZ)."""
    names: dict[int, str] = field(default_factory=dict)
    nc_declared: int | None = None
    names_format: str = "yok"  # "liste" | "sözlük" | "yok"
    warnings: list[str] = field(default_factory=list)


@dataclass
class DatasetLayout:
    name: str
    raw_dir: Path
    root: Path | None = None
    yaml_path: Path | None = None
    yaml_data: dict[str, Any] = field(default_factory=dict)
    classes: ClassInfo = field(default_factory=ClassInfo)
    splits: dict[str, SplitDirs] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.splits)


@dataclass(frozen=True)
class Sample:
    split: str
    stem: str
    image_path: Path | None
    label_path: Path | None


def parse_class_names(data: dict[str, Any]) -> ClassInfo:
    """``names`` alanını (liste veya sözlük) okur; nc ile tutarlılığı denetler."""
    info = ClassInfo()
    names = data.get("names")
    nc = data.get("nc")
    if nc is not None:
        try:
            info.nc_declared = int(nc)
        except (TypeError, ValueError):
            info.warnings.append(f"nc sayı değil: {nc!r}")
    if isinstance(names, list):
        info.names_format = "liste"
        info.names = {i: str(n) for i, n in enumerate(names)}
    elif isinstance(names, dict):
        info.names_format = "sözlük"
        try:
            info.names = {int(k): str(v) for k, v in names.items()}
        except (TypeError, ValueError):
            info.warnings.append("names sözlüğünde sayısal olmayan anahtar var")
    else:
        info.warnings.append("names alanı yok veya tanınmayan biçimde; sınıf sırası TAHMİN EDİLMEZ")
        return info
    ids = sorted(info.names)
    if ids != list(range(len(ids))):
        info.warnings.append(f"Sınıf ID'leri 0..N-1 aralığında ardışık değil: {ids}")
    if info.nc_declared is not None and info.nc_declared != len(info.names):
        info.warnings.append(f"nc={info.nc_declared} ama names içinde {len(info.names)} sınıf var")
    return info


def _find_yaml(raw_dir: Path) -> Path | None:
    """raw_dir altında (en fazla 3 seviye) data.yaml, yoksa başka .yaml/.yml arar; en sığ olanı seçer."""
    candidates: list[tuple[int, int, Path]] = []
    for dirpath, dirnames, filenames in os.walk(raw_dir):
        depth = len(Path(dirpath).relative_to(raw_dir).parts)
        if depth > 3:
            dirnames[:] = []
            continue
        for fn in filenames:
            if fn.lower().endswith((".yaml", ".yml")):
                priority = 0 if fn.lower() in ("data.yaml", "data.yml") else 1
                candidates.append((priority, depth, Path(dirpath) / fn))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0], c[1], str(c[2])))
    return candidates[0][2]


def _has_images(directory: Path) -> bool:
    try:
        with os.scandir(directory) as it:
            return any(e.is_file() and Path(e.name).suffix.lower() in IMAGE_EXTS for e in it)
    except OSError:
        return False


def _walk_dirs(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, _ in os.walk(root):
        if len(Path(dirpath).relative_to(root).parts) >= MAX_DEPTH:
            dirnames[:] = []
        dirnames.sort()
        yield Path(dirpath)


def detect_splits(root: Path) -> tuple[dict[str, SplitDirs], list[str]]:
    """root altındaki train/val/test dizinlerini yerleşim A/B/C'ye göre bulur."""
    found: dict[str, list[SplitDirs]] = {}
    for d in _walk_dirs(root):
        alias = SPLIT_ALIASES.get(d.name.lower())
        # Yerleşim A: <images>/<split>
        if d.name.lower() == "images":
            for child in sorted(p for p in d.iterdir() if p.is_dir()):
                canon = SPLIT_ALIASES.get(child.name.lower())
                if canon:
                    found.setdefault(canon, []).append(
                        SplitDirs(canon, child.name, child, d.parent / "labels" / child.name, "A"))
        # Yerleşim B: <split>/images
        if alias and (d / "images").is_dir():
            found.setdefault(alias, []).append(
                SplitDirs(alias, d.name, d / "images", d / "labels", "B"))
        # Yerleşim C: <split>/ düz
        elif (alias and not (d / "images").is_dir() and d.parent.name.lower() not in ("images", "labels")
              and _has_images(d)):
            found.setdefault(alias, []).append(SplitDirs(alias, d.name, d, d, "C"))
    chosen: dict[str, SplitDirs] = {}
    notes: list[str] = []
    for split in SPLIT_ORDER:
        options = found.get(split, [])
        if not options:
            continue
        # Aynı split için birden çok aday varsa en sığ olanı seç ve belirt
        options.sort(key=lambda s: (len(s.images_dir.relative_to(root).parts), str(s.images_dir)))
        chosen[split] = options[0]
        if len(options) > 1:
            notes.append(f"'{split}' için {len(options)} aday dizin bulundu; en sığ olan seçildi: "
                         f"{options[0].images_dir.relative_to(root).as_posix()}")
    return chosen, notes


def discover_dataset(name: str, raw_dir: Path) -> DatasetLayout:
    """Bir veri setinin gerçek klasör yapısını, YAML'ını ve sınıflarını tespit eder."""
    layout = DatasetLayout(name=name, raw_dir=raw_dir)
    if not raw_dir.is_dir():
        layout.notes.append(f"Klasör yok: {raw_dir}")
        return layout
    layout.yaml_path = _find_yaml(raw_dir)
    if layout.yaml_path is None:
        layout.notes.append("YAML dosyası bulunamadı; sınıf adları/numaraları belirlenemez (tahmin edilmedi).")
    else:
        try:
            loaded = yaml.safe_load(layout.yaml_path.read_text(encoding="utf-8", errors="replace"))
            layout.yaml_data = loaded if isinstance(loaded, dict) else {}
            if not isinstance(loaded, dict):
                layout.notes.append("YAML okundu ancak sözlük biçiminde değil.")
        except yaml.YAMLError as exc:
            layout.notes.append(f"YAML ayrıştırılamadı: {exc}")
        layout.classes = parse_class_names(layout.yaml_data)
    layout.root = layout.yaml_path.parent if layout.yaml_path else raw_dir
    layout.splits, split_notes = detect_splits(layout.root)
    if not layout.splits and layout.root != raw_dir:
        layout.splits, split_notes = detect_splits(raw_dir)
    layout.notes.extend(split_notes)
    if not layout.splits:
        layout.notes.append("train/val/test dizinleri tanınan hiçbir yerleşimde bulunamadı (A/B/C).")
    if "test" not in layout.splits and layout.splits:
        layout.notes.append("Ayrı bir test bölümü yok.")
    return layout


def yaml_path_report(layout: DatasetLayout) -> dict[str, Any]:
    """YAML'daki path/train/val/test girdilerinin bu makinede çözülüp çözülmediğini raporlar."""
    data = layout.yaml_data
    out: dict[str, Any] = {"path": data.get("path")}
    yaml_dir = layout.yaml_path.parent if layout.yaml_path else None
    base = yaml_dir
    declared = data.get("path")
    if declared:
        cand = Path(str(declared))
        if cand.is_absolute():
            base = cand
        elif yaml_dir is not None:
            base = (yaml_dir / cand)
    for key in ("train", "val", "test"):
        value = data.get(key)
        entry: dict[str, Any] = {"value": value, "resolves": None}
        if value and base is not None:
            entry["resolves"] = (Path(base) / str(value)).resolve().is_dir() if not Path(str(value)).is_absolute() \
                else Path(str(value)).is_dir()
        out[key] = entry
    out["path_is_absolute_foreign"] = bool(declared) and (
        str(declared)[1:3] in (":/", ":\\") or (Path(str(declared)).is_absolute() and not Path(str(declared)).exists())
    )
    return out


def iter_samples(split_dirs: SplitDirs) -> Iterator[Sample]:
    """Bir bölümdeki görüntü/etiket çiftlerini (stem eşleşmesiyle) sırayla üretir.

    Etiketi olmayan görüntüler ``label_path=None``, görüntüsü olmayan etiketler ``image_path=None`` ile gelir.
    Aynı stem'e sahip birden çok görüntü ayrı örnek olarak döner (etiket aynı dosyayı gösterir).
    """
    images: dict[str, list[Path]] = {}
    with os.scandir(split_dirs.images_dir) as it:
        for e in it:
            if e.is_file() and Path(e.name).suffix.lower() in IMAGE_EXTS:
                images.setdefault(Path(e.name).stem, []).append(Path(e.path))
    labels: dict[str, Path] = {}
    if split_dirs.labels_dir.is_dir():
        with os.scandir(split_dirs.labels_dir) as it:
            for e in it:
                if e.is_file() and e.name.lower().endswith(".txt"):
                    labels[Path(e.name).stem] = Path(e.path)
    for stem in sorted(images):
        for img in sorted(images[stem]):
            yield Sample(split_dirs.split, stem, img, labels.get(stem))
    for stem in sorted(set(labels) - set(images)):
        yield Sample(split_dirs.split, stem, None, labels[stem])
