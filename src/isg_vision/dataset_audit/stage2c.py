"""Aşama 2C nihai bütünlük denetimleri (salt okunur).

Her denetim {id, question, passed, detail} döndürür. Ham veri DEĞİŞTİRİLMEZ: dosyalar yalnızca okunur, ZIP arşivleri yalnızca
okunarak (CRC32 + boyut) karşılaştırılır.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
import zlib
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from isg_vision.dataset_audit.labels import parse_label_file

# Proje iskeletine ait yer tutucu dosyalar (Aşama 1 .gitkeep, macOS .DS_Store): veri değildir, ZIP'te yoktur.
# Karşılaştırmadan çıkarılır ama sonuçta AÇIKÇA listelenir.
PLACEHOLDER_NAMES = (".gitkeep", ".DS_Store")


def check(cid: str, question: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"id": cid, "question": question, "passed": bool(passed), "detail": detail}


def read_publisher_manifest(raw_dir: Path) -> list[dict[str, Any]]:
    path = raw_dir / "manifest.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_publisher_manifest(raw_dir: Path, check_hashes: bool = True) -> dict[str, Any]:
    """manifest.jsonl (yayıncı) <-> images/<split>/*.jpg + labels/<split>/*.txt eşleşmesi.

    Karşılaştırılan: dosya adı kümesi, bölüm, SHA-256 (isteğe bağlı) ve 'annotations' = etiket dosyasındaki geçerli satır sayısı.
    """
    entries = read_publisher_manifest(raw_dir)
    on_disk = {p.name: p for p in sorted((raw_dir / "images").glob("*/*")) if p.is_file()}
    by_name = {e["file"]: e for e in entries}
    res: dict[str, Any] = {"manifest_entries": len(entries), "image_files": len(on_disk),
                           "duplicate_manifest_names": len(entries) - len(by_name),
                           "missing_on_disk": sorted(set(by_name) - set(on_disk)), "not_in_manifest": sorted(set(on_disk) - set(by_name)),
                           "split_mismatch": [], "sha256_mismatch": [], "annotation_count_mismatch": [], "missing_label": [],
                           "by_split": dict(sorted(Counter(e["split"] for e in entries).items()))}
    for name, e in by_name.items():
        p = on_disk.get(name)
        if p is None:
            continue
        if p.parent.name != e["split"]:
            res["split_mismatch"].append(name)
        if check_hashes and sha256_file(p) != e["sha256"]:
            res["sha256_mismatch"].append(name)
        lbl = raw_dir / "labels" / p.parent.name / (p.stem + ".txt")
        if not lbl.is_file():
            res["missing_label"].append(name)
            continue
        if len(parse_label_file(lbl).boxes) != int(e["annotations"]):
            res["annotation_count_mismatch"].append(name)
    res["ok"] = (res["manifest_entries"] == res["image_files"] and not res["duplicate_manifest_names"]
                 and not any(res[k] for k in ("missing_on_disk", "not_in_manifest", "split_mismatch", "sha256_mismatch",
                                              "annotation_count_mismatch", "missing_label")))
    return res


def count_target_boxes(raw_dir: Path, class_map: dict[int, int], target_names: dict[int, str],
                       only_files: set[str] | None = None) -> dict[str, dict[str, int]]:
    """labels/<split>/*.txt içinden hedef sınıf kutu sayıları (bölüm bazında). only_files: yalnızca bu görüntü adları."""
    out: dict[str, Counter] = {}
    for lbl in sorted((raw_dir / "labels").glob("*/*.txt")):
        if only_files is not None and not any((lbl.stem + ext) in only_files for ext in (".jpg", ".jpeg", ".png")):
            continue
        c = out.setdefault(lbl.parent.name, Counter())
        for b in parse_label_file(lbl).boxes:
            if b.class_id in class_map:
                c[target_names[class_map[b.class_id]]] += 1
    return {s: dict(sorted(c.items())) for s, c in sorted(out.items())}


def verify_against_zip(zip_path: Path, raw_dir: Path, prefix: str) -> dict[str, Any]:
    """Ham klasördeki her dosyayı özgün ZIP girdisiyle (boyut + CRC32) karşılaştırır; arşiv YALNIZCA okunur."""
    res: dict[str, Any] = {"zip": str(zip_path), "zip_present": zip_path.is_file()}
    if not zip_path.is_file():
        res["ok"] = None
        return res
    expected: dict[str, tuple[int, int]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = PurePosixPath(info.filename.replace("\\", "/")).parts
            if prefix and parts and parts[0] == prefix:
                parts = parts[1:]
            if parts:
                expected["/".join(parts)] = (info.file_size, info.CRC)
    all_files = [p for p in raw_dir.rglob("*") if p.is_file()]
    placeholders = sorted(p.relative_to(raw_dir).as_posix() for p in all_files if p.name in PLACEHOLDER_NAMES and p.stat().st_size == 0
                          or p.name == ".DS_Store")
    disk = {p.relative_to(raw_dir).as_posix(): p for p in all_files if p.relative_to(raw_dir).as_posix() not in placeholders}
    mismatched = []
    for relp, (size, crc) in expected.items():
        p = disk.get(relp)
        if p is None:
            continue
        if p.stat().st_size != size:
            mismatched.append(relp)
            continue
        val = 0
        with p.open("rb") as fh:
            while block := fh.read(1 << 20):
                val = zlib.crc32(block, val)
        if val & 0xFFFFFFFF != crc:
            mismatched.append(relp)
    res.update({"zip_files": len(expected), "disk_files": len(disk), "placeholders_ignored": placeholders, "missing_on_disk": sorted(set(expected) - set(disk)),
                "extra_on_disk": sorted(set(disk) - set(expected)), "content_mismatch": mismatched})
    res["ok"] = not (res["missing_on_disk"] or res["extra_on_disk"] or mismatched)
    return res


def writable_files(raw_dir: Path) -> list[str]:
    """Sahibi tarafından yazılabilir ham dosyalar (çıkarma sırasında salt okunur yapılmıştı)."""
    return sorted(p.relative_to(raw_dir).as_posix() for p in raw_dir.rglob("*")
                  if p.is_file() and p.name not in PLACEHOLDER_NAMES and p.stat().st_mode & 0o200)


# ---------------------------------------------------------------- Aşama 2D: aşama numarası tutarlılığı
STAGE_SCAN_GLOBS = ("README.md", "docs/*.md", "configs/*.yaml", "configs/review/*.yaml", "reports/dataset_audit/*.md",
                    "reports/dataset_audit/final_visual_review/*.md", "src/isg_vision/**/*.py", "scripts/*.py")
STAGE_SCAN_EXCLUDE = ("src/isg_vision/dataset_audit/stage2c.py", "scripts/run_stage2c_checks.py")  # denetleyicinin kendi metinleri
NEGATIONS = ("YOK", "yok", "Yok", "yapılmayacak", "yapılmaz", "değil", "DEĞİL", "içermez", "etkilenmez")
_STAGE3_MISUSE = re.compile(r"Aşama 3\b[^\n]{0,80}?(eğitim|YOLO|birleştir|türetilmiş|sınıf dönüşüm)", re.IGNORECASE)
_PAREN_STAGE3 = re.compile(r"(eğitim|birleştir|türetilmiş|sınıf dönüşüm)[^\n]{0,40}\(Aşama 3\)", re.IGNORECASE)
_OUT_OF_RANGE = re.compile(r"Aşama (1[3-9]|[2-9][0-9])\b")


def stage_reference_problems(project_root: Path, stages: dict[int, dict[str, Any]]) -> list[str]:
    """Belgelerdeki aşama referanslarının 12 aşamalı planla tutarlılığı. Boş liste = tutarlı."""
    problems: list[str] = []
    if sorted(stages) != list(range(1, 13)):
        problems.append(f"configs/project_stages.yaml 1-12 aşamalarını içermiyor: {sorted(stages)}")
        return problems
    titles = [stages[i]["title"] for i in range(1, 13)]
    for doc in ("docs/project_plan.md", "README.md"):
        text = (project_root / doc).read_text(encoding="utf-8")
        start = text.find(titles[0])                 # sıra, plan listesinin başladığı yerden itibaren denetlenir
        text = text[start:] if start >= 0 else text
        pos = [text.find(t) for t in titles]
        missing = [t for t, p in zip(titles, pos) if p < 0]
        if missing:
            problems.append(f"{doc}: plan başlıkları eksik: {missing}")
        elif pos != sorted(pos):
            problems.append(f"{doc}: aşama başlıkları plan sırasında değil")
    files: set[Path] = set()
    for g in STAGE_SCAN_GLOBS:
        files.update(project_root.glob(g))
    for f in sorted(files):
        relp = f.relative_to(project_root).as_posix()
        if relp in STAGE_SCAN_EXCLUDE or not f.is_file():
            continue
        for no, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if (_STAGE3_MISUSE.search(line) or _PAREN_STAGE3.search(line)) and not any(n in line for n in NEGATIONS):
                problems.append(f"{relp}:{no}: Aşama 3 eğitim/veri işiyle ilişkilendirilmiş: {line.strip()[:120]}")
            if _OUT_OF_RANGE.search(line):
                problems.append(f"{relp}:{no}: planda olmayan aşama numarası: {line.strip()[:120]}")
    return problems


def pdf_page_count(path: Path) -> int:
    """Basit PDF sayfa sayımı ('/Type /Page' nesneleri; '/Pages' hariç)."""
    data = path.read_bytes()
    return len(re.findall(rb"/Type\s*/Page(?![s\w])", data))
