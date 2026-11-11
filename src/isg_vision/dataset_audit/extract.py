"""ZIP arşivlerini güvenli biçimde proje içine çıkarma.

Güvenlik kuralları:
- Orijinal ZIP'e yalnızca okuma yapılır (silinmez/taşınmaz).
- Arşiv içindeki yol hedef klasörün dışına çıkamaz (zip-slip: mutlak yol, '..', sürücü harfi).
- Sembolik bağlar ve özel dosya türleri çıkarılmaz.
- Var olan hiçbir dosyanın üzerine yazılmaz (atlanır ve raporlanır).
- Çıkarılan dosyalar salt-okunur yapılır (isteğe bağlı), böylece yanlışlıkla değiştirilemez.
- Çıkarılan hiçbir dosya ÇALIŞTIRILMAZ (ör. arşivdeki .py dosyaları yalnızca veri olarak durur).
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import stat
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

MAX_TOTAL_UNCOMPRESSED = 8 * 1024**3  # 8 GB üstü arşivler reddedilir (zip bombası koruması)
MAX_RATIO = 200  # sıkıştırma oranı bundan yüksekse şüpheli


@dataclass
class ExtractionResult:
    dataset: str
    zip_path: str
    zip_size: int = 0
    zip_sha256: str = ""
    dest_dir: str = ""
    stripped_prefix: str = ""
    entries_total: int = 0
    extracted: int = 0
    skipped_existing: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def find_zip_candidates(search_dir: Path, patterns: list[str]) -> list[Path]:
    """Yalnızca verilen klasörün DOĞRUDAN altındaki, desene uyan .zip dosyalarını listeler."""
    if not search_dir.is_dir():
        raise FileNotFoundError(f"Arama klasörü yok veya erişilemiyor: {search_dir}")
    found = []
    for item in sorted(search_dir.iterdir()):
        if item.is_file() and item.suffix.lower() == ".zip":
            if any(fnmatch.fnmatch(item.name.lower(), pat.lower()) for pat in patterns):
                found.append(item)
    return found


def _safe_relative(name: str) -> PurePosixPath | None:
    """Arşiv girdi adını güvenli göreli yola çevirir; güvensizse None."""
    normalized = name.replace("\\", "/")
    p = PurePosixPath(normalized)
    if p.is_absolute() or normalized.startswith("/"):
        return None
    if any(part == ".." for part in p.parts):
        return None
    if p.parts and len(p.parts[0]) == 2 and p.parts[0][1] == ":":  # C: gibi sürücü harfi
        return None
    return p


def _common_wrapper(names: list[str]) -> str:
    """Tüm girdiler tek bir üst klasörün altındaysa onun adını döndürür (yoksa boş)."""
    tops = {PurePosixPath(n.replace("\\", "/")).parts[0] for n in names if n.strip("/")}
    if len(tops) != 1:
        return ""
    top = next(iter(tops))
    # Yalnızca gerçekten klasör olan tek üst düzey girdi sıyrılır (dosya ise sıyrılmaz).
    has_children = any(len(PurePosixPath(n.replace("\\", "/")).parts) > 1 for n in names)
    return top if has_children else ""


def extract_zip_safely(
    zip_path: Path,
    dest_dir: Path,
    dataset: str,
    *,
    strip_wrapper: bool = True,
    read_only: bool = True,
    dry_run: bool = False,
) -> ExtractionResult:
    """ZIP'i dest_dir altına çıkarır; ayrıntılı sonuç döndürür. Üzerine yazmaz."""
    result = ExtractionResult(dataset=dataset, zip_path=str(zip_path), dry_run=dry_run)
    result.zip_size = zip_path.stat().st_size
    result.dest_dir = str(dest_dir)
    dest_root = dest_dir.resolve()

    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        result.entries_total = len(infos)
        total = sum(i.file_size for i in infos)
        if total > MAX_TOTAL_UNCOMPRESSED:
            result.errors.append(f"Açılmış boyut çok büyük ({total} bayt); çıkarma reddedildi.")
            return result
        if result.zip_size and total / result.zip_size > MAX_RATIO:
            result.errors.append("Sıkıştırma oranı şüpheli derecede yüksek; çıkarma reddedildi.")
            return result

        prefix = _common_wrapper([i.filename for i in infos]) if strip_wrapper else ""
        result.stripped_prefix = prefix
        result.zip_sha256 = sha256_file(zip_path)

        for info in infos:
            name = info.filename
            safe = _safe_relative(name)
            if safe is None:
                result.rejected.append(f"{name} (güvensiz yol)")
                continue
            file_type = stat.S_IFMT(info.external_attr >> 16)  # 0: tür bilgisi yok (bazı arşivleyiciler yazmaz)
            if file_type == stat.S_IFLNK:
                result.rejected.append(f"{name} (sembolik bağ)")
                continue
            if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                result.rejected.append(f"{name} (özel dosya türü)")
                continue
            parts = safe.parts
            if prefix and parts and parts[0] == prefix:
                parts = parts[1:]
            if not parts:
                continue
            target = dest_root.joinpath(*parts)
            resolved = Path(os.path.normpath(target))
            if dest_root != resolved and dest_root not in resolved.parents:
                result.rejected.append(f"{name} (hedef klasör dışına çıkıyor)")
                continue
            if info.is_dir():
                if not dry_run:
                    resolved.mkdir(parents=True, exist_ok=True)
                continue
            if resolved.exists():
                result.skipped_existing.append(resolved.relative_to(dest_root).as_posix())
                continue
            if dry_run:
                result.extracted += 1
                continue
            resolved.parent.mkdir(parents=True, exist_ok=True)
            try:
                with zf.open(info) as src, resolved.open("xb") as out:  # 'x': var olanı ezmez
                    while block := src.read(1 << 20):
                        out.write(block)
            except (zipfile.BadZipFile, OSError) as exc:
                result.errors.append(f"{name}: {exc}")
                continue
            if read_only:
                resolved.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            result.extracted += 1
    return result
