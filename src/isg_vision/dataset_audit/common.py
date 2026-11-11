"""Betikler arasında paylaşılan küçük yardımcılar: kayıt defteri, CSV/JSON yazımı, yol gösterimi."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from isg_vision.dataset_audit import AUDIT_DIR
from isg_vision.utils.paths import PROJECT_ROOT

REGISTRY_FILE = PROJECT_ROOT / "configs" / "datasets.yaml"


def load_registry(path: Path = REGISTRY_FILE) -> dict[str, dict[str, Any]]:
    """configs/datasets.yaml içindeki veri seti kayıtlarını okur."""
    if not path.is_file():
        raise FileNotFoundError(f"Veri seti kayıt dosyası bulunamadı: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["datasets"]


def select_datasets(registry: dict[str, dict[str, Any]], names: Sequence[str] | None) -> list[str]:
    """CLI'dan gelen veri seti adlarını doğrular; boşsa hepsini döndürür."""
    if not names:
        return list(registry)
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise SystemExit(f"Bilinmeyen veri seti: {unknown}. Geçerli adlar: {list(registry)}")
    return list(names)


def dataset_raw_dir(registry: dict[str, dict[str, Any]], name: str) -> Path:
    return PROJECT_ROOT / registry[name]["raw_dir"]


def rel(path: Path | str) -> str:
    """Proje köküne göreli (mümkünse) POSIX yol; rapora kullanıcıya özel yol yazılmasın diye."""
    p = Path(path)
    try:
        return p.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def ensure_audit_dir(out_dir: Path | None = None) -> Path:
    target = out_dir or AUDIT_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> int:
    """CSV yazar (UTF-8, başlıklı); yazılan satır sayısını döndürür."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path.name} bulunamadı. Önce ilgili analiz betiğini çalıştırın ({path})."
        )
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"{path.name} bulunamadı ({path}). Önce ilgili betiği çalıştırın.")
    return json.loads(path.read_text(encoding="utf-8"))


def human_size(num_bytes: int | float) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)
