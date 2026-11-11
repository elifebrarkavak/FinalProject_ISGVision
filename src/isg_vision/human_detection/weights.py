"""Resmî model ağırlığının açık ve denetlenebilir biçimde indirilmesi.

Bu modül, projede **ağ kullanan tek** insan-tespiti koddur ve yalnızca ``scripts/download_person_model.py`` tarafından,
kullanıcı betiği çalıştırdığında çağrılır. ``PersonDetector`` hiçbir zaman indirme yapmaz.

Güvenlik ilkeleri
-----------------
* Yalnızca ``OFFICIAL_MODELS`` izin listesindeki dosya adları indirilir; URL'ler Ultralytics'in resmî GitHub
  release sayfasındandır (``https://github.com/ultralytics/assets/releases/download/...``). Rastgele URL kabul edilmez.
* PyTorch ``.pt`` dosyaları ``pickle`` içerir; güvenilmeyen kaynaktan yüklemek kod çalıştırabilir. Bu yüzden indirilen
  dosyanın **SHA-256 özeti sabitlenmiştir** ve uyuşmazlıkta dosya reddedilir.
* Mevcut bir dosyanın üzerine yazılmaz (``force=True`` gerekir); indirme önce geçici dosyaya yapılır, doğrulanırsa taşınır.
* Ağırlıklar ``models/human_detection/`` altına yazılır; ``.gitignore`` ``*.pt`` dosyalarını Git dışında tutar.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from isg_vision.utils.paths import project_path

from .errors import WeightsDownloadError

OFFICIAL_URL_PREFIX = "https://github.com/ultralytics/assets/releases/download/"
CHUNK_BYTES = 1024 * 256
MIN_WEIGHT_BYTES = 100_000  # Ultralytics'in kendi alt sınırıyla aynı mantık: bundan küçük dosya hata sayfasıdır


@dataclass(frozen=True)
class OfficialModel:
    """İzin listesindeki resmî bir ağırlık dosyası.

    Attributes:
        name: Dosya adı.
        url: Resmî indirme adresi (``OFFICIAL_URL_PREFIX`` ile başlamalıdır).
        sha256: Beklenen SHA-256 (küçük harf hex); ``None`` ise henüz sabitlenmemiştir (ilk indirme; özet yazdırılır).
    """

    name: str
    url: str
    sha256: str | None = None


# sha256: 2026-11-11'de indirilen dosyanın özeti; GitHub release API'sindeki ``digest`` alanıyla (asset ``yolo11n.pt``,
# release ``v8.4.0``, 5613764 bayt) eşleştiği doğrulandı.
OFFICIAL_MODELS: dict[str, OfficialModel] = {
    "yolo11n.pt": OfficialModel(
        name="yolo11n.pt",
        url=OFFICIAL_URL_PREFIX + "v8.4.0/yolo11n.pt",
        sha256="0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    ),
}


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    size_bytes: int
    sha256: str
    url: str
    downloaded: bool  # False = dosya zaten vardı, indirme yapılmadı
    pinned: bool  # SHA-256 sabitlenmiş değerle doğrulandı mı


def default_weights_dir() -> Path:
    return project_path("models", "human_detection")


def sha256_file(path: Path) -> str:
    """Dosyanın SHA-256 özeti (küçük harf hex)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_official_weights(
    name: str,
    dest_dir: Path | None = None,
    *,
    force: bool = False,
    opener: Callable[..., Any] = urllib.request.urlopen,
    models: dict[str, OfficialModel] | None = None,
) -> DownloadResult:
    """İzin listesindeki bir resmî ağırlığı indirir ve doğrular.

    Args:
        name: ``OFFICIAL_MODELS`` içindeki dosya adı (ör. ``yolo11n.pt``).
        dest_dir: Hedef klasör (varsayılan ``models/human_detection``).
        force: ``True`` ise var olan dosyanın üzerine yazılır.
        opener: ``urllib.request.urlopen`` benzeri fonksiyon (testlerde ağ kullanmamak için).
        models: İzin listesi (testler için).

    Raises:
        WeightsDownloadError: İzin listesinde olmayan ad, ağ hatası, çok küçük dosya veya SHA-256 uyuşmazlığı.
    """
    registry = OFFICIAL_MODELS if models is None else models
    model = registry.get(name)
    if model is None:
        raise WeightsDownloadError(
            f"{name!r} resmî model izin listesinde değil; izin verilenler: {sorted(registry)}. Rastgele kaynaktan "
            "indirme desteklenmez."
        )
    if not model.url.startswith(OFFICIAL_URL_PREFIX):
        raise WeightsDownloadError(f"Resmî olmayan indirme adresi reddedildi: {model.url}")

    directory = default_weights_dir() if dest_dir is None else Path(dest_dir)
    target = directory / model.name
    if target.exists() and not force:
        digest = sha256_file(target)
        _verify_hash(model, digest, target, delete=False)
        return DownloadResult(target, target.stat().st_size, digest, model.url, downloaded=False, pinned=model.sha256 is not None)

    directory.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with opener(model.url, timeout=60) as response, partial.open("wb") as out:
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    except Exception as exc:  # ağ/SSL/disk hatası
        partial.unlink(missing_ok=True)
        raise WeightsDownloadError(f"İndirme başarısız ({model.url}): {type(exc).__name__}: {exc}") from exc

    if size < MIN_WEIGHT_BYTES:
        partial.unlink(missing_ok=True)
        raise WeightsDownloadError(f"İndirilen dosya çok küçük ({size} bayt); bir hata sayfası olabilir: {model.url}")
    _verify_hash(model, digest.hexdigest(), partial, delete=True)  # uyuşmazlıkta geçici dosyayı siler ve hata verir
    os.replace(partial, target)
    return DownloadResult(target, size, digest.hexdigest(), model.url, downloaded=True, pinned=model.sha256 is not None)


def _verify_hash(model: OfficialModel, actual: str, file: Path, *, delete: bool) -> None:
    if model.sha256 is None or actual == model.sha256:
        return
    if delete:
        file.unlink(missing_ok=True)
    raise WeightsDownloadError(
        f"SHA-256 uyuşmuyor ({model.name}): beklenen {model.sha256}, bulunan {actual}. Dosya reddedildi"
        + ("" if delete else "; mevcut dosyayı silip yeniden indirin veya bütünlüğünü inceleyin") + "."
    )
