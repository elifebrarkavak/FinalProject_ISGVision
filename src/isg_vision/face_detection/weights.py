"""Resmî yüz tespiti modelinin açık ve denetlenebilir biçimde indirilmesi.

Bu modül, yüz tespiti için **ağ kullanan tek** koddur ve yalnızca ``scripts/download_face_model.py`` tarafından, kullanıcı betiği
çalıştırdığında çağrılır. ``FaceDetector`` hiçbir zaman indirme yapmaz.

Güvenlik ilkeleri (``human_detection.weights`` ile aynı yaklaşım)
------------------------------------------------------------------
* Yalnızca ``OFFICIAL_MODELS`` izin listesindeki dosya adları indirilir; URL, OpenCV Zoo deposunun **sabit bir commit'ine** bağlıdır
  (``main`` dalına değil), yani içerik zamanla değişemez. Rastgele URL kabul edilmez.
* ONNX dosyası OpenCV'nin ayrıştırıcısına verilir; güvenilmeyen bir dosyayı ayrıştırmamak için indirilen dosyanın **boyutu ve
  SHA-256 özeti** yayıncının kaydıyla (Git LFS işaretçisi) sabitlenmiştir; uyuşmazlıkta dosya reddedilir.
* Mevcut bir dosyanın üzerine yazılmaz (``force=True`` gerekir); indirme önce geçici dosyaya yapılır, doğrulanırsa taşınır.
* Model ``models/face_detection/`` altına yazılır; ``.gitignore`` ``*.onnx`` dosyalarını Git dışında tutar.

Kaynak ve lisans kaydı
----------------------
* Model: YuNet (``face_detection_yunet_2026may.onnx``); OpenCV Zoo ``models/face_detection_yunet`` klasörü, commit
  ``48b10fa505aeb51fbd127fc60a5c660040e823b7`` (2026-10-20). Klasördeki tüm dosyalar **MIT** lisanslıdır (``LICENSE``:
  "Copyright (c) 2020 Shiqi Yu"); modelin kaynağı ``ShiqiYu/libfacedetection.train`` (o deponun lisansı da MIT'tir).
* Yayıncının LFS işaretçisi: ``oid sha256:ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0``, ``size 229738``.
* ``face_detection_yunet_2026may.onnx``, ``face_detection_yunet_2023mar.onnx`` dosyasından, sabit H/W boyutlarının sembolik
  boyutlarla değiştirilmesiyle yeniden dışa aktarılmış sürümdür (README). Bu projede iki dosya aynı görüntülerde birebir aynı
  kutuları ve skorları verdi (``docs/face_detection.md``).
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

from .errors import FaceWeightsDownloadError

OPENCV_ZOO_COMMIT = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
OFFICIAL_URL_PREFIX = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{OPENCV_ZOO_COMMIT}/models/face_detection_yunet/"
CHUNK_BYTES = 1024 * 256
MIN_MODEL_BYTES = 100_000  # bundan küçük dosya büyük olasılıkla bir hata sayfası veya LFS işaretçisidir (131 bayt)


@dataclass(frozen=True)
class OfficialFaceModel:
    """İzin listesindeki resmî bir model dosyası.

    Attributes:
        name: Dosya adı.
        url: Resmî indirme adresi (``OFFICIAL_URL_PREFIX`` ile başlamalıdır).
        sha256: Beklenen SHA-256 (küçük harf hex).
        size_bytes: Beklenen dosya boyutu (bayt).
        license_name: Lisans adı (bilgi amaçlı).
    """

    name: str
    url: str
    sha256: str
    size_bytes: int
    license_name: str = "MIT"


OFFICIAL_MODELS: dict[str, OfficialFaceModel] = {
    "face_detection_yunet_2026may.onnx": OfficialFaceModel(
        name="face_detection_yunet_2026may.onnx",
        url=OFFICIAL_URL_PREFIX + "face_detection_yunet_2026may.onnx",
        sha256="ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
        size_bytes=229738,
    ),
}


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    size_bytes: int
    sha256: str
    url: str
    downloaded: bool  # False = dosya zaten vardı, indirme yapılmadı


def default_weights_dir() -> Path:
    return project_path("models", "face_detection")


def sha256_file(path: Path) -> str:
    """Dosyanın SHA-256 özeti (küçük harf hex)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_official_model(
    name: str,
    dest_dir: Path | None = None,
    *,
    force: bool = False,
    opener: Callable[..., Any] = urllib.request.urlopen,
    models: dict[str, OfficialFaceModel] | None = None,
) -> DownloadResult:
    """İzin listesindeki bir resmî model dosyasını indirir ve doğrular.

    Args:
        name: ``OFFICIAL_MODELS`` içindeki dosya adı.
        dest_dir: Hedef klasör (varsayılan ``models/face_detection``).
        force: ``True`` ise var olan dosyanın üzerine yazılır.
        opener: ``urllib.request.urlopen`` benzeri fonksiyon (testlerde ağ kullanmamak için).
        models: İzin listesi (testler için).

    Raises:
        FaceWeightsDownloadError: İzin listesinde olmayan ad, ağ hatası, çok küçük dosya veya boyut/SHA-256 uyuşmazlığı.
    """
    registry = OFFICIAL_MODELS if models is None else models
    model = registry.get(name)
    if model is None:
        raise FaceWeightsDownloadError(
            f"{name!r} resmî model izin listesinde değil; izin verilenler: {sorted(registry)}. Rastgele kaynaktan indirme desteklenmez."
        )
    if not model.url.startswith(OFFICIAL_URL_PREFIX):
        raise FaceWeightsDownloadError(f"Resmî olmayan indirme adresi reddedildi: {model.url}")

    directory = default_weights_dir() if dest_dir is None else Path(dest_dir)
    target = directory / model.name
    if target.exists() and not force:
        digest = sha256_file(target)
        _verify(model, digest, target.stat().st_size, target, delete=False)
        return DownloadResult(target, target.stat().st_size, digest, model.url, downloaded=False)

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
        raise FaceWeightsDownloadError(f"İndirme başarısız ({model.url}): {type(exc).__name__}: {exc}") from exc

    if size < MIN_MODEL_BYTES:
        partial.unlink(missing_ok=True)
        raise FaceWeightsDownloadError(f"İndirilen dosya çok küçük ({size} bayt); bir hata sayfası veya LFS işaretçisi olabilir: {model.url}")
    _verify(model, digest.hexdigest(), size, partial, delete=True)  # uyuşmazlıkta geçici dosyayı siler ve hata verir
    os.replace(partial, target)
    return DownloadResult(target, size, digest.hexdigest(), model.url, downloaded=True)


def _verify(model: OfficialFaceModel, actual_hash: str, actual_size: int, file: Path, *, delete: bool) -> None:
    problems = []
    if actual_size != model.size_bytes:
        problems.append(f"boyut uyuşmuyor: beklenen {model.size_bytes}, bulunan {actual_size}")
    if actual_hash != model.sha256:
        problems.append(f"SHA-256 uyuşmuyor: beklenen {model.sha256}, bulunan {actual_hash}")
    if not problems:
        return
    if delete:
        file.unlink(missing_ok=True)
    raise FaceWeightsDownloadError(
        f"{model.name}: " + "; ".join(problems) + ". Dosya reddedildi"
        + ("" if delete else "; mevcut dosyayı silip yeniden indirin veya bütünlüğünü inceleyin") + "."
    )
