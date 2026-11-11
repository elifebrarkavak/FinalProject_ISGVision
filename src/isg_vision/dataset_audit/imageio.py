"""Görüntü bilgisi okuma (boyut, bozulma, EXIF yönü). Dosyalar yalnızca okunur."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = 200_000_000  # çok büyük (şüpheli) görüntülere karşı sınır


@dataclass
class ImageInfo:
    width: int = 0
    height: int = 0
    format: str = ""
    exif_orientation: int = 1
    readable: bool = True
    error: str = ""


def read_image_info(path: Path, full_decode: bool = True) -> ImageInfo:
    """Görüntü boyutunu ve okunabilirliğini döndürür.

    full_decode=True ise pikseller tamamen çözülür (bozuk dosyaları yakalar); False ise yalnızca başlık okunur.
    EXIF yönü 5-8 ise (90° döndürme) genişlik/yükseklik YOLO'nun görecegi şekilde yer değiştirir.
    """
    info = ImageInfo()
    try:
        with Image.open(path) as im:
            info.format = im.format or ""
            w, h = im.size
            try:
                info.exif_orientation = int(im.getexif().get(274, 1) or 1)
            except Exception:  # noqa: BLE001 - bozuk EXIF görüntüyü bozuk saymaz
                info.exif_orientation = 1
            if full_decode:
                im.load()
        if info.exif_orientation in (5, 6, 7, 8):
            w, h = h, w
        info.width, info.height = w, h
    except (OSError, UnidentifiedImageError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        info.readable = False
        info.error = f"{type(exc).__name__}: {exc}"[:200]
    return info
