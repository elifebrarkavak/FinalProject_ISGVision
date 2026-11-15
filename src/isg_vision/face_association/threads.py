"""OpenCV iş parçacığı sayısı yönetimi (Aşama 5B; ``--opencv-threads``).

Sorun (Aşama 5A'da bulundu, 5B'de araştırıldı; ölçümler ``docs/face_association.md``): ``ultralytics`` içe aktarılınca ``cv2.setNumThreads(0)`` çağırır
(``ultralytics/utils/__init__.py``; gerekçesi PyTorch DataLoader ile uyumsuzluk). Bu, süreç genelinde OpenCV'nin paralel bölgelerini kapatır ve OpenCV
DNN ile çalışan YuNet'i yaklaşık 2,7 kat yavaşlatır (1280x720: ≈ 11 -> ≈ 30 ms). YOLO ve YuNet AYNI süreçte çalıştığı için bu ayar entegre hattı etkiler.

Bu modül ayarı DEĞİŞTİRMEZ, yalnızca isteğe bağlı bir ayar yolu sunar: ``keep`` (varsayılan; hiçbir şey yapma), ``restore`` (``cv2.setNumThreads(-1)``:
OpenCV varsayılanına dön) veya bir tam sayı ``N`` (``cv2.setNumThreads(N)``). **Varsayılan hiçbir şey yapmamaktır**: ölçüm ve regresyon kanıtı
olmadan süreç genelinde bir ayar varsayılan yapılmaz (kullanıcı kararı bekler).

Dikkat: bu OpenCV yapısında ``cv2.getNumThreads()`` ``setNumThreads(1)`` sonrasında da 10 gösterebilir (ölçüldü), yani sayaç yavaşlamanın güvenilir
göstergesi DEĞİLDİR; ``ultralytics``'in ``setNumThreads(0)`` ayarı ise ``getNumThreads() == 1`` verir. Karar için süre ölçümü esastır.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import cv2

MODE_KEEP = "keep"
MODE_RESTORE = "restore"
RESTORE_VALUE = -1  # OpenCV: negatif değer = varsayılan iş parçacığı sayısı
MAX_THREADS = 256


@dataclass(frozen=True)
class ThreadSetting:
    """Bir iş parçacığı ayarı uygulamasının kaydı.

    Attributes:
        requested: İstenen mod (``keep``/``restore``/sayı metni).
        value: ``cv2.setNumThreads``'e verilen değer; ``keep`` ise ``None`` (hiçbir şey çağrılmadı).
        before / after: Uygulamadan önce/sonra ``cv2.getNumThreads()`` (yavaşlamanın güvenilir göstergesi DEĞİL; modül açıklaması).
    """

    requested: str
    value: int | None
    before: int
    after: int

    @property
    def changed(self) -> bool:
        return self.value is not None


def parse_threads(text: str) -> int | None:
    """``keep`` -> ``None``; ``restore`` -> ``-1``; ``0..MAX_THREADS`` tam sayı -> o sayı. Aksi halde ``ValueError``."""
    normalized = str(text).strip().lower()
    if normalized == MODE_KEEP:
        return None
    if normalized == MODE_RESTORE:
        return RESTORE_VALUE
    try:
        value = int(normalized)
    except ValueError:
        raise ValueError(f"keep, restore veya 0..{MAX_THREADS} arası bir tam sayı olmalıdır: {text!r}") from None
    if not 0 <= value <= MAX_THREADS:
        raise ValueError(f"iş parçacığı sayısı 0..{MAX_THREADS} arasında olmalıdır (restore = OpenCV varsayılanı): {text!r}")
    return value


def threads_argument(text: str) -> str:
    """``argparse`` türü: değeri doğrular ve normalleştirilmiş metni döndürür."""
    try:
        parse_threads(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return str(text).strip().lower()


def apply_opencv_threads(requested: str = MODE_KEEP) -> ThreadSetting:
    """İstenen modu uygular (``keep`` ise ``cv2.setNumThreads`` ÇAĞRILMAZ) ve öncesi/sonrasını kaydeder.

    ``ultralytics`` içe aktarıldıktan SONRA çağrılmalıdır (aksi halde ``ultralytics`` ayarı yeniden 0 yapar).
    """
    value = parse_threads(requested)
    before = int(cv2.getNumThreads())
    if value is not None:
        cv2.setNumThreads(value)
    return ThreadSetting(str(requested).strip().lower(), value, before, int(cv2.getNumThreads()))
