"""Yüz tespiti hataları.

İki ayrı hata ailesi vardır (kamera ve insan tespiti modüllerindeki düzenle aynı):

* ``FaceConfigError`` — kullanıcının verdiği değerler geçersiz (eşik, genişlik, model yolu biçimi...). Model yüklenmeden önce,
  yapılandırma aşamasında oluşur.
* ``FaceDetectionError`` ve alt sınıfları — çalışma anı hataları (model bulunamadı/bozuk, geçersiz görüntü, çıkarım hatası...).

Mesajlar kullanıcıya doğrudan gösterilebilecek şekilde, olası nedeni ve çözümü içerecek biçimde yazılır.
"""

from __future__ import annotations


class FaceConfigError(ValueError):
    """Geçersiz yüz tespiti yapılandırması (güven/NMS eşiği, genişlik, azami yüz sayısı, model yolu biçimi vb.)."""


class FaceDetectionError(Exception):
    """Yüz tespiti çalışma anı hatalarının ortak tabanı."""


class FaceModelLoadError(FaceDetectionError):
    """Model dosyası bulunamadı, bütünlük denetimini geçemedi, bozuk veya OpenCV tarafından yüklenemedi."""


class FaceInvalidFrameError(FaceDetectionError, ValueError):
    """Tespite verilen görüntü geçersiz (``None``, boş, yanlış tür/boyut/kanal sayısı)."""


class FaceInferenceError(FaceDetectionError):
    """Model çıkarımı sırasında beklenmeyen bir hata oluştu veya model beklenmeyen biçimde çıktı verdi."""


class FaceWeightsDownloadError(FaceDetectionError):
    """Resmî model dosyası indirilemedi veya bütünlük (boyut/SHA-256) doğrulaması başarısız oldu."""
