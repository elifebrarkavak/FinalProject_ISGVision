"""İnsan tespiti hataları.

İki ayrı hata ailesi vardır (kamera modülündeki düzenle aynı):

* ``DetectionConfigError`` — kullanıcının verdiği değerler geçersiz (eşik, boyut, cihaz adı, model yolu biçimi...).
  Model yüklenmeden önce, yapılandırma aşamasında oluşur.
* ``DetectionError`` ve alt sınıfları — çalışma anı hataları (model yüklenemedi, cihaz yok, geçersiz görüntü...).

Mesajlar kullanıcıya doğrudan gösterilebilecek şekilde, olası nedeni ve çözümü içerecek biçimde yazılır.
"""

from __future__ import annotations


class DetectionConfigError(ValueError):
    """Geçersiz tespit yapılandırması (güven/IoU eşiği, görüntü boyutu, cihaz adı, model yolu biçimi vb.)."""


class DetectionError(Exception):
    """İnsan tespiti çalışma anı hatalarının ortak tabanı."""


class ModelLoadError(DetectionError):
    """Model ağırlığı bulunamadı, bozuk, uyumsuz veya ``person`` sınıfını içermiyor."""


class DeviceUnavailableError(DetectionError):
    """Açıkça istenen cihaz (ör. ``mps``) bu makinede kullanılamıyor. Sessizce CPU'ya düşülmez."""


class InvalidFrameError(DetectionError, ValueError):
    """Tespite verilen görüntü geçersiz (``None``, boş, yanlış tür/boyut/kanal sayısı)."""


class InferenceError(DetectionError):
    """Model çıkarımı sırasında beklenmeyen hata oluştu."""


class WeightsDownloadError(DetectionError):
    """Resmî model ağırlığı indirilemedi veya bütünlük doğrulaması başarısız oldu."""
