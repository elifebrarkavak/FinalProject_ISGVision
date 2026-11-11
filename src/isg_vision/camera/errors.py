"""Kamera/video kaynağı hataları.

İki ayrı hata ailesi vardır:

* ``SourceConfigError`` — kullanıcının verdiği değerler geçersiz (indeks, çözünürlük, yol biçimi...).
  Kaynak açılmadan önce, yapılandırma aşamasında oluşur.
* ``SourceError`` ve alt sınıfları — kaynak açılırken veya kare okunurken oluşan çalışma anı hataları.

Mesajlar kullanıcıya doğrudan gösterilebilecek şekilde, olası nedeni ve çözümü içerecek biçimde yazılır.
"""

from __future__ import annotations


class SourceConfigError(ValueError):
    """Geçersiz kaynak yapılandırması (kamera indeksi, çözünürlük, video yolu biçimi vb.)."""


class SourceError(Exception):
    """Kamera veya video kaynağında çalışma anı hatalarının ortak tabanı."""


class CameraOpenError(SourceError):
    """Kamera açılamadı (bulunamadı, izin verilmedi, başka uygulama kullanıyor olabilir)."""


class VideoFileError(SourceError):
    """Video dosyası bulunamadı, dosya değil, açılamadı veya hiç kare içermiyor."""


class FrameReadError(SourceError):
    """Kaynak açıldı ancak kare okunamadı (ilk kare gelmedi veya okuma sürekli başarısız)."""


class SourceDisconnectedError(FrameReadError):
    """Kare akışı başladıktan sonra kaynakla bağlantı koptu (ör. kamera çıkarıldı)."""


class DisplayError(Exception):
    """OpenCV görüntü penceresi oluşturulamadı veya kullanılamadı (ör. grafik oturumu yok)."""
