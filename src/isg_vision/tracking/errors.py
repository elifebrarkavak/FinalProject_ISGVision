"""İnsan takibi hataları.

Kamera ve insan tespiti modüllerindeki düzenle aynı iki aile:

* ``TrackingConfigError`` — kullanıcının verdiği takip eşikleri/ayarları geçersiz (yapılandırma aşamasında oluşur).
* ``TrackingError`` ve alt sınıfları — çalışma anı hataları (bağımlılık eksik, takipçi başlatılamadı, geçersiz girdi, kare sırası bozuk).

Mesajlar kullanıcıya doğrudan gösterilebilecek şekilde, olası nedeni ve çözümü içerecek biçimde yazılır.
"""

from __future__ import annotations


class TrackingConfigError(ValueError):
    """Geçersiz takip yapılandırması (eşik aralığı, eşikler arası ilişki, tampon uzunluğu vb.)."""


class TrackingError(Exception):
    """İnsan takibi çalışma anı hatalarının ortak tabanı."""


class TrackerDependencyError(TrackingError):
    """ByteTrack'in gerektirdiği bir paket (``lap``, ``ultralytics``) kurulu değil veya kullanılamıyor.

    Hiçbir paket otomatik kurulmaz; kurulum kullanıcıya ait, açık bir adımdır.
    """


class TrackerBackendError(TrackingError):
    """ByteTrack takipçisi başlatılamadı, beklenmeyen biçimde çöktü veya beklenmeyen/tutarsız çıktı verdi."""


class InvalidTrackingInputError(TrackingError, ValueError):
    """Takipçiye verilen tespit sonucu geçersiz (yanlış tür, geçersiz görüntü boyutu, görüntü dışı kutu)."""


class FrameOrderError(TrackingError, ValueError):
    """Kare sırası bozuk: aynı ``frame_index`` ikinci kez veya daha küçük bir ``frame_index``/zaman damgası geldi.

    Takipçi durumu bu hatada DEĞİŞMEZ (kare işlenmemiş sayılır).
    """
