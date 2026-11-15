"""Yüz–kişi ilişkilendirme hataları.

Kamera, insan tespiti ve yüz tespiti modüllerindeki düzenle aynı iki aile:

* ``AssociationConfigError`` — kullanıcının verdiği eşikler/ayarlar geçersiz (yapılandırma aşamasında oluşur).
* ``FaceAssociationError`` ve alt sınıfları — çalışma anı hataları.

Önemli ayrım: ``AssociationInputError`` yalnızca YAPISAL olarak geçersiz girdi içindir (``None``, yanlış tür, eksik öznitelik). Aynı
karenin yüz ve takip sonuçlarının birbiriyle TUTARSIZ olması (``frame_index``/``image_size``/zaman damgası uyuşmazlığı, tekrarlanan
Track ID...) bir hata fırlatmaz: ilişkilendirici hiçbir eşleştirme yapmadan ``invalid_input`` durumlu bir sonuç döndürür (canlı
döngü, tek bir tutarsız kare yüzünden çökmez ve tutarsız girdiden tahmini eşleşme üretilmez).
"""

from __future__ import annotations


class AssociationConfigError(ValueError):
    """Geçersiz ilişkilendirme yapılandırması (eşik aralığı, eşikler arası ilişki, kare sayıları vb.)."""


class FaceAssociationError(Exception):
    """Yüz–kişi ilişkilendirme çalışma anı hatalarının ortak tabanı."""


class AssociationInputError(FaceAssociationError, ValueError):
    """İlişkilendiriciye verilen nesne yapısal olarak geçersiz (``None``, yanlış tür, eksik öznitelik)."""


class AssociationResultError(FaceAssociationError, ValueError):
    """Bir ilişkilendirme sonucu değişmezleri ihlal ediyor (ör. bir yüz iki kişiye atanmış). Bir hata göstergesidir; sonuç üretilmez."""
