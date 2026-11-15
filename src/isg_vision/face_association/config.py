"""Yüz–kişi ilişkilendirme yapılandırması.

Bu modül ``cv2``/``ultralytics``/model import etmez, ağ kullanmaz; yalnızca değerleri doğrular.

**Varsayılan değerler evrensel optimum DEĞİL, gerekçeli başlangıç noktalarıdır** (gerçek model çıktılarıyla — ``zidane.jpg``,
``bus.jpg`` — ve sentetik senaryolarla sınanmıştır; ``docs/face_association.md``). Etiketli gerçek çoklu-kişi verisiyle ayarlanmamıştır.

Geometrik eşikler (kişi kutusu ``P`` içinde yüz kutusu ``F``)
-------------------------------------------------------------
* ``min_containment`` — ``alan(F ∩ P) / alan(F)``: yüzün ne kadarı kişi kutusunun içinde. Bir yüz, sahibinin kutusunun neredeyse
  tamamen içindedir; Kalman gecikmesi/kutu sıkılığı için bir miktar pay bırakılır (0,8).
* ``max_center_y_ratio`` — yüz merkezinin kişi kutusu yüksekliğine göre dikey konumu (üstten 0, alttan 1). Yüz kişinin ÜST bölgesindedir:
  ayakta, tam boy kişide ≈ 0,05–0,15; masa başında/dizüstü kamerasında yarım boy kişide ≈ 0,15–0,35. 0,55'in altında yüz, kutunun alt
  yarısındadır ve o kişinin yüzü olamaz (ör. başka birinin yüzü bu kişinin gövdesinin önünde).
* ``min_face_height_ratio`` / ``max_face_height_ratio`` — yüz yüksekliğinin kişi kutusu yüksekliğine oranı için makul aralık. Kişi
  kutusuna göre çok küçük (< 0,03) bir yüz (ör. tişörtteki baskı) veya çok büyük (> 0,8) bir yüz (kutudan büyük yüz) bu kişiye bağlanmaz.

Karar eşikleri
--------------
* ``min_score`` — eşleşme için en düşük geometrik skor (0..1). Altında kalan aday "belirsiz"dir (eşleştirilmez).
* ``ambiguity_margin`` — bir yüz (veya kişi) için en iyi aday ile ikinci aday arasındaki en az skor farkı. Fark bundan küçükse
  durum "belirsiz"dir ve HİÇBİR eşleşme yapılmaz (örtüşen kutularda tahmin yürütülmez).

Süreklilik (ID switch / devralma koruması) eşikleri
---------------------------------------------------
* ``jump_iou`` — aynı Track ID'nin ardışık iki karedeki kutuları arasındaki IoU bunun altındaysa kutu "zıpladı" (ID devri/anlık
  yer değiştirme şüphesi): süreklilik dönemi (epoch) yenilenir.
* ``max_gap_frames`` — bir Track ID bu kadar kareden fazla görünmeyip geri gelirse (ByteTrack tamponundan kurtarma) eski kişi ile
  aynı olduğu varsayılmaz: yeni epoch. (Ayrıca kutu IoU'su ``jump_iou``'nun altındaysa da yeni epoch.)
* ``overlap_guard`` — iki aktif kutunun ortak alanı, küçük kutunun alanının bu oranından büyükse "örtüşme" vardır; örtüşen izler
  güvenilir sayılmaz ve örtüşme bittikten sonra (``overlap_cooldown_frames`` kare sakin kalınca) epoch yenilenir.
* ``min_trusted_frames`` — bir epoch içinde bir iz en az bu kadar kare görülmeden "güvenilir" sayılmaz (yeni iz varsayılan olarak güvensizdir).
* ``state_ttl_frames`` — görünmeyen izlerin durumu bu kadar kare sonra silinir (bellek sınırı; sonra gelen aynı Track ID yeni sayılır).
* ``stationary_iou`` / ``stationary_flag_frames`` — kutusu bu kadar kare boyunca (neredeyse) hiç değişmeyen iz BAYRAKLANIR
  (``possibly_static``; poster/yorgan benzeri kalıcı yanlış pozitif adayı). Bayrak bir yargı, silme veya eşleşme engeli DEĞİLDİR
  (masa başında oturan gerçek kişi de bayraklanabilir).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import AssociationConfigError

DEFAULT_MIN_CONTAINMENT = 0.8
DEFAULT_MAX_CENTER_Y_RATIO = 0.55
DEFAULT_MIN_FACE_HEIGHT_RATIO = 0.03
DEFAULT_MAX_FACE_HEIGHT_RATIO = 0.8
DEFAULT_MIN_SCORE = 0.5
DEFAULT_AMBIGUITY_MARGIN = 0.15
DEFAULT_BORDER_FACTOR = 0.9
DEFAULT_JUMP_IOU = 0.3
DEFAULT_MAX_GAP_FRAMES = 3
DEFAULT_OVERLAP_GUARD = 0.25
DEFAULT_OVERLAP_COOLDOWN_FRAMES = 10
DEFAULT_MIN_TRUSTED_FRAMES = 3
DEFAULT_STATE_TTL_FRAMES = 150
DEFAULT_STATIONARY_IOU = 0.95
DEFAULT_STATIONARY_FLAG_FRAMES = 150


@dataclass(frozen=True)
class AssociationConfig:
    """Doğrulanmış yüz–kişi ilişkilendirme yapılandırması (alan anlamları için modül açıklamasına bakın).

    Attributes:
        min_containment: Yüzün kişi kutusu içinde kalması gereken en düşük oran, ``0 < x <= 1``.
        max_center_y_ratio: Yüz merkezinin kişi kutusundaki en büyük dikey konumu, ``0 < x <= 1``.
        min_face_height_ratio / max_face_height_ratio: Yüz/kişi yükseklik oranı aralığı, ``0 < min < max <= 1``.
        min_score: Eşleşme için en düşük geometrik skor, ``0 <= x <= 1``.
        ambiguity_margin: İlk iki aday arasındaki en az skor farkı, ``0 < x <= 1``.
        border_factor: Kadraj kenarındaki (``at_border``) yüzün skoruna uygulanan çarpan, ``0 < x <= 1`` (kısmen görünen yüzün
            geometrisi daha az güvenilirdir).
        jump_iou, max_gap_frames, overlap_guard, overlap_cooldown_frames, min_trusted_frames, state_ttl_frames, stationary_iou,
        stationary_flag_frames: Süreklilik izleyicisi eşikleri (modül açıklaması).
        require_frame_index: ``True`` ise yüz sonucunda ``frame_index`` yoksa girdi tutarsız sayılır (aynı kare doğrulanamaz).
    """

    min_containment: float = DEFAULT_MIN_CONTAINMENT
    max_center_y_ratio: float = DEFAULT_MAX_CENTER_Y_RATIO
    min_face_height_ratio: float = DEFAULT_MIN_FACE_HEIGHT_RATIO
    max_face_height_ratio: float = DEFAULT_MAX_FACE_HEIGHT_RATIO
    min_score: float = DEFAULT_MIN_SCORE
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN
    border_factor: float = DEFAULT_BORDER_FACTOR
    jump_iou: float = DEFAULT_JUMP_IOU
    max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES
    overlap_guard: float = DEFAULT_OVERLAP_GUARD
    overlap_cooldown_frames: int = DEFAULT_OVERLAP_COOLDOWN_FRAMES
    min_trusted_frames: int = DEFAULT_MIN_TRUSTED_FRAMES
    state_ttl_frames: int = DEFAULT_STATE_TTL_FRAMES
    stationary_iou: float = DEFAULT_STATIONARY_IOU
    stationary_flag_frames: int = DEFAULT_STATIONARY_FLAG_FRAMES
    require_frame_index: bool = True

    def __post_init__(self) -> None:
        for name in (
            "min_containment", "max_center_y_ratio", "min_face_height_ratio", "max_face_height_ratio", "ambiguity_margin",
            "border_factor", "jump_iou", "overlap_guard", "stationary_iou",
        ):
            _check_open_closed(name, getattr(self, name))  # 0 < x <= 1
        _check_closed(self, "min_score")  # 0 <= x <= 1
        if not self.min_face_height_ratio < self.max_face_height_ratio:
            raise AssociationConfigError(
                f"min_face_height_ratio ({self.min_face_height_ratio}) max_face_height_ratio'dan ({self.max_face_height_ratio}) küçük olmalıdır."
            )
        for name in ("max_gap_frames", "overlap_cooldown_frames"):
            _check_int(name, getattr(self, name), minimum=0)
        for name in ("min_trusted_frames", "state_ttl_frames", "stationary_flag_frames"):
            _check_int(name, getattr(self, name), minimum=1)
        if self.state_ttl_frames <= self.max_gap_frames:
            raise AssociationConfigError(
                f"state_ttl_frames ({self.state_ttl_frames}) max_gap_frames'ten ({self.max_gap_frames}) büyük olmalıdır "
                "(aksi halde izin geçmişi, boşluk toleransından önce silinirdi)."
            )
        if not isinstance(self.require_frame_index, bool):
            raise AssociationConfigError(f"require_frame_index True/False olmalıdır: {self.require_frame_index!r}")
        for name in (
            "min_containment", "max_center_y_ratio", "min_face_height_ratio", "max_face_height_ratio", "min_score", "ambiguity_margin",
            "border_factor", "jump_iou", "overlap_guard", "stationary_iou",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))


def _check_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AssociationConfigError(f"{name} sonlu bir sayı olmalıdır: {value!r}")
    return float(value)


def _check_open_closed(name: str, value: object) -> None:
    number = _check_number(name, value)
    if not 0.0 < number <= 1.0:
        raise AssociationConfigError(f"{name} 0'dan büyük ve 1'e eşit veya küçük olmalıdır: {value!r}")


def _check_closed(config: AssociationConfig, name: str) -> None:
    number = _check_number(name, getattr(config, name))
    if not 0.0 <= number <= 1.0:
        raise AssociationConfigError(f"{name} 0 ile 1 arasında olmalıdır: {getattr(config, name)!r}")


def _check_int(name: str, value: object, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AssociationConfigError(f"{name} {minimum} veya daha büyük bir tam sayı olmalıdır: {value!r}")
