"""Takip (ByteTrack) yapılandırması ve eşiklerin birbirini nasıl etkilediğinin analizi.

Bu modül ``ultralytics``/``torch`` import etmez ve ağ bağlantısı kurmaz; yalnızca değerleri doğrular.

Varsayılan değerler, kurulu ``ultralytics==8.4.210`` ile gelen ``cfg/trackers/bytetrack.yaml`` dosyasının değerleridir
(evrensel optimum değil, makul başlangıç noktasıdır; sahneye/kameraya göre ayarlanması beklenir).

Eşiklerin gerçek anlamı (Ultralytics 8.4.210 ``BYTETracker.update`` kaynak kodundan okundu; tahmin değildir)
------------------------------------------------------------------------------------------------------------
Her karede YOLO tespitleri güven skoruna (``s``) göre ikiye ayrılır:

* **Yüksek güven** ``s >= track_high_thresh``: birinci eşleştirmeye (Kalman tahmini ile IoU, ``match_thresh``) girer.
  Eşleşmeyen yüksek güvenli tespit, ``s >= new_track_thresh`` ise **yeni iz** doğurur. **Yeni iz yalnızca yüksek
  güvenli tespitten doğar**; bu yüzden ``new_track_thresh < track_high_thresh`` anlamsızdır (bu modül reddeder).
* **Düşük güven** ``track_low_thresh < s < track_high_thresh``: yalnızca ikinci eşleştirmeye girer (yalnızca IoU, eşik 0,5);
  **mevcut** izleri sürdürmeye/kurtarmaya yarar, asla yeni iz doğurmaz (ByteTrack'in temel fikri).
* ``s <= track_low_thresh``: tamamen yok sayılır.

Kritik sonuç: YOLO kendi güven eşiğiyle (``DetectionConfig.conf``, 4A varsayılanı 0,25) ``conf``'un altındaki kutuları
ByteTrack'e HİÇ VERMEZ. Bu yüzden ``conf >= track_high_thresh`` ise ikinci (düşük güven) aşama **fiilen devre dışıdır**.
Varsayılan yapılandırma (``conf=0.25``, ``track_high_thresh=0.25``) tam olarak bu durumdadır: bilinçli, güvenli bir tercihtir
(yanlış pozitifleri artırmaz), ancak ByteTrack'in düşük güvenli kurtarma özelliğini kullanmaz. Bu özelliği denemek
için YOLO ``conf`` değeri ``track_high_thresh``'nin altına indirilmelidir (ör. ``--conf 0.10 --track-high-thresh 0.35``);
bunun yanlış pozitif etkisi ölçülmeden varsayılan yapılmaz. ``analyze_thresholds`` bu ilişkiyi çalışma anında açıklar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import TrackingConfigError

DEFAULT_TRACK_HIGH_THRESH = 0.25
DEFAULT_TRACK_LOW_THRESH = 0.1
DEFAULT_NEW_TRACK_THRESH = 0.25
DEFAULT_TRACK_BUFFER = 30
DEFAULT_MATCH_THRESH = 0.8
DEFAULT_FUSE_SCORE = True
MAX_TRACK_BUFFER = 1000


@dataclass(frozen=True)
class TrackingConfig:
    """Doğrulanmış ByteTrack yapılandırması.

    Attributes:
        track_high_thresh: Yüksek/düşük güven ayrımı, ``0 < x <= 1``. Birinci eşleştirmeye girecek tespitlerin alt sınırı.
        track_low_thresh: İkinci (kurtarma) eşleştirmesi için alt sınır, ``0 < x < track_high_thresh``. Bunun altı yok sayılır.
        new_track_thresh: Yeni iz başlatmak için gereken en düşük güven, ``track_high_thresh <= x <= 1``.
            **Yanlış pozitiflerden yeni iz doğmasını azaltmanın tek tracker-içi kolu budur** (yüksek = daha az yeni iz,
            ama gerçek bir insanın iz alması da gecikir/kaçabilir).
        track_buffer: Kaybolan izin tutulacağı **işlenen kare** sayısı, ``1..1000``. Süre DEĞİL kare sayısıdır:
            30 kare, 30 FPS'te ~1 sn, 15 FPS'te ~2 sn'dir (Ultralytics bunu FPS'e göre ölçeklemez). Büyük değer kısa örtüşmeleri
            daha iyi tolere eder ama başka kişiye yanlış bağlanma (ID switch) riskini artırır. Ölçülen etkin kurtarma penceresi
            ``track_buffer + 1`` karedir (Ultralytics 8.4.210: silinen iz havuzda bir kare daha kalır); ``track_buffer + 2`` kare
            kayıp kalan kişi yeni ID alır.
        match_thresh: Birinci eşleştirmede kabul edilen en büyük maliyet (``1 - IoU``, ``fuse_score`` ile skorla birleştirilmiş),
            ``0 < x <= 1``. Büyük değer daha gevşek eşleştirme demektir (0,8 ≈ IoU en az 0,2).
        fuse_score: ``True`` ise eşleştirme maliyeti tespit skoruyla birleştirilir (zayıf tespitler daha zor eşleşir).
    """

    track_high_thresh: float = DEFAULT_TRACK_HIGH_THRESH
    track_low_thresh: float = DEFAULT_TRACK_LOW_THRESH
    new_track_thresh: float = DEFAULT_NEW_TRACK_THRESH
    track_buffer: int = DEFAULT_TRACK_BUFFER
    match_thresh: float = DEFAULT_MATCH_THRESH
    fuse_score: bool = DEFAULT_FUSE_SCORE

    def __post_init__(self) -> None:
        for name in ("track_high_thresh", "track_low_thresh", "new_track_thresh", "match_thresh"):
            _check_unit_interval(name, getattr(self, name))
        if not self.track_low_thresh < self.track_high_thresh:
            raise TrackingConfigError(
                f"track_low_thresh ({self.track_low_thresh}) track_high_thresh'ten ({self.track_high_thresh}) küçük olmalıdır."
            )
        if self.new_track_thresh < self.track_high_thresh:
            raise TrackingConfigError(
                f"new_track_thresh ({self.new_track_thresh}) track_high_thresh'ten ({self.track_high_thresh}) küçük olamaz: "
                "yeni iz yalnızca yüksek güvenli tespitten doğar, bu ayar sessizce etkisiz kalırdı."
            )
        if isinstance(self.track_buffer, bool) or not isinstance(self.track_buffer, int) or not 1 <= self.track_buffer <= MAX_TRACK_BUFFER:
            raise TrackingConfigError(f"track_buffer 1 ile {MAX_TRACK_BUFFER} arasında bir tam sayı (kare sayısı) olmalıdır: {self.track_buffer!r}")
        if not isinstance(self.fuse_score, bool):
            raise TrackingConfigError(f"fuse_score True/False olmalıdır: {self.fuse_score!r}")
        for name in ("track_high_thresh", "track_low_thresh", "new_track_thresh", "match_thresh"):
            object.__setattr__(self, name, float(getattr(self, name)))

    def bytetrack_args(self) -> dict[str, float | int | bool]:
        """Ultralytics ``BYTETracker``'ın beklediği (``cfg/trackers/bytetrack.yaml`` ile aynı anahtarlar) sözlük."""
        return {
            "tracker_type": "bytetrack",
            "track_high_thresh": self.track_high_thresh,
            "track_low_thresh": self.track_low_thresh,
            "new_track_thresh": self.new_track_thresh,
            "track_buffer": self.track_buffer,
            "match_thresh": self.match_thresh,
            "fuse_score": self.fuse_score,
        }


@dataclass(frozen=True)
class ThresholdAnalysis:
    """YOLO ``conf`` ile takip eşiklerinin birlikte ne anlama geldiği.

    Attributes:
        detection_conf: YOLO'nun kutuları attığı güven eşiği (``DetectionConfig.conf``).
        low_stage_active: Düşük güvenli (ikinci) eşleştirme aşamasına tespit ulaşabilir mi.
        low_band: İkinci aşamaya girebilecek skor aralığı ``(alt, üst)`` (alt dahil değil, üst dahil değil); aşama kapalıysa ``None``.
        effective_new_track_floor: Yeni iz doğurabilecek en düşük skor (``max(conf, track_high_thresh, new_track_thresh)``).
        notes: Kullanıcıya gösterilecek açıklamalar (Türkçe).
    """

    detection_conf: float
    low_stage_active: bool
    low_band: tuple[float, float] | None
    effective_new_track_floor: float
    notes: tuple[str, ...]


def analyze_thresholds(detection_conf: float, config: TrackingConfig) -> ThresholdAnalysis:
    """``conf`` ve takip eşiklerinin etkileşimini hesaplar (ağ/model gerektirmez)."""
    _check_unit_interval("detection_conf", detection_conf)
    conf = float(detection_conf)
    high, low, new = config.track_high_thresh, config.track_low_thresh, config.new_track_thresh
    floor = max(conf, high, new)
    notes: list[str] = []

    low_active = conf < high
    band: tuple[float, float] | None = (max(conf, low), high) if low_active else None
    if low_active:
        assert band is not None
        notes.append(
            f"Düşük güvenli kurtarma aşaması AÇIK: skoru {band[0]:.2f}–{band[1]:.2f} arasındaki tespitler yalnızca mevcut izleri "
            "sürdürmek için kullanılır, yeni iz doğurmaz. Bu aralık yanlış pozitif de içerebilir (etkisi ölçülmeden varsayılan yapılmamıştır)."
        )
    else:
        notes.append(
            f"Düşük güvenli kurtarma aşaması fiilen KAPALI: YOLO conf ({conf:.2f}) >= track_high_thresh ({high:.2f}); "
            f"conf altındaki kutular ByteTrack'e hiç verilmiyor. Açmak için YOLO --conf değerini {high:.2f}'nin altına indirin."
        )
    if conf > high:
        notes.append(f"YOLO conf ({conf:.2f}), track_high_thresh'ten ({high:.2f}) yüksek: etkin alt sınır YOLO conf'tur.")
    notes.append(
        f"Yeni iz için etkin en düşük güven: {floor:.2f}. İz, ancak ardışık İKİ karede eşleşirse görüntülenir "
        "(tek karelik tespitler iz olarak gösterilmez; oturumun ilk karesi bunun istisnasıdır)."
    )
    return ThresholdAnalysis(conf, low_active, band, floor, tuple(notes))


def _check_unit_interval(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TrackingConfigError(f"{name} sayı olmalıdır (0 < {name} <= 1): {value!r}")
    if not 0.0 < float(value) <= 1.0:
        raise TrackingConfigError(f"{name} 0'dan büyük ve 1'e eşit veya küçük olmalıdır: {value!r}")
