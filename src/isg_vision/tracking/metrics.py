"""Takip sürekliliği için küçük, bağımsız yardımcı: ID switch ve ID devralma sayımı.

**Bu bir doğruluk ölçümü (IDF1 / HOTA / MOTA) DEĞİLDİR.** Bu metrikler etiketli (ground-truth) takip kimlikleri ve kutu eşleştirmesi
ister; bu projede etiketli takip verisi yoktur, dolayısıyla IDF1/HOTA/MOTA **ölçülmemiştir**. Bu modül yalnızca, gerçek kimliğin
bilindiği KONTROLLÜ durumlarda (ör. betiklenmiş sahte tespit dizileri veya ileride hazırlanacak etiketli bir değerlendirme videosu)
"takipçi bir kişinin numarasını değiştirdi mi?" sorusunu sayar.

Girdi: her kare için ``{gerçek_kimlik: track_id}`` sözlüğü (o karede takip edilen kimlikler). Tanımlar:

* **ID switch**: bir gerçek kimliğin, kendisine daha önce atanmış olan ``track_id``'den farklı bir ``track_id`` alması
  (kimliğin takip edilmediği kareler atlanır; yani kayıptan sonra yeni numarayla dönmek de ID switch sayılır).
* **ID devralma (takeover)**: bir ``track_id``'nin, daha önce başka bir gerçek kimliğe atanmış olduğu halde yeni bir gerçek kimliğe
  verilmesi (ör. örtüşmeden sonra B kişisinin A'nın numarasını alması). Kimlik karışmasının en tehlikeli biçimidir.
* **Çift atama**: aynı karede aynı ``track_id``'nin iki gerçek kimliğe verilmesi (``TrackingResult`` benzersiz ID garantisi bunu engeller;
  yine de sayılır).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class IdSwitchReport:
    """Süreklilik sayımının sonucu.

    Attributes:
        switches: Toplam ID switch sayısı.
        takeovers: Toplam ID devralma sayısı.
        duplicate_frames: Aynı ``track_id``'nin birden çok kimliğe verildiği kare sayısı.
        switches_per_identity: Kimlik başına ID switch.
        track_ids_per_identity: Kimlik başına görülen farklı ``track_id``'ler (ilk görülme sırasıyla).
        frames_tracked_per_identity: Kimlik başına takip edilen kare sayısı.
    """

    switches: int
    takeovers: int
    duplicate_frames: int
    switches_per_identity: dict[str, int]
    track_ids_per_identity: dict[str, tuple[int, ...]]
    frames_tracked_per_identity: dict[str, int]

    @property
    def is_continuous(self) -> bool:
        """Hiç ID switch, devralma veya çift atama yoksa ``True``."""
        return self.switches == 0 and self.takeovers == 0 and self.duplicate_frames == 0


def count_id_switches(frames: Sequence[Mapping[str, int]]) -> IdSwitchReport:
    """Kare dizisindeki ID switch / devralma / çift atamaları sayar (bkz. modül açıklaması)."""
    last_id: dict[str, int] = {}
    owner: dict[int, str] = {}  # track_id -> en son atandığı gerçek kimlik
    ids_seen: dict[str, list[int]] = {}
    tracked_frames: dict[str, int] = {}
    switches_per: dict[str, int] = {}
    takeovers = duplicates = 0

    for assignment in frames:
        values = list(assignment.values())
        if len(values) != len(set(values)):
            duplicates += 1
        for identity, track_id in assignment.items():
            tracked_frames[identity] = tracked_frames.get(identity, 0) + 1
            switches_per.setdefault(identity, 0)
            seen = ids_seen.setdefault(identity, [])
            previous = last_id.get(identity)
            if previous is not None and previous != track_id:
                switches_per[identity] += 1
            if track_id not in seen:
                seen.append(track_id)
            previous_owner = owner.get(track_id)
            if previous_owner is not None and previous_owner != identity:
                takeovers += 1
            owner[track_id] = identity
            last_id[identity] = track_id

    return IdSwitchReport(
        switches=sum(switches_per.values()),
        takeovers=takeovers,
        duplicate_frames=duplicates,
        switches_per_identity=switches_per,
        track_ids_per_identity={k: tuple(v) for k, v in ids_seen.items()},
        frames_tracked_per_identity=tracked_frames,
    )
