"""Yüz–kişi ilişkilendirme sonuç veri yapıları.

Koordinat kuralı projenin tamamıyla aynıdır (``human_detection.results``): ``bbox_xyxy = (x1, y1, x2, y2)``, piksel, sol-üst
başlangıç, ``x1 < x2``, ``y1 < y2``, görüntü sınırlarına kırpılmış ``float``.

**Bu sonuçlarda kimlik YOKTUR.** ``track_id`` geçici ve oturum kapsamlı bir takip numarasıdır (``(session_id, track_id)``); çalışan adı,
personel numarası, yüz kimliği, embedding veya biyometrik şablon bulunmaz. ``score`` GEOMETRİK uygunluktur (``geometry`` modülü),
kimlik doğruluğu veya canlılık değildir; yüz tespit güveni (``face_confidence``) de kimlik doğruluğu DEĞİLDİR.

Durumlar (açıkça ayrılmıştır; ``STATUS_*``)
-------------------------------------------
Kişi için: ``matched`` (eşleşti), ``no_face`` (yüz bulunamadı), ``ambiguous`` (belirsiz eşleşme), ``invalid_input``.
Yüz için: ``matched``, ``unmatched_face`` (eşleşmeyen yüz), ``ambiguous``, ``invalid_input``.

Değişmezler (``FaceAssociationResult`` oluşturulurken DENETLENİR; ihlalde ``AssociationResultError``)
-------------------------------------------------------------------------------------------------
* Bir yüz aynı karede EN FAZLA bir kişiye, bir kişi EN FAZLA bir yüze atanır.
* Eşleşme karşılıklıdır: kişinin ``face_index``'i, ``matched`` durumlu ve o kişiye (``session_id``, ``track_id``) işaret eden yüzdür.
* ``matched`` olmayan hiçbir kişi/yüzde yüz/kişi alanı bulunmaz; ``invalid_input`` sonucunda hiçbir eşleşme yoktur.

Bu modül yalnızca standart kütüphane (+ ``human_detection.results``) kullanır; NumPy/OpenCV import etmez.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

from isg_vision.human_detection.results import BBox

from .errors import AssociationResultError

__all__ = [
    "BREAK_GAP", "BREAK_JUMP", "BREAK_NEW_TRACK", "BREAK_OVERLAP_END", "BREAK_SESSION_RESET", "BREAK_REASONS",
    "FACE_STATUSES", "PERSON_STATUSES", "REASON_CANDIDATES_TAKEN", "REASON_CONTESTED", "REASON_GEOMETRY_OK",
    "REASON_INPUT_INCONSISTENT", "REASON_LOW_SCORE", "REASON_NO_CANDIDATE", "STATUS_AMBIGUOUS", "STATUS_INVALID",
    "STATUS_MATCHED", "STATUS_NO_FACE", "STATUS_UNMATCHED_FACE", "UNTRUSTED_NEW_EPOCH", "UNTRUSTED_OVERLAP",
    "UNTRUSTED_OVERLAP_COOLDOWN", "AssociatedFace", "AssociatedPerson", "BBox", "FaceAssociationResult", "TrackContinuity",
]

STATUS_MATCHED = "matched"  # eşleşti
STATUS_NO_FACE = "no_face"  # yüz bulunamadı (kişi için)
STATUS_AMBIGUOUS = "ambiguous"  # belirsiz eşleşme: eşleştirilmedi
STATUS_UNMATCHED_FACE = "unmatched_face"  # eşleşmeyen yüz (yüz için)
STATUS_INVALID = "invalid_input"  # geçersiz veya tutarsız girdi: hiçbir eşleştirme yapılmadı
PERSON_STATUSES = (STATUS_MATCHED, STATUS_NO_FACE, STATUS_AMBIGUOUS, STATUS_INVALID)
FACE_STATUSES = (STATUS_MATCHED, STATUS_UNMATCHED_FACE, STATUS_AMBIGUOUS, STATUS_INVALID)

REASON_GEOMETRY_OK = "geometry_ok"  # eşleşti: sert elemeler geçti, skor yeterli, rakip yok/yeterince geride
REASON_NO_CANDIDATE = "no_candidate"  # sert elemeleri geçen karşı taraf yok
REASON_CANDIDATES_TAKEN = "candidates_taken"  # adayların hepsi başka bir eşleşmeyle (güvenle) alındı
REASON_CONTESTED = "contested"  # ilk iki aday arasındaki fark ``ambiguity_margin``'den küçük (veya karşılıklı en iyi değil)
REASON_LOW_SCORE = "low_score"  # tek/en iyi aday var ama skor ``min_score``'un altında
REASON_INPUT_INCONSISTENT = "input_inconsistent"
REASONS = (REASON_GEOMETRY_OK, REASON_NO_CANDIDATE, REASON_CANDIDATES_TAKEN, REASON_CONTESTED, REASON_LOW_SCORE, REASON_INPUT_INCONSISTENT)

BREAK_NEW_TRACK = "new_track"  # bu (oturum, iz) ilk kez görüldü
BREAK_SESSION_RESET = "session_reset"  # takip oturumu değişti/sıfırlandı: tüm geçmiş geçersiz
BREAK_JUMP = "jump"  # ardışık karelerde kutu zıpladı (IoU < jump_iou): ID devri/anlık yer değiştirme şüphesi
BREAK_GAP = "gap"  # iz uzun süre (veya konum değiştirerek) görünmeyip geri geldi: aynı kişi varsayılmaz
BREAK_OVERLAP_END = "overlap_end"  # örtüşme bitti: örtüşme sırasında ID değişmiş olabilir
BREAK_REASONS = (BREAK_NEW_TRACK, BREAK_SESSION_RESET, BREAK_JUMP, BREAK_GAP, BREAK_OVERLAP_END)

UNTRUSTED_NEW_EPOCH = "new_epoch"  # epoch'ta henüz yeterli kare görülmedi (veya bu karede epoch yenilendi)
UNTRUSTED_OVERLAP = "overlap"  # başka bir izle örtüşüyor
UNTRUSTED_OVERLAP_COOLDOWN = "overlap_cooldown"  # örtüşme yeni bitti, sakinleşme süresi dolmadı


def _check_box(name: str, box: object) -> BBox:
    try:
        values = tuple(float(v) for v in box)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        raise ValueError(f"{name} dört sayıdan oluşmalıdır: {box!r}") from None
    if len(values) != 4 or not all(math.isfinite(v) for v in values) or values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError(f"{name} için dört sonlu sayı ve x1<x2, y1<y2 gerekir: {box!r}")
    return values  # type: ignore[return-value]


def _check_unit(name: str, value: float | None, *, allow_none: bool = True) -> None:
    if value is None:
        if allow_none:
            return
        raise ValueError(f"{name} gereklidir")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} 0..1 aralığında sonlu bir sayı olmalıdır: {value!r}")


@dataclass(frozen=True)
class TrackContinuity:
    """Bir izin süreklilik/güvenilirlik durumu (ID switch ve ID devralma koruması; ``continuity`` modülü).

    **Ne için?** Eşleşme her karede YALNIZCA o karenin geometrisinden (yüz, kişi kutusunun kafa bölgesinde mi) kurulur ve geçmişten
    HİÇBİR eşleşme taşınmaz; bu yüzden eski bir kişinin yüzü başka birine kalıcı olarak yapışamaz. Ancak sonraki aşamalar (kimlik
    tanıma, çok kareli kanıt) bir Track ID altında kareler boyunca kanıt biriktirebilir; Track ID kişi değiştirmişse bu birikim
    yanlış kişiye ait olur. ``epoch`` ve ``trusted`` bunu önler: kanıt YALNIZCA aynı ``(session_id, track_id, epoch)`` içinde ve
    ``trusted`` karelerde birikmelidir; ``epoch`` değişince veya iz güvensizken eski birikim bu ize ait SAYILMAZ.

    Attributes:
        epoch: Bu ``(session_id, track_id)`` için süreklilik dönemi numarası (``>= 1``). Şüpheli ID devri/zıplama/uzun kayıp/örtüşme
            sonrası artar; yeni iz 1'den başlar. Aynı numara aynı kişi anlamına GELMEZ; yalnızca "süreklilik bozulmadı" demektir.
        frames_in_epoch: Bu epoch'ta izin görüldüğü kare sayısı (bu kare dahil).
        trusted: ``True`` ise bu karede iz sürekli sayılır (epoch'ta en az ``min_trusted_frames`` kare, örtüşme/sakinleşme yok, bu
            karede epoch yenilenmedi). ``False`` güvenli varsayılandır.
        untrusted_reasons: ``trusted=False`` ise nedenler (``UNTRUSTED_*``); aksi halde boş.
        breaks: Bu karede epoch'u yenileyen olaylar (``BREAK_*``); çoğu karede boştur.
        overlap_fraction: Bu iz ile başka bir aktif iz arasındaki en büyük örtüşme (ortak alan / küçük kutu alanı, ``0..1``).
        overlapping_track_ids: Örtüşme eşiğini aşan diğer aktif izlerin ``track_id``'leri.
        matched_streak: Bu epoch'ta art arda yüz eşleşmesi alınan kare sayısı (bu kare dahil; eşleşme yoksa 0).
        frames_since_match: Bu epoch'ta son eşleşmeden beri geçen kare (bu karede eşleştiyse 0; epoch'ta hiç eşleşme yoksa ``None``).
        stationary_frames: Kutusu art arda neredeyse hiç değişmeyen kare sayısı.
        possibly_static: ``stationary_frames`` eşiği aştı: kalıcı hareketsiz iz (poster/yorgan benzeri yanlış pozitif ADAYI, ya da
            masa başındaki gerçek kişi). **Yargı/silme/engel DEĞİLDİR; yalnızca bayraktır.**
    """

    epoch: int
    frames_in_epoch: int
    trusted: bool
    untrusted_reasons: tuple[str, ...] = ()
    breaks: tuple[str, ...] = ()
    overlap_fraction: float = 0.0
    overlapping_track_ids: tuple[int, ...] = ()
    matched_streak: int = 0
    frames_since_match: int | None = None
    stationary_frames: int = 0
    possibly_static: bool = False

    def __post_init__(self) -> None:
        if self.epoch < 1 or self.frames_in_epoch < 1 or self.matched_streak < 0 or self.stationary_frames < 0:
            raise ValueError("epoch/frames_in_epoch >= 1, matched_streak/stationary_frames >= 0 olmalıdır")
        if self.frames_since_match is not None and self.frames_since_match < 0:
            raise ValueError(f"frames_since_match negatif olamaz: {self.frames_since_match}")
        if self.trusted and self.untrusted_reasons:
            raise ValueError("trusted=True iken untrusted_reasons boş olmalıdır")
        if not self.trusted and not self.untrusted_reasons:
            raise ValueError("trusted=False iken en az bir untrusted_reason gerekir")
        unknown = [b for b in self.breaks if b not in BREAK_REASONS]
        if unknown:
            raise ValueError(f"bilinmeyen süreklilik kesintisi: {unknown}")
        _check_unit("overlap_fraction", self.overlap_fraction, allow_none=False)


@dataclass(frozen=True)
class AssociatedPerson:
    """Bir aktif izin (kişinin) bu karedeki yüz ilişkilendirmesi.

    Attributes:
        session_id / track_id: Takip anahtarı ``(session_id, track_id)``; geçici, kimlik DEĞİL.
        frame_index / timestamp: Aynı karenin sıra numarası ve ``FramePacket.timestamp``'ı.
        person_bbox / person_confidence: Takipçinin kutusu ve YOLO güveni (``TrackedPerson``'dan).
        status: ``STATUS_MATCHED`` / ``STATUS_NO_FACE`` / ``STATUS_AMBIGUOUS`` / ``STATUS_INVALID``.
        reason: Durumun kısa nedeni (``REASON_*``).
        face_index / face_bbox / face_confidence / face_at_border: Yalnızca ``matched`` iken dolu: eşleşen yüzün ``FaceDetectionResult``
            içindeki sırası, kutusu, tespit güveni (kimlik doğruluğu DEĞİL) ve kadraj kenarı bayrağı.
        score: Yalnızca ``matched`` iken: geometrik uygunluk skoru (kimlik doğruluğu DEĞİL).
        margin: Yalnızca ``matched`` iken: bu eşleşmenin en iyi ve ikinci aday arasındaki (yüz ve kişi tarafının küçüğü) skor farkı;
            rakip aday yoksa ``None`` (``ambiguity_margin``'den küçük fark eşleşmeyi ``ambiguous`` yapardı).
        candidate_face_indices: Sert elemeleri geçen aday yüzlerin sırası (en iyiden başlayarak; çözümleme/ayıklama içindir).
        continuity: Süreklilik/güvenilirlik durumu (``invalid_input`` sonucunda ``None``).
    """

    session_id: int
    track_id: int
    frame_index: int
    timestamp: float | None
    person_bbox: BBox
    person_confidence: float
    status: str
    reason: str
    face_index: int | None = None
    face_bbox: BBox | None = None
    face_confidence: float | None = None
    face_at_border: bool = False
    score: float | None = None
    margin: float | None = None
    candidate_face_indices: tuple[int, ...] = ()
    continuity: TrackContinuity | None = None

    def __post_init__(self) -> None:
        for name in ("session_id", "track_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} 1 veya daha büyük bir tam sayı olmalıdır: {value!r}")
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError(f"frame_index negatif olmayan bir tam sayı olmalıdır: {self.frame_index!r}")
        if self.status not in PERSON_STATUSES:
            raise ValueError(f"status {PERSON_STATUSES} değerlerinden biri olmalıdır: {self.status!r}")
        if self.reason not in REASONS:
            raise ValueError(f"reason {REASONS} değerlerinden biri olmalıdır: {self.reason!r}")
        object.__setattr__(self, "person_bbox", _check_box("person_bbox", self.person_bbox))
        _check_unit("person_confidence", self.person_confidence, allow_none=False)
        matched = self.status == STATUS_MATCHED
        face_fields = (self.face_index, self.face_bbox, self.face_confidence, self.score)
        if matched:
            if any(f is None for f in face_fields):
                raise ValueError("matched durumunda face_index/face_bbox/face_confidence/score dolu olmalıdır")
            if self.reason != REASON_GEOMETRY_OK:
                raise ValueError(f"matched durumunun nedeni {REASON_GEOMETRY_OK!r} olmalıdır: {self.reason!r}")
            if self.face_index < 0:  # type: ignore[operator]
                raise ValueError(f"face_index negatif olamaz: {self.face_index}")
            object.__setattr__(self, "face_bbox", _check_box("face_bbox", self.face_bbox))
            _check_unit("face_confidence", self.face_confidence, allow_none=False)
            _check_unit("score", self.score, allow_none=False)
            _check_unit("margin", self.margin)
        else:
            if any(f is not None for f in face_fields) or self.face_at_border or self.margin is not None:
                raise ValueError(f"{self.status} durumunda eşleşen yüz alanları boş olmalıdır")
            if self.reason == REASON_GEOMETRY_OK:
                raise ValueError(f"{self.status} durumunun nedeni {REASON_GEOMETRY_OK!r} olamaz")
        if self.status == STATUS_INVALID and self.continuity is not None:
            raise ValueError("invalid_input durumunda continuity None olmalıdır (durum güncellenmedi)")
        object.__setattr__(self, "candidate_face_indices", tuple(self.candidate_face_indices))

    @property
    def key(self) -> tuple[int, int]:
        """Oturumlar arası benzersiz takip anahtarı ``(session_id, track_id)`` (kimlik DEĞİL)."""
        return (self.session_id, self.track_id)

    @property
    def is_matched(self) -> bool:
        return self.status == STATUS_MATCHED

    @property
    def trusted_match(self) -> bool:
        """Yüz eşleşmesi var VE iz bu karede sürekli/güvenilir sayılıyor (kanıt biriktirmeye uygun)."""
        return self.is_matched and self.continuity is not None and self.continuity.trusted


@dataclass(frozen=True)
class AssociatedFace:
    """Bir yüzün bu karedeki kişi ilişkilendirmesi.

    Attributes:
        face_index: Yüzün ``FaceDetectionResult.faces`` içindeki sırası.
        face_bbox / face_confidence / at_border: Yüz kutusu, tespit güveni (kimlik doğruluğu DEĞİL), kadraj kenarı bayrağı.
        status: ``STATUS_MATCHED`` / ``STATUS_UNMATCHED_FACE`` / ``STATUS_AMBIGUOUS`` / ``STATUS_INVALID``.
        reason: Durumun kısa nedeni (``REASON_*``).
        session_id / track_id / score: Yalnızca ``matched`` iken: yüzün atandığı iz ve geometrik skor.
        candidate_track_ids: Sert elemeleri geçen aday izlerin ``track_id``'leri (en iyiden başlayarak).
    """

    face_index: int
    frame_index: int
    timestamp: float | None
    face_bbox: BBox
    face_confidence: float
    at_border: bool
    status: str
    reason: str
    session_id: int | None = None
    track_id: int | None = None
    score: float | None = None
    candidate_track_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.face_index, bool) or not isinstance(self.face_index, int) or self.face_index < 0:
            raise ValueError(f"face_index negatif olmayan bir tam sayı olmalıdır: {self.face_index!r}")
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError(f"frame_index negatif olmayan bir tam sayı olmalıdır: {self.frame_index!r}")
        if self.status not in FACE_STATUSES:
            raise ValueError(f"status {FACE_STATUSES} değerlerinden biri olmalıdır: {self.status!r}")
        if self.reason not in REASONS:
            raise ValueError(f"reason {REASONS} değerlerinden biri olmalıdır: {self.reason!r}")
        object.__setattr__(self, "face_bbox", _check_box("face_bbox", self.face_bbox))
        _check_unit("face_confidence", self.face_confidence, allow_none=False)
        link = (self.session_id, self.track_id, self.score)
        if self.status == STATUS_MATCHED:
            if any(f is None for f in link):
                raise ValueError("matched durumunda session_id/track_id/score dolu olmalıdır")
            if self.reason != REASON_GEOMETRY_OK:
                raise ValueError(f"matched durumunun nedeni {REASON_GEOMETRY_OK!r} olmalıdır: {self.reason!r}")
            _check_unit("score", self.score, allow_none=False)
        else:
            if any(f is not None for f in link):
                raise ValueError(f"{self.status} durumunda session_id/track_id/score boş olmalıdır")
            if self.reason == REASON_GEOMETRY_OK:
                raise ValueError(f"{self.status} durumunun nedeni {REASON_GEOMETRY_OK!r} olamaz")
        object.__setattr__(self, "candidate_track_ids", tuple(self.candidate_track_ids))

    @property
    def is_matched(self) -> bool:
        return self.status == STATUS_MATCHED


@dataclass(frozen=True)
class FaceAssociationResult:
    """Tek bir karenin birleşik (insan takibi + yüz tespiti + ilişkilendirme) sonucu.

    Sayılar birbirine KARIŞTIRILMAMALIDIR: ``person_count`` (aktif izler), ``face_count`` (yüz tespitleri), ``matched_count`` (güvenle
    eşleşen çiftler), ``ambiguous_*`` (belirsiz; eşleştirilmedi), ``no_face_count`` (yüzü bulunamayan kişiler) ve
    ``unmatched_face_count`` (hiçbir kişiye bağlanamayan yüzler). ``face_count`` benzersiz kişi sayısı DEĞİLDİR.

    Attributes:
        persons: Aktif izler (``track_id``'ye göre artan sırada).
        faces: Yüzler (``face_index`` sırasında).
        image_size: ``(genişlik, yükseklik)``; kutular bu görüntüdedir.
        frame_index / timestamp: Karenin sıra numarası ve zaman damgası (``None``: bilinmiyor; yalnızca geçersiz girdide olabilir).
        session_id: Takip oturumu (``TrackingResult.session_id``).
        association_ms: Yalnızca ilişkilendirme adımının süresi (ms); YOLO, ByteTrack ve YuNet süreleri dahil DEĞİLDİR.
        input_valid: ``False`` ise yüz ve takip sonuçları aynı kareye ait değil/tutarsız; HİÇBİR eşleştirme yapılmamıştır.
        invalid_reason: ``input_valid=False`` iken tutarsızlığın açıklaması.
        breaks: Bu karede süreklilik kesintisi yaşayan izler ``(track_id, neden)`` (ID switch/devralma şüphesi dahil; ``BREAK_*``).
    """

    persons: tuple[AssociatedPerson, ...]
    faces: tuple[AssociatedFace, ...]
    image_size: tuple[int, int]
    frame_index: int | None
    timestamp: float | None
    session_id: int
    association_ms: float = 0.0
    input_valid: bool = True
    invalid_reason: str | None = None
    breaks: tuple[tuple[int, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "persons", tuple(self.persons))
        object.__setattr__(self, "faces", tuple(self.faces))
        object.__setattr__(self, "breaks", tuple(self.breaks))
        self._check_invariants()

    # ------------------------------------------------------------------ değişmezler
    def _check_invariants(self) -> None:
        if not self.input_valid and not self.invalid_reason:
            raise AssociationResultError("input_valid=False iken invalid_reason gerekir")
        if self.input_valid and self.invalid_reason:
            raise AssociationResultError("input_valid=True iken invalid_reason boş olmalıdır")
        if not self.input_valid:  # tutarsız girdi (yinelenen Track ID/yüz sırası dahil) bu sonuçta GÖSTERİLEBİLİR; yalnızca eşleşme olmadığı denetlenir
            if any(p.status != STATUS_INVALID for p in self.persons) or any(f.status != STATUS_INVALID for f in self.faces):
                raise AssociationResultError("geçersiz girdide tüm kişi ve yüzler invalid_input durumunda olmalıdır (eşleştirme yapılmaz)")
            return
        track_ids = [p.track_id for p in self.persons]
        if len(set(track_ids)) != len(track_ids):
            raise AssociationResultError(f"aynı karede yinelenen track_id: {track_ids}")
        face_indices = [f.face_index for f in self.faces]
        if len(set(face_indices)) != len(face_indices):
            raise AssociationResultError(f"aynı karede yinelenen face_index: {face_indices}")
        faces_by_index = {f.face_index: f for f in self.faces}
        persons_by_track = {p.track_id: p for p in self.persons}
        if any(p.status == STATUS_INVALID for p in self.persons) or any(f.status == STATUS_INVALID for f in self.faces):
            raise AssociationResultError("geçerli girdide invalid_input durumlu kişi/yüz olamaz")
        used_faces: set[int] = set()
        for person in self.persons:
            if not person.is_matched:
                continue
            assert person.face_index is not None  # noqa: S101 (AssociatedPerson doğruladı)
            if person.face_index in used_faces:
                raise AssociationResultError(f"yüz {person.face_index} aynı karede birden fazla kişiye atanmış")
            used_faces.add(person.face_index)
            face = faces_by_index.get(person.face_index)
            if face is None or not face.is_matched or face.track_id != person.track_id or face.session_id != person.session_id:
                raise AssociationResultError(f"eşleşme karşılıklı değil: kişi {person.track_id} -> yüz {person.face_index}")
            if face.face_bbox != person.face_bbox:
                raise AssociationResultError(f"eşleşen yüz kutusu tutarsız: kişi {person.track_id}, yüz {person.face_index}")
        for face in self.faces:
            if not face.is_matched:
                continue
            person = persons_by_track.get(face.track_id)  # type: ignore[arg-type]
            if person is None or not person.is_matched or person.face_index != face.face_index:
                raise AssociationResultError(f"eşleşme karşılıklı değil: yüz {face.face_index} -> kişi {face.track_id}")

    # ------------------------------------------------------------------ sayılar
    @property
    def person_count(self) -> int:
        return len(self.persons)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def matched_count(self) -> int:
        """Güvenle eşleşen (kişi, yüz) çifti sayısı."""
        return sum(1 for p in self.persons if p.is_matched)

    @property
    def trusted_match_count(self) -> int:
        """Eşleşen VE süreklilik bakımından güvenilir sayılan çift sayısı."""
        return sum(1 for p in self.persons if p.trusted_match)

    @property
    def no_face_count(self) -> int:
        return sum(1 for p in self.persons if p.status == STATUS_NO_FACE)

    @property
    def ambiguous_person_count(self) -> int:
        return sum(1 for p in self.persons if p.status == STATUS_AMBIGUOUS)

    @property
    def ambiguous_face_count(self) -> int:
        return sum(1 for f in self.faces if f.status == STATUS_AMBIGUOUS)

    @property
    def unmatched_face_count(self) -> int:
        return sum(1 for f in self.faces if f.status == STATUS_UNMATCHED_FACE)

    @property
    def possibly_static_count(self) -> int:
        return sum(1 for p in self.persons if p.continuity is not None and p.continuity.possibly_static)

    # ------------------------------------------------------------------ erişim
    def person(self, track_id: int) -> AssociatedPerson | None:
        return next((p for p in self.persons if p.track_id == track_id), None)

    def face(self, face_index: int) -> AssociatedFace | None:
        return next((f for f in self.faces if f.face_index == face_index), None)

    def pairs(self) -> tuple[tuple[int, int], ...]:
        """Eşleşen ``(track_id, face_index)`` çiftleri (``track_id`` artan sırada)."""
        return tuple((p.track_id, p.face_index) for p in self.persons if p.is_matched and p.face_index is not None)

    def __iter__(self) -> Iterator[AssociatedPerson]:
        return iter(self.persons)

    def __len__(self) -> int:
        return len(self.persons)
