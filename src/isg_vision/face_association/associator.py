"""``TrackingResult`` + ``FaceDetectionResult`` -> ``FaceAssociationResult`` (aynı karenin yüz–kişi ilişkilendirmesi).

``FacePersonAssociator`` YOLO'yu, ByteTrack'i veya YuNet'i ÇALIŞTIRMAZ; yalnızca aynı karenin iki hazır sonucunu birleştirir. Sorumluluklar ayrıdır:

* ``matching`` — tek karenin geometrik eşleştirmesi (saf, durumsuz; geçmiş KULLANMAZ).
* ``continuity`` — iz bazında ID switch/devralma/örtüşme/oturum değişimi izleme (eşleşme KURMAZ; yalnızca güven ve epoch üretir).
* bu modül — girdi doğrulama, iki parçayı birleştirme, sonuç oluşturma, sayaçlar.

Girdi tutarlılığı
-----------------
Yüz ve takip sonuçları AYNI kareye ait olmalıdır: ``frame_index`` eşit, ``image_size`` eşit, (ikisinde de varsa) ``timestamp`` eşit; kutular
görüntü içinde; Track ID'ler ve yüz sıraları benzersiz; kare sırası (aynı oturumda) artan. Bunlardan biri bozuksa **hiçbir eşleştirme
yapılmaz**: ``input_valid=False`` ve tüm kişi/yüzler ``invalid_input`` durumlu bir sonuç döner (süreklilik durumu güncellenmez). Yapısal olarak
geçersiz nesne (``None``/yanlış tür/eksik öznitelik) ise ``AssociationInputError`` fırlatır.

Oturum
------
``TrackingResult.session_id`` değişirse veya ``reset_reason`` doluysa süreklilik geçmişi silinir (``continuity``); eski oturumun hiçbir izi
yeni oturuma taşınmaz. ``reset()`` yeni kaynak/oturum başlarken çağrılır (``run_face_association`` bunu kendiliğinden yapar).
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .config import AssociationConfig
from .continuity import ContinuityMonitor, TrackObservation
from .errors import AssociationInputError
from .matching import FaceBox, PersonBox, match_faces_to_persons
from .results import (
    REASON_INPUT_INCONSISTENT,
    STATUS_INVALID,
    AssociatedFace,
    AssociatedPerson,
    FaceAssociationResult,
)

BOX_TOLERANCE_PX = 1e-3

_TRACKING_ATTRS = ("tracks", "image_size", "frame_index", "timestamp", "session_id")
_FACES_ATTRS = ("faces", "image_size", "frame_index", "timestamp")
_TRACK_ATTRS = ("track_id", "session_id", "bbox_xyxy", "confidence")
_FACE_ATTRS = ("bbox_xyxy", "confidence")


@dataclass(frozen=True)
class AssociatorStats:
    """``FacePersonAssociator``'ın başlangıçtan (veya ``reset_stats``'tan) bu yana sayaçları.

    Attributes:
        frames: İşlenen kare (geçersiz girdili kareler dahil).
        invalid_frames: Girdisi tutarsız olduğu için HİÇ eşleştirme yapılmayan kare sayısı.
        person_total / face_total: Tüm karelerdeki kişi ve yüz sayısı toplamı (benzersiz kişi sayısı DEĞİL).
        matched / trusted_matched: Eşleşen (kişi, yüz) çiftleri; ikincisi süreklilik bakımından da güvenilir olanlar.
        ambiguous_persons / ambiguous_faces / no_face / unmatched_faces: Durum sayıları (toplam).
        breaks: Süreklilik kesintisi sayıları (``BREAK_*`` -> sayı).
    """

    frames: int
    invalid_frames: int
    person_total: int
    face_total: int
    matched: int
    trusted_matched: int
    ambiguous_persons: int
    ambiguous_faces: int
    no_face: int
    unmatched_faces: int
    breaks: dict[str, int]


class FacePersonAssociator:
    """Her karede yüzleri aktif izlere (geçici Track ID) bağlar; belirsizde eşleştirmez, süreklilik bozulunca geçmişi geçersiz kılar.

    Args:
        config: Doğrulanmış yapılandırma (``None`` = varsayılanlar).
        clock: Süre ölçümü için sayaç (varsayılan ``time.perf_counter``).
    """

    def __init__(self, config: AssociationConfig | None = None, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self.config = config or AssociationConfig()
        self._clock = clock
        self._monitor = ContinuityMonitor(self.config)
        self._counters: Counter[str] = Counter()

    # ------------------------------------------------------------------ durum
    @property
    def stats(self) -> AssociatorStats:
        c = self._counters
        return AssociatorStats(
            frames=c["frames"], invalid_frames=c["invalid_frames"], person_total=c["person_total"], face_total=c["face_total"],
            matched=c["matched"], trusted_matched=c["trusted_matched"], ambiguous_persons=c["ambiguous_persons"],
            ambiguous_faces=c["ambiguous_faces"], no_face=c["no_face"], unmatched_faces=c["unmatched_faces"],
            breaks=dict(self._monitor.break_counts),
        )

    def reset(self) -> None:
        """Süreklilik geçmişini siler (yeni kaynak/oturum). Sayaçlar korunur."""
        self._monitor.reset()

    # ------------------------------------------------------------------ ilişkilendirme
    def associate(self, tracking: Any, faces: Any) -> FaceAssociationResult:
        """Aynı karenin takip ve yüz sonuçlarını ilişkilendirir (modül açıklaması).

        Args:
            tracking: ``PersonTracker.update`` çıktısı (``TrackingResult``).
            faces: Aynı karenin ``FaceDetector.detect`` çıktısı (``FaceDetectionResult``).

        Raises:
            AssociationInputError: Nesneler yapısal olarak geçersiz (``None``, yanlış tür, eksik öznitelik).
        """
        start = self._clock()
        tracks, face_list = _structure(tracking, faces)
        problem = self._consistency_problem(tracking, faces, tracks, face_list)
        if problem is not None:
            result = self._invalid_result(tracking, tracks, face_list, problem, (self._clock() - start) * 1000.0)
            self._count(result)
            return result

        persons_in = [PersonBox(t.track_id, tuple(t.bbox_xyxy)) for t in tracks]  # type: ignore[arg-type]
        faces_in = [FaceBox(i, tuple(f.bbox_xyxy), bool(getattr(f, "at_border", False))) for i, f in enumerate(face_list)]  # type: ignore[arg-type]
        outcome = match_faces_to_persons(persons_in, faces_in, self.config)
        matched_ids = {tid for tid, o in outcome.persons.items() if o.face_index is not None}
        continuity, breaks = self._monitor.update(
            session_id=tracking.session_id,
            frame_index=tracking.frame_index,
            reset_reason=getattr(tracking, "reset_reason", None),
            tracks=[TrackObservation(t.track_id, tuple(t.bbox_xyxy)) for t in tracks],  # type: ignore[arg-type]
            matched_track_ids=matched_ids,
        )

        frame_index, timestamp = tracking.frame_index, tracking.timestamp
        persons: list[AssociatedPerson] = []
        for track in sorted(tracks, key=lambda t: t.track_id):
            o = outcome.persons[track.track_id]
            face = face_list[o.face_index] if o.face_index is not None else None
            persons.append(
                AssociatedPerson(
                    session_id=track.session_id, track_id=track.track_id, frame_index=frame_index, timestamp=timestamp,
                    person_bbox=tuple(track.bbox_xyxy), person_confidence=float(track.confidence),  # type: ignore[arg-type]
                    status=o.status, reason=o.reason,
                    face_index=o.face_index, face_bbox=tuple(face.bbox_xyxy) if face is not None else None,  # type: ignore[arg-type]
                    face_confidence=float(face.confidence) if face is not None else None,
                    face_at_border=bool(getattr(face, "at_border", False)) if face is not None else False,
                    score=o.score, margin=o.margin, candidate_face_indices=o.candidates, continuity=continuity[track.track_id],
                )
            )
        session_of = {t.track_id: t.session_id for t in tracks}
        face_results = tuple(
            AssociatedFace(
                face_index=i, frame_index=frame_index, timestamp=timestamp, face_bbox=tuple(f.bbox_xyxy),  # type: ignore[arg-type]
                face_confidence=float(f.confidence), at_border=bool(getattr(f, "at_border", False)),
                status=outcome.faces[i].status, reason=outcome.faces[i].reason,
                session_id=session_of[outcome.faces[i].track_id] if outcome.faces[i].track_id is not None else None,  # type: ignore[index]
                track_id=outcome.faces[i].track_id, score=outcome.faces[i].score,
                candidate_track_ids=outcome.faces[i].candidates,
            )
            for i, f in enumerate(face_list)
        )
        result = FaceAssociationResult(
            persons=tuple(persons), faces=face_results, image_size=tuple(tracking.image_size), frame_index=frame_index,  # type: ignore[arg-type]
            timestamp=timestamp, session_id=tracking.session_id, association_ms=max(0.0, (self._clock() - start) * 1000.0), breaks=breaks,
        )
        self._count(result)
        return result

    # ------------------------------------------------------------------ iç işlevler
    def _consistency_problem(self, tracking: Any, faces: Any, tracks: tuple[Any, ...], face_list: tuple[Any, ...]) -> str | None:
        """Aynı kareye ait olmayan/tutarsız girdi için açıklama; tutarlıysa ``None``."""
        t_size, f_size = tuple(tracking.image_size), tuple(faces.image_size)
        if len(t_size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in t_size):
            return f"takip image_size geçersiz: {tracking.image_size!r}"
        if t_size != f_size:
            return f"image_size uyuşmuyor: takip {t_size[0]}x{t_size[1]}, yüz {f_size[0]}x{f_size[1]}"
        if faces.frame_index is None:
            if self.config.require_frame_index:
                return "yüz sonucunda frame_index yok (aynı kare doğrulanamaz)"
        elif faces.frame_index != tracking.frame_index:
            return f"frame_index uyuşmuyor: takip {tracking.frame_index}, yüz {faces.frame_index}"
        if tracking.timestamp is not None and faces.timestamp is not None and tracking.timestamp != faces.timestamp:
            return f"timestamp uyuşmuyor: takip {tracking.timestamp}, yüz {faces.timestamp}"
        if not self._monitor.accepts(tracking.session_id, tracking.frame_index, getattr(tracking, "reset_reason", None)):
            return (f"kare sırası bozuk: frame_index {tracking.frame_index}, aynı oturumda (oturum {tracking.session_id}) "
                    f"son işlenen {self._monitor.last_frame_index}")
        width, height = t_size
        track_ids = [t.track_id for t in tracks]
        if len(set(track_ids)) != len(track_ids):
            return f"aynı karede yinelenen track_id: {track_ids}"
        for track in tracks:
            if track.session_id != tracking.session_id:
                return f"iz {track.track_id} başka bir oturuma ait (iz {track.session_id}, sonuç {tracking.session_id})"
            if not _inside(track.bbox_xyxy, width, height):
                return f"iz {track.track_id} kutusu görüntü dışında/geçersiz: {tuple(track.bbox_xyxy)}"
        for position, face in enumerate(face_list):
            face_index = getattr(face, "face_index", None)
            if face_index is not None and face_index != position:
                return f"yüz sırası tutarsız: konum {position}, face_index {face_index}"
            if getattr(face, "frame_index", None) is not None and face.frame_index != tracking.frame_index:
                return f"yüz {position} başka bir kareye ait (frame_index {face.frame_index}, takip {tracking.frame_index})"
            if not _inside(face.bbox_xyxy, width, height):
                return f"yüz {position} kutusu görüntü dışında/geçersiz: {tuple(face.bbox_xyxy)}"
        return None

    def _invalid_result(
        self, tracking: Any, tracks: tuple[Any, ...], face_list: tuple[Any, ...], problem: str, elapsed_ms: float
    ) -> FaceAssociationResult:
        frame_index = tracking.frame_index if isinstance(tracking.frame_index, int) and tracking.frame_index >= 0 else 0
        persons = []
        for track in sorted(tracks, key=lambda t: t.track_id):
            try:
                persons.append(
                    AssociatedPerson(
                        session_id=track.session_id, track_id=track.track_id, frame_index=frame_index, timestamp=tracking.timestamp,
                        person_bbox=tuple(track.bbox_xyxy), person_confidence=float(track.confidence),  # type: ignore[arg-type]
                        status=STATUS_INVALID, reason=REASON_INPUT_INCONSISTENT,
                    )
                )
            except ValueError:
                continue  # geçersiz kutulu iz: sonuçta gösterilemez (yine de hiçbir eşleşme yok)
        faces = []
        for position, face in enumerate(face_list):
            try:
                faces.append(
                    AssociatedFace(
                        face_index=position, frame_index=frame_index, timestamp=tracking.timestamp, face_bbox=tuple(face.bbox_xyxy),  # type: ignore[arg-type]
                        face_confidence=float(face.confidence), at_border=bool(getattr(face, "at_border", False)),
                        status=STATUS_INVALID, reason=REASON_INPUT_INCONSISTENT,
                    )
                )
            except ValueError:
                continue
        return FaceAssociationResult(
            persons=tuple(persons), faces=tuple(faces), image_size=tuple(tracking.image_size), frame_index=tracking.frame_index,  # type: ignore[arg-type]
            timestamp=tracking.timestamp, session_id=tracking.session_id, association_ms=max(0.0, elapsed_ms),
            input_valid=False, invalid_reason=problem,
        )

    def _count(self, result: FaceAssociationResult) -> None:
        c = self._counters
        c["frames"] += 1
        if not result.input_valid:
            c["invalid_frames"] += 1
            return
        c["person_total"] += result.person_count
        c["face_total"] += result.face_count
        c["matched"] += result.matched_count
        c["trusted_matched"] += result.trusted_match_count
        c["ambiguous_persons"] += result.ambiguous_person_count
        c["ambiguous_faces"] += result.ambiguous_face_count
        c["no_face"] += result.no_face_count
        c["unmatched_faces"] += result.unmatched_face_count


def _inside(box: Any, width: int, height: int) -> bool:
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return False
    finite = all(v == v and abs(v) != float("inf") for v in (x1, y1, x2, y2))
    return finite and x2 > x1 and y2 > y1 and x1 >= -BOX_TOLERANCE_PX and y1 >= -BOX_TOLERANCE_PX and x2 <= width + BOX_TOLERANCE_PX and y2 <= height + BOX_TOLERANCE_PX


def _structure(tracking: Any, faces: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Yapısal doğrulama (öznitelik tabanlı: modül yeniden yüklense de geçerli nesne reddedilmez); ``(izler, yüzler)`` döndürür."""
    if tracking is None or faces is None:
        raise AssociationInputError("tracking ve faces None olamaz.")
    for name, obj, attrs in (("tracking", tracking, _TRACKING_ATTRS), ("faces", faces, _FACES_ATTRS)):
        missing = [a for a in attrs if not hasattr(obj, a)]
        if missing:
            raise AssociationInputError(f"{name} için {attrs} öznitelikleri gerekir: {type(obj).__name__} (eksik: {missing})")
    try:
        tracks = tuple(tracking.tracks)
        face_list = tuple(faces.faces)
    except TypeError:
        raise AssociationInputError("tracking.tracks ve faces.faces yinelenebilir olmalıdır.") from None
    for index, track in enumerate(tracks):
        missing = [a for a in _TRACK_ATTRS if not hasattr(track, a)]
        if missing:
            raise AssociationInputError(f"tracking.tracks[{index}] bir TrackedPerson olmalıdır (eksik: {missing}): {type(track).__name__}")
    for index, face in enumerate(face_list):
        missing = [a for a in _FACE_ATTRS if not hasattr(face, a)]
        if missing:
            raise AssociationInputError(f"faces.faces[{index}] bir FaceDetection olmalıdır (eksik: {missing}): {type(face).__name__}")
    if isinstance(tracking.session_id, bool) or not isinstance(tracking.session_id, int) or tracking.session_id < 1:
        raise AssociationInputError(f"tracking.session_id 1 veya daha büyük bir tam sayı olmalıdır: {tracking.session_id!r}")
    if isinstance(tracking.frame_index, bool) or not isinstance(tracking.frame_index, int) or tracking.frame_index < 0:
        raise AssociationInputError(f"tracking.frame_index negatif olmayan bir tam sayı olmalıdır: {tracking.frame_index!r}")
    return tracks, face_list


__all__ = ["AssociatorStats", "FacePersonAssociator"]
