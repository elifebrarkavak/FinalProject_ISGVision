"""Tek bir karede yüzleri kişilere (aktif izlere) bağlayan SAF, durumsuz karar mantığı.

Girdi: aynı karenin kişi kutuları ve yüz kutuları. Çıktı: her kişi ve her yüz için durum + neden. Bu modül geçmiş KULLANMAZ; bu yüzden
önceki karelerden hiçbir eşleşme taşınamaz (ID switch/devralma güvenliğinin temeli; süreklilik ayrıca ``continuity``'de izlenir).

Algoritma ("karşılıklı en iyi + belirsizlik payı"; ``AssociationConfig`` eşikleri)
------------------------------------------------------------------------------------
1. Her (yüz ``f``, kişi ``p``) çifti için ``geometry.score_pair`` hesaplanır. Sert elemelere (içerme, dikey konum, boyut) takılan çift
   **aday değildir**.
2. Bir çift ``(f, p)`` ancak ŞUNLARIN HEPSİ sağlanırsa kabul edilir:
   a. ``p``, ``f`` için en yüksek skorlu adaydır VE ikinci en iyi kişi adayı varsa fark ``>= ambiguity_margin``;
   b. ``f``, ``p`` için en yüksek skorlu adaydır VE ikinci en iyi yüz adayı varsa fark ``>= ambiguity_margin``;
   c. skor ``>= min_score``.
3. (a) ve (b) birlikte tek-birebirliği GARANTİ eder: bir yüzün en iyi kişisi tektir ve bir kişinin en iyi yüzü tektir. Eşit skorlar
   fark 0 verir, yani HER ZAMAN belirsizdir (sıralama sırasına göre sessizce biri seçilmez).
4. Kabul edilmeyenler: kalan adayı (başka bir eşleşmeyle alınmamış aday) varsa ``ambiguous`` (tahmini eşleşme YAPILMAZ), yoksa kişi için
   ``no_face`` / yüz için ``unmatched_face``.

Bilinçli olarak YAPILMAYANLAR: "eleme yoluyla" atama (yüz ``f``'nin iki adayından biri başka yüzle güvenle eşleştiği için ``f``'yi
diğerine atamak bir TAHMİNDİR ve örtüşmede yanlış kişiye yol açabilir) ve çok kareli oylama.
"""

from __future__ import annotations

from dataclasses import dataclass

from isg_vision.human_detection.results import BBox

from .config import AssociationConfig
from .geometry import PairScore, score_pair
from .results import (
    REASON_CANDIDATES_TAKEN,
    REASON_CONTESTED,
    REASON_GEOMETRY_OK,
    REASON_LOW_SCORE,
    REASON_NO_CANDIDATE,
    STATUS_AMBIGUOUS,
    STATUS_MATCHED,
    STATUS_NO_FACE,
    STATUS_UNMATCHED_FACE,
)

_EPS = 1e-9


@dataclass(frozen=True)
class PersonBox:
    track_id: int
    bbox: BBox


@dataclass(frozen=True)
class FaceBox:
    index: int
    bbox: BBox
    at_border: bool = False


@dataclass(frozen=True)
class PersonOutcome:
    track_id: int
    status: str
    reason: str
    face_index: int | None = None
    score: float | None = None
    margin: float | None = None
    candidates: tuple[int, ...] = ()  # aday yüz sıraları, en iyiden başlayarak


@dataclass(frozen=True)
class FaceOutcome:
    face_index: int
    status: str
    reason: str
    track_id: int | None = None
    score: float | None = None
    candidates: tuple[int, ...] = ()  # aday track_id'ler, en iyiden başlayarak


@dataclass(frozen=True)
class MatchOutcome:
    persons: dict[int, PersonOutcome]
    faces: dict[int, FaceOutcome]
    pair_scores: dict[tuple[int, int], PairScore]  # (face_index, track_id) -> skor (yalnızca adaylar)


def match_faces_to_persons(persons: list[PersonBox], faces: list[FaceBox], config: AssociationConfig) -> MatchOutcome:
    """Bir karenin yüzlerini kişilere bağlar (modül açıklamasındaki algoritma); girdi listeleri değiştirilmez."""
    pair_scores: dict[tuple[int, int], PairScore] = {}
    for face in faces:
        for person in persons:
            scored = score_pair(face.bbox, person.bbox, config, at_border=face.at_border)
            if scored is not None:
                pair_scores[(face.index, person.track_id)] = scored

    # aday listeleri: (skor azalan, ardından id artan) — eşit skor fark 0 verir ve belirsizdir
    face_cands: dict[int, list[tuple[float, int]]] = {f.index: [] for f in faces}
    person_cands: dict[int, list[tuple[float, int]]] = {p.track_id: [] for p in persons}
    for (face_index, track_id), pair in pair_scores.items():
        face_cands[face_index].append((pair.score, track_id))
        person_cands[track_id].append((pair.score, face_index))
    for cands in face_cands.values():
        cands.sort(key=lambda item: (-item[0], item[1]))
    for cands in person_cands.values():
        cands.sort(key=lambda item: (-item[0], item[1]))

    margin = config.ambiguity_margin
    accepted: dict[int, int] = {}  # face_index -> track_id
    accepted_margin: dict[int, float | None] = {}
    rejection: dict[int, str] = {}  # face_index -> neden (aday vardı, kabul edilmedi)
    for face in faces:
        cands = face_cands[face.index]
        if not cands:
            continue
        best_score, track_id = cands[0]
        face_gap = best_score - cands[1][0] if len(cands) > 1 else None
        if face_gap is not None and face_gap < margin - _EPS:
            rejection[face.index] = REASON_CONTESTED
            continue
        person_list = person_cands[track_id]
        person_gap = person_list[0][0] - person_list[1][0] if len(person_list) > 1 else None
        if person_list[0][1] != face.index or (person_gap is not None and person_gap < margin - _EPS):
            rejection[face.index] = REASON_CONTESTED
            continue
        if best_score < config.min_score - _EPS:
            rejection[face.index] = REASON_LOW_SCORE
            continue
        accepted[face.index] = track_id
        gaps = [g for g in (face_gap, person_gap) if g is not None]
        accepted_margin[face.index] = min(gaps) if gaps else None

    taken_persons = set(accepted.values())
    face_outcomes: dict[int, FaceOutcome] = {}
    for face in faces:
        cands = tuple(t for _, t in face_cands[face.index])
        if face.index in accepted:
            track_id = accepted[face.index]
            face_outcomes[face.index] = FaceOutcome(
                face.index, STATUS_MATCHED, REASON_GEOMETRY_OK, track_id, pair_scores[(face.index, track_id)].score, cands
            )
        elif not cands:
            face_outcomes[face.index] = FaceOutcome(face.index, STATUS_UNMATCHED_FACE, REASON_NO_CANDIDATE, None, None, cands)
        elif all(t in taken_persons for t in cands):
            face_outcomes[face.index] = FaceOutcome(face.index, STATUS_UNMATCHED_FACE, REASON_CANDIDATES_TAKEN, None, None, cands)
        else:
            face_outcomes[face.index] = FaceOutcome(
                face.index, STATUS_AMBIGUOUS, rejection.get(face.index, REASON_CONTESTED), None, None, cands
            )

    face_to_person = {f: t for f, t in accepted.items()}
    person_to_face = {t: f for f, t in face_to_person.items()}
    taken_faces = set(accepted)
    person_outcomes: dict[int, PersonOutcome] = {}
    for person in persons:
        cands = tuple(f for _, f in person_cands[person.track_id])
        if person.track_id in person_to_face:
            face_index = person_to_face[person.track_id]
            person_outcomes[person.track_id] = PersonOutcome(
                person.track_id, STATUS_MATCHED, REASON_GEOMETRY_OK, face_index,
                pair_scores[(face_index, person.track_id)].score, accepted_margin[face_index], cands,
            )
            continue
        remaining = [f for f in cands if f not in taken_faces]
        if not cands:
            person_outcomes[person.track_id] = PersonOutcome(person.track_id, STATUS_NO_FACE, REASON_NO_CANDIDATE, candidates=cands)
        elif not remaining:
            person_outcomes[person.track_id] = PersonOutcome(person.track_id, STATUS_NO_FACE, REASON_CANDIDATES_TAKEN, candidates=cands)
        else:
            reason = rejection.get(remaining[0], REASON_CONTESTED)
            person_outcomes[person.track_id] = PersonOutcome(person.track_id, STATUS_AMBIGUOUS, reason, candidates=cands)
    return MatchOutcome(person_outcomes, face_outcomes, pair_scores)
