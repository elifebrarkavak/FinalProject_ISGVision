"""Sentetik sahnelerde (``scenes``) ilişkilendirme DOĞRULUĞUNU gerçek (ground-truth) kutulara göre ölçer.

**Kanıt sınırı:** sahneler montajdır (``scenes`` modül açıklaması); buradaki sayılar gerçek kamerada/gerçek çoklu-kişi videosunda doğruluk DEĞİLDİR.
Ayrıca gerçek takip metrikleri (IDF1/HOTA/MOTA) bu projede ölçülmemiştir; bu modül yalnızca bilinen kimlikli sahnede "yüz doğru kişiye bağlandı mı" ve
"ID değişti mi" sorularını sayar (``tracking.metrics.count_id_switches`` ile).

Eşleme kuralları (ground-truth ↔ sonuç)
---------------------------------------
* Bir izin sahibi, kutusu ile gerçek kişi kutusu arasındaki IoU'su en yüksek (``>= OWNER_IOU``) kişidir; yoksa izin sahibi yoktur (``None``; ör. yanlış pozitif iz).
* Bir yüzün sahibi, gerçek (görünür) yüz kutusu ile IoU'su en yüksek (``>= OWNER_IOU``) kişidir; yoksa merkezi bir fotoğraf/poster kırpıntısının içindeyse
  ``decoy:<ad>``; aksi halde ``None`` (sahte tespit).
* Eşleşen (iz, yüz) çifti DOĞRU: iz sahibi = yüz sahibi = aynı gerçek kişi. YANLIŞ (``wrong_pairs``): gerçek bir yüz, sahibi olmayan veya başka bir kişiye
  ait ize bağlanmışsa (ya da yüzün sahibi yoksa/izin sahibi başka bir kişiyse).
* SAHTE-YÜZ çiftleri (fotoğraf/poster kırpıntısındaki yüz) ayrı sayılır ve YANLIŞ SAYILMAZ (geometri bunu ayıramaz; tasarım sınırı): ``decoy_pairs`` = sahte yüz,
  sahibi olmayan veya sahte bir izle eşleşti (ör. poster kişisi + poster yüzü); ``decoy_on_person_pairs`` = sahte yüz GERÇEK bir kişinin izine bağlandı (tehlikeli durum).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from isg_vision.tracking.metrics import IdSwitchReport, count_id_switches

from .geometry import iou
from .results import STATUS_AMBIGUOUS, STATUS_NO_FACE, FaceAssociationResult
from .scenes import Box, SceneFrame

OWNER_IOU = 0.3
MIN_VISIBLE_FOR_EXPECTED = 0.5


@dataclass(frozen=True)
class SceneEvaluation:
    """Bir senaryonun ilişkilendirme değerlendirmesi (sentetik montaj; gerçek doğruluk değildir).

    Attributes:
        frames / invalid_frames: Değerlendirilen kare; girdisi tutarsız olduğu için eşleştirme yapılmayan kare.
        correct_pairs: İz sahibi = yüz sahibi olan eşleşmeler. wrong_pairs: Gerçek yüzün yanlış ize bağlandığı eşleşmeler (``wrong_examples``'ta ilk örnekler).
        decoy_pairs: Sahte yüz (fotoğraf/poster) ile sahte/sahipsiz bir izin eşleşmesi. decoy_on_person_pairs: Sahte yüzün GERÇEK bir kişinin izine bağlandığı eşleşmeler.
        untrusted_decoy_on_person_pairs: Bunlardan, eşleşme anında iz ``trusted=False`` olanlar.
        expected: Görünür gerçek yüzü olan ve bir izin sahibi olduğu (kişi, kare) sayısı. recalled: Bunlardan doğru eşleşenler. face_not_detected: YuNet'in yüzü
            bulamadığı. ambiguous: Eşleştirilmeyen "belirsiz" olanlar. missed: Yüz tespit edildi ama eşleşmedi (no_face/unmatched).
        track_ids_per_actor: Her gerçek kişi için görülen Track ID'ler (ID switch göstergesi). id_report: ``count_id_switches`` sonucu.
        breaks: Süreklilik kesintisi sayıları (``BREAK_*``).
        untrusted_wrong_pairs: ``wrong_pairs`` içinde, eşleşme sırasında iz ``trusted=False`` olanlar (süreklilik uyarısı yanlışı önceden işaret etti mi).
    """

    frames: int
    invalid_frames: int
    correct_pairs: int
    wrong_pairs: int
    decoy_pairs: int
    expected: int
    recalled: int
    face_not_detected: int
    ambiguous: int
    missed: int
    track_ids_per_actor: dict[str, tuple[int, ...]]
    id_report: IdSwitchReport
    breaks: dict[str, int]
    untrusted_wrong_pairs: int
    decoy_on_person_pairs: int = 0
    untrusted_decoy_on_person_pairs: int = 0
    wrong_examples: tuple[str, ...] = field(default_factory=tuple)

    @property
    def recall(self) -> float | None:
        return self.recalled / self.expected if self.expected else None

    @property
    def wrong_rate(self) -> float | None:
        """Yanlış eşleşme oranı (yanlış / tüm eşleşmeler; sahte-yüz çiftleri hariç)."""
        total = self.correct_pairs + self.wrong_pairs
        return self.wrong_pairs / total if total else None


def _owner(box: Box, boxes: dict[str, Box]) -> str | None:
    best_name, best = None, OWNER_IOU
    for name, other in boxes.items():
        value = iou(box, other)
        if value >= best:
            best_name, best = name, value
    return best_name


def evaluate_scene(frames: Sequence[SceneFrame], results: Sequence[FaceAssociationResult]) -> SceneEvaluation:
    """Her karede ilişkilendirme sonucunu gerçek kutularla karşılaştırır (``len(frames) == len(results)`` olmalıdır)."""
    if len(frames) != len(results):
        raise ValueError(f"kare sayıları uyuşmuyor: {len(frames)} sahne karesi, {len(results)} sonuç")
    correct = wrong = decoy = expected = recalled = not_detected = ambiguous = missed = untrusted_wrong = invalid = decoy_on_person = untrusted_decoy = 0
    examples: list[str] = []
    assignments: list[dict[str, int]] = []
    ids_seen: dict[str, list[int]] = {}
    breaks: dict[str, int] = {}

    for index, (scene, result) in enumerate(zip(frames, results)):
        for _, reason in result.breaks:
            breaks[reason] = breaks.get(reason, 0) + 1
        if not result.input_valid:
            invalid += 1
            assignments.append({})
            continue
        person_boxes = {name: t.person_box for name, t in scene.truth.items()}
        person_boxes.update({f"decoy:{name}": box for name, box in scene.decoys.items()})
        face_boxes = {name: t.face_box for name, t in scene.truth.items() if t.face_box is not None}

        track_owner: dict[int, str | None] = {p.track_id: _owner(p.person_bbox, person_boxes) for p in result.persons}
        # kişi başına tek iz (en yüksek IoU) -> ID switch için
        frame_assignment: dict[str, int] = {}
        best_iou: dict[str, float] = {}
        for person in result.persons:
            owner = track_owner[person.track_id]
            if owner is None or owner.startswith("decoy:"):
                continue
            value = iou(person.person_bbox, person_boxes[owner])
            if value > best_iou.get(owner, 0.0):
                best_iou[owner], frame_assignment[owner] = value, person.track_id
        assignments.append(frame_assignment)
        for actor, track_id in frame_assignment.items():
            seen = ids_seen.setdefault(actor, [])
            if track_id not in seen:
                seen.append(track_id)

        for person in result.persons:
            if not person.is_matched or person.face_bbox is None:
                continue
            t_owner = track_owner[person.track_id]
            f_owner = _owner(person.face_bbox, {n: b for n, b in face_boxes.items()})
            if f_owner is None:
                cx, cy = (person.face_bbox[0] + person.face_bbox[2]) / 2, (person.face_bbox[1] + person.face_bbox[3]) / 2
                for name, box in scene.decoys.items():
                    if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
                        f_owner = f"decoy:{name}"
                        break
            if f_owner is not None and f_owner.startswith("decoy:"):
                if t_owner is not None and not t_owner.startswith("decoy:"):
                    decoy_on_person += 1  # sahte yüz GERÇEK bir kişinin izine bağlandı
                    untrusted_decoy += 0 if person.trusted_match else 1
                    examples.append(f"kare {index}: iz {person.track_id} ({t_owner}) <- sahte yüz ({f_owner})")
                else:
                    decoy += 1
            elif t_owner is not None and t_owner == f_owner:
                correct += 1
            else:
                wrong += 1
                examples.append(f"kare {index}: iz {person.track_id} (sahip {t_owner}) <- yüz sahibi {f_owner}")
                untrusted_wrong += 0 if person.trusted_match else 1

        detected_faces = [f.face_bbox for f in result.faces]
        for actor, truth in scene.truth.items():
            if truth.face_box is None or truth.visible_fraction < MIN_VISIBLE_FOR_EXPECTED:
                continue
            tracks = [p for p in result.persons if track_owner[p.track_id] == actor]
            if not tracks:
                continue
            expected += 1
            person = max(tracks, key=lambda p: iou(p.person_bbox, truth.person_box))
            if person.is_matched and person.face_bbox is not None and iou(person.face_bbox, truth.face_box) >= OWNER_IOU:
                recalled += 1
            elif not any(iou(box, truth.face_box) >= OWNER_IOU for box in detected_faces):
                not_detected += 1
            elif person.status == STATUS_AMBIGUOUS:
                ambiguous += 1
            elif person.status == STATUS_NO_FACE or not person.is_matched:
                missed += 1

    return SceneEvaluation(
        frames=len(frames), invalid_frames=invalid, correct_pairs=correct, wrong_pairs=wrong, decoy_pairs=decoy, expected=expected, recalled=recalled,
        face_not_detected=not_detected, ambiguous=ambiguous, missed=missed,
        track_ids_per_actor={k: tuple(v) for k, v in ids_seen.items()}, id_report=count_id_switches(assignments), breaks=breaks,
        untrusted_wrong_pairs=untrusted_wrong, decoy_on_person_pairs=decoy_on_person, untrusted_decoy_on_person_pairs=untrusted_decoy,
        wrong_examples=tuple(examples[:10]),
    )
