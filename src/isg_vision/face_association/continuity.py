"""Track ID süreklilik izleyicisi: ID switch, ID devralma, örtüşme ve oturum değişimine karşı koruma.

Neden gerekli? ``(session_id, track_id)`` aynı fiziksel kişiyi HER ZAMAN temsil etmez (Aşama 4B, ``KL-4B-001/003/009/010``): ByteTrack
görünüm kullanmaz, ID'ler konuma yapışır; örtüşmede ID değişebilir; tampon içinde kaybolan ize başka biri girerse eski ID'yi devralır;
kadrajdan çıkıp dönen kişi yeni ID alabilir. Bu modül yüz eşleşmesini KURMAZ (o ``matching``'in işidir ve geçmiş kullanmaz); yalnızca
her iz için "süreklilik bozuldu mu?" sorusunu izler ve sonraki aşamalara şunu söyler: bu izin ESKİ karelerdeki kanıtını (ör. yüz
eşleşmeleri, ileride yüz embedding'i) bu karedekiyle birleştirme.

Politika (``AssociationConfig``)
--------------------------------
* **Yeni epoch** (``TrackContinuity.epoch`` artar, ``frames_in_epoch`` 1'e döner, eşleşme geçmişi sıfırlanır) şu durumlarda:
  - ``new_track``: iz ilk kez görüldü (yeni Track ID önceki kişinin kimliğini/eşleşmesini DEVRALMAZ);
  - ``session_reset``: takip oturumu değişti/sıfırlandı (``session_id`` değişti veya ``reset_reason`` var): tüm geçmiş silinir;
  - ``jump``: ardışık karelerde aynı ID'nin kutusu ``jump_iou``'nun altında örtüştü (ID devri/anlık yer değiştirme şüphesi);
  - ``gap``: iz ``max_gap_frames``'ten uzun görünmeyip döndü ya da döndüğünde kutusu son bilinen kutuyla ``jump_iou`` altında örtüşüyor
    (kaybolan kişinin yerine giren başkası ID'yi devralmış olabilir);
  - ``overlap_end``: izin kutusu başka bir izle örtüşüyordu ve ``overlap_cooldown_frames`` kare boyunca örtüşmedi (örtüşme sırasında ID
    değişmiş olabilir; örtüşmeden ÖNCEKİ kanıt epoch'u korur çünkü o zaman kimlik doğruydu).
* **Güven** (``trusted``): epoch'ta en az ``min_trusted_frames`` kare + şu anda örtüşme/sakinleşme yok + bu karede epoch yenilenmedi.
  Yeni iz her zaman güvensizle başlar (güvenli varsayılan).
* **Hareketsizlik bayrağı** (``possibly_static``): kutu ``stationary_flag_frames`` kare boyunca ``stationary_iou`` üstünde değişmedi.
  Yalnızca bilgi (poster/yorgan/oturan kişi ayrımı YAPILMAZ).

Bellek sınırlıdır: görünmeyen izlerin durumu ``state_ttl_frames`` kare sonra silinir (sonra dönen aynı Track ID yeni sayılır).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from isg_vision.human_detection.results import BBox

from .config import AssociationConfig
from .geometry import iou, overlap_fraction
from .results import (
    BREAK_GAP,
    BREAK_JUMP,
    BREAK_NEW_TRACK,
    BREAK_OVERLAP_END,
    BREAK_SESSION_RESET,
    UNTRUSTED_NEW_EPOCH,
    UNTRUSTED_OVERLAP,
    UNTRUSTED_OVERLAP_COOLDOWN,
    TrackContinuity,
)


@dataclass
class _TrackState:
    epoch: int
    frames_in_epoch: int
    last_frame: int
    last_box: BBox
    anchor_box: BBox
    stationary_frames: int = 0
    matched_streak: int = 0
    frames_since_match: int | None = None
    overlap_active: bool = False
    quiet_frames: int = 0


@dataclass(frozen=True)
class TrackObservation:
    """Süreklilik izleyicisine verilen bir aktif izin bu karedeki gözlemi."""

    track_id: int
    bbox: BBox


class ContinuityMonitor:
    """Aktif izlerin kareler arası sürekliliğini izler (``update`` her GEÇERLİ karede tam bir kez çağrılır)."""

    def __init__(self, config: AssociationConfig | None = None) -> None:
        self.config = config or AssociationConfig()
        self._session_id: int | None = None
        self._last_frame: int | None = None
        self._states: dict[int, _TrackState] = {}
        self.break_counts: Counter[str] = Counter()

    # ------------------------------------------------------------------ durum
    @property
    def session_id(self) -> int | None:
        return self._session_id

    @property
    def last_frame_index(self) -> int | None:
        return self._last_frame

    @property
    def tracked_count(self) -> int:
        """Durumu bellekte tutulan iz sayısı (görünmeyenler dahil; ``state_ttl_frames`` ile sınırlı)."""
        return len(self._states)

    def reset(self) -> None:
        """Tüm geçmişi siler (yeni kaynak/oturum). Kesinti sayaçları korunur."""
        self._session_id = None
        self._last_frame = None
        self._states.clear()

    def accepts(self, session_id: int, frame_index: int, reset_reason: str | None) -> bool:
        """Kare, durum güncellemesi için geçerli mi? (Aynı oturumda ``frame_index`` kesin artmalıdır.)"""
        if reset_reason is not None or session_id != self._session_id:
            return True  # yeni oturum başlatır
        return self._last_frame is None or frame_index > self._last_frame

    # ------------------------------------------------------------------ güncelleme
    def update(
        self,
        *,
        session_id: int,
        frame_index: int,
        reset_reason: str | None,
        tracks: Sequence[TrackObservation],
        matched_track_ids: Collection[int],
    ) -> tuple[dict[int, TrackContinuity], tuple[tuple[int, str], ...]]:
        """Bu karenin aktif izleri için süreklilik durumunu hesaplar ve belleği günceller.

        Returns:
            ``({track_id: TrackContinuity}, ((track_id, kesinti_nedeni), ...))``.
        """
        cfg = self.config
        session_reset = False
        if reset_reason is not None or (self._session_id is not None and session_id != self._session_id):
            had_state = bool(self._states) or self._last_frame is not None
            self._states.clear()
            session_reset = had_state
        self._session_id = session_id

        overlaps = _overlaps(tracks, cfg.overlap_guard)
        continuity: dict[int, TrackContinuity] = {}
        breaks_out: list[tuple[int, str]] = []
        matched = set(matched_track_ids)

        for obs in sorted(tracks, key=lambda t: t.track_id):
            state = self._states.get(obs.track_id)
            reasons: list[str] = []
            if state is None:
                reasons.append(BREAK_SESSION_RESET if session_reset else BREAK_NEW_TRACK)
                state = _TrackState(epoch=1, frames_in_epoch=1, last_frame=frame_index, last_box=obs.bbox, anchor_box=obs.bbox)
                self._states[obs.track_id] = state
                fresh = True
            else:
                fresh = False
                gap = frame_index - state.last_frame - 1
                if gap > cfg.max_gap_frames or iou(state.last_box, obs.bbox) < cfg.jump_iou:
                    reasons.append(BREAK_GAP if gap > 0 else BREAK_JUMP)

            overlapping_ids, max_fraction = overlaps.get(obs.track_id, ((), 0.0))
            in_overlap = bool(overlapping_ids)
            if in_overlap:
                state.overlap_active = True
                state.quiet_frames = 0
            elif state.overlap_active and not fresh:
                state.quiet_frames += 1
                if state.quiet_frames >= cfg.overlap_cooldown_frames:
                    if not reasons:
                        reasons.append(BREAK_OVERLAP_END)
                    state.overlap_active = False
                    state.quiet_frames = 0
            if reasons and not fresh:
                state.epoch += 1
                state.frames_in_epoch = 1
                state.matched_streak = 0
                state.frames_since_match = None
                if BREAK_OVERLAP_END not in reasons:
                    state.overlap_active = in_overlap
                    state.quiet_frames = 0
            elif not fresh:
                state.frames_in_epoch += 1

            if iou(state.anchor_box, obs.bbox) >= cfg.stationary_iou:
                state.stationary_frames += 1
            else:
                state.anchor_box = obs.bbox
                state.stationary_frames = 0
            state.last_frame, state.last_box = frame_index, obs.bbox

            if obs.track_id in matched:
                state.matched_streak += 1
                state.frames_since_match = 0
            else:
                state.matched_streak = 0
                if state.frames_since_match is not None:
                    state.frames_since_match += 1

            untrusted: list[str] = []
            if reasons or state.frames_in_epoch < cfg.min_trusted_frames:
                untrusted.append(UNTRUSTED_NEW_EPOCH)
            if in_overlap:
                untrusted.append(UNTRUSTED_OVERLAP)
            elif state.overlap_active:
                untrusted.append(UNTRUSTED_OVERLAP_COOLDOWN)
            continuity[obs.track_id] = TrackContinuity(
                epoch=state.epoch,
                frames_in_epoch=state.frames_in_epoch,
                trusted=not untrusted,
                untrusted_reasons=tuple(untrusted),
                breaks=tuple(reasons),
                overlap_fraction=max_fraction,
                overlapping_track_ids=overlapping_ids,
                matched_streak=state.matched_streak,
                frames_since_match=state.frames_since_match,
                stationary_frames=state.stationary_frames,
                possibly_static=state.stationary_frames >= cfg.stationary_flag_frames,
            )
            for reason in reasons:
                breaks_out.append((obs.track_id, reason))
                self.break_counts[reason] += 1

        stale = [tid for tid, st in self._states.items() if frame_index - st.last_frame > cfg.state_ttl_frames]
        for tid in stale:
            del self._states[tid]
        self._last_frame = frame_index
        return continuity, tuple(breaks_out)


def _overlaps(tracks: Sequence[TrackObservation], guard: float) -> dict[int, tuple[tuple[int, ...], float]]:
    """Her iz için: eşiği aşan örtüşen diğer izlerin ``track_id``'leri ve en büyük örtüşme oranı."""
    ids: dict[int, list[int]] = {t.track_id: [] for t in tracks}
    best: dict[int, float] = {t.track_id: 0.0 for t in tracks}
    for i, a in enumerate(tracks):
        for b in tracks[i + 1 :]:
            fraction = overlap_fraction(a.bbox, b.bbox)
            best[a.track_id] = max(best[a.track_id], fraction)
            best[b.track_id] = max(best[b.track_id], fraction)
            if fraction >= guard:
                ids[a.track_id].append(b.track_id)
                ids[b.track_id].append(a.track_id)
    return {tid: (tuple(sorted(others)), best[tid]) for tid, others in ids.items()}
