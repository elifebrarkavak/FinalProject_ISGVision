"""``DetectionResult`` -> Ultralytics ByteTrack -> ``TrackingResult`` adaptörü (YOLO'yu İKİNCİ kez ÇALIŞTIRMAZ).

Mimari karar (kurulu ``ultralytics==8.4.210`` kaynak kodundan doğrulandı; tahmin değildir)
---------------------------------------------------------------------------------------
``YOLO.track()`` bir ``predict`` çağrısının içine takipçiyi geri çağrımlarla (callback) bağlar; yani onu kullanmak her kareyi
YOLO'ya İKİNCİ kez göndermeyi gerektirirdi (4A'nın ``PersonDetector.detect`` zaten çıkarım yapıyor). Bunun yerine takipçi
sınıfı doğrudan kullanılır::

    from ultralytics.trackers.byte_tracker import BYTETracker
    tracker = BYTETracker(args)            # args: track_high_thresh, track_low_thresh, new_track_thresh, track_buffer, match_thresh, fuse_score
    rows = tracker.update(boxes)           # boxes: .conf, .cls, .xywh veren, boolean indekslenebilen NumPy tabanlı nesne
    # rows: (N, 8) float32 = [x1, y1, x2, y2, track_id, score, cls, idx]   (idx = boxes içindeki tespit sırası)

``boxes`` olarak ``ultralytics.engine.results.Boxes`` (NumPy dizisinden ``[x1,y1,x2,y2,conf,cls]``) kullanılır; bu, ``YOLO.track()``'in
kendi içinde ``result.boxes.cpu().numpy()`` ile verdiği nesnenin aynısıdır. Görüntü (``img``) verilmez: ByteTrack hareket
telafisi (GMC) veya ReID kullanmaz, bu yüzden kare kopyalanmaz.

Sürüm bağımlılığı riski (açıkça belgelenmiştir)
-----------------------------------------------
``BYTETracker`` ve ``Boxes`` Ultralytics'in *dahili* (kararlılık sözü verilmemiş) sınıflarıdır; ``tracked_stracks`` / ``lost_stracks`` /
``STrack.start_frame`` gibi alanlar kaynak kodda belgelenmiş öznitelikler olsa da sürümler arasında değişebilir. Bu yüzden:
``requirements.txt`` ``ultralytics==8.4.210``'a sabitlenmiştir; farklı sürümde ``backend_info.version_note`` uyarı verir; beklenen
çıktı biçimi bozulursa ``TrackerBackendError`` fırlatılır (sessizce yanlış sonuç üretilmez). Ultralytics sürümü yükseltilirse
``tests/test_person_tracking_bytetrack_real.py`` (gerçek ByteTrack) yeniden çalıştırılmalıdır.

Bağımlılık ve ağ
----------------
ByteTrack ``lap`` paketini ister: ``ultralytics.trackers.utils.matching`` modül düzeyinde ``import lap`` yapar ve paket yoksa ``pip`` ile
OTOMATİK KURMAYI dener. Bunu engellemek için ``check_tracker_dependencies`` Ultralytics'i import etmeden ÖNCE ``lap``'ı denetler ve yoksa
açıklayıcı bir ``TrackerDependencyError`` verir; ayrıca ``YOLO_OFFLINE=1`` / ``YOLO_AUTOINSTALL=False`` ayarlanır. Hiçbir paket indirilmez/kurulmaz.

Takip politikası (backend'den bağımsız; ``PersonTracker`` sahibidir)
--------------------------------------------------------------------
* Her işlenen kare için ``update`` TAM BİR KEZ çağrılır — tespit olmayan kareler dahil (boş girdi ile). Aksi halde kayıp izler yaşlanmaz.
* ``frame_index`` kesin artan olmalıdır; tekrar/geri giden kare ``FrameOrderError`` verir ve durum DEĞİŞMEZ. Atlanan numaralar
  (boşluk) kabul edilir, ``stats.gap_frames`` olarak sayılır; ByteTrack boşluktaki kareleri görmez (ardışık sayar).
* Kaynak etiketi (``source``) veya görüntü boyutu değişirse takipçi OTOMATİK sıfırlanır (``reset_reason``), ``session_id`` artar.
* Çıktıdaki kutular Kalman tahminidir ve görüntü sınırlarına kırpılır; geçersiz kalan iz atılır ve sayılır (``dropped_tracks``).
* Aynı karede aynı ``track_id``'nin iki kez çıkması bir iç tutarsızlıktır ve ``TrackerBackendError`` ile bildirilir.

Bu modül import edildiğinde ``ultralytics``/``torch`` yüklenmez, hiçbir şey indirilmez, ağ kullanılmaz.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from isg_vision.human_detection.results import DetectionResult, PersonDetection, clip_box_xyxy

from .config import TrackingConfig
from .errors import (
    FrameOrderError,
    InvalidTrackingInputError,
    TrackerBackendError,
    TrackerDependencyError,
    TrackingError,
)
from .results import LostTrack, TrackedPerson, TrackingResult

TESTED_ULTRALYTICS_VERSION = "8.4.210"
REQUIRED_LAP_VERSION = "0.5.14"  # requirements.txt'te sabitlenen sürüm (Ultralytics en az 0.5.12 ister)
LAP_INSTALL_HINT = (
    f"Kurulum (ağ gerektirir; projede YALNIZCA bu açık adımdır, otomatik kurulum kapalıdır): "
    f".venv/bin/python -m pip install lap=={REQUIRED_LAP_VERSION}"
)
MAX_EXACT_TRACK_ID = 2**24  # ByteTrack çıktısı float32'dir; bu sayıdan sonra tam sayılar kesin gösterilemez
BOX_TOLERANCE_PX = 1e-3

RESET_SOURCE_CHANGED = "source_changed"
RESET_IMAGE_SIZE_CHANGED = "image_size_changed"


# ---------------------------------------------------------------------- bağımlılık denetimi
def prepare_offline_environment() -> None:
    """Ultralytics'in ağ yoklamasını ve otomatik paket kurulumunu kapatır (Ultralytics import edilmeden ÖNCE çağrılmalıdır)."""
    os.environ.setdefault("YOLO_OFFLINE", "1")  # 4A ile aynı koruma: import sırasında DNS yoklaması/indirme yok
    os.environ["YOLO_AUTOINSTALL"] = "False"  # gizli pip kurulumu hiçbir koşulda istenmez


def check_tracker_dependencies() -> dict[str, str]:
    """ByteTrack'in gerçekten ihtiyaç duyduğu paketleri, Ultralytics'i import etmeden denetler.

    Returns:
        ``{"lap": sürüm, "ultralytics": sürüm}``.

    Raises:
        TrackerDependencyError: ``lap`` veya ``ultralytics`` kurulu değil / kullanılamıyor. Hiçbir şey kurulmaz.
    """
    versions: dict[str, str] = {}
    for package in ("ultralytics", "lap"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            hint = LAP_INSTALL_HINT if package == "lap" else "ultralytics requirements.txt ile kurulur (ultralytics==8.4.210)."
            extra = (
                " ByteTrack (ultralytics.trackers.utils.matching) bu paketi modül düzeyinde ister ve yoksa kendiliğinden pip ile "
                "kurmaya çalışırdı; bu proje bunu bilinçli olarak engeller."
                if package == "lap"
                else ""
            )
            raise TrackerDependencyError(f"ByteTrack için '{package}' paketi kurulu değil.{extra} {hint}") from None
    try:
        lap = importlib.import_module("lap")  # ikili eklenti gerçekten yükleniyor mu (Ultralytics de __version__'ı doğrular)
        if not getattr(lap, "__version__", None):
            raise AttributeError("lap.__version__ yok")
    except Exception as exc:
        raise TrackerDependencyError(
            f"'lap' paketi kurulu görünüyor ({versions['lap']}) ama kullanılamıyor ({type(exc).__name__}: {exc}). {LAP_INSTALL_HINT}"
        ) from exc
    return versions


# ---------------------------------------------------------------------- backend
@dataclass(frozen=True)
class BackendFrame:
    """Backend'in bir kare için döndürdüğü ham sonuç (``PersonTracker`` bunu ``TrackingResult``'a çevirir).

    Attributes:
        rows: ``(N, 8)`` float32: ``[x1, y1, x2, y2, track_id, score, cls, idx]`` (ByteTrack çıktı biçimi).
        ages: ``track_id -> işlenen kare cinsinden yaş`` (``rows`` içindeki izler için).
        lost: ``(track_id, frames_since_seen)`` çiftleri (tamponda bekleyen kayıp izler).
        tentative: Onaylanmamış yeni iz sayısı.
    """

    rows: np.ndarray
    ages: dict[int, int]
    lost: tuple[tuple[int, int], ...]
    tentative: int


class TrackerBackend(Protocol):
    def update(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray, image_hw: tuple[int, int]) -> BackendFrame: ...

    def reset(self) -> None: ...


BackendFactory = Callable[[TrackingConfig], TrackerBackend]


class UltralyticsByteTrackBackend:
    """Gerçek Ultralytics ``BYTETracker``'ı saran ince katman (algoritma yeniden yazılmamıştır)."""

    def __init__(self, config: TrackingConfig) -> None:
        self.versions = check_tracker_dependencies()
        prepare_offline_environment()
        try:
            from ultralytics.engine.results import Boxes  # noqa: PLC0415 (bilinçli gecikmeli import)
            from ultralytics.trackers.basetrack import TrackState  # noqa: PLC0415
            from ultralytics.trackers.byte_tracker import BYTETracker  # noqa: PLC0415
            from ultralytics.utils import IterableSimpleNamespace  # noqa: PLC0415
        except Exception as exc:
            raise TrackerDependencyError(
                f"Ultralytics ByteTrack içe aktarılamadı ({type(exc).__name__}: {exc}). Kurulu sürüm: {self.versions['ultralytics']}; "
                f"test edilen sürüm: {TESTED_ULTRALYTICS_VERSION}."
            ) from exc
        try:
            self._tracker = BYTETracker(IterableSimpleNamespace(**config.bytetrack_args()))
        except Exception as exc:
            raise TrackerBackendError(f"ByteTrack takipçisi başlatılamadı ({type(exc).__name__}): {exc}") from exc
        self._boxes_cls = Boxes
        self._lost_state = TrackState.Lost
        self.version_note: str | None = None
        if self.versions["ultralytics"] != TESTED_ULTRALYTICS_VERSION:
            self.version_note = (
                f"Ultralytics {self.versions['ultralytics']} kurulu; takip modülü {TESTED_ULTRALYTICS_VERSION} ile test edildi. "
                "BYTETracker dahili bir API'dir; sürüm farkında davranış değişebilir, gerçek ByteTrack testlerini yeniden çalıştırın."
            )

    def update(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray, image_hw: tuple[int, int]) -> BackendFrame:
        n = len(conf)
        data = np.empty((n, 6), dtype=np.float32)
        data[:, :4] = xyxy
        data[:, 4] = conf
        data[:, 5] = cls
        raw = np.asarray(self._tracker.update(self._boxes_cls(data, image_hw)), dtype=np.float32)
        if raw.size == 0:
            rows = np.zeros((0, 8), dtype=np.float32)  # boş çıktı 1B (0,) gelir; biçim burada normalleştirilir
        elif raw.ndim == 2 and raw.shape[1] == 8:
            rows = raw
        else:
            raise TrackerBackendError(f"ByteTrack beklenmeyen biçimde çıktı verdi: şekil {raw.shape} (beklenen (N, 8)).")
        tracker = self._tracker
        frame_id = tracker.frame_id
        ages = {int(s.track_id): int(frame_id - s.start_frame + 1) for s in tracker.tracked_stracks}
        # Ultralytics 8.4.210: bu karede silinen (Removed) iz, havuzda bir kare daha kalır (merge_track_pools) ve o karede hâlâ yeniden
        # eşleşebilir (etkin kurtarma penceresi track_buffer+1 kare; ölçüldü). "Kayıp" sayısına yalnızca gerçekten Lost durumundakiler girer.
        lost = tuple(
            (int(s.track_id), int(frame_id - s.end_frame)) for s in tracker.lost_stracks if s.state == self._lost_state
        )
        tentative = sum(1 for s in tracker.tracked_stracks if not s.is_activated)
        return BackendFrame(rows, ages, lost, tentative)

    def reset(self) -> None:
        self._tracker.reset()  # iz havuzlarını boşaltır, Kalman süzgecini yeniler, kimlikleri 1'den başlatır


def create_ultralytics_backend(config: TrackingConfig) -> TrackerBackend:
    return UltralyticsByteTrackBackend(config)


# ---------------------------------------------------------------------- takipçi
@dataclass(frozen=True)
class TrackerStats:
    """``PersonTracker``'ın başlangıçtan bu yana özeti.

    Attributes:
        frames: İşlenen toplam kare.
        sessions: Başlatılan oturum sayısı (ilk oturum dahil; hiç kare işlenmediyse 0).
        resets: Otomatik veya elle yapılan sıfırlama sayısı.
        gap_frames: ``frame_index`` numaralarındaki boşluklar nedeniyle ByteTrack'in GÖRMEDİĞİ kare sayısı.
        dropped_tracks: Geçersiz kutu nedeniyle atılan toplam iz.
        distinct_tracks: Görülen farklı ``(session_id, track_id)`` sayısı. Gerçek kişi sayısından belirgin büyükse
            izler parçalanıyor (ID switch/yeniden başlatma) veya yanlış pozitifler iz alıyor olabilir; tek başına doğruluk ölçüsü DEĞİLDİR.
        max_active: Tek karedeki en büyük aktif iz sayısı.
    """

    frames: int
    sessions: int
    resets: int
    gap_frames: int
    dropped_tracks: int
    distinct_tracks: int
    max_active: int


class PersonTracker:
    """``DetectionResult``'ları ByteTrack ile kareler arasında ilişkilendirip geçici ``track_id`` atar.

    Args:
        config: Doğrulanmış takip yapılandırması (``None`` = varsayılanlar).
        backend_factory: ``TrackingConfig -> backend``; testlerde sahte/başarısız backend vermek için. Varsayılan gerçek ByteTrack.
        clock: Süre ölçümü için sayaç (varsayılan ``time.perf_counter``).

    ``load`` bağımlılık ve takipçi başlatma hatalarını kamera açılmadan önce ortaya çıkarmak içindir (``update`` gerekirse kendisi çağırır).
    Bir ``TrackingError`` (özellikle ``TrackerBackendError``) sonrasında iç durum tanımsız sayılmalı, ``reset()`` çağrılmalıdır;
    ``FrameOrderError`` ve ``InvalidTrackingInputError`` durumu DEĞİŞTİRMEZ (kare reddedilir).
    """

    def __init__(
        self,
        config: TrackingConfig | None = None,
        *,
        backend_factory: BackendFactory = create_ultralytics_backend,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config or TrackingConfig()
        self._backend_factory = backend_factory
        self._clock = clock
        self._backend: TrackerBackend | None = None
        self.load_count = 0  # takipçi gerçekten kaç kez oluşturuldu (testlerde denetim için)
        self._session_id = 1
        self._frames_in_session = 0
        self._last_frame_index: int | None = None
        self._last_timestamp: float | None = None
        self._image_size: tuple[int, int] | None = None
        self._source: str | None = None
        self._frames = self._sessions = self._resets = self._gap_frames = self._dropped_tracks = self._max_active = 0
        self._seen_keys: set[tuple[int, int]] = set()

    # ------------------------------------------------------------------ durum
    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    @property
    def session_id(self) -> int:
        """Geçerli oturum numarası (her sıfırlamada artar; ilk oturum 1)."""
        return self._session_id

    @property
    def backend_info(self) -> dict[str, Any]:
        """Yüklenmiş backend'in paket sürümleri ve varsa sürüm uyarısı (yüklenmediyse boş)."""
        backend = self._backend
        return {"versions": dict(getattr(backend, "versions", {}) or {}), "version_note": getattr(backend, "version_note", None)}

    @property
    def stats(self) -> TrackerStats:
        return TrackerStats(
            frames=self._frames,
            sessions=self._sessions,
            resets=self._resets,
            gap_frames=self._gap_frames,
            dropped_tracks=self._dropped_tracks,
            distinct_tracks=len(self._seen_keys),
            max_active=self._max_active,
        )

    # ------------------------------------------------------------------ yükleme / sıfırlama / kapatma
    def load(self) -> None:
        """Takipçiyi (yalnızca bir kez) oluşturur; bağımlılık/başlatma hatalarını burada ortaya çıkarır."""
        if self._backend is not None:
            return
        try:
            backend = self._backend_factory(self.config)
        except TrackingError:
            raise
        except Exception as exc:
            raise TrackerBackendError(f"Takipçi başlatılamadı ({type(exc).__name__}): {exc}") from exc
        self._backend = backend
        self.load_count += 1

    def reset(self) -> None:
        """Takip durumunu sıfırlar: iz havuzları boşalır, ``track_id``'ler 1'den başlar, oturum numarası artar (kare işlendiyse).

        Yeni kaynak veya yeni oturum başlarken çağrılmalıdır (``run_tracking`` bunu kendiliğinden yapar).
        """
        self._reset_state()

    def _reset_state(self) -> None:
        if self._backend is not None:
            self._backend.reset()
        if self._frames_in_session > 0:
            self._session_id += 1
            self._resets += 1
        self._frames_in_session = 0
        self._last_frame_index = None
        self._last_timestamp = None
        self._image_size = None
        self._source = None

    def close(self) -> None:
        """Takipçi referansını bırakır (birden fazla kez çağrılması güvenlidir)."""
        self._backend = None

    def __enter__(self) -> PersonTracker:
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ güncelleme
    def update(self, result: DetectionResult, *, source: str | None = None) -> TrackingResult:
        """Bir karenin tespitlerini takipçiye verir ve takip sonucunu döndürür. Tespitsiz kareler de verilmelidir.

        Args:
            result: ``PersonDetector.detect`` çıktısı (aynı karenin; YOLO burada ÇALIŞTIRILMAZ).
            source: Kaynak etiketi (``FramePacket.source``). Önceki karelerdekinden farklıysa takipçi önce sıfırlanır.

        Raises:
            InvalidTrackingInputError: Sonuç geçersiz (yanlış tür, geçersiz boyut, görüntü dışı kutu). Durum değişmez.
            FrameOrderError: ``frame_index`` tekrar etti veya geri gitti (ya da zaman damgası geri gitti). Durum değişmez.
            TrackerDependencyError / TrackerBackendError: ByteTrack kullanılamıyor veya beklenmeyen çıktı verdi.
        """
        start = self._clock()
        detections = _validate_result(result)
        self.load()

        reset_reason = self._pending_reset_reason(result, source)
        if reset_reason is None:
            self._check_order(result)  # durum değişmeden önce; hata durumu bozmaz
        frame_index = self._frame_index_for(result, reset=reset_reason is not None)

        if reset_reason is not None:
            self._reset_state()
        backend = self._backend
        assert backend is not None  # noqa: S101 (load() sonrası)

        width, height = result.image_size
        xyxy = np.asarray([d.bbox_xyxy for d in detections], dtype=np.float32).reshape(-1, 4)
        conf = np.asarray([d.confidence for d in detections], dtype=np.float32)
        cls = np.asarray([d.class_id for d in detections], dtype=np.float32)
        try:
            frame = backend.update(xyxy, conf, cls, (height, width))
        except TrackingError:
            raise
        except Exception as exc:
            raise TrackerBackendError(f"ByteTrack güncellemesi başarısız oldu ({type(exc).__name__}): {exc}") from exc

        tracks, dropped = self._build_tracks(frame, detections, result.image_size, frame_index, result.timestamp)
        elapsed_ms = max(0.0, (self._clock() - start) * 1000.0)

        if self._last_frame_index is not None and frame_index > self._last_frame_index + 1:
            self._gap_frames += frame_index - self._last_frame_index - 1
        if self._frames_in_session == 0:
            self._sessions += 1
        self._frames_in_session += 1
        self._frames += 1
        self._last_frame_index = frame_index
        self._last_timestamp = result.timestamp if result.timestamp is not None else self._last_timestamp
        self._image_size = result.image_size
        self._source = source if source is not None else self._source
        self._dropped_tracks += dropped
        self._max_active = max(self._max_active, len(tracks))
        self._seen_keys.update(t.key for t in tracks)

        return TrackingResult(
            tracks=tracks,
            lost_tracks=tuple(LostTrack(tid, max(1, since)) for tid, since in sorted(frame.lost)),
            tentative_count=int(frame.tentative),
            detection_count=len(detections),
            image_size=result.image_size,
            frame_index=frame_index,
            timestamp=result.timestamp,
            session_id=self._session_id,
            tracking_ms=elapsed_ms,
            dropped_tracks=dropped,
            reset_reason=reset_reason,
        )

    # ------------------------------------------------------------------ iç işlevler
    def _pending_reset_reason(self, result: DetectionResult, source: str | None) -> str | None:
        if self._frames_in_session == 0:
            return None
        if source is not None and self._source is not None and source != self._source:
            return RESET_SOURCE_CHANGED
        if self._image_size is not None and result.image_size != self._image_size:
            return RESET_IMAGE_SIZE_CHANGED
        return None

    def _check_order(self, result: DetectionResult) -> None:
        last = self._last_frame_index
        if last is not None and result.frame_index is not None and result.frame_index <= last:
            raise FrameOrderError(
                f"Kare sırası bozuk: frame_index {result.frame_index} geldi, bir önceki {last}. Aynı kare tekrar işlenemez ve kareler "
                "geriye gidemez; kare reddedildi, takip durumu değişmedi. Yeni bir kaynak/oturum için önce reset() çağırın."
            )
        if self._last_timestamp is not None and result.timestamp is not None and result.timestamp < self._last_timestamp:
            raise FrameOrderError(
                f"Kare zaman damgası geriye gitti ({result.timestamp} < {self._last_timestamp}); kare reddedildi, takip durumu değişmedi."
            )

    def _frame_index_for(self, result: DetectionResult, *, reset: bool) -> int:
        if result.frame_index is not None:
            return int(result.frame_index)
        last = None if reset else self._last_frame_index
        return 0 if last is None else last + 1  # kare numarası verilmediyse takipçi kendi sırasını yürütür

    def _build_tracks(
        self,
        frame: BackendFrame,
        detections: tuple[PersonDetection, ...],
        image_size: tuple[int, int],
        frame_index: int,
        timestamp: float | None,
    ) -> tuple[tuple[TrackedPerson, ...], int]:
        width, height = image_size
        tracks: list[TrackedPerson] = []
        seen: set[int] = set()
        dropped = 0
        for row in frame.rows:
            if not np.all(np.isfinite(row)):
                raise TrackerBackendError(f"ByteTrack sonlu olmayan bir değer üretti: {row.tolist()}")
            track_id, det_index = int(round(float(row[4]))), int(round(float(row[7])))
            if not 1 <= track_id <= MAX_EXACT_TRACK_ID:
                raise TrackerBackendError(f"Geçersiz Track ID {track_id} (1..{MAX_EXACT_TRACK_ID} dışında); takipçiyi sıfırlayın.")
            if not 0 <= det_index < len(detections):
                raise TrackerBackendError(f"ByteTrack, {len(detections)} tespitlik karede geçersiz tespit sırası döndürdü: {det_index}")
            if track_id in seen:
                raise TrackerBackendError(f"Aynı karede Track ID {track_id} iki ayrı izde göründü (tutarsız takipçi çıktısı).")
            seen.add(track_id)
            box = clip_box_xyxy(row[:4].tolist(), width, height)
            if box is None:
                dropped += 1
                continue
            det = detections[det_index]
            tracks.append(
                TrackedPerson(
                    track_id=track_id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox_xyxy=box,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    session_id=self._session_id,
                    detection_index=det_index,
                    age_frames=max(1, frame.ages.get(track_id, 1)),
                )
            )
        tracks.sort(key=lambda t: t.track_id)
        return tuple(tracks), dropped


_RESULT_ATTRS = ("detections", "image_size", "frame_index", "timestamp")
_DETECTION_ATTRS = ("bbox_xyxy", "confidence", "class_id", "class_name")


def _validate_result(result: object) -> tuple[PersonDetection, ...]:
    """Tespit sonucunu doğrular ve tespitleri demet olarak döndürür; geçersizse ``InvalidTrackingInputError``.

    Denetim ÖZNİTELİK tabanlıdır (``DetectionResult``/``PersonDetection`` sınıf kimliğine bağlı değil): modül yeniden yüklense (IDE/notebook
    hot-reload) bile geçerli bir sonuç reddedilmez; öznitelikleri eksik veya değerleri geçersiz nesneler yine reddedilir.
    """
    missing = [a for a in _RESULT_ATTRS if not hasattr(result, a)]
    if missing:
        raise InvalidTrackingInputError(
            f"Takipçi bir DetectionResult (detections, image_size, frame_index, timestamp) bekler: {type(result).__name__} "
            f"(eksik öznitelikler: {missing})"
        )
    result_ = result  # type: ignore[assignment]
    size = result_.image_size  # type: ignore[attr-defined]
    if (
        not isinstance(size, tuple)
        or len(size) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in size)
    ):
        raise InvalidTrackingInputError(f"image_size (genişlik, yükseklik) pozitif tam sayılar olmalıdır: {size!r}")
    width, height = size
    try:
        detections = tuple(result_.detections)  # type: ignore[attr-defined]
    except TypeError:
        raise InvalidTrackingInputError(f"detections yinelenebilir olmalıdır: {type(result_.detections).__name__}") from None  # type: ignore[attr-defined]
    for index, det in enumerate(detections):
        absent = [a for a in _DETECTION_ATTRS if not hasattr(det, a)]
        if absent:
            raise InvalidTrackingInputError(f"detections[{index}] bir PersonDetection olmalıdır: {type(det).__name__} (eksik öznitelikler: {absent})")
        try:
            x1, y1, x2, y2 = det.bbox_xyxy
            finite = all(math.isfinite(v) for v in (x1, y1, x2, y2))
            inside = finite and x2 > x1 and y2 > y1 and x1 >= -BOX_TOLERANCE_PX and y1 >= -BOX_TOLERANCE_PX and x2 <= width + BOX_TOLERANCE_PX and y2 <= height + BOX_TOLERANCE_PX
            conf_ok = math.isfinite(det.confidence) and 0.0 <= det.confidence <= 1.0
        except (TypeError, ValueError):
            raise InvalidTrackingInputError(f"detections[{index}] kutusu/güveni sayısal değil veya biçimi geçersiz: {det!r}") from None
        if not inside:
            raise InvalidTrackingInputError(
                f"detections[{index}] kutusu {det.bbox_xyxy} görüntü dışında veya geçersiz (görüntü {width}x{height}); "
                "tespit ile takip aynı kareye/boyuta ait olmalıdır."
            )
        if not conf_ok:
            raise InvalidTrackingInputError(f"detections[{index}] güven skoru geçersiz: {det.confidence!r}")
    return detections  # type: ignore[return-value]
