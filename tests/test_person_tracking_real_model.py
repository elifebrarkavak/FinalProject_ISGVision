"""GERÇEK YOLO11n + GERÇEK ByteTrack entegrasyon testleri (mock model/takipçi YOK).

Mock testleri gerçek çıkarımın/takibin çalıştığını kanıtlamaz; bu dosya ``PersonDetector`` (gerçek ``yolo11n.pt``) ile ``PersonTracker``
(gerçek Ultralytics ByteTrack) zincirini birlikte sınar.

* Ağırlık (``models/human_detection/yolo11n.pt``), ``lap`` veya ``ultralytics`` örnek görüntüleri yoksa testler açıkça ATLANIR; hiçbir şey
  indirilmez/kurulmaz, internet gerekmez.
* Görüntüler yüklü ``ultralytics`` paketinin içinde gelen örnek dosyalardır (``bus.jpg``: 4 yaya + otobüs, ``zidane.jpg``: 2 kişi); dışarıdan
  görüntü indirilmez. "Hareket eden kişi" dizileri bu GERÇEK kişi kesitlerinin sentetik bir tuval üzerinde kaydırılmasıyla üretilir.
* **Bunlar SENTETİK/kayıtlı örnek görüntü sınamalarıdır.** Gerçek kamerada, gerçek çalışma ortamında takip başarısını, IDF1/HOTA/MOTA'yı veya
  "%85 hedefini" kanıtlamaz; etiketli gerçek takip verisi yoktur. Beklenen sayılar SHA-256'sı sabit ağırlık, ``ultralytics==8.4.210`` ve CPU
  ile ölçülmüştür; MPS testleri yalnızca değişmezleri (invariants) denetler (cihazlar sınır tespitlerinde küçük farklar verebilir).
"""

from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

os.environ.setdefault("YOLO_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from camera_test_support import FakeCapture, FakeClock, FakeFactory, make_frame  # noqa: E402,F401

from isg_vision.camera.capture import FrameSource  # noqa: E402
from isg_vision.camera.config import SourceConfig  # noqa: E402
from isg_vision.human_detection.benchmark import make_synthetic_video  # noqa: E402
from isg_vision.human_detection.config import DEFAULT_MODEL_PATH, DetectionConfig, torch_mps_status  # noqa: E402
from isg_vision.human_detection.person_detector import PersonDetector  # noqa: E402
from isg_vision.tracking.bytetrack_adapter import PersonTracker  # noqa: E402
from isg_vision.tracking.config import TrackingConfig  # noqa: E402
from isg_vision.tracking.metrics import count_id_switches  # noqa: E402
from isg_vision.tracking.pipeline import REASON_END_OF_STREAM, REASON_MAX_FRAMES, run_tracking  # noqa: E402

ROOT_SCRIPT = ROOT / "scripts" / "run_person_tracking.py"
_ultra = importlib.util.find_spec("ultralytics")
ASSETS = Path(_ultra.origin).parent / "assets" if _ultra and _ultra.origin else None
WEIGHTS_PRESENT = DEFAULT_MODEL_PATH.is_file()
SAMPLES_PRESENT = bool(ASSETS and (ASSETS / "bus.jpg").is_file() and (ASSETS / "zidane.jpg").is_file())
LAP_PRESENT = importlib.util.find_spec("lap") is not None
READY = WEIGHTS_PRESENT and SAMPLES_PRESENT and LAP_PRESENT
SKIP_REASON = (
    "Gerçek YOLO+ByteTrack testi: models/human_detection/yolo11n.pt (scripts/download_person_model.py), 'lap' (pip install lap==0.5.14) "
    "ve ultralytics örnek görüntüleri gerekir; atlandı"
)


def person_crops(detector: PersonDetector, bus: np.ndarray, scale: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """bus.jpg'de gerçek YOLO ile bulunan en güvenli iki kişinin (kenar boşluklu) kesitleri."""
    result = detector.detect(bus)
    people = sorted(result.detections, key=lambda d: -d.confidence)
    assert len(people) >= 2, "bus.jpg'de en az iki kişi bulunmalıydı"

    def crop(d, pad=10):  # type: ignore[no-untyped-def]
        x1, y1, x2, y2 = (int(v) for v in d.bbox_xyxy)
        return cv2.resize(bus[max(0, y1 - pad) : y2 + pad, max(0, x1 - pad) : x2 + pad].copy(), None, fx=scale, fy=scale)

    return crop(people[0]), crop(people[1])


class Canvas:
    """Gri tuval üzerinde bilinen konumlarda (gerçek kimlik bilinen) kişi kesitleri."""

    W, H = 960, 720

    def __init__(self, crop_a: np.ndarray, crop_b: np.ndarray) -> None:
        self.crops = {"A": crop_a, "B": crop_b}

    def frame(self, pos: dict[str, tuple[int, int]]) -> np.ndarray:
        canvas = np.full((self.H, self.W, 3), 110, np.uint8)
        for name, (x, y) in pos.items():
            crop = self.crops[name]
            canvas[y : y + crop.shape[0], x : x + crop.shape[1]] = crop
        return canvas

    def center(self, name: str, pos: tuple[int, int]) -> tuple[float, float]:
        h, w = self.crops[name].shape[:2]
        return (pos[0] + w / 2, pos[1] + h / 2)


def identity_of(det, gt: dict[str, tuple[float, float]]) -> str:  # type: ignore[no-untyped-def]
    """Ham tespitin merkezine en yakın gerçek konumun kimliği (kimlik bilinen sentetik dizi için)."""
    cx, cy = (det.bbox_xyxy[0] + det.bbox_xyxy[2]) / 2, (det.bbox_xyxy[1] + det.bbox_xyxy[3]) / 2
    return min(gt, key=lambda k: (gt[k][0] - cx) ** 2 + (gt[k][1] - cy) ** 2)


class _RealMixin:
    """Ortak kurulum ve testler (``TestCase`` DEĞİL: yalnızca somut alt sınıflarda keşfedilir; böylece atlanan test sayısı şişmez)."""

    device = "cpu"
    detector: PersonDetector
    bus: np.ndarray
    zidane: np.ndarray

    @classmethod
    def setUpClass(cls) -> None:
        assert ASSETS is not None
        cls.bus = cv2.imread(str(ASSETS / "bus.jpg"))
        cls.zidane = cv2.imread(str(ASSETS / "zidane.jpg"))
        cls.detector = PersonDetector(DetectionConfig(device=cls.device))
        cls.detector.load()
        cls.crop_a, cls.crop_b = person_crops(cls.detector, cls.bus)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.detector.close()

    # -- yardımcılar
    def track_sequence(self, frames: list[np.ndarray], gt_per_frame: list[dict[str, tuple[float, float]]] | None = None):  # type: ignore[no-untyped-def]
        """Kare dizisini gerçek YOLO + gerçek ByteTrack'ten geçirir; (sonuçlar, kimlik atamaları, YOLO çağrı sayısı) döndürür."""
        tracker = PersonTracker()
        tracker.load()
        results, assignments = [], []
        with mock.patch.object(self.detector._model, "predict", wraps=self.detector._model.predict) as predict_spy:
            for i, frame in enumerate(frames):
                detection = self.detector.detect(frame, frame_index=i, timestamp=i / 30.0)
                tracking = tracker.update(detection, source="gercek-model-testi")
                results.append(tracking)
                if gt_per_frame is not None:
                    assignments.append({identity_of(detection.detections[t.detection_index], gt_per_frame[i]): t.track_id for t in tracking.tracks})
        return results, assignments, predict_spy.call_count

    def assert_invariants(self, results, n_frames: int, yolo_calls: int) -> None:  # type: ignore[no-untyped-def]
        self.assertEqual(yolo_calls, n_frames, "YOLO kare başına tam bir kez çalışmalı (takip ek çıkarım yapmaz)")
        for r in results:
            self.assertEqual(len(set(r.track_ids)), len(r.track_ids), "aynı karede yinelenen Track ID")
            self.assertLessEqual(r.active_count, r.detection_count, "aktif iz sayısı tespit sayısını aşamaz")
            self.assertEqual(r.dropped_tracks, 0)

    # -- testler
    def test_static_image_persons_get_stable_unique_ids(self) -> None:
        for name, image in (("bus", self.bus), ("zidane", self.zidane)):
            with self.subTest(image=name):
                results, _, calls = self.track_sequence([image] * 15)
                self.assert_invariants(results, 15, calls)
                n = results[0].detection_count
                self.assertGreaterEqual(n, 2)
                self.assertEqual(results[0].active_count, n, "oturumun ilk karesinde tüm tespitler hemen iz alır")
                self.assertTrue(all(r.track_ids == results[0].track_ids for r in results), "sabit görüntüde ID'ler kare boyunca değişmemeli")
                self.assertEqual(len({k for r in results for k in r.track_ids}), n)

    def test_boxes_follow_real_detections_on_static_image(self) -> None:
        results, _, _ = self.track_sequence([self.bus] * 10)
        last = results[-1]
        detection = self.detector.detect(self.bus)
        for t in last.tracks:
            raw = detection.detections[t.detection_index].bbox_xyxy
            for a, b in zip(t.bbox_xyxy, raw):
                self.assertLess(abs(a - b), 3.0, "durağan kişide takip kutusu ham YOLO kutusundan en çok birkaç piksel sapmalı")

    def test_single_moving_real_person_keeps_one_id(self) -> None:
        canvas = Canvas(self.crop_a, self.crop_b)
        frames = [canvas.frame({"A": (20 + 5 * i, 60)}) for i in range(30)]
        results, _, calls = self.track_sequence(frames)
        self.assert_invariants(results, 30, calls)
        ids = {k for r in results for k in r.track_ids}
        self.assertEqual(len(ids), 1, f"tek kişi tek ID almalı: {ids}")
        self.assertGreaterEqual(sum(1 for r in results if r.active_count == 1), 28)

    def test_real_people_reappearing_after_short_empty_frames_keep_ids(self) -> None:
        canvas = Canvas(self.crop_a, self.crop_b)
        n = 40
        gt, frames = [], []
        for i in range(n):
            pos = {"A": (20 + 8 * i, 60), "B": (canvas.W - self.crop_b.shape[1] - 20 - 8 * i, 330)}
            if 15 <= i < 20:  # 5 kare tamamen boş (kadraj dışı gibi)
                frames.append(canvas.frame({}))
                gt.append({})
            else:
                frames.append(canvas.frame(pos))
                gt.append({k: canvas.center(k, v) for k, v in pos.items()})
        results, assignments, calls = self.track_sequence(frames, gt)
        self.assert_invariants(results, n, calls)
        self.assertTrue(all(r.active_count == 0 for r in results[15:20]), "boş karelerde aktif iz olmamalı")
        self.assertTrue(all(r.lost_count == 2 for r in results[15:20]), "iki kişi kayıp tamponunda (aktif sayılmaz)")
        self.assertEqual(count_id_switches(assignments).switches, 0)

    def test_empty_scene_is_handled(self) -> None:
        empty = np.full((480, 640, 3), 110, np.uint8)
        results, _, calls = self.track_sequence([empty] * 12)
        self.assert_invariants(results, 12, calls)
        self.assertTrue(all((r.detection_count, r.active_count, r.lost_count) == (0, 0, 0) for r in results))

    def test_image_change_between_runs_is_a_new_session_without_carry_over(self) -> None:
        tracker = PersonTracker()
        tracker.load()
        for i in range(6):
            tracker.update(self.detector.detect(self.bus, frame_index=i), source="bus")
        out = tracker.update(self.detector.detect(self.zidane, frame_index=0), source="zidane")
        self.assertEqual(out.reset_reason, "source_changed")
        self.assertEqual((out.session_id, out.lost_count), (2, 0))
        self.assertEqual(sorted(out.track_ids), list(range(1, out.active_count + 1)), "yeni oturumda numaralar 1'den başlar")

    def test_loop_over_synthetic_video_with_real_model_and_tracker(self) -> None:
        """4A'nın sentetik montajıyla (örnek görüntülerden) entegrasyon/hız: takip DOĞRULUĞU çıkarılamaz."""
        with tempfile.TemporaryDirectory() as tmp:
            video = make_synthetic_video(Path(tmp) / "montaj.mp4", frames=45, size=(640, 360))
            tracker = PersonTracker()
            with mock.patch.object(self.detector._model, "predict", wraps=self.detector._model.predict) as spy, FrameSource(SourceConfig.video(video)) as src:
                summary = run_tracking(src, self.detector, tracker)
        self.assertEqual(summary.reason, REASON_END_OF_STREAM)
        self.assertEqual(summary.frames_processed, 45)
        self.assertEqual(spy.call_count, 45, "45 kare için tam 45 YOLO çıkarımı")
        self.assertGreaterEqual(summary.active_max, 4)
        self.assertGreaterEqual(summary.detections_max, summary.active_max)
        self.assertLessEqual(summary.distinct_tracks, 3 * summary.detections_max, "iz sayısı patlamamalı")
        self.assertIsNotNone(summary.tracking)
        self.assertLess(summary.tracking.median_ms, 5.0, "birkaç kişi için ByteTrack güncellemesi milisaniye mertebesinde olmalı")  # type: ignore[union-attr]
        self.assertEqual(summary.dropped_tracks, 0)


@unittest.skipUnless(READY, SKIP_REASON)
class RealYoloByteTrackCpuTest(_RealMixin, unittest.TestCase):
    device = "cpu"

    def test_two_moving_real_people_in_opposite_directions_keep_their_ids(self) -> None:
        """Gerçek kimlik bilinir (kesit konumları); ID switch sayılır. Dikey ayrık geçiş: bu dizide ID switch yok (idealize sentetik).

        Kesin ID sayıları yalnızca CPU'da ölçüldü (MPS sınır tespitlerinde farklı olabilir); bu yüzden yalnızca CPU sınıfındadır."""
        canvas = Canvas(self.crop_a, self.crop_b)
        n = 50
        gt, frames = [], []
        for i in range(n):
            pos = {"A": (20 + 8 * i, 60), "B": (canvas.W - self.crop_b.shape[1] - 20 - 8 * i, 330)}
            frames.append(canvas.frame(pos))
            gt.append({k: canvas.center(k, v) for k, v in pos.items()})
        results, assignments, calls = self.track_sequence(frames, gt)
        self.assert_invariants(results, n, calls)
        report = count_id_switches(assignments)
        self.assertEqual(report.switches, 0, report)
        self.assertEqual(report.takeovers, 0, report)
        self.assertEqual(len(report.track_ids_per_identity["A"]) + len(report.track_ids_per_identity["B"]), 2, "her kişi tek ID")
        self.assertGreaterEqual(min(report.frames_tracked_per_identity.values()), n - 3)

    def test_models_and_tracker_run_offline_without_network_or_pip(self) -> None:
        """Yükleme dahil tüm zincir: ağ bağlantısı veya alt süreç (pip) denenirse test düşer."""
        with mock.patch.object(socket.socket, "connect", side_effect=AssertionError("ağ bağlantısı denendi!")), mock.patch.object(
            subprocess, "Popen", side_effect=AssertionError("alt süreç (pip?) denendi!")
        ), mock.patch.object(subprocess, "check_output", side_effect=AssertionError("pip denendi!")):
            detector = PersonDetector(DetectionConfig(device="cpu"))
            detector.load()
            tracker = PersonTracker()
            tracker.load()
            for i in range(4):
                tracker.update(detector.detect(self.bus, frame_index=i), source="offline")
            detector.close()
        self.assertEqual(os.environ.get("YOLO_OFFLINE"), "1")
        self.assertEqual(os.environ.get("YOLO_AUTOINSTALL"), "False")

    def test_the_model_weights_are_untouched(self) -> None:
        import hashlib

        digest = hashlib.sha256(DEFAULT_MODEL_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1", "yolo11n.pt SHA-256 değişmemeli")

    def test_script_runs_end_to_end_on_cpu_with_real_model(self) -> None:
        """Gerçek betik (alt süreç): gerçek YOLO + gerçek ByteTrack, sentetik montaj videosu, penceresiz."""
        with tempfile.TemporaryDirectory() as tmp:
            video = make_synthetic_video(Path(tmp) / "montaj.mp4", frames=30, size=(640, 360))
            proc = subprocess.run(
                [sys.executable, str(ROOT_SCRIPT), "--source", "video", "--video-path", str(video), "--device", "cpu", "--no-display"],
                capture_output=True, text=True, timeout=300,
            )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        for needle in ("Takip: ByteTrack", "video sona erdi", "İşlenen kare: 30", "ByteTrack takip süresi", "Görülen farklı Track ID", "İz özeti"):
            self.assertIn(needle, proc.stdout)
        self.assertNotIn("Traceback", proc.stderr)

    def test_script_max_frames_on_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = make_synthetic_video(Path(tmp) / "montaj.mp4", frames=30, size=(640, 360))
            proc = subprocess.run(
                [sys.executable, str(ROOT_SCRIPT), "--source", "video", "--video-path", str(video), "--device", "cpu", "--no-display", "--max-frames", "10",
                 "--conf", "0.25"],
                capture_output=True, text=True, timeout=300,
            )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("--max-frames sınırına ulaşıldı", proc.stdout)
        self.assertIn("İşlenen kare: 10", proc.stdout)


@unittest.skipUnless(READY and torch_mps_status()[0], SKIP_REASON + " (ve MPS bu makinede kullanılabilir olmalı)")
class RealYoloByteTrackMpsTest(_RealMixin, unittest.TestCase):
    """MPS: yalnızca değişmezler (cihazlar sınır tespitlerinde farklı olabilir)."""

    device = "mps"

    def test_script_runs_on_mps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = make_synthetic_video(Path(tmp) / "montaj.mp4", frames=30, size=(640, 360))
            proc = subprocess.run(
                [sys.executable, str(ROOT_SCRIPT), "--source", "video", "--video-path", str(video), "--device", "mps", "--no-display"],
                capture_output=True, text=True, timeout=300,
            )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("cihaz: mps", proc.stdout)
        self.assertIn("YOLO cihazı: mps (ByteTrack: CPU)", proc.stdout)


if __name__ == "__main__":
    unittest.main()
