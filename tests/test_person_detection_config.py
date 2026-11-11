"""İnsan tespiti yapılandırması, cihaz seçimi ve resmî ağırlık indirme/doğrulama testleri (ağ ve gerçek model yok)."""

from __future__ import annotations

import hashlib
import io
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from isg_vision.human_detection import weights  # noqa: E402
from isg_vision.human_detection.config import (  # noqa: E402
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_IOU,
    DEFAULT_MODEL_PATH,
    DEVICES,
    DetectionConfig,
    resolve_device,
    torch_mps_status,
)
from isg_vision.human_detection.errors import (  # noqa: E402
    DetectionConfigError,
    DeviceUnavailableError,
    WeightsDownloadError,
)
from person_detection_test_support import TempDirMixin  # noqa: E402


class ValidConfigTest(unittest.TestCase):
    def test_defaults_are_valid_and_documented_values(self) -> None:
        cfg = DetectionConfig()
        self.assertEqual((cfg.conf, cfg.iou, cfg.imgsz, cfg.device), (DEFAULT_CONF, DEFAULT_IOU, DEFAULT_IMGSZ, "auto"))
        self.assertEqual(cfg.model_path, DEFAULT_MODEL_PATH)
        self.assertEqual(cfg.model_name, "yolo11n.pt")

    def test_custom_valid_values(self) -> None:
        cfg = DetectionConfig(model_path="~/agirlik.pt", conf=0.35, iou=0.5, imgsz=320, device="cpu", max_det=10)
        self.assertEqual((cfg.conf, cfg.iou, cfg.imgsz, cfg.device, cfg.max_det), (0.35, 0.5, 320, "cpu", 10))
        self.assertIsInstance(cfg.model_path, Path)
        self.assertNotIn("~", str(cfg.model_path))  # ev dizini açıldı

    def test_boundary_values_accepted(self) -> None:
        DetectionConfig(conf=1.0, iou=1.0, imgsz=32, max_det=1)
        DetectionConfig(conf=0.001, iou=0.001, imgsz=4096)

    def test_creating_config_does_not_touch_filesystem_or_network(self) -> None:
        # Var olmayan bir ağırlık yolu yapılandırma aşamasında hata vermez (varlık yükleme sırasında denetlenir).
        DetectionConfig(model_path="/olmayan/klasor/model.pt")

    def test_config_is_frozen(self) -> None:
        with self.assertRaises(Exception):
            DetectionConfig().conf = 0.9  # type: ignore[misc]


class InvalidModelPathTest(unittest.TestCase):
    def test_empty_path(self) -> None:
        for bad in ("", "   "):
            with self.subTest(bad=bad), self.assertRaises(DetectionConfigError):
                DetectionConfig(model_path=bad)

    def test_wrong_type(self) -> None:
        for bad in (None, 5, True, 3.5):
            with self.subTest(bad=bad), self.assertRaises(DetectionConfigError):
                DetectionConfig(model_path=bad)  # type: ignore[arg-type]

    def test_wrong_extension(self) -> None:
        for bad in ("model.onnx", "model", "model.txt", "klasor/"):
            with self.subTest(bad=bad), self.assertRaises(DetectionConfigError):
                DetectionConfig(model_path=bad)

    def test_uppercase_extension_accepted(self) -> None:
        DetectionConfig(model_path="MODEL.PT")


class InvalidThresholdTest(unittest.TestCase):
    def test_invalid_confidence(self) -> None:
        for bad in (0, 0.0, -0.1, 1.01, 2, float("nan"), float("inf"), "0.5", None, True):
            with self.subTest(bad=bad), self.assertRaises(DetectionConfigError):
                DetectionConfig(conf=bad)  # type: ignore[arg-type]

    def test_invalid_iou(self) -> None:
        for bad in (0, 0.0, -0.5, 1.5, float("nan"), float("-inf"), "0.7", None, False):
            with self.subTest(bad=bad), self.assertRaises(DetectionConfigError):
                DetectionConfig(iou=bad)  # type: ignore[arg-type]

    def test_error_message_names_the_field(self) -> None:
        with self.assertRaisesRegex(DetectionConfigError, "conf"):
            DetectionConfig(conf=5)
        with self.assertRaisesRegex(DetectionConfigError, "iou"):
            DetectionConfig(iou=5)


class InvalidImgszAndMaxDetTest(unittest.TestCase):
    def test_invalid_imgsz(self) -> None:
        for bad in (0, -32, 16, 31, 100, 641, 8192, 4128, 640.0, "640", None, True):
            with self.subTest(bad=bad), self.assertRaises(DetectionConfigError):
                DetectionConfig(imgsz=bad)  # type: ignore[arg-type]

    def test_imgsz_must_be_multiple_of_stride_message(self) -> None:
        with self.assertRaisesRegex(DetectionConfigError, "32"):
            DetectionConfig(imgsz=100)

    def test_invalid_max_det(self) -> None:
        for bad in (0, -1, 1001, 1.5, "10", None, True):
            with self.subTest(bad=bad), self.assertRaises(DetectionConfigError):
                DetectionConfig(max_det=bad)  # type: ignore[arg-type]


class DeviceSelectionTest(unittest.TestCase):
    MPS_OK = staticmethod(lambda: (True, "kullanılabilir"))
    MPS_NO = staticmethod(lambda: (False, "test: MPS yok"))

    def test_valid_device_names(self) -> None:
        self.assertEqual(DEVICES, ("auto", "cpu", "mps"))
        for name in DEVICES:
            DetectionConfig(device=name)

    def test_invalid_device_names(self) -> None:
        for bad in ("cuda", "gpu", "MPS", "", None, 0):
            with self.subTest(bad=bad), self.assertRaises(DetectionConfigError):
                DetectionConfig(device=bad)  # type: ignore[arg-type]

    def test_cpu_never_consults_torch(self) -> None:
        def boom() -> tuple[bool, str]:
            raise AssertionError("cpu için MPS yoklanmamalı")

        resolved = resolve_device("cpu", mps_available=boom)
        self.assertEqual((resolved.device, resolved.warning), ("cpu", None))

    def test_mps_when_available(self) -> None:
        resolved = resolve_device("mps", mps_available=self.MPS_OK)
        self.assertEqual((resolved.requested, resolved.device, resolved.warning), ("mps", "mps", None))

    def test_explicit_mps_unavailable_raises_instead_of_silent_cpu(self) -> None:
        with self.assertRaises(DeviceUnavailableError) as ctx:
            resolve_device("mps", mps_available=self.MPS_NO)
        message = str(ctx.exception)
        self.assertIn("MPS", message)
        self.assertIn("test: MPS yok", message)  # gerçek neden gösterilir
        self.assertIn("--device cpu", message)  # çıkış yolu söylenir

    def test_auto_prefers_mps(self) -> None:
        resolved = resolve_device("auto", mps_available=self.MPS_OK)
        self.assertEqual((resolved.device, resolved.warning), ("mps", None))

    def test_auto_falls_back_to_cpu_with_visible_warning(self) -> None:
        resolved = resolve_device("auto", mps_available=self.MPS_NO)
        self.assertEqual(resolved.device, "cpu")
        self.assertEqual(resolved.requested, "auto")
        self.assertIsNotNone(resolved.warning)
        self.assertIn("MPS kullanılamıyor", resolved.warning or "")

    def test_invalid_request_rejected(self) -> None:
        with self.assertRaises(DetectionConfigError):
            resolve_device("cuda", mps_available=self.MPS_OK)

    def test_real_torch_status_returns_pair(self) -> None:
        available, reason = torch_mps_status()
        self.assertIsInstance(available, bool)
        self.assertIsInstance(reason, str)
        self.assertTrue(reason)


class ImportSideEffectTest(unittest.TestCase):
    def test_importing_package_does_not_load_torch_or_ultralytics(self) -> None:
        code = (
            "import sys; sys.path.insert(0, %r); import isg_vision.human_detection as h, "
            "isg_vision.human_detection.config, isg_vision.human_detection.person_detector, isg_vision.human_detection.weights; "
            "bad = [m for m in ('torch', 'ultralytics') if m in sys.modules]; "
            "print(','.join(bad)); sys.exit(1 if bad else 0)"
        ) % str(ROOT / "src")
        proc = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, f"torch/ultralytics import sırasında yüklendi: {proc.stdout} {proc.stderr}")


# ---------------------------------------------------------------------- resmî ağırlık indirme/doğrulama
PAYLOAD = b"A" * 200_000  # MIN_WEIGHT_BYTES'tan büyük sahte "ağırlık"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()
OFFICIAL = weights.OFFICIAL_URL_PREFIX + "v0/test.pt"


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _opener(data: bytes = PAYLOAD, *, error: Exception | None = None):  # type: ignore[no-untyped-def]
    calls: list[str] = []

    def opener(url: str, timeout: float = 0) -> _FakeResponse:
        calls.append(url)
        if error is not None:
            raise error
        return _FakeResponse(data)

    opener.calls = calls  # type: ignore[attr-defined]
    return opener


def _registry(sha: str | None = PAYLOAD_SHA, url: str = OFFICIAL) -> dict[str, weights.OfficialModel]:
    return {"test.pt": weights.OfficialModel("test.pt", url, sha)}


class OfficialWeightsTest(TempDirMixin, unittest.TestCase):
    def test_allow_list_contains_default_model_with_pinned_hash_and_official_url(self) -> None:
        spec = weights.OFFICIAL_MODELS["yolo11n.pt"]
        self.assertTrue(spec.url.startswith(weights.OFFICIAL_URL_PREFIX))
        self.assertRegex(spec.sha256 or "", r"^[0-9a-f]{64}$")

    def test_download_success_writes_file_and_verifies_hash(self) -> None:
        opener = _opener()
        result = weights.download_official_weights("test.pt", self.tmp, opener=opener, models=_registry())
        self.assertTrue(result.downloaded)
        self.assertTrue(result.pinned)
        self.assertEqual(result.sha256, PAYLOAD_SHA)
        self.assertEqual((self.tmp / "test.pt").read_bytes(), PAYLOAD)
        self.assertEqual(opener.calls, [OFFICIAL])
        self.assertFalse(list(self.tmp.glob("*.part")), "geçici dosya kalmamalı")

    def test_unlisted_name_is_refused_without_network(self) -> None:
        opener = _opener()
        with self.assertRaisesRegex(WeightsDownloadError, "izin listesinde değil"):
            weights.download_official_weights("baska.pt", self.tmp, opener=opener, models=_registry())
        self.assertEqual(opener.calls, [])

    def test_non_official_url_is_refused_without_network(self) -> None:
        opener = _opener()
        bad = _registry(url="https://example.com/test.pt")
        with self.assertRaisesRegex(WeightsDownloadError, "Resmî olmayan"):
            weights.download_official_weights("test.pt", self.tmp, opener=opener, models=bad)
        self.assertEqual(opener.calls, [])

    def test_hash_mismatch_rejects_and_removes_partial_file(self) -> None:
        with self.assertRaisesRegex(WeightsDownloadError, "SHA-256 uyuşmuyor"):
            weights.download_official_weights("test.pt", self.tmp, opener=_opener(b"B" * 200_000), models=_registry())
        self.assertEqual(list(self.tmp.iterdir()), [], "uyuşmayan dosya hedefe taşınmamalı, geçici dosya silinmeli")

    def test_tiny_response_is_rejected_as_error_page(self) -> None:
        with self.assertRaisesRegex(WeightsDownloadError, "çok küçük"):
            weights.download_official_weights("test.pt", self.tmp, opener=_opener(b"<html>404</html>"), models=_registry(None))
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_network_error_is_wrapped_and_cleans_up(self) -> None:
        with self.assertRaisesRegex(WeightsDownloadError, "İndirme başarısız"):
            weights.download_official_weights("test.pt", self.tmp, opener=_opener(error=OSError("ağ yok")), models=_registry())
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_existing_file_is_not_overwritten_or_redownloaded(self) -> None:
        (self.tmp / "test.pt").write_bytes(PAYLOAD)
        opener = _opener(b"C" * 200_000)
        result = weights.download_official_weights("test.pt", self.tmp, opener=opener, models=_registry())
        self.assertFalse(result.downloaded)
        self.assertEqual(opener.calls, [])
        self.assertEqual((self.tmp / "test.pt").read_bytes(), PAYLOAD)

    def test_existing_corrupt_file_is_reported_not_deleted(self) -> None:
        (self.tmp / "test.pt").write_bytes(b"bozuk")
        with self.assertRaisesRegex(WeightsDownloadError, "SHA-256 uyuşmuyor"):
            weights.download_official_weights("test.pt", self.tmp, opener=_opener(), models=_registry())
        self.assertEqual((self.tmp / "test.pt").read_bytes(), b"bozuk")  # kullanıcının dosyası silinmez

    def test_force_overwrites(self) -> None:
        (self.tmp / "test.pt").write_bytes(b"eski")
        result = weights.download_official_weights("test.pt", self.tmp, force=True, opener=_opener(), models=_registry())
        self.assertTrue(result.downloaded)
        self.assertEqual((self.tmp / "test.pt").read_bytes(), PAYLOAD)

    def test_unpinned_model_downloads_and_reports_hash(self) -> None:
        result = weights.download_official_weights("test.pt", self.tmp, opener=_opener(), models=_registry(None))
        self.assertFalse(result.pinned)
        self.assertEqual(result.sha256, PAYLOAD_SHA)

    def test_sha256_file_matches_hashlib(self) -> None:
        path = self.tmp / "x.bin"
        path.write_bytes(PAYLOAD)
        self.assertEqual(weights.sha256_file(path), PAYLOAD_SHA)

    def test_downloaded_default_weight_matches_pinned_hash_if_present(self) -> None:
        path = weights.default_weights_dir() / "yolo11n.pt"
        if not path.is_file():
            self.skipTest("yolo11n.pt indirilmemiş (scripts/download_person_model.py)")
        self.assertEqual(weights.sha256_file(path), weights.OFFICIAL_MODELS["yolo11n.pt"].sha256)


if __name__ == "__main__":
    unittest.main()
