"""Resmî yüz modeli indirme/bütünlük testleri — AĞ KULLANILMAZ (sahte ``opener``); gerçek dosya yalnızca SHA-256 için okunur."""

from __future__ import annotations

import hashlib
import io
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from person_detection_test_support import TempDirMixin  # noqa: E402

from isg_vision.face_detection import weights as w  # noqa: E402
from isg_vision.face_detection.config import DEFAULT_MODEL_NAME, DEFAULT_MODEL_PATH  # noqa: E402
from isg_vision.face_detection.errors import FaceWeightsDownloadError  # noqa: E402

PAYLOAD = (b"onnx-benzeri-veri" * 20000)[:150_000]
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


def registry(**overrides) -> dict[str, w.OfficialFaceModel]:  # type: ignore[no-untyped-def]
    fields = {"name": "test_model.onnx", "url": w.OFFICIAL_URL_PREFIX + "test_model.onnx", "sha256": PAYLOAD_SHA, "size_bytes": len(PAYLOAD)}
    fields.update(overrides)
    return {fields["name"]: w.OfficialFaceModel(**fields)}


class FakeResponse(io.BytesIO):
    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class Opener:
    def __init__(self, data: bytes | Exception = PAYLOAD) -> None:
        self.data = data
        self.calls: list[tuple[str, float]] = []

    def __call__(self, url: str, timeout: float = 0) -> FakeResponse:
        self.calls.append((url, timeout))
        if isinstance(self.data, Exception):
            raise self.data
        return FakeResponse(self.data)


class RegistryTest(unittest.TestCase):
    def test_default_model_is_in_the_allowlist(self) -> None:
        self.assertIn(DEFAULT_MODEL_NAME, w.OFFICIAL_MODELS)
        self.assertEqual(DEFAULT_MODEL_PATH.name, DEFAULT_MODEL_NAME)

    def test_urls_are_pinned_to_an_immutable_commit_not_a_branch(self) -> None:
        self.assertTrue(re.fullmatch(r"[0-9a-f]{40}", w.OPENCV_ZOO_COMMIT))
        self.assertIn(w.OPENCV_ZOO_COMMIT, w.OFFICIAL_URL_PREFIX)
        for model in w.OFFICIAL_MODELS.values():
            self.assertTrue(model.url.startswith(w.OFFICIAL_URL_PREFIX), model.url)
            self.assertTrue(model.url.startswith("https://"), "yalnızca https")
            self.assertNotIn("/main/", model.url)
            self.assertNotIn("/master/", model.url)

    def test_registry_entries_have_pinned_hash_size_and_license(self) -> None:
        for model in w.OFFICIAL_MODELS.values():
            self.assertRegex(model.sha256, r"[0-9a-f]{64}")
            self.assertGreater(model.size_bytes, w.MIN_MODEL_BYTES)
            self.assertEqual(model.license_name, "MIT")

    def test_publisher_recorded_values_for_the_default_model(self) -> None:
        # Değerler OpenCV Zoo'nun Git LFS işaretçisinden (commit 48b10fa...) alınmıştır; bkz. weights.py modül açıklaması.
        model = w.OFFICIAL_MODELS["face_detection_yunet_2026may.onnx"]
        self.assertEqual(model.sha256, "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0")
        self.assertEqual(model.size_bytes, 229738)
        self.assertEqual(w.OPENCV_ZOO_COMMIT, "48b10fa505aeb51fbd127fc60a5c660040e823b7")

    def test_downloaded_model_file_matches_the_pinned_values_when_present(self) -> None:
        if not DEFAULT_MODEL_PATH.is_file():
            self.skipTest("model dosyası indirilmemiş (scripts/download_face_model.py)")
        model = w.OFFICIAL_MODELS[DEFAULT_MODEL_NAME]
        self.assertEqual(DEFAULT_MODEL_PATH.stat().st_size, model.size_bytes)
        self.assertEqual(w.sha256_file(DEFAULT_MODEL_PATH), model.sha256)

    def test_sha256_file_matches_hashlib(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.bin"
            p.write_bytes(PAYLOAD)
            self.assertEqual(w.sha256_file(p), PAYLOAD_SHA)


class DownloadTest(TempDirMixin, unittest.TestCase):
    def test_successful_download_verifies_and_writes_atomically(self) -> None:
        opener = Opener()
        result = w.download_official_model("test_model.onnx", self.tmp, opener=opener, models=registry())
        self.assertEqual((result.downloaded, result.size_bytes, result.sha256, result.path), (True, len(PAYLOAD), PAYLOAD_SHA, self.tmp / "test_model.onnx"))
        self.assertEqual((self.tmp / "test_model.onnx").read_bytes(), PAYLOAD)
        self.assertEqual(len(opener.calls), 1)
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), ["test_model.onnx"], "geçici .part dosyası kalmamalı")

    def test_destination_directory_is_created(self) -> None:
        target = self.tmp / "yeni" / "alt"
        w.download_official_model("test_model.onnx", target, opener=Opener(), models=registry())
        self.assertTrue((target / "test_model.onnx").is_file())

    def test_name_outside_the_allowlist_is_rejected_without_network(self) -> None:
        opener = Opener()
        with self.assertRaises(FaceWeightsDownloadError) as ctx:
            w.download_official_model("baska.onnx", self.tmp, opener=opener, models=registry())
        self.assertIn("izin listesinde değil", str(ctx.exception))
        self.assertEqual(opener.calls, [])

    def test_non_official_url_is_rejected_without_network(self) -> None:
        opener = Opener()
        reg = registry(url="https://example.com/m.onnx")
        with self.assertRaises(FaceWeightsDownloadError) as ctx:
            w.download_official_model("test_model.onnx", self.tmp, opener=opener, models=reg)
        self.assertIn("Resmî olmayan", str(ctx.exception))
        self.assertEqual(opener.calls, [])

    def test_hash_mismatch_is_rejected_and_nothing_is_left_behind(self) -> None:
        with self.assertRaises(FaceWeightsDownloadError) as ctx:
            w.download_official_model("test_model.onnx", self.tmp, opener=Opener(PAYLOAD[:-1] + b"X"), models=registry())
        self.assertIn("SHA-256", str(ctx.exception))
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_size_mismatch_is_rejected(self) -> None:
        reg = registry(size_bytes=len(PAYLOAD) + 1)
        with self.assertRaises(FaceWeightsDownloadError) as ctx:
            w.download_official_model("test_model.onnx", self.tmp, opener=Opener(), models=reg)
        self.assertIn("boyut", str(ctx.exception))
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_tiny_response_such_as_an_lfs_pointer_is_rejected(self) -> None:
        pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 232589\n"
        with self.assertRaises(FaceWeightsDownloadError) as ctx:
            w.download_official_model("test_model.onnx", self.tmp, opener=Opener(pointer), models=registry())
        self.assertIn("çok küçük", str(ctx.exception))
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_network_error_is_wrapped_and_cleans_up(self) -> None:
        with self.assertRaises(FaceWeightsDownloadError) as ctx:
            w.download_official_model("test_model.onnx", self.tmp, opener=Opener(OSError("ağ yok")), models=registry())
        self.assertIn("ağ yok", str(ctx.exception))
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_existing_valid_file_is_not_downloaded_again(self) -> None:
        (self.tmp / "test_model.onnx").write_bytes(PAYLOAD)
        opener = Opener()
        result = w.download_official_model("test_model.onnx", self.tmp, opener=opener, models=registry())
        self.assertFalse(result.downloaded)
        self.assertEqual(opener.calls, [])

    def test_existing_corrupt_file_is_reported_not_overwritten_or_deleted(self) -> None:
        (self.tmp / "test_model.onnx").write_bytes(b"bozuk" * 40000)
        opener = Opener()
        with self.assertRaises(FaceWeightsDownloadError) as ctx:
            w.download_official_model("test_model.onnx", self.tmp, opener=opener, models=registry())
        self.assertIn("yeniden indirin", str(ctx.exception))
        self.assertEqual(opener.calls, [], "izinsiz üzerine yazılmamalı")
        self.assertTrue((self.tmp / "test_model.onnx").exists(), "bozuk dosya sessizce silinmemeli")

    def test_force_replaces_an_existing_file(self) -> None:
        (self.tmp / "test_model.onnx").write_bytes(b"eski" * 40000)
        result = w.download_official_model("test_model.onnx", self.tmp, force=True, opener=Opener(), models=registry())
        self.assertTrue(result.downloaded)
        self.assertEqual((self.tmp / "test_model.onnx").read_bytes(), PAYLOAD)

    def test_failed_forced_download_keeps_the_previous_file(self) -> None:
        old = b"eski" * 40000
        (self.tmp / "test_model.onnx").write_bytes(old)
        with self.assertRaises(FaceWeightsDownloadError):
            w.download_official_model("test_model.onnx", self.tmp, force=True, opener=Opener(OSError("x")), models=registry())
        self.assertEqual((self.tmp / "test_model.onnx").read_bytes(), old)

    def test_download_uses_a_timeout(self) -> None:
        opener = Opener()
        w.download_official_model("test_model.onnx", self.tmp, opener=opener, models=registry())
        self.assertGreater(opener.calls[0][1], 0)


class OnlyTheDownloadModuleUsesTheNetworkTest(unittest.TestCase):
    def test_detector_and_pipeline_modules_never_import_network_libraries(self) -> None:
        for name in ("detector", "pipeline", "cli", "results", "config", "visualization", "benchmark"):
            source = (ROOT / "src" / "isg_vision" / "face_detection" / f"{name}.py").read_text(encoding="utf-8")
            for forbidden in ("urllib", "requests", "http.client", "socket", "urlopen"):
                with self.subTest(module=name, forbidden=forbidden):
                    self.assertNotIn(f"import {forbidden}", source)
                    self.assertNotIn(f"from {forbidden}", source)


if __name__ == "__main__":
    unittest.main()
