"""Aşama 1 temel testleri: klasör yapısı, paket importu, ortam."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isg_vision.utils.paths import (  # noqa: E402
    CONFIG_FILES,
    PROJECT_ROOT,
    REQUIRED_DIRS,
    REQUIRED_FILES,
)

SUBPACKAGES = (
    "camera",
    "human_detection",
    "tracking",
    "face_detection",
    "face_recognition",
    "ppe_detection",
    "integration",
    "storage",
    "utils",
)


class ProjectStructureTest(unittest.TestCase):
    def test_project_root_is_repo_root(self) -> None:
        self.assertEqual(PROJECT_ROOT, ROOT)

    def test_required_directories_exist(self) -> None:
        missing = [d for d in REQUIRED_DIRS if not (ROOT / d).is_dir()]
        self.assertEqual(missing, [], f"Eksik klasörler: {missing}")

    def test_required_files_exist(self) -> None:
        missing = [f for f in REQUIRED_FILES if not (ROOT / f).is_file()]
        self.assertEqual(missing, [], f"Eksik dosyalar: {missing}")

    def test_config_files_exist(self) -> None:
        for name in CONFIG_FILES:
            self.assertTrue((ROOT / name).is_file(), name)

    def test_subpackages_have_init(self) -> None:
        for name in SUBPACKAGES:
            self.assertTrue(
                (ROOT / "src" / "isg_vision" / name / "__init__.py").is_file(), name
            )

    def test_gitignore_protects_sensitive_paths(self) -> None:
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for pattern in ("/.venv/", "/data/**", "*.pt", ".env", "*.npy"):
            self.assertIn(pattern, text, f".gitignore içinde '{pattern}' yok")


class PythonEnvironmentTest(unittest.TestCase):
    def test_supported_python_version(self) -> None:
        self.assertGreaterEqual(sys.version_info[:2], (3, 11))
        self.assertLess(sys.version_info[:2], (3, 13))


class PackageImportTest(unittest.TestCase):
    def test_package_importable(self) -> None:
        import isg_vision

        self.assertTrue(isg_vision.__version__)

    def test_default_config_is_valid_yaml(self) -> None:
        import yaml

        data = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
        self.assertIn("paths", data)
        for value in data["paths"].values():
            self.assertFalse(Path(value).is_absolute(), "Yollar göreli olmalı")


if __name__ == "__main__":
    unittest.main()
