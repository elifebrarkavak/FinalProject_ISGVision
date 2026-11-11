"""pre-commit hook'unun biyometrik/yasaklı dosyaları engellediğini doğrular.

Tüm denemeler geçici bir Git deposunda yapılır; gerçek proje deposuna dokunulmaz.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOKS_DIR = ROOT / "scripts" / "git-hooks"
HOOK = HOOKS_DIR / "pre-commit"


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=repo, capture_output=True, text=True, env=env,
    )


class PreCommitHookTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)
        git(self.repo, "init", "-q")
        git(self.repo, "config", "core.hooksPath", str(HOOKS_DIR))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def commit_file(self, rel: str, content: bytes = b"x") -> subprocess.CompletedProcess[str]:
        path = self.repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        git(self.repo, "add", "-f", rel)  # -f: .gitignore'u aşsa bile hook'un durdurduğunu test eder
        return git(self.repo, "commit", "-q", "-m", "test")

    def test_hook_is_executable(self) -> None:
        self.assertTrue(HOOK.is_file())
        self.assertTrue(os.access(HOOK, os.X_OK), "pre-commit çalıştırılabilir olmalı")

    def test_blocks_sensitive_files(self) -> None:
        blocked = (
            "data/employees/ali.jpg",
            "data/employees/ali.txt",
            "data/raw/site_safety/img1.jpg",
            "some/where/employees/photo.png",
            "face_db/ali.png",
            "embeddings/ali.npy",
            "ali_face.npy",
            "models/ppe_detection/best.pt",
            "dataset.zip",
            "reports/dataset_audit/visual_samples/site_safety/contact_sheet_01.jpg",
            ".env",
        )
        for rel in blocked:
            with self.subTest(file=rel):
                result = self.commit_file(rel)
                self.assertNotEqual(result.returncode, 0, f"{rel} engellenmeliydi")
                self.assertIn("COMMIT ENGELLENDİ", result.stderr)
                git(self.repo, "reset", "-q")

    def test_allows_normal_files(self) -> None:
        for rel in (
            "src/isg_vision/utils/paths.py",
            "README.md",
            "data/employees/.gitkeep",
            ".env.example",
            "docs/architecture.png",
        ):
            with self.subTest(file=rel):
                result = self.commit_file(rel)
                self.assertEqual(result.returncode, 0, result.stderr)


class VSCodeSettingsTest(unittest.TestCase):
    def test_settings_point_to_project_venv(self) -> None:
        settings = json.loads((ROOT / ".vscode" / "settings.json").read_text(encoding="utf-8"))
        self.assertEqual(
            settings["python.defaultInterpreterPath"], "${workspaceFolder}/.venv/bin/python"
        )
        self.assertTrue(settings["python.testing.unittestEnabled"])


if __name__ == "__main__":
    unittest.main()
