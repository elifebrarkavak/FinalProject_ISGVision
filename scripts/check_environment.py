#!/usr/bin/env python
"""Geliştirme ortamını doğrular.

Veri seti indirmez, kamera açmaz, yapay zekâ modeli yüklemez.
Çıkış kodu: 0 = kritik hata yok, 1 = en az bir kritik hata var.
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

REQUIRED_PYTHON = (3, 11)
# (görünen ad, import adı)
LIBRARIES = (
    ("numpy", "numpy"),
    ("opencv", "cv2"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("ultralytics", "ultralytics"),
    ("pyyaml", "yaml"),
    ("mediapipe", "mediapipe"),
)

passed: list[str] = []
failed: list[str] = []  # kritik
warnings: list[str] = []


def ok(msg: str) -> None:
    passed.append(msg)


def fail(msg: str) -> None:
    failed.append(msg)


def check_python() -> None:
    v = sys.version_info
    text = f"Python {v.major}.{v.minor}.{v.micro} ({sys.executable})"
    if (v.major, v.minor) == REQUIRED_PYTHON:
        ok(text)
    else:
        fail(f"{text} -> Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]} gerekli")
    if sys.prefix == sys.base_prefix:
        warnings.append("Sanal ortam (.venv) etkin görünmüyor: source .venv/bin/activate")
    else:
        ok(f"Sanal ortam etkin: {sys.prefix}")


def check_platform() -> None:
    ok(f"İşletim sistemi: {platform.platform()}")
    ok(f"İşlemci mimarisi: {platform.machine()}")
    if platform.system() == "Darwin" and platform.machine() != "arm64":
        warnings.append("Apple Silicon (arm64) değil; Rosetta altında çalışıyor olabilir.")


def check_libraries() -> None:
    for display, module_name in LIBRARIES:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - her import hatası raporlanmalı
            fail(f"{display}: import edilemedi ({type(exc).__name__}: {exc})")
            continue
        ok(f"{display}: {getattr(module, '__version__', 'sürüm bilinmiyor')}")


def check_package() -> None:
    try:
        import isg_vision
    except Exception as exc:  # noqa: BLE001
        fail(f"isg_vision paketi import edilemedi ({type(exc).__name__}: {exc})")
        return
    ok(f"isg_vision paketi: {isg_vision.__version__}")


def check_structure() -> None:
    try:
        from isg_vision.utils.paths import CONFIG_FILES, REQUIRED_DIRS, REQUIRED_FILES
    except Exception as exc:  # noqa: BLE001
        fail(f"Klasör listesi okunamadı ({exc})")
        return
    missing_dirs = [d for d in REQUIRED_DIRS if not (PROJECT_ROOT / d).is_dir()]
    if missing_dirs:
        fail("Eksik klasörler: " + ", ".join(missing_dirs))
    else:
        ok(f"Klasör yapısı tam ({len(REQUIRED_DIRS)} klasör)")
    missing_files = [f for f in REQUIRED_FILES if not (PROJECT_ROOT / f).is_file()]
    if missing_files:
        fail("Eksik dosyalar: " + ", ".join(missing_files))
    else:
        ok(f"Gerekli dosyalar tam ({len(REQUIRED_FILES)} dosya)")
    missing_cfg = [f for f in CONFIG_FILES if not (PROJECT_ROOT / f).is_file()]
    if missing_cfg:
        fail("Eksik yapılandırma dosyaları: " + ", ".join(missing_cfg))
    else:
        ok("Yapılandırma dosyaları mevcut: " + ", ".join(CONFIG_FILES))


def check_git_hook() -> None:
    """pre-commit hook'unun etkin olup olmadığına bakar (kritik değil, uyarı)."""
    import subprocess

    if not (PROJECT_ROOT / ".git").exists():
        warnings.append("Git deposu bulunamadı (git init yapılmamış).")
        return
    try:
        value = subprocess.run(
            ["git", "config", "--get", "core.hooksPath"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
        ).stdout.strip()
    except OSError:
        warnings.append("git çalıştırılamadı; hook durumu kontrol edilemedi.")
        return
    if value == "scripts/git-hooks":
        ok("Git pre-commit hook etkin (core.hooksPath=scripts/git-hooks)")
    else:
        warnings.append(
            "Git pre-commit hook etkin değil: git config core.hooksPath scripts/git-hooks"
        )


def check_mps() -> None:
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch eksikliği zaten kritik olarak raporlandı
        return
    built = torch.backends.mps.is_built()
    available = torch.backends.mps.is_available()
    if available:
        ok("PyTorch MPS: kullanılabilir")
    else:
        # Kritik değil: CPU fallback mevcut.
        warnings.append(
            f"PyTorch MPS kullanılamıyor (derlenmiş={built}); CPU fallback kullanılacak."
        )


def main() -> int:
    print("=== ISG Vision ortam kontrolü ===\n")
    check_python()
    check_platform()
    check_libraries()
    check_package()
    check_structure()
    check_git_hook()
    check_mps()

    print(f"BAŞARILI ({len(passed)}):")
    for line in passed:
        print(f"  [OK]    {line}")
    print(f"\nUYARI ({len(warnings)}):")
    for line in warnings:
        print(f"  [UYARI] {line}")
    print(f"\nKRİTİK HATA ({len(failed)}):")
    for line in failed:
        print(f"  [HATA]  {line}")

    if failed:
        print("\nSonuç: BAŞARISIZ - kritik eksikler var.")
        return 1
    print("\nSonuç: BAŞARILI - geliştirme ortamı hazır.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
