#!/usr/bin/env python
"""Apple Silicon (MPS) desteğini kontrol eder.

Yalnızca küçük bir tensor işlemi yapar; benchmark veya model eğitimi yapmaz.
Çıkış kodu: 0 = MPS kullanılabilir, 1 = MPS kullanılamıyor (CPU fallback).
"""

from __future__ import annotations

import platform
import sys


def main() -> int:
    print("=== MPS (Apple Silicon) kontrolü ===")
    print(f"İşlemci mimarisi : {platform.machine()} ({platform.platform()})")

    try:
        import torch
    except ImportError as exc:
        print(f"[HATA] PyTorch import edilemedi: {exc}")
        print("       Sanal ortamı etkinleştirdiğinizden emin olun: source .venv/bin/activate")
        return 1

    print(f"PyTorch sürümü   : {torch.__version__}")

    built = bool(torch.backends.mps.is_built())
    available = bool(torch.backends.mps.is_available())
    print(f"MPS derlenmiş    : {'evet' if built else 'hayır'}")
    print(f"MPS kullanılabilir: {'evet' if available else 'hayır'}")

    if not available:
        if not built:
            print("[UYARI] Bu PyTorch derlemesi MPS desteği içermiyor.")
        else:
            print("[UYARI] MPS derlenmiş ancak kullanılamıyor (macOS 12.3+ ve Apple Silicon gerekir).")
        print("CPU fallback: kod device='cpu' ile çalışmaya devam edebilir (daha yavaş).")
        return 1

    x = torch.ones(3, 3, device="mps")
    y = (x * 2 + 1).sum().item()
    print(f"Test işlemi      : 3x3 birler matrisi * 2 + 1 toplamı = {y} (beklenen 27.0)")
    if y != 27.0:
        print("[HATA] MPS tensor işlemi beklenen sonucu vermedi.")
        return 1

    print("[OK] MPS kullanılabilir. CPU fallback de her zaman mevcuttur.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
