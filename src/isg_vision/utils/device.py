"""PyTorch cihaz seçimi: MPS (Apple Silicon) > CUDA > CPU.

CUDA zorunlu değildir; MPS yoksa CPU'ya düşülür.
"""

from __future__ import annotations


def select_device(preferred: str = "auto") -> str:
    """Kullanılabilir en uygun cihaz adını döndürür ("mps", "cuda" veya "cpu").

    ``preferred`` "auto" ise sırayla MPS, CUDA, CPU denenir. Belirli bir cihaz
    istenip kullanılamıyorsa CPU'ya düşülür.
    """
    import torch  # torch yalnızca çağrı anında import edilir

    available = {
        "mps": bool(torch.backends.mps.is_available()),
        "cuda": bool(torch.cuda.is_available()),
        "cpu": True,
    }
    if preferred != "auto":
        return preferred if available.get(preferred, False) else "cpu"
    for name in ("mps", "cuda"):
        if available[name]:
            return name
    return "cpu"
