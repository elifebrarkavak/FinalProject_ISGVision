"""Bağımsız FPS ölçümü.

İki farklı kavram vardır ve karıştırılmamalıdır:

* **Kaynak FPS** — kameranın veya video dosyasının *bildirdiği* kare hızı (``FrameSource.source_fps``).
* **Ölçülen FPS** — uygulamanın gerçekten saniyede kaç kareyi işlediği/gösterdiği; bu sınıf ölçer.

``FpsMeter`` kamera veya OpenCV'ye bağımlı değildir; yalnızca zaman damgaları alır. Zaman kaynağı olarak
``time.monotonic`` kullanılır (sistem saati ayarlarından etkilenmez).
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable


class FpsMeter:
    """Kayan pencere ortalamasıyla FPS ölçer.

    Args:
        window: Anlık FPS için kullanılan son kare sayısı (en az 2). Tek kare aralığı yerine pencere
            ortalaması kullanılır; böylece ekrandaki değer aşırı dalgalanmaz.
        clock: Zaman kaynağı (varsayılan ``time.monotonic``; testlerde sahte saat verilebilir).
    """

    def __init__(self, window: int = 30, clock: Callable[[], float] = time.monotonic) -> None:
        if isinstance(window, bool) or not isinstance(window, int) or window < 2:
            raise ValueError(f"FPS pencere boyutu en az 2 olan bir tam sayı olmalıdır: {window!r}")
        self._window = window
        self._clock = clock
        self._times: deque[float] = deque(maxlen=window)
        self._first: float | None = None
        self._frames = 0

    def tick(self, now: float | None = None) -> float:
        """Bir karenin işlendiğini kaydeder ve güncel (pencere) FPS değerini döndürür.

        Args:
            now: Kare zamanı (monotonic saniye); verilmezse saat okunur. Geriye gitmemelidir.
        """
        now = self._clock() if now is None else float(now)
        if self._times and now < self._times[-1]:
            raise ValueError("FPS zaman damgaları geriye gidemez (monotonic zaman kullanın).")
        if self._first is None:
            self._first = now
        self._times.append(now)
        self._frames += 1
        return self.fps

    @property
    def fps(self) -> float:
        """Son ``window`` karenin ortalama FPS değeri. En az iki kare ölçülene kadar 0.0."""
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0

    @property
    def average_fps(self) -> float:
        """İlk kareden bu yana tüm ölçüm süresinin ortalama FPS değeri (oturum özeti için)."""
        if self._first is None or self._frames < 2:
            return 0.0
        span = self._times[-1] - self._first
        return (self._frames - 1) / span if span > 0 else 0.0

    @property
    def frames(self) -> int:
        """Kaydedilen toplam kare sayısı."""
        return self._frames

    @property
    def elapsed(self) -> float:
        """İlk ve son kare arasında geçen süre (saniye)."""
        if self._first is None:
            return 0.0
        return self._times[-1] - self._first

    def reset(self) -> None:
        """Tüm ölçümleri sıfırlar."""
        self._times.clear()
        self._first = None
        self._frames = 0
