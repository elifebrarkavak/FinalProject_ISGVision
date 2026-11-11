"""Kamera testleri için ortak yardımcılar (test modülü değildir; ``test_*.py`` desenine uymaz).

* ``FakeCapture`` / ``FakeFactory``: ``cv2.VideoCapture`` yerine kullanılan sahte kaynak.
  Gerçek kamera açılmaz, kullanıcı izni istenmez.
* ``write_synthetic_video``: geçici dizinde sentetik video üretir (internet/model/veri seti gerekmez).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def make_frame(index: int, width: int = 64, height: int = 48) -> np.ndarray:
    """Tek renkli BGR kare; değeri sıra numarasından türetilir (içerikten sıra doğrulanabilsin diye)."""
    return np.full((height, width, 3), (index * 10) % 256, dtype=np.uint8)


class FakeCapture:
    """``cv2.VideoCapture`` benzeri sahte nesne.

    Args:
        script: Sırayla dönecek okuma sonuçları. ``ndarray`` = başarılı kare, ``None`` = başarısız okuma,
            ``Exception`` örneği = okuma sırasında fırlatılır. Liste bitince okuma başarısız döner.
        opened: ``isOpened()`` değeri.
        props: ``get()`` ile okunacak özellik değerleri (``cv2.CAP_PROP_*`` -> değer).
        accept_set: ``True`` ise ``set()`` değeri ``props`` içine yazar (cihaz isteneni uyguluyor);
            ``False`` ise değer yok sayılır (cihaz desteklemiyor).
    """

    def __init__(
        self,
        script: list[Any] | tuple[Any, ...] = (),
        *,
        opened: bool = True,
        props: dict[int, float] | None = None,
        accept_set: bool = True,
    ) -> None:
        self._script = list(script)
        self._opened = opened
        self.props: dict[int, float] = dict(props or {})
        self.accept_set = accept_set
        self.set_calls: list[tuple[int, float]] = []
        self.read_calls = 0
        self.release_calls = 0

    def isOpened(self) -> bool:  # noqa: N802 (OpenCV adı)
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.read_calls += 1
        if not self._script:
            return False, None
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        if item is None:
            return False, None
        return True, item

    def get(self, prop: int) -> float:
        return float(self.props.get(prop, 0.0))

    def set(self, prop: int, value: float) -> bool:
        self.set_calls.append((prop, value))
        if self.accept_set:
            self.props[prop] = float(value)
        return self.accept_set

    def release(self) -> None:
        self.release_calls += 1
        self._opened = False


class FakeFactory:
    """``cv2.VideoCapture(...)`` çağrısını taklit eder; çağrı argümanlarını kaydeder."""

    def __init__(self, capture: FakeCapture | BaseException) -> None:
        self.capture = capture
        self.calls: list[tuple[Any, ...]] = []

    def __call__(self, *args: Any) -> FakeCapture:
        self.calls.append(args)
        if isinstance(self.capture, BaseException):
            raise self.capture
        return self.capture


def write_synthetic_video(
    path: Path,
    n_frames: int = 10,
    fps: float = 25.0,
    size: tuple[int, int] = (64, 48),
    fourcc: str = "mp4v",
) -> Path:
    """Geçici/sentetik video yazar (kare ``i`` değeri ``i*10``). ``size`` = (genişlik, yükseklik)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Sentetik video yazıcısı açılamadı: {path} ({fourcc})")
    try:
        for i in range(n_frames):
            writer.write(make_frame(i, width=size[0], height=size[1]))
    finally:
        writer.release()
    return path


class FakeClock:
    """Elle ilerletilen sahte monotonic saat (gerçek zaman beklenmeden FPS/oynatma hızı testi için)."""

    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeWindow:
    """``PreviewWindow`` yerine kullanılan sahte pencere; gerçek GUI penceresi açılmaz.

    Args:
        clock: Verilirse ``show(delay_ms)`` saati ``delay_ms`` kadar ilerletir (``waitKey`` beklemesini taklit eder).
        keys: ``{çağrı_sırası(1'den): tuş_kodu}``; diğer çağrılar ``-1`` (tuş yok) döndürür.
        close_after: Bu sayıda ``show`` çağrısından sonra ``closed_by_user`` ``True`` olur.
        raise_on_call: ``{çağrı_sırası: Exception}``; o çağrıda fırlatılır.
        extra_cost_s: Her ``show`` çağrısında saati ek olarak ilerletir (yavaş işlemeyi taklit eder).
    """

    def __init__(
        self,
        clock: FakeClock | None = None,
        *,
        keys: dict[int, int] | None = None,
        close_after: int | None = None,
        raise_on_call: dict[int, Exception] | None = None,
        extra_cost_s: float = 0.0,
    ) -> None:
        self.clock = clock
        self.keys = keys or {}
        self.close_after = close_after
        self.raise_on_call = raise_on_call or {}
        self.extra_cost_s = extra_cost_s
        self.images: list[np.ndarray] = []
        self.delays: list[int] = []
        self.open_error: Exception | None = None
        self.open_calls = 0
        self.close_calls = 0

    def open(self) -> None:
        self.open_calls += 1
        if self.open_error is not None:
            raise self.open_error

    def close(self) -> None:
        self.close_calls += 1

    def show(self, image: np.ndarray, delay_ms: int = 1) -> int:
        call = len(self.images) + 1
        if call in self.raise_on_call:
            raise self.raise_on_call[call]
        self.images.append(image)
        self.delays.append(delay_ms)
        if self.clock is not None:
            self.clock.advance(max(delay_ms / 1000.0, self.extra_cost_s))
        return self.keys.get(call, -1)

    @property
    def closed_by_user(self) -> bool:
        return self.close_after is not None and len(self.images) >= self.close_after
