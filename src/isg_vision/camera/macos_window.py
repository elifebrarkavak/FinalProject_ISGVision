"""macOS (Cocoa): OpenCV penceresinin kırmızı kapatma düğmesini etkinleştirir.

**Sorun (ölçülerek doğrulandı, OpenCV 5.0.0 / macOS arm64):** OpenCV'nin Cocoa arka ucu ``cv2.namedWindow`` ile oluşturduğu
``NSWindow``'u ``styleMask = 0xd`` (Titled | Miniaturizable | Resizable) ile açar; ``Closable`` biti (``0x2``) **yoktur**.
Bu yüzden pencerenin kırmızı kapatma düğmesi devre dışıdır (``NSButton.isEnabled == NO``) ve tıklamak **hiçbir şey yapmaz**.
Bu, ``WINDOW_AUTOSIZE``, ``WINDOW_NORMAL`` ve ``WINDOW_GUI_NORMAL`` bayrakları için aynıdır. Yani sorun "kapanmanın algılanmaması"
değil, kapanmanın hiç gerçekleşmemesidir; ``cv2.getWindowProperty`` kapanmayan pencere için doğru olarak ``1.0`` döndürür.

**Düzeltme:** pencere oluşturulduktan sonra ``Closable`` biti ``NSWindow.setStyleMask:`` ile eklenir. Bundan sonra düğme etkindir;
tıklandığında pencere kapanır ve ``cv2.getWindowProperty(..., WND_PROP_VISIBLE)`` ``0.0`` döndürür; ``PreviewWindow.closed_by_user``
bunu zaten algılar.

Uygulama ``ctypes`` ile Objective-C çalışma zamanını kullanır (``pyobjc`` bağımlılığı eklenmez). Her hata ``False`` ile sonuçlanır;
pencere işlevi isteğe bağlı bir iyileştirmedir ve Q/Esc/Ctrl+C çıkışlarını etkilemez. Modül import edildiğinde hiçbir sistem
kütüphanesi yüklenmez ve pencere açılmaz; Objective-C ancak ``enable_close_button`` çağrılınca (macOS'ta, ana iş parçacığında) kullanılır.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from typing import Any, Protocol

NS_WINDOW_STYLE_MASK_CLOSABLE = 1 << 1  # NSWindowStyleMaskClosable

_LIBOBJC = "/usr/lib/libobjc.A.dylib"
_APPKIT = "/System/Library/Frameworks/AppKit.framework/AppKit"


class WindowRuntime(Protocol):
    """``enable_close_button`` için gereken en küçük pencere arayüzü (testlerde sahte sürümü kullanılır)."""

    def find_window(self, title: str) -> Any | None:
        """Başlığı ``title`` olan pencereyi bulur; yoksa ``None``."""

    def style_mask(self, window: Any) -> int:
        """Pencerenin ``styleMask`` değerini döndürür."""

    def set_style_mask(self, window: Any, mask: int) -> None:
        """Pencerenin ``styleMask`` değerini ayarlar."""


class CocoaRuntime:
    """Gerçek ``NSWindow`` nesneleri için ``ctypes`` + Objective-C çalışma zamanı sarmalayıcısı.

    Yalnızca macOS'ta ve ana iş parçacığında kullanılmalıdır. ``NSApp`` henüz oluşturulmamışsa (ör. hiç OpenCV penceresi yoksa)
    ``NSApplication`` **oluşturulmaz**; pencere yok sayılır.
    """

    def __init__(self) -> None:
        self._objc = ctypes.cdll.LoadLibrary(_LIBOBJC)
        self._appkit = ctypes.cdll.LoadLibrary(_APPKIT)
        self._objc.sel_registerName.restype = ctypes.c_void_p
        self._objc.sel_registerName.argtypes = [ctypes.c_char_p]

    def _send(self, receiver: Any, selector: str, *args: Any, restype: Any = ctypes.c_void_p, argtypes: tuple = ()) -> Any:
        function = ctypes.cast(
            self._objc.objc_msgSend, ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)
        )
        return function(receiver, self._objc.sel_registerName(selector.encode("ascii")), *args)

    def find_window(self, title: str) -> Any | None:
        app = ctypes.c_void_p.in_dll(self._appkit, "NSApp").value
        if not app:  # NSApplication yok → OpenCV penceresi de yok; yenisini oluşturma
            return None
        windows = self._send(app, "windows")
        if not windows:
            return None
        count = self._send(windows, "count", restype=ctypes.c_ulong)
        for index in range(count):
            window = self._send(windows, "objectAtIndex:", index, argtypes=(ctypes.c_ulong,))
            ns_title = self._send(window, "title") if window else None
            raw = self._send(ns_title, "UTF8String", restype=ctypes.c_char_p) if ns_title else None
            if raw is not None and raw.decode("utf-8", "replace") == title:
                return window
        return None

    def style_mask(self, window: Any) -> int:
        return int(self._send(window, "styleMask", restype=ctypes.c_ulong))

    def set_style_mask(self, window: Any, mask: int) -> None:
        self._send(window, "setStyleMask:", mask, restype=None, argtypes=(ctypes.c_ulong,))


def enable_close_button(
    title: str,
    *,
    runtime: WindowRuntime | None = None,
    platform: str | None = None,
) -> bool:
    """Başlığı ``title`` olan OpenCV penceresinin kırmızı kapatma düğmesini etkinleştirir.

    Args:
        title: Pencere başlığı (OpenCV'de pencere adı).
        runtime: Test için sahte çalışma zamanı; verilmezse gerçek ``CocoaRuntime`` kullanılır.
        platform: Test için platform adı; verilmezse ``sys.platform``.

    Returns:
        Kapatma düğmesi etkin duruma getirildiyse (veya zaten etkinse) ``True``. macOS dışında, ana iş parçacığı dışında,
        pencere bulunamazsa veya herhangi bir hata olursa ``False``; bu fonksiyon **asla istisna fırlatmaz**.
    """
    if (platform or sys.platform) != "darwin":
        return False
    if threading.current_thread() is not threading.main_thread():  # Cocoa pencere işlemleri ana iş parçacığında yapılır
        return False
    try:
        runtime = runtime if runtime is not None else CocoaRuntime()
        window = runtime.find_window(title)
        if window is None:
            return False
        mask = runtime.style_mask(window)
        if not mask & NS_WINDOW_STYLE_MASK_CLOSABLE:
            runtime.set_style_mask(window, mask | NS_WINDOW_STYLE_MASK_CLOSABLE)
            mask = runtime.style_mask(window)  # etkisini doğrula
        return bool(mask & NS_WINDOW_STYLE_MASK_CLOSABLE)
    except Exception:  # noqa: BLE001 - OSError (kütüphane yok), AttributeError, ctypes hataları vb.; pencere işlevi isteğe bağlıdır
        return False
