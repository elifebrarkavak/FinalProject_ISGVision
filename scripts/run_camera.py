#!/usr/bin/env python
"""Kamera veya video dosyasından görüntü alır ve canlı pencerede gösterir (Aşama 3: kamera altyapısı).

Pencerede kaynak, ölçülen FPS, gerçek çözünürlük ve durum bilgisi gösterilir. Hiçbir yapay zekâ modeli
çalıştırılmaz; görüntü kaydedilmez. Çıkış: Q veya Esc, pencereyi kapatma ya da Ctrl+C.

Örnekler:
    .venv/bin/python scripts/run_camera.py --source camera
    .venv/bin/python scripts/run_camera.py --source camera --camera-index 1 --width 640 --height 480
    .venv/bin/python scripts/run_camera.py --source video --video-path /yol/video.mp4
    .venv/bin/python scripts/run_camera.py --help

Mantık src/isg_vision/camera/cli.py içindedir; bu dosya yalnızca ince bir başlatıcıdır.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.camera.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
