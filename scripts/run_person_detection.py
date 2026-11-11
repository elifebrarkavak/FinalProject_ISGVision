#!/usr/bin/env python
"""Kamera veya video dosyasındaki tüm insanları YOLO ile tespit eder ve canlı gösterir (Aşama 4A: çoklu insan tespiti).

Pencerede insan kutuları, güven skorları, kişi sayısı, ölçülen FPS ve çıkarım süresi gösterilir. İnsanlara takip numarası
veya kimlik atanmaz; görüntü kaydedilmez; ağ kullanılmaz. Çıkış: Q veya Esc, pencereyi kapatma ya da Ctrl+C.

Örnekler:
    .venv/bin/python scripts/run_person_detection.py --source camera
    .venv/bin/python scripts/run_person_detection.py --source video --video-path /yol/video.mp4
    .venv/bin/python scripts/run_person_detection.py --source camera --device mps --conf 0.35
    .venv/bin/python scripts/run_person_detection.py --source camera --no-display --max-frames 100
    .venv/bin/python scripts/run_person_detection.py --help

Ağırlık dosyası yoksa önce (ağ kullanan tek adım): .venv/bin/python scripts/download_person_model.py
Mantık src/isg_vision/human_detection/cli.py içindedir; bu dosya yalnızca ince bir başlatıcıdır.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.human_detection.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
