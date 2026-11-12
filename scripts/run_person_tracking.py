#!/usr/bin/env python
"""Kamera veya video dosyasındaki tüm insanları YOLO ile tespit edip ByteTrack ile takip eder (Aşama 4B: çoklu insan takibi).

Pencerede her aktif kişi için kutu ve geçici ``ID n``, tespit/aktif/kayıp iz sayıları, ölçülen FPS, YOLO çıkarım ve takip süreleri gösterilir.
YOLO her karede yalnızca bir kez çalışır. Track ID geçici bir takip numarasıdır, çalışan kimliği DEĞİLDİR; görüntü kaydedilmez; ağ
kullanılmaz; hiçbir paket kurulmaz. Çıkış: Q veya Esc, pencereyi kapatma ya da Ctrl+C.

Örnekler:
    .venv/bin/python scripts/run_person_tracking.py --source camera --camera-index 0 --device mps
    .venv/bin/python scripts/run_person_tracking.py --source camera --camera-index 0 --device cpu
    .venv/bin/python scripts/run_person_tracking.py --source video --video-path /yol/video.mp4 --device mps
    .venv/bin/python scripts/run_person_tracking.py --source camera --device mps --no-display --max-frames 300
    .venv/bin/python scripts/run_person_tracking.py --source camera --device mps --conf 0.25
    .venv/bin/python scripts/run_person_tracking.py --help

Ağırlık dosyası yoksa önce (ağ kullanan tek adım): .venv/bin/python scripts/download_person_model.py
ByteTrack 'lap' paketini ister (requirements.txt'te sabitlidir); kurulu değilse hiçbir şey otomatik kurulmaz.
Mantık src/isg_vision/tracking/cli.py içindedir; bu dosya yalnızca ince bir başlatıcıdır.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.tracking.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
