#!/usr/bin/env python
"""Kamera veya video dosyasındaki tüm yüzleri YuNet ile tespit eder ve canlı gösterir (Aşama 5A: çoklu yüz tespiti).

Pencerede yüz kutuları, güven skorları, toplam yüz sayısı, ölçülen FPS ve çıkarım süresi gösterilir. Yüzler bir kişiye veya Track
ID'ye BAĞLANMAZ (Aşama 5B'nin işi); yüz tanıma, çalışan adı veya biyometrik veri yoktur; görüntü kaydedilmez; ağ kullanılmaz.
Yüz tespiti, canlı insan doğrulaması DEĞİLDİR: poster/fotoğraf/ekrandaki yüz de tespit edilebilir. Çıkış: Q veya Esc,
pencereyi kapatma ya da Ctrl+C.

Örnekler:
    .venv/bin/python scripts/run_face_detection.py --source camera
    .venv/bin/python scripts/run_face_detection.py --source video --video-path /yol/video.mp4
    .venv/bin/python scripts/run_face_detection.py --source camera --conf 0.7 --detect-width 960
    .venv/bin/python scripts/run_face_detection.py --source camera --no-display --max-frames 100
    .venv/bin/python scripts/run_face_detection.py --help

Model dosyası yoksa önce (ağ kullanan tek adım): .venv/bin/python scripts/download_face_model.py
Mantık src/isg_vision/face_detection/cli.py içindedir; bu dosya yalnızca ince bir başlatıcıdır.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.face_detection.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
