#!/usr/bin/env python
"""Kamera veya video dosyasında insanları takip eder, yüzleri tespit eder ve yüzleri geçici Track ID'lere bağlar (Aşama 5B).

Pipeline (her model kare başına yalnızca BİR kez): YOLO11n insan tespiti -> ByteTrack geçici Track ID -> YuNet yüz tespiti -> geometrik yüz–kişi
ilişkilendirme. Pencerede: insan kutusu + Track ID, yüz kutusu + bağlı Track ID (eşleşmeyen/belirsiz yüze ID yazılmaz), toplam insan/yüz/eşleşme
sayıları, FPS, YOLO/ByteTrack/YuNet/ilişkilendirme süreleri. Belirsizde eşleştirilmez. Gerçek çalışan kimliği, yüz tanıma, embedding veya biyometrik
veri YOKTUR; Track ID kimlik değildir; yüz–kişi bağı canlı insan doğrulaması değildir (poster/ekrandaki yüz de bağlanabilir). Görüntü kaydedilmez,
ağ kullanılmaz, paket kurulmaz. Çıkış: Q veya Esc, pencereyi kapatma ya da Ctrl+C.

Örnekler:
    .venv/bin/python scripts/run_face_person_association.py --source camera --camera-index 0 --device mps
    .venv/bin/python scripts/run_face_person_association.py --source video --video-path /yol/video.mp4 --device mps
    .venv/bin/python scripts/run_face_person_association.py --source camera --device mps --no-display --max-frames 300
    .venv/bin/python scripts/run_face_person_association.py --source camera --device mps --opencv-threads restore
    .venv/bin/python scripts/run_face_person_association.py --help

Ağırlıklar yoksa önce (ağ kullanan tek adımlar): scripts/download_person_model.py ve scripts/download_face_model.py
Mantık src/isg_vision/face_association/cli.py içindedir; bu dosya yalnızca ince bir başlatıcıdır.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.face_association.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
