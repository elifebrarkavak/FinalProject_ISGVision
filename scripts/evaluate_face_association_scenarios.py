#!/usr/bin/env python
"""Sentetik montaj senaryolarında (T1-T9b) yüz–kişi ilişkilendirme DOĞRULUĞUNU ground-truth'a göre ölçer ve eşik duyarlılığını tablolar (Aşama 5B).

GERÇEK YOLO11n + GERÇEK ByteTrack + GERÇEK YuNet çalışır; her senaryonun kare sonuçları bir kez toplanır, ilişkilendirme farklı eşiklerle yeniden (modelsiz) çalıştırılır.
**KANIT SINIRI:** girdi MONTAJDIR (tek fotoğraf karesinden kırpılmış üç gerçek kişi kaydırılır; ``face_association.scenes``): gerçek çoklu-kişi videosu veya gerçek kamera
değildir; buradaki sayılar gerçek kamerada doğruluk, ID switch oranı veya IDF1/HOTA/MOTA DEĞİLDİR. Ağ kullanılmaz, paket kurulmaz, hiçbir görüntü diske yazılmaz.

Örnekler:
    .venv/bin/python scripts/evaluate_face_association_scenarios.py
    .venv/bin/python scripts/evaluate_face_association_scenarios.py --device mps --out results/face_association/scenarios_mps.json
    .venv/bin/python scripts/evaluate_face_association_scenarios.py --scenarios swap_places pass_in_front
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isg_vision.camera.capture import FrameSource  # noqa: E402
from isg_vision.camera.config import SourceConfig  # noqa: E402
from isg_vision.face_association import AssociationConfig, FacePersonAssociator, scenes  # noqa: E402
from isg_vision.face_association.evaluation import SceneEvaluation, evaluate_scene  # noqa: E402
from isg_vision.face_association.pipeline import run_face_association  # noqa: E402
from isg_vision.face_detection.detector import FaceDetector  # noqa: E402
from isg_vision.human_detection.config import DetectionConfig  # noqa: E402
from isg_vision.human_detection.person_detector import PersonDetector  # noqa: E402
from isg_vision.tracking.bytetrack_adapter import PersonTracker  # noqa: E402
from isg_vision.utils.paths import project_path  # noqa: E402

import cv2  # noqa: E402


class ArrayCapture:
    """Bellekteki kareleri ``cv2.VideoCapture`` gibi sunan küçük kaynak (video dosyası sıkıştırması olmadan; ``FrameSource`` ve döngü GERÇEKTİR)."""

    def __init__(self, frames: list, size: tuple[int, int]) -> None:
        self._frames, self._size, self._i = list(frames), size, 0

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def read(self):  # type: ignore[no-untyped-def]
        if self._i >= len(self._frames):
            return False, None
        frame = self._frames[self._i]
        self._i += 1
        return True, frame

    def get(self, prop: int) -> float:
        return {cv2.CAP_PROP_FRAME_WIDTH: float(self._size[0]), cv2.CAP_PROP_FRAME_HEIGHT: float(self._size[1]), cv2.CAP_PROP_FPS: 30.0}.get(prop, 0.0)

    def set(self, prop: int, value: float) -> bool:
        return True

    def release(self) -> None:
        pass


def collect(frames, detector, face_detector):  # type: ignore[no-untyped-def]
    """Kareleri gerçek döngüden geçirip her karenin ``(takip, yüz)`` sonuçlarını toplar."""
    size = (frames[0].shape[1], frames[0].shape[0])
    source = FrameSource(SourceConfig.camera(0, width=size[0], height=size[1]), capture_factory=lambda *_: ArrayCapture(frames, size), max_read_failures=1, sleep=lambda s: None)
    tracker = PersonTracker()
    tracker.load()
    pairs = []
    with source:
        summary = run_face_association(source, detector, tracker, face_detector, FacePersonAssociator(), None, max_frames=len(frames),
                                       on_result=lambda p, d, t, f, a: pairs.append((t, f, a)))
    return pairs, summary


def evaluation_row(name: str, ev: SceneEvaluation) -> dict:
    return {
        "scenario": name, "frames": ev.frames, "correct_pairs": ev.correct_pairs, "wrong_pairs": ev.wrong_pairs, "decoy_pairs": ev.decoy_pairs,
        "decoy_on_person_pairs": ev.decoy_on_person_pairs, "untrusted_decoy_on_person_pairs": ev.untrusted_decoy_on_person_pairs, "expected": ev.expected,
        "recalled": ev.recalled, "recall": ev.recall, "face_not_detected": ev.face_not_detected, "ambiguous": ev.ambiguous, "missed": ev.missed,
        "id_switches": ev.id_report.switches, "id_takeovers": ev.id_report.takeovers, "track_ids_per_actor": {k: list(v) for k, v in ev.track_ids_per_actor.items()},
        "breaks": ev.breaks, "wrong_examples": list(ev.wrong_examples),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"), help="YOLO cihazı (varsayılan: cpu; kayıtlı ölçümler CPU ile deterministiktir)")
    parser.add_argument("--scenarios", nargs="+", default=list(scenes.SCENARIOS), choices=list(scenes.SCENARIOS))
    parser.add_argument("--margins", nargs="+", type=float, default=[0.05, 0.15, 0.30], help="Duyarlılık: ambiguity_margin değerleri")
    parser.add_argument("--containments", nargs="+", type=float, default=[0.6, 0.8, 0.95], help="Duyarlılık: min_containment değerleri")
    parser.add_argument("--out", type=Path, default=project_path("results", "face_association", "scenario_evaluation.json"))
    args = parser.parse_args(argv)

    detector = PersonDetector(DetectionConfig(device=args.device))
    detector.load()
    face_detector = FaceDetector()
    face_detector.load()
    actors = scenes.load_actors()
    print(f"[BİLGİ] YOLO cihazı: {detector.device} | SENTETİK montaj senaryoları (gerçek çoklu-kişi videosu değildir)")
    report: dict = {"device": detector.device, "synthetic": True, "scenarios": [], "sensitivity": []}
    header = f"{'senaryo':<18}{'kare':>5}{'doğru':>7}{'YANLIŞ':>8}{'sahte':>7}{'sahte/kişi':>11}{'beklenen':>9}{'doğru-bağ':>10}{'yüz-yok':>8}{'belirsiz':>9}{'kaçan':>6}{'IDsw':>5}{'devral':>7}"
    print(header)
    cache = {}
    for name in args.scenarios:
        frames = scenes.render_scenario(actors, scenes.SCENARIOS[name]())
        pairs, summary = collect([f.image for f in frames], detector, face_detector)
        cache[name] = (frames, pairs)
        ev = evaluate_scene(frames, [a for _, _, a in pairs])
        row = evaluation_row(name, ev)
        row["fps_end_to_end_headless"] = summary.end_to_end_fps
        report["scenarios"].append(row)
        print(f"{name:<18}{ev.frames:>5}{ev.correct_pairs:>7}{ev.wrong_pairs:>8}{ev.decoy_pairs:>7}{ev.decoy_on_person_pairs:>11}{ev.expected:>9}{ev.recalled:>10}"
              f"{ev.face_not_detected:>8}{ev.ambiguous:>9}{ev.missed:>6}{ev.id_report.switches:>5}{ev.id_report.takeovers:>7}  ids={row['track_ids_per_actor']} breaks={ev.breaks}")
        for example in ev.wrong_examples[:3]:
            print(f"      YANLIŞ: {example}")

    print("\n[Duyarlılık] ilişkilendirme aynı takip/yüz sonuçlarıyla yeniden çalıştırıldı (model yok); toplamlar tüm senaryolar:")
    print(f"{'min_containment':>16}{'margin':>8}{'doğru':>7}{'YANLIŞ':>8}{'sahte/kişi':>11}{'beklenen':>9}{'doğru-bağ':>10}{'belirsiz':>9}{'kaçan':>6}")
    for containment in args.containments:
        for margin in args.margins:
            totals = dict(correct=0, wrong=0, decoy_on=0, expected=0, recalled=0, ambiguous=0, missed=0)
            for name, (frames, pairs) in cache.items():
                associator = FacePersonAssociator(AssociationConfig(min_containment=containment, ambiguity_margin=margin))
                ev = evaluate_scene(frames, [associator.associate(t, f) for t, f, _ in pairs])
                totals["correct"] += ev.correct_pairs
                totals["wrong"] += ev.wrong_pairs
                totals["decoy_on"] += ev.decoy_on_person_pairs
                totals["expected"] += ev.expected
                totals["recalled"] += ev.recalled
                totals["ambiguous"] += ev.ambiguous
                totals["missed"] += ev.missed
            report["sensitivity"].append({"min_containment": containment, "ambiguity_margin": margin, **totals})
            print(f"{containment:>16}{margin:>8}{totals['correct']:>7}{totals['wrong']:>8}{totals['decoy_on']:>11}{totals['expected']:>9}{totals['recalled']:>10}{totals['ambiguous']:>9}{totals['missed']:>6}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[BİLGİ] JSON: {args.out}")
    detector.close()
    face_detector.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
