"""Performans/iş parçacığı ölçüm aracı: matris çalıştırma, özetleme, alt süreç çalıştırıcı, video üretimi [Mock/birim: sahte çalıştırıcı; gerçek ölçüm yok].

Gerçek ölçümler ``scripts/benchmark_face_person_association.py`` ile yapılır ve ``docs/face_association.md``/``reports/face_association/STAGE_5B_REPORT.md``'ye
yazılır; bu dosya yalnızca ölçüm ARACININ mantığını sınar (sahte sayılar HİÇBİR YERE ölçüm olarak yazılmaz).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isg_vision.face_association import benchmark, scenes  # noqa: E402

SCRIPT = ROOT / "scripts" / "benchmark_face_person_association.py"


def fake_pipeline_result(device: str, threads: str, *, fps: float, face_ms: float, signature: str = "abc") -> dict:
    lat = lambda v: {"median_ms": v, "mean_ms": v, "p95_ms": v, "n": 10}  # noqa: E731
    return {"mode": "pipeline", "device": device, "opencv_threads_request": threads, "end_to_end_fps": fps, "cv2_threads_after_setting": 10 if threads == "restore" else 1,
            "latency": {"yolo": lat(15.0), "tracking": lat(0.3), "face": lat(face_ms), "association": lat(0.1), "compute": lat(15.4 + face_ms)}, "signature": signature}


class FakeRunner:
    def __init__(self, *, signature_for=lambda device, threads: "abc") -> None:  # type: ignore[no-untyped-def]
        self.calls: list[list[str]] = []
        self.signature_for = signature_for

    def __call__(self, args: list[str]) -> dict:
        self.calls.append(args)
        if args[1] == "pipeline":
            device, threads = args[args.index("--device") + 1], args[args.index("--opencv-threads") + 1]
            return fake_pipeline_result(device, threads, fps=45.0 if threads == "restore" else 20.0, face_ms=10.0 if threads == "restore" else 29.0,
                                        signature=self.signature_for(device, threads))
        return {"mode": "probe", "set_threads": int(args[args.index("--set-threads") + 1]), "stages": []}


class MatrixTest(unittest.TestCase):
    def test_runs_every_variant_for_every_repeat_and_every_probe(self) -> None:
        runner = FakeRunner()
        report = benchmark.run_matrix(runner, Path("v.mp4"), devices=("mps", "cpu"), threads=("keep", "restore"), repeats=3, probe_thread_values=(0, 4, -1), clock=lambda: 1.0)
        pipeline_calls = [c for c in runner.calls if c[1] == "pipeline"]
        probe_calls = [c for c in runner.calls if c[1] == "probe"]
        self.assertEqual(len(pipeline_calls), 2 * 2 * 3)
        self.assertEqual(len(probe_calls), 3)
        self.assertEqual(len(report["pipeline_runs"]), 12)
        self.assertEqual([r["repeat"] for r in report["pipeline_runs"][:4]], [1, 1, 1, 1])
        self.assertIn("SENTETİK", report["synthetic_disclaimer"])
        self.assertEqual(report["video"], "v.mp4")

    def test_variant_order_rotates_between_repeats_to_balance_order_effects(self) -> None:
        runner = FakeRunner()
        benchmark.run_matrix(runner, Path("v.mp4"), devices=("mps", "cpu"), threads=("keep", "restore"), repeats=2, probe_thread_values=())
        firsts = [(c[c.index("--device") + 1], c[c.index("--opencv-threads") + 1]) for c in runner.calls if c[1] == "pipeline"]
        self.assertNotEqual(firsts[0], firsts[4], "ikinci tekrar farklı varyantla başlamalı")

    def test_arguments_are_passed_to_the_worker(self) -> None:
        runner = FakeRunner()
        benchmark.run_matrix(runner, Path("/x/v.mp4"), devices=("cpu",), threads=("restore",), repeats=1, max_frames=50, probe_thread_values=(2,))
        pipeline = runner.calls[0]
        self.assertEqual(pipeline[:2], ["--worker", "pipeline"])
        self.assertIn("/x/v.mp4", pipeline)
        self.assertEqual(pipeline[pipeline.index("--max-frames") + 1], "50")
        self.assertEqual(runner.calls[1][:2], ["--worker", "probe"])
        self.assertEqual(runner.calls[1][runner.calls[1].index("--set-threads") + 1], "2")

    def test_invalid_repeats(self) -> None:
        with self.assertRaises(ValueError):
            benchmark.run_matrix(FakeRunner(), Path("v.mp4"), repeats=0)

    def test_summary_uses_medians_per_variant_and_reports_signature_consistency(self) -> None:
        runs = [fake_pipeline_result("cpu", "keep", fps=f, face_ms=29.0) for f in (19.0, 20.0, 40.0)] + \
               [fake_pipeline_result("cpu", "restore", fps=f, face_ms=10.0, signature=s) for f, s in ((33.0, "a"), (34.0, "a"), (35.0, "b"))]
        rows = {(r["device"], r["opencv_threads"]): r for r in benchmark.summarize(runs)}
        keep, restore = rows[("cpu", "keep")], rows[("cpu", "restore")]
        self.assertEqual(keep["end_to_end_fps_median"], 20.0, "ortalama değil MEDYAN")
        self.assertEqual((keep["runs"], keep["face_ms_median"]), (3, 29.0))
        self.assertEqual(keep["signatures"], ["abc"])
        self.assertEqual(restore["signatures"], ["a", "b"], "imza farkı görünür kalmalı")
        text = benchmark.summary_to_markdown(benchmark.summarize(runs))
        self.assertIn("| cpu | keep | 3 | 20.0 |", text)
        self.assertIn("HAYIR", text)

    def test_markdown_handles_missing_values(self) -> None:
        text = benchmark.summary_to_markdown([{"device": "cpu", "opencv_threads": "keep", "runs": 1, "end_to_end_fps_median": None, "yolo_ms_median": None,
                                               "tracking_ms_median": None, "face_ms_median": None, "association_ms_median": None, "compute_ms_median": None,
                                               "signatures": ["x"]}])
        self.assertIn("—", text)


class SignatureTest(unittest.TestCase):
    def result(self, frame: int, people: list[tuple[int, int | None, str]]):  # type: ignore[no-untyped-def]
        return SimpleNamespace(frame_index=frame, persons=[SimpleNamespace(track_id=t, face_index=f, status=s) for t, f, s in people])

    def test_signature_is_stable_and_sensitive_to_associations(self) -> None:
        a = [self.result(0, [(1, 0, "matched")]), self.result(1, [(1, 0, "matched")])]
        same = [self.result(0, [(1, 0, "matched")]), self.result(1, [(1, 0, "matched")])]
        different = [self.result(0, [(1, 0, "matched")]), self.result(1, [(1, None, "no_face")])]
        self.assertEqual(benchmark.result_signature(a), benchmark.result_signature(same))
        self.assertNotEqual(benchmark.result_signature(a), benchmark.result_signature(different))
        self.assertEqual(len(benchmark.result_signature(a)), 16)


class SubprocessRunnerTest(unittest.TestCase):
    def make_script(self, directory: Path, body: str) -> Path:
        path = directory / "w.py"
        path.write_text(textwrap.dedent(body), encoding="utf-8")
        return path

    def test_returns_the_last_json_line_ignoring_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = self.make_script(Path(tmp), 'import sys, json\nprint("gürültü")\nprint(json.dumps({"ok": 1, "args": sys.argv[1:]}))\n')
            result = benchmark.subprocess_runner(script)(["--worker", "probe"])
        self.assertEqual(result, {"ok": 1, "args": ["--worker", "probe"]})

    def test_failure_raises_with_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = self.make_script(Path(tmp), 'import sys\nsys.stderr.write("patladı")\nsys.exit(3)\n')
            with self.assertRaises(RuntimeError) as ctx:
                benchmark.subprocess_runner(script)([])
        self.assertIn("patladı", str(ctx.exception))
        self.assertIn("kod 3", str(ctx.exception))

    def test_missing_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = self.make_script(Path(tmp), 'print("json yok")\n')
            with self.assertRaises(RuntimeError):
                benchmark.subprocess_runner(script)([])


class VideoAndEnvironmentTest(unittest.TestCase):
    def test_environment_info_has_the_basics(self) -> None:
        info = benchmark.environment_info()
        for key in ("platform", "python", "cv2", "cpu_count"):
            self.assertIn(key, info)

    def test_benchmark_video_is_written_and_readable(self) -> None:
        try:
            scenes.load_actors()
        except scenes.SceneError as exc:
            self.skipTest(f"örnek görüntüler yok: {exc}")
        with tempfile.TemporaryDirectory() as tmp:
            path = benchmark.make_benchmark_video(Path(tmp) / "b.mp4", frames=12)
            cap = cv2.VideoCapture(str(path))
            try:
                self.assertTrue(cap.isOpened())
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 12)
                self.assertEqual((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), scenes.CANVAS)
            finally:
                cap.release()


class ScriptTest(unittest.TestCase):
    def run_script(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=120, env=dict(os.environ, YOLO_OFFLINE="1"))

    def test_help(self) -> None:
        proc = self.run_script("--help")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("SENTETİK", proc.stdout)
        self.assertIn("--repeats", proc.stdout)

    def test_pipeline_worker_requires_a_video(self) -> None:
        self.assertEqual(self.run_script("--worker", "pipeline").returncode, 2)

    def test_invalid_thread_setting_is_rejected(self) -> None:
        self.assertEqual(self.run_script("--threads", "bozuk").returncode, 2)


if __name__ == "__main__":
    unittest.main()
