"""Aşama 2 analiz araçları için testler. Yalnızca küçük SENTETİK veri kullanılır.

Gerçek veri setleri (data/raw) bu testlerde okunmaz ve DEĞİŞTİRİLMEZ; ayrıca sınıf sonunda
data/raw dosya listesinin (yol, boyut, mtime) test öncesi ve sonrası aynı olduğu doğrulanır.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isg_vision.dataset_audit.duplicates import (  # noqa: E402
    base_name, find_exact, find_near, index_dataset,
)
from isg_vision.dataset_audit.extract import extract_zip_safely  # noqa: E402
from isg_vision.dataset_audit.labels import conflicting_overlaps, parse_label_file  # noqa: E402
from isg_vision.dataset_audit.layout import discover_dataset, iter_samples, parse_class_names  # noqa: E402
from isg_vision.dataset_audit.mapping import check_completeness, mapping_rows  # noqa: E402
from isg_vision.dataset_audit.scan import scan_dataset  # noqa: E402
from isg_vision.dataset_audit.visualize import (  # noqa: E402
    annotate_image, index_by_class, select_samples,
)

RAW = ROOT / "data" / "raw"


def raw_snapshot() -> dict[str, tuple[int, int]]:
    snap = {}
    for p in RAW.rglob("*"):
        if p.is_file():
            st = p.stat()
            snap[str(p)] = (st.st_size, st.st_mtime_ns)
    return snap


def make_image(path: Path, size: tuple[int, int] = (64, 48), seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, "JPEG")


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def codes(result) -> set[str]:
    return {i.code for i in result.issues}


class RealDataUntouchedMixin(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._before = raw_snapshot()

    @classmethod
    def tearDownClass(cls) -> None:
        assert raw_snapshot() == cls._before, "data/raw içeriği test sırasında değişti!"


class LabelParsingTest(RealDataUntouchedMixin):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def label(self, text: str, **kw):
        return parse_label_file(write(self.dir / "a.txt", text), valid_class_ids={0, 1, 2}, **kw)

    def test_valid_labels(self) -> None:
        r = self.label("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n", image_size=(640, 480))
        self.assertEqual(len(r.boxes), 2)
        self.assertEqual(r.issues, [])

    def test_undefined_class_id(self) -> None:
        self.assertIn("CLASS_ID_UNDEFINED", codes(self.label("7 0.5 0.5 0.2 0.2\n")))

    def test_invalid_class_id_negative_and_float(self) -> None:
        self.assertIn("CLASS_ID_INVALID", codes(self.label("-1 0.5 0.5 0.2 0.2\n")))
        self.assertIn("CLASS_ID_INVALID", codes(self.label("1.5 0.5 0.5 0.2 0.2\n")))

    def test_empty_label_is_info_not_error(self) -> None:
        r = self.label("\n  \n")
        self.assertTrue(r.empty)
        self.assertEqual({i.severity for i in r.issues}, {"info"})

    def test_nonpositive_box_size(self) -> None:
        self.assertIn("BOX_NONPOSITIVE_SIZE", codes(self.label("0 0.5 0.5 0.0 0.2\n")))
        self.assertIn("BOX_NONPOSITIVE_SIZE", codes(self.label("0 0.5 0.5 0.2 -0.1\n")))

    def test_coordinates_outside_normalized_range(self) -> None:
        self.assertIn("COORD_OUT_OF_RANGE", codes(self.label("0 1.5 0.5 0.2 0.2\n")))
        self.assertIn("COORD_OUT_OF_RANGE", codes(self.label("0 0.5 0.5 1.4 0.2\n")))

    def test_box_exceeding_image_bounds(self) -> None:
        r = self.label("0 0.95 0.5 0.2 0.2\n")  # x2 = 1.05
        self.assertIn("BOX_OUTSIDE_IMAGE", codes(r))
        self.assertNotIn("COORD_OUT_OF_RANGE", codes(r))  # merkez/boyut geçerli, yalnızca kenar taşıyor

    def test_rounding_noise_within_tolerance_is_ok(self) -> None:
        self.assertNotIn("BOX_OUTSIDE_IMAGE", codes(self.label("0 0.5 0.5 1.0000005 0.2\n")))

    def test_unsupported_segmentation_format(self) -> None:
        r = self.label("0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")
        self.assertIn("UNSUPPORTED_FORMAT", codes(r))
        self.assertEqual(r.boxes, [])  # poligon detection kutusu gibi İŞLENMEZ
        self.assertEqual(r.unsupported_rows, 1)

    def test_bad_column_count_and_non_numeric(self) -> None:
        self.assertIn("BAD_COLUMN_COUNT", codes(self.label("0 0.5 0.5\n")))
        self.assertIn("NON_NUMERIC", codes(self.label("0 a 0.5 0.2 0.2\n")))

    def test_duplicate_and_near_duplicate_boxes(self) -> None:
        r = self.label("0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.2 0.2\n1 0.3001 0.3 0.2 0.2\n")
        self.assertIn("DUPLICATE_BOX", codes(r))
        self.assertIn("NEAR_DUPLICATE_BOX", codes(r))

    def test_very_small_box(self) -> None:
        self.assertIn("BOX_VERY_SMALL", codes(self.label("0 0.5 0.5 0.001 0.2\n", image_size=(640, 480))))

    def test_conflicting_overlap(self) -> None:
        r = self.label("0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n2 0.1 0.1 0.05 0.05\n")
        found = conflicting_overlaps(r.boxes, [(0, 1)])
        self.assertEqual(len(found), 1)
        self.assertEqual(conflicting_overlaps(r.boxes, [(0, 2)]), [])

    def test_unreadable_label_file(self) -> None:
        p = self.dir / "bin.txt"
        p.write_bytes(b"\xff\xfe\x00bad")
        r = parse_label_file(p)
        self.assertFalse(r.readable)
        self.assertIn("LABEL_UNREADABLE", codes(r))

    def test_label_file_not_modified(self) -> None:
        p = write(self.dir / "keep.txt", "0 1.5 0.5 0.2 0.2\n")
        before = hashlib.sha256(p.read_bytes()).hexdigest()
        parse_label_file(p, {0})
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), before)


class YamlClassTest(RealDataUntouchedMixin):
    def test_names_as_list(self) -> None:
        info = parse_class_names({"nc": 2, "names": ["a", "b"]})
        self.assertEqual(info.names, {0: "a", 1: "b"})
        self.assertEqual(info.names_format, "liste")
        self.assertEqual(info.warnings, [])

    def test_names_as_dict_keeps_order_and_ids(self) -> None:
        info = parse_class_names({"names": {0: "helmet", 2: "vest", 1: "x"}})
        self.assertEqual(info.names_format, "sözlük")
        self.assertEqual(info.names[2], "vest")

    def test_nc_mismatch_warns(self) -> None:
        info = parse_class_names({"nc": 3, "names": ["a", "b"]})
        self.assertTrue(any("nc=3" in w for w in info.warnings))

    def test_non_consecutive_ids_warn(self) -> None:
        info = parse_class_names({"names": {0: "a", 2: "b"}})
        self.assertTrue(any("ardışık" in w for w in info.warnings))

    def test_missing_names_does_not_guess(self) -> None:
        info = parse_class_names({"nc": 2})
        self.assertEqual(info.names, {})
        self.assertTrue(info.warnings)


class LayoutDiscoveryTest(RealDataUntouchedMixin):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def add_pair(self, img_dir: Path, lbl_dir: Path, stem: str, label: str | None = "0 0.5 0.5 0.2 0.2\n", seed: int = 0):
        make_image(img_dir / f"{stem}.jpg", seed=seed)
        if label is not None:
            write(lbl_dir / f"{stem}.txt", label)

    def test_layout_a_images_train(self) -> None:
        ds = self.root / "a"
        write(ds / "data.yaml", "names: [x]\n")
        for sp in ("train", "val"):
            self.add_pair(ds / "images" / sp, ds / "labels" / sp, "i1")
        lay = discover_dataset("a", ds)
        self.assertEqual(set(lay.splits), {"train", "val"})
        self.assertEqual(lay.splits["train"].layout, "A")
        self.assertEqual(len(list(iter_samples(lay.splits["train"]))), 1)
        self.assertTrue(any("test" in n for n in lay.notes))  # test bölümü yok notu

    def test_layout_b_train_images_and_valid_alias(self) -> None:
        ds = self.root / "b"
        write(ds / "data.yaml", "nc: 1\nnames: ['x']\n")
        for sp in ("train", "valid", "test"):
            self.add_pair(ds / sp / "images", ds / sp / "labels", "i1")
        lay = discover_dataset("b", ds)
        self.assertEqual(set(lay.splits), {"train", "val", "test"})
        self.assertEqual(lay.splits["val"].original_name, "valid")
        self.assertEqual(lay.splits["val"].layout, "B")

    def test_nested_root(self) -> None:
        ds = self.root / "c" / "wrapper" / "inner"
        write(ds / "data.yaml", "names: [x]\n")
        self.add_pair(ds / "train" / "images", ds / "train" / "labels", "i1")
        lay = discover_dataset("c", self.root / "c")
        self.assertTrue(lay.found)
        self.assertEqual(lay.root, ds)

    def test_flat_layout_c(self) -> None:
        ds = self.root / "d"
        write(ds / "data.yaml", "names: [x]\n")
        self.add_pair(ds / "train", ds / "train", "i1")
        lay = discover_dataset("d", ds)
        self.assertEqual(lay.splits["train"].layout, "C")

    def test_missing_yaml_no_guessing(self) -> None:
        ds = self.root / "e"
        self.add_pair(ds / "train" / "images", ds / "train" / "labels", "i1")
        lay = discover_dataset("e", ds)
        self.assertIsNone(lay.yaml_path)
        self.assertEqual(lay.classes.names, {})
        self.assertTrue(any("YAML" in n for n in lay.notes))
        self.assertTrue(lay.found)  # klasör yapısı yine de bulunur

    def test_nonexistent_dataset_dir(self) -> None:
        lay = discover_dataset("zz", self.root / "yok")
        self.assertFalse(lay.found)
        self.assertTrue(lay.notes)

    def test_scan_reports_missing_empty_orphan_unreadable_collision(self) -> None:
        ds = self.root / "f"
        write(ds / "data.yaml", "names: [x]\n")
        img, lbl = ds / "train" / "images", ds / "train" / "labels"
        self.add_pair(img, lbl, "ok")
        self.add_pair(img, lbl, "nolabel", label=None)
        self.add_pair(img, lbl, "empty", label="")
        write(lbl / "orphan.txt", "0 0.5 0.5 0.2 0.2\n")
        write(img / "broken.jpg", "not an image")
        write(lbl / "broken.txt", "0 0.5 0.5 0.2 0.2\n")
        make_image(img / "coll.jpg")
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(img / "coll.png")
        write(lbl / "coll.txt", "0 0.5 0.5 0.2 0.2\n")
        lay = discover_dataset("f", ds)
        found: dict[str, set[str]] = {}
        for rec in scan_dataset(lay, full_decode=True):
            found[rec.sample.stem] = {i.code for i in rec.issues}
        self.assertEqual(found["ok"], set())
        self.assertIn("LABEL_MISSING", found["nolabel"])
        self.assertIn("LABEL_EMPTY", found["empty"])
        self.assertIn("ORPHAN_LABEL", found["orphan"])
        self.assertIn("IMAGE_UNREADABLE", found["broken"])
        self.assertIn("STEM_COLLISION", found["coll"])

    def test_scan_does_not_modify_files(self) -> None:
        ds = self.root / "g"
        write(ds / "data.yaml", "names: [x]\n")
        self.add_pair(ds / "train" / "images", ds / "train" / "labels", "i1", label="0 2 2 2 2\n")
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in ds.rglob("*") if p.is_file()}
        list(scan_dataset(discover_dataset("g", ds), full_decode=True))
        after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in ds.rglob("*") if p.is_file()}
        self.assertEqual(before, after)


class DuplicateDetectionTest(RealDataUntouchedMixin):
    def test_same_image_in_different_folders_is_exact_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "ds"
            write(ds / "data.yaml", "names: [x]\n")
            make_image(ds / "train" / "images" / "a.jpg", seed=1)
            (ds / "valid" / "images").mkdir(parents=True)
            (ds / "valid" / "images" / "b.jpg").write_bytes((ds / "train" / "images" / "a.jpg").read_bytes())
            make_image(ds / "valid" / "images" / "c.jpg", seed=2)
            lay = discover_dataset("ds", ds)
            entries, unreadable = index_dataset(lay, lambda p: Path(p).relative_to(tmp).as_posix())
            self.assertEqual(unreadable, [])
            exact = find_exact(entries)
            self.assertEqual(len(exact), 1)
            self.assertEqual(exact[0]["relation"], "within_dataset_cross_split")
            self.assertEqual(exact[0]["split_pair"], "train|val")

    def test_resized_copy_is_near_but_not_exact_and_flipped_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "ds"
            write(ds / "data.yaml", "names: [x]\n")
            rng = np.random.default_rng(5)
            base = cv2.GaussianBlur(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8), (0, 0), 6)
            (ds / "train" / "images").mkdir(parents=True)
            cv2.imwrite(str(ds / "train" / "images" / "orig.jpg"), base)
            cv2.imwrite(str(ds / "train" / "images" / "small.jpg"), cv2.resize(base, (80, 60)))
            cv2.imwrite(str(ds / "train" / "images" / "flip.jpg"), base[:, ::-1])
            cv2.imwrite(str(ds / "train" / "images" / "other.jpg"), cv2.GaussianBlur(
                np.random.default_rng(99).integers(0, 255, (120, 160, 3), dtype=np.uint8), (0, 0), 6))
            entries, _ = index_dataset(discover_dataset("ds", ds), lambda p: Path(p).name)
            self.assertEqual(find_exact(entries), [])
            rows, truncated = find_near(entries, threshold=6)
            self.assertFalse(truncated)
            pairs = {frozenset((r["path_a"], r["path_b"])): r["match_type"] for r in rows}
            self.assertIn(frozenset(("orig.jpg", "small.jpg")), pairs)
            self.assertIn(frozenset(("orig.jpg", "flip.jpg")), pairs)
            self.assertFalse(any("other.jpg" in k for k in pairs))

    def test_base_name_hint(self) -> None:
        self.assertEqual(base_name("x/00596_jpg.rf.e056867eae13eca16cc7cb8fe55be7df.jpg"), "00596")
        self.assertEqual(base_name("x/plain.jpg"), "plain")


class SafeExtractionTest(RealDataUntouchedMixin):
    def make_zip(self, path: Path, entries: dict[str, bytes], symlink: str | None = None) -> None:
        with zipfile.ZipFile(path, "w") as zf:
            for name, data in entries.items():
                zf.writestr(name, data)
            if symlink:
                info = zipfile.ZipInfo(symlink)
                info.external_attr = (0o120777 << 16)
                zf.writestr(info, "target")

    def test_extract_strips_wrapper_and_protects_zip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            z = Path(tmp) / "d.zip"
            self.make_zip(z, {"wrap/data.yaml": b"names: [a]\n", "wrap/train/images/a.jpg": b"x"})
            before = hashlib.sha256(z.read_bytes()).hexdigest()
            dest = Path(tmp) / "out"
            res = extract_zip_safely(z, dest, "d")
            self.assertEqual(res.stripped_prefix, "wrap")
            self.assertTrue((dest / "data.yaml").is_file())
            self.assertEqual(res.extracted, 2)
            self.assertEqual(hashlib.sha256(z.read_bytes()).hexdigest(), before)  # ZIP değişmedi
            self.assertFalse((dest / "data.yaml").stat().st_mode & 0o222)  # salt-okunur

    def test_zip_slip_and_absolute_paths_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            z = Path(tmp) / "evil.zip"
            self.make_zip(z, {"ok.txt": b"1", "../escape.txt": b"2", "/abs.txt": b"3", "a/../../escape2.txt": b"4",
                              "C:/drive.txt": b"5"})
            dest = Path(tmp) / "sub" / "out"
            res = extract_zip_safely(z, dest, "evil", strip_wrapper=False)
            self.assertEqual(len(res.rejected), 4)
            self.assertFalse((Path(tmp) / "escape.txt").exists())
            self.assertFalse((Path(tmp) / "sub" / "escape.txt").exists())
            self.assertFalse((Path(tmp) / "escape2.txt").exists())
            self.assertTrue((dest / "ok.txt").is_file())

    def test_symlink_entry_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            z = Path(tmp) / "s.zip"
            self.make_zip(z, {"a.txt": b"1"}, symlink="link")
            res = extract_zip_safely(z, Path(tmp) / "o", "s", strip_wrapper=False)
            self.assertTrue(any("sembolik" in r for r in res.rejected))
            self.assertFalse((Path(tmp) / "o" / "link").exists())

    def test_never_overwrites_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            z = Path(tmp) / "d.zip"
            self.make_zip(z, {"a.txt": b"NEW", "b.txt": b"B"})
            dest = Path(tmp) / "out"
            write(dest / "a.txt", "OLD")
            res = extract_zip_safely(z, dest, "d", strip_wrapper=False)
            self.assertEqual((dest / "a.txt").read_text(), "OLD")
            self.assertEqual(res.skipped_existing, ["a.txt"])
            self.assertEqual(res.extracted, 1)

    def test_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            z = Path(tmp) / "d.zip"
            self.make_zip(z, {"a.txt": b"1"})
            res = extract_zip_safely(z, Path(tmp) / "out", "d", strip_wrapper=False, dry_run=True)
            self.assertEqual(res.extracted, 1)
            self.assertFalse((Path(tmp) / "out" / "a.txt").exists())


class MappingTest(RealDataUntouchedMixin):
    TARGETS = {0: "helmet", 1: "no_helmet"}

    def test_rows_and_invalid_status(self) -> None:
        rows = mapping_rows(self.TARGETS, [{"dataset": "d", "id": 0, "name": "H", "target": 0, "status": "confirmed"},
                                           {"dataset": "d", "id": 1, "name": "G", "target": None, "status": "incompatible"}])
        self.assertEqual(rows[0]["proposed_target_class"], "helmet")
        self.assertEqual(rows[1]["proposed_target_class"], "(hedef dışı)")
        self.assertEqual(rows[0]["user_review"], "Pending")
        with self.assertRaises(ValueError):
            mapping_rows(self.TARGETS, [{"dataset": "d", "id": 0, "name": "H", "target": 0, "status": "maybe"}])
        with self.assertRaises(ValueError):
            mapping_rows(self.TARGETS, [{"dataset": "d", "id": 0, "name": "H", "target": 9, "status": "confirmed"}])

    def test_completeness_check(self) -> None:
        entries = [{"dataset": "d", "id": 0, "name": "H", "target": 0, "status": "confirmed"}]
        self.assertEqual(check_completeness(entries, {"d": {0: "H"}}), [])
        self.assertTrue(any("EKSİK" in p for p in check_completeness(entries, {"d": {0: "H", 1: "X"}})))
        self.assertTrue(any("uyuşmuyor" in p for p in check_completeness(entries, {"d": {0: "Other"}})))

    def test_real_mapping_file_matches_real_yaml_classes(self) -> None:
        """Gerçek eşleme dosyası, YAML sınıflarıyla tam tutarlı olmalı (veri varsa; yalnızca okuma)."""
        from isg_vision.dataset_audit.common import dataset_raw_dir, load_registry
        from isg_vision.dataset_audit.mapping import load_mapping

        registry = load_registry()
        classes = {}
        for name in registry:
            raw = dataset_raw_dir(registry, name)
            if not raw.is_dir() or not any(raw.glob("data.yaml")):
                self.skipTest("Gerçek veri seti çıkarılmamış; eşleme-YAML karşılaştırması atlandı")
            classes[name] = discover_dataset(name, raw).classes.names
        targets, entries = load_mapping()
        self.assertEqual(check_completeness(entries, classes), [])
        mapping_rows(targets, entries)


class VisualizationTest(RealDataUntouchedMixin):
    def test_selection_is_deterministic_and_source_not_modified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "ds"
            write(ds / "data.yaml", "names: [a, b]\n")
            for i in range(12):
                sp = "train" if i < 8 else "val"
                make_image(ds / sp / "images" / f"i{i}.jpg", seed=i)
                write(ds / sp / "labels" / f"i{i}.txt", f"{i % 2} 0.5 0.5 0.3 0.3\n")
            lay = discover_dataset("ds", ds)
            by_class, everything = index_by_class(lay)
            s1 = select_samples(by_class, everything, n=6, seed=42)
            s2 = select_samples(by_class, everything, n=6, seed=42)
            self.assertEqual([c.image_path for c, _ in s1], [c.image_path for c, _ in s2])
            self.assertEqual(len(s1), 6)
            self.assertEqual({c.split for c, _ in s1}, {"train", "val"})
            classes_seen = {cid for cid, cands in by_class.items() for c, _ in s1 if c in cands}
            self.assertEqual(classes_seen, {0, 1})

            img = cv2.imread(str(ds / "train" / "images" / "i0.jpg"))
            snapshot = img.copy()
            out = annotate_image(img, parse_label_file(ds / "train" / "labels" / "i0.txt").boxes, {0: "a", 1: "b"},
                                 [(0, 0, 255), (0, 255, 0)])
            self.assertTrue(np.array_equal(img, snapshot))      # girdi dizisi değişmedi (kopya üzerine çizildi)
            self.assertFalse(np.array_equal(out, snapshot))     # çıktıya kutu çizildi


class ScriptCliTest(RealDataUntouchedMixin):
    SCRIPTS = ("extract_datasets", "inspect_datasets", "validate_labels", "visualize_annotations",
               "analyze_class_distribution", "detect_duplicates", "generate_dataset_report")

    def test_help_runs_for_every_script(self) -> None:
        for name in self.SCRIPTS:
            with self.subTest(script=name):
                res = subprocess.run([sys.executable, str(ROOT / "scripts" / f"{name}.py"), "--help"],
                                     capture_output=True, text=True, timeout=120)
                self.assertEqual(res.returncode, 0, res.stderr)
                self.assertIn("usage", res.stdout.lower())

    def test_validate_labels_handles_missing_dataset_dir_gracefully(self) -> None:
        # Bilinmeyen veri seti adı: açık hata mesajıyla çıkar
        res = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_labels.py"), "--datasets", "yok_boyle_bir_set"],
                             capture_output=True, text=True, timeout=120)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Bilinmeyen veri seti", res.stderr + res.stdout)


if __name__ == "__main__":
    unittest.main()
