import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage

from rsna.data import DatasetIndex, load_manifest, load_or_refresh, save_manifest


def write_dicom(path, sop):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = sop
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = sop
    ds.SeriesInstanceUID = "1.2.2"
    ds.StudyInstanceUID = "1.2.1"
    ds.PatientID = "P1"
    ds.Rows = 2
    ds.Columns = 2
    ds.save_as(path, enforce_file_format=True)
    return path


class PersistentDatasetIndexTests(unittest.TestCase):
    def test_cold_warm_added_modified_removed_and_material_lookups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cache = Path(tmp) / "raw", Path(tmp) / "cache" / "index.sqlite3"
            first = write_dicom(root / "s" / "a.dcm", "1.2.3.1")
            parse = "rsna.data.cache.read_dicom_metadata"
            with patch(parse, wraps=__import__("rsna.data.cache", fromlist=["read_dicom_metadata"]).read_dicom_metadata) as reader:
                cold = load_or_refresh(root, cache)
                self.assertEqual(("cold-build", 1), (cold.report.mode, reader.call_count))
                warm = load_or_refresh(root, cache)
                self.assertEqual(("warm-load", 1), (warm.report.mode, reader.call_count))
            self.assertIsInstance(warm.index, DatasetIndex)
            self.assertEqual(cold.index.index_id, warm.index.index_id)
            self.assertIsNotNone(warm.index.study_by_uid("1.2.1"))
            self.assertEqual(1, len(warm.index.series_by_uid("1.2.2")))
            self.assertIsNotNone(warm.index.slice_by_path("s/a.dcm"))
            self.assertEqual(1, len(warm.index.slices_by_sop_uid("1.2.3.1")))

            write_dicom(root / "s" / "b.dcm", "1.2.3.2")
            added = load_or_refresh(root, cache)
            self.assertEqual((1, 1), (added.report.added, added.report.reparsed))
            write_dicom(first, "1.2.3.3")
            modified = load_or_refresh(root, cache)
            self.assertEqual((1, 1), (modified.report.modified, modified.report.reparsed))
            (root / "s" / "b.dcm").unlink()
            removed = load_or_refresh(root, cache)
            self.assertEqual(1, removed.report.removed)
            self.assertIsNone(removed.index.slice_by_path("s/b.dcm"))
            (root / "notes.jpg").write_bytes(b"not DICOM")
            ignored = load_or_refresh(root, cache)
            self.assertEqual((1, 0), (ignored.report.added, ignored.report.reparsed))

    def test_cache_binding_corruption_old_schema_and_manifest_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cache = Path(tmp) / "raw", Path(tmp) / "index.sqlite3"
            write_dicom(root / "a.dcm", "1.2.3.1")
            first = load_or_refresh(root, cache, dataset_version_id="dv1")
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                load_or_refresh(root, cache, dataset_version_id="dv2", cache_policy="strict")
            db = sqlite3.connect(cache)
            try:
                db.execute("update files set record='{}' where path='a.dcm'")
                db.commit()
            finally:
                db.close()
            with self.assertRaisesRegex(ValueError, "payload hash"):
                load_or_refresh(root, cache, dataset_version_id="dv1", cache_policy="strict")
            first = load_or_refresh(root, cache, dataset_version_id="dv1")
            db = sqlite3.connect(cache)
            try:
                db.execute("update manifest set value='0' where key='schema_version'")
                db.commit()
            finally:
                db.close()
            with self.assertRaisesRegex(ValueError, "schema version"):
                load_or_refresh(root, cache, dataset_version_id="dv1", cache_policy="strict")
            rebuilt = load_or_refresh(root, cache, dataset_version_id="dv1")
            self.assertEqual("rebuild", rebuilt.report.mode)
            index = rebuilt.index
            manifest = Path(tmp) / "manifest.json"
            save_manifest(index, manifest)
            manifest.write_text("{broken", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_manifest(manifest)
            save_manifest(index, manifest)
            self.assertEqual(index.index_id, load_manifest(manifest).index_id)
            valid_bytes = manifest.read_bytes()
            with patch("rsna.data.manifest.os.replace", side_effect=OSError("simulated manifest crash")):
                with self.assertRaisesRegex(OSError, "manifest crash"):
                    save_manifest(index, manifest)
            self.assertEqual(valid_bytes, manifest.read_bytes())

    def test_root_move_order_rebuild_and_validate_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root, cache = base / "raw", base / "cache.sqlite3"
            write_dicom(root / "z.dcm", "1.2.3.2")
            write_dicom(root / "a.dcm", "1.2.3.1")
            built = load_or_refresh(root, cache)
            moved = base / "moved"
            shutil.copytree(root, moved)
            self.assertEqual(built.index.index_id, load_or_refresh(moved, cache).index.index_id)
            before = cache.read_bytes()
            checked = load_or_refresh(moved, cache, validate_only=True)
            self.assertEqual("validate-only", checked.report.mode)
            self.assertEqual(before, cache.read_bytes())
            rebuilt = load_or_refresh(moved, cache, rebuild=True)
            self.assertEqual(2, rebuilt.report.reparsed)
            self.assertEqual(built.index.index_id, rebuilt.index.index_id)
            reversed_root = base / "reversed"
            write_dicom(reversed_root / "a.dcm", "1.2.3.1")
            write_dicom(reversed_root / "z.dcm", "1.2.3.2")
            self.assertEqual(rebuilt.index.index_id, load_or_refresh(reversed_root, base / "other.sqlite3").index.index_id)

    def test_failed_atomic_cache_replace_keeps_old_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cache = Path(tmp) / "raw", Path(tmp) / "cache.sqlite3"
            write_dicom(root / "a.dcm", "1.2.3.1")
            load_or_refresh(root, cache)
            old = cache.read_bytes()
            write_dicom(root / "b.dcm", "1.2.3.2")
            with patch("rsna.data.cache.os.replace", side_effect=OSError("simulated crash")):
                with self.assertRaisesRegex(OSError, "simulated crash"):
                    load_or_refresh(root, cache)
            self.assertEqual(old, cache.read_bytes())

    def test_cli_reports_cold_warm_validate_and_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp) / "raw", Path(tmp) / "artifacts" / "dataset-index"
            write_dicom(root / "a.dcm", "1.2.3.1")
            def run(*flags):
                return subprocess.run([sys.executable, "-m", "rsna", "data-index",
                    "--input", str(root), "--output", str(output), *flags],
                    check=True, capture_output=True, text=True).stdout
            self.assertIn("Mode: cold-build", run())
            warm = run()
            self.assertIn("Mode: warm-load", warm)
            self.assertIn("Files parsed: 0", warm)
            self.assertIn("Mode: validate-only", run("--validate-only"))
            self.assertIn("Mode: rebuild", run("--rebuild"))
            self.assertTrue((output / "index.sqlite3").is_file())
            self.assertTrue((output / "manifest.json").is_file())
            self.assertFalse(output.is_relative_to(root))


if __name__ == "__main__":
    unittest.main()


