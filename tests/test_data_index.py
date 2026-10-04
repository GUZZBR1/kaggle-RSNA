import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from rsna.data import discover_dataset, load_manifest, load_or_refresh, read_dicom_metadata, save_manifest


def write_dicom(path, *, study="1.2.10", series="1.2.20", sop=None, patient="P1", missing=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    sop = sop or generate_uid()
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = sop
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = MRImageStorage
    for name, value in {"SOPInstanceUID": sop, "StudyInstanceUID": study,
                        "SeriesInstanceUID": series, "PatientID": patient,
                        "InstanceNumber": 1, "ImagePositionPatient": [0, 0, 1],
                        "ImageOrientationPatient": [1, 0, 0, 0, 1, 0],
                        "SliceThickness": 3, "SpacingBetweenSlices": 3,
                        "PixelSpacing": [0.5, 0.5], "Rows": 2, "Columns": 2,
                        "Modality": "MR", "SeriesDescription": "T1",
                        "ProtocolName": "knee", "Laterality": "L",
                        "PatientPosition": "HFS"}.items():
        if name not in missing:
            setattr(ds, name, value)
    ds.save_as(path, enforce_file_format=True)
    return path


class DatasetIndexTests(unittest.TestCase):
    def test_persistent_index_cold_warm_and_incremental_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cache = Path(tmp) / "data", Path(tmp) / "cache" / "index.sqlite3"
            first_file = write_dicom(root / "a.dcm", sop="1.2.1")
            cold = load_or_refresh(root, cache)
            self.assertEqual("cold-build", cold.report.mode)
            self.assertEqual(1, cold.report.reparsed)
            warm = load_or_refresh(root, cache)
            self.assertEqual("warm-load", warm.report.mode)
            self.assertEqual(0, warm.report.reparsed)
            self.assertEqual(cold.index.index_id, warm.index.index_id)
            write_dicom(root / "b.dcm", sop="1.2.2")
            added = load_or_refresh(root, cache)
            self.assertEqual((1, 1), (added.report.added, added.report.reparsed))
            write_dicom(first_file, sop="1.2.3")
            changed = load_or_refresh(root, cache)
            self.assertEqual((1, 1), (changed.report.modified, changed.report.reparsed))
            (root / "b.dcm").unlink()
            removed = load_or_refresh(root, cache)
            self.assertEqual(1, removed.report.removed)
            self.assertEqual(1, removed.index.statistics["n_slices"])

    def test_cache_validation_corruption_binding_and_root_portability(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root, cache = base / "one", base / "cache.sqlite3"
            write_dicom(root / "s" / "a.dcm", sop="1.2.88")
            original = load_or_refresh(root, cache, dataset_version_id="v1")
            moved = base / "moved"
            shutil.copytree(root, moved)
            self.assertEqual(original.index.index_id, load_or_refresh(moved, cache,
                dataset_version_id="v1").index.index_id)
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                load_or_refresh(root, cache, dataset_version_id="v2", cache_policy="strict")
            db = sqlite3.connect(cache)
            try:
                db.execute("update files set record='{}' where path='s/a.dcm'")
                db.commit()
            finally:
                db.close()
            with self.assertRaises(ValueError):
                load_or_refresh(root, cache, cache_policy="strict")
            rebuilt = load_or_refresh(root, cache)
            self.assertEqual("rebuild", rebuilt.report.mode)
            db = sqlite3.connect(cache)
            try:
                db.execute("update manifest set value='0' where key='schema_version'")
                db.commit()
            finally:
                db.close()
            with self.assertRaisesRegex(ValueError, "schema version"):
                load_or_refresh(root, cache, cache_policy="strict")
            self.assertEqual("rebuild", load_or_refresh(root, cache).report.mode)

    def test_validate_only_does_not_write_and_lookup_maps_are_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cache = Path(tmp) / "data", Path(tmp) / "idx.sqlite3"
            write_dicom(root / "a.dcm", sop="1.2.44")
            built = load_or_refresh(root, cache)
            before = cache.stat().st_mtime_ns
            checked = load_or_refresh(root, cache, validate_only=True)
            self.assertEqual("validate-only", checked.report.mode)
            self.assertEqual(before, cache.stat().st_mtime_ns)
            self.assertIsNotNone(built.index.slice_by_path("a.dcm"))
            self.assertEqual(1, len(built.index.slices_by_sop_uid("1.2.44")))

    def test_failed_atomic_replace_preserves_previous_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cache = Path(tmp) / "data", Path(tmp) / "index.sqlite3"
            write_dicom(root / "a.dcm", sop="1.2.31")
            load_or_refresh(root, cache)
            old_bytes = cache.read_bytes()
            write_dicom(root / "b.dcm", sop="1.2.32")
            with patch("rsna.data.cache.os.replace", side_effect=OSError("simulated crash")):
                with self.assertRaisesRegex(OSError, "simulated crash"):
                    load_or_refresh(root, cache)
            self.assertEqual(old_bytes, cache.read_bytes())
            self.assertEqual(2, load_or_refresh(root, cache).index.statistics["n_slices"])

    def test_one_series_three_slices_and_metadata_only_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for n in range(3):
                write_dicom(root / "train" / "study" / "series" / f"{n}.dcm", sop=f"1.2.30.{n+1}")
            index = discover_dataset(root)
            self.assertEqual((1, 1, 3), (index.statistics["n_studies"], index.statistics["n_series"], index.statistics["n_slices"]))
            self.assertNotIn("PixelData", read_dicom_metadata(root / "train/study/series/0.dcm"))
            self.assertEqual(3, index.studies[0].series[0].n_slices)

    def test_two_series_and_two_studies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dicom(root / "a.dcm", study="1.2.1", series="1.2.11", sop="1.2.101")
            write_dicom(root / "b.dcm", study="1.2.1", series="1.2.12", sop="1.2.102")
            write_dicom(root / "c.dcm", study="1.2.2", series="1.2.21", sop="1.2.201")
            index = discover_dataset(root)
            self.assertEqual(2, index.statistics["n_studies"])
            self.assertEqual(3, index.statistics["n_series"])

    def test_missing_uids_invalid_non_dicom_and_invalid_policies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dicom(root / "missing.dcm", missing=("StudyInstanceUID", "SeriesInstanceUID"))
            (root / "broken.dcm").write_bytes(b"not a dicom")
            (root / "notes.txt").write_text("not a DICOM candidate")
            index = discover_dataset(root)
            self.assertEqual(1, index.statistics["invalid_files"])
            self.assertEqual(2, index.statistics["missing_uid_count"])
            self.assertTrue(index.warnings)
            self.assertEqual(1, discover_dataset(root, on_invalid="skip-invalid").statistics["invalid_files"])
            with self.assertRaises(ValueError):
                discover_dataset(root, on_invalid="strict")

    def test_duplicate_sop_uid_and_patient_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dicom(root / "a.dcm", sop="1.2.900", patient="P1")
            write_dicom(root / "b.dcm", sop="1.2.900", patient="P2")
            index = discover_dataset(root)
            self.assertEqual(1, index.statistics["duplicate_sop_uid_count"])
            self.assertTrue(any("conflicting PatientID" in w for w in index.studies[0].warnings))

    def test_series_uid_reused_across_studies_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dicom(root / "a.dcm", study="1.2.1", series="1.2.88", sop="1.2.1.1")
            write_dicom(root / "b.dcm", study="1.2.2", series="1.2.88", sop="1.2.2.1")
            index = discover_dataset(root)
            self.assertTrue(any("SeriesInstanceUID 1.2.88" in w for w in index.warnings))

    def test_empty_tree_and_csv_metadata_are_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.csv").write_text("id,label\n")
            index = discover_dataset(root)
            self.assertEqual(0, index.statistics["n_studies"])
            self.assertEqual(("train.csv",), index.metadata_files)

    def test_cli_smoke_with_synthetic_dicom(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            output = Path(tmp) / "output"
            write_dicom(root / "s" / "r" / "1.dcm", sop="1.2.555")
            result = subprocess.run([sys.executable, "-m", "rsna", "data-index",
                "--input", str(root), "--output", str(output)], check=True, capture_output=True, text=True)
            self.assertIn("Studies: 1, Series: 1, Slices: 1", result.stdout)
            self.assertEqual(1, result.stdout.count("Studies:"))
            self.assertTrue((output / "manifest.json").is_file())
            self.assertEqual(discover_dataset(root).index_id, load_manifest(output / "manifest.json").index_id)
            warm = subprocess.run([sys.executable, "-m", "rsna", "data-index",
                "--input", str(root), "--output", str(output)], check=True, capture_output=True, text=True)
            self.assertIn("Mode: warm-load", warm.stdout)
            self.assertIn("Files parsed: 0", warm.stdout)

    def test_manifest_and_ids_are_deterministic_and_root_move_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "one"
            target = Path(tmp) / "moved" / "one"
            write_dicom(source / "s" / "r" / "x.dcm", sop="1.2.765")
            index1 = discover_dataset(source)
            shutil.copytree(source, target)
            index2 = discover_dataset(target)
            self.assertEqual(index1.index_id, index2.index_id)
            self.assertEqual(index1.studies[0].study_id, index2.studies[0].study_id)
            path = Path(tmp) / "manifest.json"
            first_hash = save_manifest(index1, path)
            first_bytes = path.read_bytes()
            self.assertEqual(index1, load_manifest(path))
            second_hash = save_manifest(discover_dataset(source), path)
            self.assertEqual(first_hash, second_hash)
            self.assertEqual(first_bytes, path.read_bytes())

    def test_cached_invalid_dicom_is_not_reparsed_until_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cache = Path(tmp) / "data", Path(tmp) / "index.sqlite3"
            bad = root / "bad.dcm"
            bad.parent.mkdir(parents=True)
            bad.write_bytes(b"invalid DICOM")
            cold = load_or_refresh(root, cache)
            warm = load_or_refresh(root, cache)
            self.assertEqual(1, cold.index.statistics["invalid_files"])
            self.assertEqual(0, warm.report.reparsed)
            bad.write_bytes(b"changed invalid DICOM content")
            changed = load_or_refresh(root, cache)
            self.assertEqual(1, changed.report.modified)
            self.assertEqual(1, changed.report.reparsed)
            self.assertTrue(changed.report.warnings)
if __name__ == "__main__":
    unittest.main()




