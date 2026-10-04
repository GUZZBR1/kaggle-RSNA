import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from rsna import DatasetVersion
from rsna.data import discover_dataset, load_manifest, read_dicom_metadata, save_manifest
from rsna.identity import digest
from rsna.targets import TARGET_REGISTRY_ID


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

    def test_dataset_version_optional_artifact_binding_preserves_legacy_identity(self):
        legacy = DatasetVersion("d", "v1", "a" * 64, "none", ("normal",), synthetic=True)
        bound = DatasetVersion("d", "v1", "a" * 64, "none", ("normal",),
                               dataset_index_artifact_id="b" * 64, synthetic=True)
        self.assertEqual(64, len(bound.dataset_version_id))
        self.assertNotEqual(legacy.dataset_version_id, bound.dataset_version_id)
        self.assertEqual(bound.dataset_version_id,
                         DatasetVersion.from_dict(bound.to_dict()).dataset_version_id)
        legacy_payload = {"schema_version": legacy.schema_version, "name": legacy.name,
            "version": legacy.version, "source_manifest_sha256": legacy.source_manifest_sha256,
            "preprocessing_version": legacy.preprocessing_version,
            "preprocessing": legacy.preprocessing, "class_names": legacy.class_names,
            "synthetic": legacy.synthetic, "target_schema_version": legacy.target_schema_version,
            "target_registry_id": TARGET_REGISTRY_ID}
        self.assertEqual(digest(legacy_payload), legacy.dataset_version_id)


if __name__ == "__main__":
    unittest.main()
