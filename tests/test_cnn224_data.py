from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - training extra is optional
    torch = None

from rsna.data.index import discover_dataset
from rsna.data.models import DatasetIndex, SeriesRecord, SliceRecord, StudyRecord
from rsna.labels import LabelRecord
from rsna.targets import TARGETS
from rsna.baseline.data import load_study_tensors


def _labels(uid: str, patient: str = "P1") -> LabelRecord:
    return LabelRecord(uid, {target: None for target in TARGETS}, "manual_review",
                       allow_partial=True, patient_id=patient)


def _indexed(root: Path, *, count: int = 24, duplicate_sop: bool = False,
             bad_study_at: int | None = None, bad_series_at: int | None = None) -> DatasetIndex:
    study_uid, series_uid, patient = "1.2.3", "1.2.4", "P1"
    slices = []
    for index in range(count):
        (root / "series").mkdir(parents=True, exist_ok=True)
        path = root / "series" / f"{index:03}.dcm"
        path.write_bytes(b"fixture")
        sop_index = 0 if duplicate_sop and index == count - 1 else index
        slices.append(SliceRecord(path.relative_to(root).as_posix(), path.stat().st_size, {
            "StudyInstanceUID": "9.9.9" if index == bad_study_at else study_uid,
            "SeriesInstanceUID": "9.9.8" if index == bad_series_at else series_uid,
            "SOPInstanceUID": f"1.2.5.{sop_index}",
            "InstanceNumber": index + 1,
        }))
    series = SeriesRecord(series_uid, study_uid, tuple(slices))
    study = StudyRecord(study_uid, patient, (series,))
    return DatasetIndex("root", "1", (study,), (), {"n_studies": 1, "n_series": 1, "n_slices": count})


@unittest.skipIf(torch is None, "torch is required for baseline tensor preprocessing")
class CNN224DataAdapterTests(unittest.TestCase):
    def test_uniform_24_slice_tensor_shape_normalization_and_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = _indexed(root, count=48)
            decoded = []

            def decode(path: Path):
                decoded.append(path.name)
                value = int(path.stem)
                return torch.full((3, 5), value, dtype=torch.int16)

            records = load_study_tensors(index, root, {"1.2.3": "1.2.4"}, [_labels("1.2.3")],
                                         input_size=16, pixel_decoder=decode)
            self.assertEqual(1, len(records))
            record = records[0]
            self.assertEqual((24, 1, 16, 16), tuple(record.inputs.shape))
            self.assertTrue(bool(torch.isfinite(record.inputs).all()))
            self.assertGreaterEqual(float(record.inputs.min()), 0.0)
            self.assertLessEqual(float(record.inputs.max()), 1.0)
            self.assertEqual("1.2.3", record.study_id)
            self.assertEqual(("1.2.4",), record.series_ids)
            self.assertEqual(24, len(record.sop_ids))
            selected_indices = [(slot * 47 * 2 + 23) // 46 for slot in range(24)]
            self.assertEqual([f"{i:03}.dcm" for i in selected_indices], decoded)
            self.assertTrue(bool((record.inputs[0] < record.inputs[-1]).all()))

    def test_default_resolution_is_224(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = _indexed(root)
            record = load_study_tensors(index, root, {"1.2.3": "1.2.4"}, {"1.2.3": _labels("1.2.3")},
                                        pixel_decoder=lambda _: torch.arange(16).reshape(4, 4))[0]
            self.assertEqual((24, 1, 224, 224), tuple(record.inputs.shape))

    def test_memmap_store_has_expected_shape_values_and_finite_record_view(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = _indexed(root)
            tensor_store_path = root / "cache" / "cnn224-tensors.npy"

            def decode(path: Path):
                value = int(path.stem)
                return torch.full((3, 5), value, dtype=torch.float32)

            records = load_study_tensors(
                index, root, {"1.2.3": "1.2.4"}, [_labels("1.2.3")],
                input_size=8, pixel_decoder=decode, tensor_store_path=tensor_store_path,
            )
            stored = np.load(tensor_store_path, mmap_mode="r")
            self.assertEqual((1, 24, 1, 8, 8), stored.shape)
            self.assertEqual((24, 1, 8, 8), tuple(records[0].inputs.shape))
            self.assertTrue(bool(torch.isfinite(records[0].inputs).all()))
            self.assertTrue(np.isfinite(stored).all())
            self.assertTrue(np.allclose(records[0].inputs.numpy(), stored[0]))
            self.assertTrue(np.allclose(stored[0, 0], 0.0))
            self.assertTrue(np.allclose(stored[0, -1], 1.0))
            records[0].inputs[0, 0, 0, 0] = 0.375
            self.assertAlmostEqual(0.375, float(np.load(tensor_store_path, mmap_mode="r")[0, 0, 0, 0, 0]))

    def test_short_series_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = _indexed(root, count=23)
            with self.assertRaisesRegex(ValueError, "fewer than requested"):
                load_study_tensors(index, root, {"1.2.3": "1.2.4"}, [_labels("1.2.3")],
                                   input_size=8, pixel_decoder=lambda _: torch.ones((2, 2)))

    def test_series_choice_and_labels_must_explicitly_cover_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = _indexed(root)
            with self.assertRaisesRegex(ValueError, "series mapping must exactly cover"):
                load_study_tensors(index, root, {}, [_labels("1.2.3")], pixel_decoder=lambda _: torch.ones((2, 2)))
            with self.assertRaisesRegex(ValueError, "labels must exactly cover"):
                load_study_tensors(index, root, {"1.2.3": "1.2.4"}, [], pixel_decoder=lambda _: torch.ones((2, 2)))
            with self.assertRaisesRegex(ValueError, "exactly one selected series"):
                load_study_tensors(index, root, {"1.2.3": "not-the-series"}, [_labels("1.2.3")],
                                   pixel_decoder=lambda _: torch.ones((2, 2)))

    def test_duplicate_uids_mismatched_study_or_series_are_rejected(self):
        for options, message in (({"duplicate_sop": True}, "duplicate SOPInstanceUID"),
                                 ({"bad_study_at": 0}, "mismatched StudyInstanceUID"),
                                 ({"bad_series_at": 0}, "mismatched SeriesInstanceUID")):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                index = _indexed(root, **options)
                with self.assertRaisesRegex(ValueError, message):
                    load_study_tensors(index, root, {"1.2.3": "1.2.4"}, [_labels("1.2.3")],
                                       input_size=8, pixel_decoder=lambda _: torch.ones((2, 2)))

    def test_missing_file_invalid_pixels_nonfinite_and_inconsistent_shapes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = _indexed(root)
            (root / "series" / "000.dcm").unlink()
            with self.assertRaises(FileNotFoundError):
                load_study_tensors(index, root, {"1.2.3": "1.2.4"}, [_labels("1.2.3")],
                                   input_size=8, pixel_decoder=lambda _: torch.ones((2, 2)))

        cases = (
            (lambda path: torch.ones((1, 2, 2)), "2D grayscale"),
            (lambda path: torch.full((2, 2), float("nan")), "non-finite"),
        )
        for decoder, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                index = _indexed(root)
                with self.assertRaisesRegex(ValueError, message):
                    load_study_tensors(index, root, {"1.2.3": "1.2.4"}, [_labels("1.2.3")],
                                       input_size=8, pixel_decoder=decoder)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = _indexed(root)

            def changing_shape(path: Path):
                return torch.ones((2, 2)) if path.name != "001.dcm" else torch.ones((2, 3))

            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                load_study_tensors(index, root, {"1.2.3": "1.2.4"}, [_labels("1.2.3")],
                                   input_size=8, pixel_decoder=changing_shape)

    def test_label_identity_and_patient_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = _indexed(root)
            with self.assertRaisesRegex(ValueError, "mapping keys must match"):
                load_study_tensors(index, root, {"1.2.3": "1.2.4"}, {"1.2.3": _labels("1.2.9")},
                                   input_size=8, pixel_decoder=lambda _: torch.ones((2, 2)))
            with self.assertRaisesRegex(ValueError, "PatientID does not match"):
                load_study_tensors(index, root, {"1.2.3": "1.2.4"},
                                   {"1.2.3": _labels("1.2.3", "P2")}, input_size=8,
                                   pixel_decoder=lambda _: torch.ones((2, 2)))

    def test_default_decoder_reads_dicom_pixel_array(self):
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            series_dir = root / "series"
            series_dir.mkdir()
            for index in range(24):
                sop = f"1.2.5.{index + 1}"
                meta = FileMetaDataset()
                meta.MediaStorageSOPClassUID = MRImageStorage
                meta.MediaStorageSOPInstanceUID = sop
                meta.TransferSyntaxUID = ExplicitVRLittleEndian
                ds = FileDataset(str(series_dir / f"{index:03}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
                ds.SOPClassUID = MRImageStorage
                ds.SOPInstanceUID = sop
                ds.StudyInstanceUID = "1.2.3"
                ds.SeriesInstanceUID = "1.2.4"
                ds.PatientID = "P1"
                ds.InstanceNumber = index + 1
                ds.Rows, ds.Columns = 2, 3
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = "MONOCHROME2"
                ds.BitsAllocated = 16
                ds.BitsStored = 16
                ds.HighBit = 15
                ds.PixelRepresentation = 0
                ds.PixelData = np.full((2, 3), index, dtype=np.uint16).tobytes()
                ds.save_as(series_dir / f"{index:03}.dcm", enforce_file_format=True)
            index = discover_dataset(root, on_invalid="strict")
            record = load_study_tensors(index, root, {"1.2.3": "1.2.4"}, [_labels("1.2.3")],
                                        input_size=8)[0]
            self.assertEqual((24, 1, 8, 8), tuple(record.inputs.shape))
            self.assertLess(float(record.inputs[0].mean()), float(record.inputs[-1].mean()))


if __name__ == "__main__":
    unittest.main()
