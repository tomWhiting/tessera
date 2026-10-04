import json
import os
from pathlib import Path
import tempfile
import unittest

from make_reference import publish, read_inputs


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.spec_path = (
            Path(__file__).resolve().parents[1] / "specs" / "bge-base-en-v1.5.json"
        )
        self.probe = "What is machine learning?"

    def test_profile_values_come_from_spec(self):
        source = json.loads(self.spec_path.read_text())
        spec, capability, tolerance = read_inputs(self.spec_path, "smoke", self.probe)
        self.assertEqual(spec["model"], source["model"])
        self.assertEqual(capability, source["profiles"]["smoke"]["capability"])
        self.assertEqual(tolerance["minimum_cosine"], 0.999)

    def test_unknown_profile_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown profile"):
            read_inputs(self.spec_path, "missing", self.probe)

    def test_empty_probe_is_refused(self):
        with self.assertRaisesRegex(ValueError, "probe text must not be empty"):
            read_inputs(self.spec_path, "smoke", " ")

    def test_mutable_revision_is_refused(self):
        source = json.loads(self.spec_path.read_text())
        source["model"]["revision"] = "main"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spec.json"
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, "immutable"):
                read_inputs(path, "smoke", self.probe)

    def test_existing_output_survives_publication_race(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reference.json"
            output.write_bytes(b"preserve this")
            with self.assertRaises(FileExistsError):
                publish(output, {"expected": [1.0]})
            self.assertEqual(output.read_bytes(), b"preserve this")
            self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_dangling_symlink_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reference.json"
            target = Path(directory) / "missing.json"
            output.symlink_to(target)
            with self.assertRaises(FileExistsError):
                publish(output, {"expected": [1.0]})
            self.assertEqual(Path(os.readlink(output)), target)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_publication_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            value = {"probe": "é", "expected": [0.25, -0.5]}
            publish(first, value)
            publish(second, value)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(json.loads(first.read_text()), value)

    def test_nonfinite_output_is_refused_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                publish(
                    Path(directory) / "reference.json", {"expected": [float("nan")]}
                )
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
