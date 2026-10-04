import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import make_reference
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

    def spec_without_reference(self, directory):
        path = Path(directory) / "specs" / "model.json"
        path.parent.mkdir()
        source = json.loads(self.spec_path.read_text())
        path.write_text(json.dumps(source))
        return path

    def test_missing_reference_requires_explicit_tolerances(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.spec_without_reference(directory)
            with self.assertRaisesRegex(ValueError, "requires.*tolerance"):
                read_inputs(path, "smoke", self.probe)

    def test_missing_reference_uses_explicit_tolerances(self):
        expected = {"absolute": 0.001, "relative": 0.01, "minimum_cosine": 0.999}
        with tempfile.TemporaryDirectory() as directory:
            path = self.spec_without_reference(directory)
            spec, capability, tolerance = read_inputs(
                path, "smoke", self.probe, tolerance_arguments=expected
            )
            self.assertEqual(tolerance, expected)
            self.assertEqual(capability["semantic_mode"], "query")
            self.assertEqual(
                spec["model"], json.loads(self.spec_path.read_text())["model"]
            )

    def test_missing_reference_refuses_partial_tolerances(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.spec_without_reference(directory)
            with self.assertRaisesRegex(ValueError, "requires.*tolerance"):
                read_inputs(
                    path, "smoke", self.probe, tolerance_arguments={"absolute": 0.001}
                )

    def test_existing_reference_keeps_its_tolerances(self):
        arguments = {"absolute": 0.2, "relative": 0.3, "minimum_cosine": 0.4}
        spec, capability, tolerance = read_inputs(
            self.spec_path, "smoke", self.probe, tolerance_arguments=arguments
        )
        self.assertEqual(
            tolerance, {"absolute": 0.001, "relative": 0.01, "minimum_cosine": 0.999}
        )
        self.assertEqual(capability["semantic_mode"], "query")
        self.assertEqual(spec["model"]["representation"], "dense")

    def test_unknown_profile_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown profile"):
            read_inputs(self.spec_path, "missing", self.probe)

    def test_empty_probe_is_refused(self):
        with self.assertRaisesRegex(ValueError, "probe text must not be empty"):
            read_inputs(self.spec_path, "smoke", " ")

    def test_probe_file_preserves_whitespace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.txt"
            path.write_bytes("  café\tline\r\n\r\n".encode("utf-8"))
            self.assertEqual(
                make_reference.read_probe(None, path), "  café\tline\r\n\r\n"
            )

    def test_probe_sources_are_exclusive(self):
        for text, file in [(None, None), ("text", Path("probe.txt"))]:
            with self.assertRaisesRegex(ValueError, "exactly one"):
                make_reference.read_probe(text, file)

    def test_code_source_requires_both_pins(self):
        for repository, revision in [("owner/code", None), (None, "a" * 40)]:
            with self.assertRaisesRegex(ValueError, "both"):
                make_reference.validate_code_source(repository, revision, {})

    def test_code_source_refuses_floating_revision(self):
        with self.assertRaisesRegex(ValueError, "immutable"):
            make_reference.validate_code_source("owner/code", "main", {})

    def test_code_source_refuses_another_repository(self):
        config = {"auto_map": {"AutoConfig": "other/code--config.Config"}}
        with self.assertRaisesRegex(ValueError, "repository"):
            make_reference.validate_code_source("owner/code", "a" * 40, config)

    def test_code_source_accepts_only_the_named_repository(self):
        config = {"auto_map": {"AutoConfig": "owner/code--config.Config"}}
        self.assertEqual(
            make_reference.validate_code_source("owner/code", "a" * 40, config),
            ("owner/code", "a" * 40),
        )

    def test_default_code_source_is_disabled(self):
        self.assertIsNone(make_reference.validate_code_source(None, None, {}))

    def test_direct_module_loading_preserves_both_code_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            (snapshot / "modules.json").write_text(
                json.dumps(
                    [
                        {
                            "idx": 0,
                            "type": "sentence_transformers.models.Transformer",
                            "path": "",
                        },
                        {
                            "idx": 1,
                            "type": "sentence_transformers.models.Pooling",
                            "path": "1_Pooling",
                        },
                    ]
                )
            )
            transformer, pooling = Mock(), Mock()
            modules = make_reference.load_pinned_modules(
                snapshot,
                snapshot,
                ("owner/code", "a" * 40),
                classes={"Transformer": transformer, "Pooling": pooling},
            )
            arguments = transformer.load.call_args.kwargs
            self.assertEqual(arguments["model_kwargs"]["code_revision"], "a" * 40)
            self.assertEqual(arguments["config_kwargs"]["code_revision"], "a" * 40)
            self.assertEqual(arguments["processor_kwargs"]["code_revision"], "a" * 40)
            self.assertTrue(arguments["trust_remote_code"])
            self.assertTrue(arguments["local_files_only"])
            self.assertFalse(arguments["token"])
            self.assertEqual(pooling.load.call_args.kwargs["subfolder"], "1_Pooling")
            self.assertEqual(
                modules, [transformer.load.return_value, pooling.load.return_value]
            )

    def test_direct_module_loading_refuses_external_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            (snapshot / "modules.json").write_text(
                json.dumps(
                    [
                        {"idx": 0, "type": "other.CustomModule", "path": ""},
                    ]
                )
            )
            with self.assertRaisesRegex(ValueError, "built-in"):
                make_reference.load_pinned_modules(
                    snapshot, snapshot, ("owner/code", "a" * 40), classes={}
                )

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
