from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch
from pathlib import Path
import json
import tempfile
import unittest

import make_retrieval_reference as reference


class RetrievalReferenceTests(unittest.TestCase):
    def test_v2_recipes_use_their_spec_identity_and_dimension(self):
        for name in ["splade-pp-en-v2", "colbert-v2"]:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as folder:
                path = Path(__file__).resolve().parents[1] / "specs" / f"{name}.json"
                spec = json.loads(path.read_text())
                capability = spec["profiles"]["smoke"]["capability"]
                snapshot = Path(folder)
                (snapshot / "config.json").write_text('{"model_type": "bert"}')
                tokenizer = Mock()
                tokenizer.encode.return_value = [1, 2, 3]
                transformers = MagicMock()
                transformers.__version__ = "pinned"
                transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
                transformers.AutoModelForMaskedLM.from_pretrained.return_value = (
                    Mock(),
                    {},
                )
                torch = MagicMock()
                torch.__version__ = "pinned"
                session, query, document, operations = (Mock() for _ in range(4))
                output = {"representation": spec["model"]["representation"]}
                with (
                    patch.dict(
                        "sys.modules", {"torch": torch, "transformers": transformers}
                    ),
                    patch.object(
                        reference, "sparse_output", return_value=output
                    ) as sparse,
                    patch.object(
                        reference,
                        "load_colbert",
                        return_value=(
                            session,
                            query,
                            document,
                            operations,
                            {
                                "attend_to_mask_tokens": {
                                    "value": False,
                                    "source": "metadata",
                                }
                            },
                        ),
                    ) as load,
                    patch.object(
                        reference, "colbert_output", return_value=output
                    ) as matrix,
                    patch.object(
                        reference, "colbert_source_version", return_value="pinned"
                    ),
                ):
                    result = reference.make_reference(
                        spec,
                        capability,
                        {"absolute": 0.001, "relative": 0.01, "minimum_cosine": 0.999},
                        "smoke",
                        reference.PROBES["smoke"],
                        snapshot,
                    )
                self.assertEqual(result["model_id"], name)
                self.assertEqual(result["revision"], spec["model"]["revision"])
                self.assertEqual(result["expected"], output)
                if name == "colbert-v2":
                    load.assert_called_once_with(snapshot, 128, 128, torch)
                    self.assertEqual(matrix.call_args.args[-1], 128)
                    self.assertIn(
                        '"attend_to_mask_tokens": {"source": "metadata", "value": false}',
                        result["provenance"]["producer"],
                    )
                    sparse.assert_not_called()
                else:
                    sparse.assert_called_once()
                    load.assert_not_called()
                    matrix.assert_not_called()

    def test_colbert_projection_can_have_128_columns(self):
        dtype = object()
        matrix = SimpleNamespace(dtype=dtype, ndim=2, tolist=lambda: [[1.0] * 128])
        operations = SimpleNamespace(query=Mock(return_value=[matrix]))
        result = reference.colbert_output(
            object(),
            (object(), object()),
            "late_interaction_query",
            operations,
            SimpleNamespace(float32=dtype),
            128,
        )
        self.assertEqual(result["columns"], 128)
        self.assertEqual(result["values"], [1.0] * 128)

    def test_colbert_marker_defaults_are_verified_and_recorded(self):
        defaults = SimpleNamespace(query_token_id="[unused0]", doc_token_id="[unused1]")
        tokenizer = Mock()
        vocabulary = {"[unused0]": 1, "[unused1]": 2}
        tokenizer.get_vocab.return_value = vocabulary
        tokenizer.convert_tokens_to_ids.side_effect = vocabulary.get
        tokenizer.unk_token_id = 100
        artifact = {
            "dim": 128,
            "query_maxlen": 32,
            "mask_punctuation": True,
            "attend_to_mask_tokens": False,
        }
        for explicit in [False, True]:
            with self.subTest(explicit=explicit):
                selected = dict(artifact)
                if explicit:
                    selected.update(
                        query_token_id="[unused0]", doc_token_id="[unused1]"
                    )
                settings, recorded = reference.colbert_settings(
                    selected, 128, defaults, tokenizer
                )
                self.assertEqual(settings["dim"], 128)
                for name, token in [
                    ("query_token_id", "[unused0]"),
                    ("doc_token_id", "[unused1]"),
                ]:
                    self.assertEqual(
                        recorded[name],
                        {
                            "value": token,
                            "source": "metadata" if explicit else "upstream_default",
                            "token_id": vocabulary[token],
                        },
                    )
        for name in ["query_token_id", "doc_token_id"]:
            with (
                self.subTest(conflict=name),
                self.assertRaisesRegex(ValueError, "settings"),
            ):
                reference.colbert_settings(
                    {**artifact, name: "[unused2]"}, 128, defaults, tokenizer
                )
        tokenizer.get_vocab.return_value = {"[unused1]": 2}
        with self.assertRaisesRegex(ValueError, "marker"):
            reference.colbert_settings(artifact, 128, defaults, tokenizer)

    def test_all_short_profiles_are_ready_for_reference_generation(self):
        for name in ["splade-pp-en-v1", "colbert-small"]:
            path = Path(__file__).resolve().parents[1] / "specs" / f"{name}.json"
            for profile, probe in reference.PROBES.items():
                spec, capability, tolerance = reference.read_inputs(
                    path,
                    profile,
                    probe,
                    {"absolute": 0.001, "relative": 0.01, "minimum_cosine": 0.999},
                    representations=("sparse", "multi_vector"),
                )
                self.assertEqual(spec["model"]["id"], name)
                self.assertEqual(capability["max_sequence_tokens"], 128)
                reference.validate_tolerance(tolerance)

    def test_sparse_uses_masked_language_head_log_relu_mask_and_max(self):
        dtype = object()
        vector = SimpleNamespace(
            dtype=dtype, ndim=1, tolist=lambda: [0.0, 0.5, 0.0, 1.5]
        )
        pooled = MagicMock()
        pooled.__getitem__.return_value = vector
        torch = SimpleNamespace(
            float32=dtype,
            relu=Mock(return_value=MagicMock()),
            log=Mock(return_value=MagicMock()),
            max=Mock(return_value=SimpleNamespace(values=pooled)),
        )
        model = Mock(return_value=SimpleNamespace(logits=object()))
        inputs = {"input_ids": object(), "attention_mask": MagicMock()}
        result = reference.sparse_output(model, inputs, torch)
        model.assert_called_once_with(**inputs)
        torch.relu.assert_called_once_with(model.return_value.logits)
        torch.log.assert_called_once_with(1 + torch.relu.return_value)
        inputs["attention_mask"].unsqueeze.assert_called_once_with(-1)
        torch.max.assert_called_once_with(
            torch.log.return_value * inputs["attention_mask"].unsqueeze.return_value,
            dim=1,
        )
        self.assertEqual(
            result,
            {
                "representation": "sparse",
                "vocabulary_size": 4,
                "indices": [1, 3],
                "values": [0.5, 1.5],
            },
        )

    def test_sparse_empty_and_nonfinite_outputs_are_refused(self):
        for values in [[0.0], [float("nan")], [float("inf")]]:
            with self.assertRaisesRegex(ValueError, "positive|finite"):
                reference.sparse_values(values)

    def test_upstream_query_and_document_calls_keep_their_masks(self):
        dtype = object()
        matrix = SimpleNamespace(
            dtype=dtype,
            ndim=2,
            shape=(2, 96),
            tolist=lambda: [[1.0] + [0.0] * 95, [-1.0] + [0.0] * 95],
        )
        query_result = MagicMock()
        query_result.__getitem__.return_value = matrix
        operations = SimpleNamespace(
            query=Mock(return_value=query_result), doc=Mock(return_value=[matrix])
        )
        session, ids, mask = object(), object(), object()
        torch = SimpleNamespace(float32=dtype)
        for role in ["late_interaction_query", "late_interaction_document"]:
            result = reference.colbert_output(
                session, (ids, mask), role, operations, torch, 96
            )
            self.assertEqual(result["rows"], 2)
            self.assertEqual(result["columns"], 96)
            self.assertEqual(result["values"], matrix.tolist()[0] + matrix.tolist()[1])
        operations.query.assert_called_once_with(session, ids, mask)
        operations.doc.assert_called_once_with(session, ids, mask, keep_dims=False)

    def test_invalid_matrix_shape_or_dimension_is_refused(self):
        for matrix in [[], [[1.0]], [[0.0] * 96, [0.0] * 95]]:
            with self.assertRaisesRegex(ValueError, "matrix"):
                reference.matrix_values(matrix, 96)

    def test_probe_count_is_untruncated_and_reserves_marker_space(self):
        tokenizer = Mock()
        tokenizer.encode.return_value = list(range(31))
        self.assertEqual(
            reference.probe_count(tokenizer, "text", 32, extra_tokens=1), 31
        )
        tokenizer.encode.assert_called_once_with(
            "text", add_special_tokens=True, truncation=False
        )
        tokenizer.encode.return_value = list(range(32))
        with self.assertRaisesRegex(ValueError, "marker"):
            reference.probe_count(tokenizer, "text", 32, extra_tokens=1)

    def test_loading_errors_are_named_instead_of_initializing_missing_weights(self):
        for info in [
            {"missing_keys": ["linear.weight"]},
            {"unexpected_keys": ["extra"]},
            {"mismatched_keys": ["linear.weight"]},
            {"error_msgs": ["broken"]},
        ]:
            with self.assertRaisesRegex(ValueError, "loading"):
                reference.validate_loading_info(info)

    def test_tolerance_must_meet_the_current_comparator_bounds(self):
        valid = {"absolute": 0.001, "relative": 0.01, "minimum_cosine": 0.999}
        reference.validate_tolerance(valid)
        for name, value in [
            ("absolute", 0.002),
            ("relative", 0.02),
            ("minimum_cosine", 0.998),
            ("minimum_cosine", float("nan")),
        ]:
            with self.assertRaisesRegex(ValueError, "tolerance"):
                reference.validate_tolerance({**valid, name: value})

    def test_reference_records_model_pin_and_upstream_code_pin(self):
        identity = {
            "id": "colbert-small",
            "repository": "owner/model",
            "revision": "a" * 40,
            "representation": "multi_vector",
        }
        output = {
            "representation": "multi_vector",
            "rows": 1,
            "columns": 96,
            "values": [1.0] * 96,
        }
        result = reference.document(
            identity, {}, {}, "smoke", "text", 3, output, "versions"
        )
        self.assertEqual(result["provenance"]["source_revision"], identity["revision"])
        self.assertIn(reference.COLBERT_REVISION, result["provenance"]["producer"])
        self.assertEqual(result["expected"], output)


if __name__ == "__main__":
    unittest.main()
