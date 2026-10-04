from types import SimpleNamespace
from unittest.mock import MagicMock, Mock
import unittest

import make_retrieval_reference as reference


class RetrievalReferenceTests(unittest.TestCase):
    def test_sparse_uses_masked_language_head_log_relu_mask_and_max(self):
        dtype = object()
        vector = SimpleNamespace(dtype=dtype, ndim=1, tolist=lambda: [0.0, 0.5, 0.0, 1.5])
        pooled = MagicMock()
        pooled.__getitem__.return_value = vector
        torch = SimpleNamespace(float32=dtype, relu=Mock(return_value=MagicMock()),
                                log=Mock(return_value=MagicMock()),
                                max=Mock(return_value=SimpleNamespace(values=pooled)))
        model = Mock(return_value=SimpleNamespace(logits=object()))
        inputs = {"input_ids": object(), "attention_mask": MagicMock()}
        result = reference.sparse_output(model, inputs, torch)
        model.assert_called_once_with(**inputs)
        torch.relu.assert_called_once_with(model.return_value.logits)
        torch.log.assert_called_once_with(1 + torch.relu.return_value)
        inputs["attention_mask"].unsqueeze.assert_called_once_with(-1)
        torch.max.assert_called_once_with(
            torch.log.return_value * inputs["attention_mask"].unsqueeze.return_value, dim=1)
        self.assertEqual(result, {"representation": "sparse", "vocabulary_size": 4,
                                  "indices": [1, 3], "values": [0.5, 1.5]})

    def test_sparse_empty_and_nonfinite_outputs_are_refused(self):
        for values in [[0.0], [float("nan")], [float("inf")]]:
            with self.assertRaisesRegex(ValueError, "positive|finite"):
                reference.sparse_values(values)

    def test_upstream_query_and_document_calls_keep_their_masks(self):
        dtype = object()
        matrix = SimpleNamespace(dtype=dtype, ndim=2, shape=(2, 96),
                                 tolist=lambda: [[1.0] + [0.0] * 95, [-1.0] + [0.0] * 95])
        query_result = MagicMock(); query_result.__getitem__.return_value = matrix
        operations = SimpleNamespace(query=Mock(return_value=query_result), doc=Mock(return_value=[matrix]))
        session, ids, mask = object(), object(), object()
        torch = SimpleNamespace(float32=dtype)
        for role in ["late_interaction_query", "late_interaction_document"]:
            result = reference.colbert_output(session, (ids, mask), role, operations, torch)
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
        tokenizer = Mock(); tokenizer.encode.return_value = list(range(31))
        self.assertEqual(reference.probe_count(tokenizer, "text", 32, extra_tokens=1), 31)
        tokenizer.encode.assert_called_once_with("text", add_special_tokens=True, truncation=False)
        tokenizer.encode.return_value = list(range(32))
        with self.assertRaisesRegex(ValueError, "marker"):
            reference.probe_count(tokenizer, "text", 32, extra_tokens=1)

    def test_loading_errors_are_named_instead_of_initializing_missing_weights(self):
        for info in [{"missing_keys": ["linear.weight"]}, {"unexpected_keys": ["extra"]},
                     {"mismatched_keys": ["linear.weight"]}, {"error_msgs": ["broken"]}]:
            with self.assertRaisesRegex(ValueError, "loading"):
                reference.validate_loading_info(info)

    def test_reference_records_model_pin_and_upstream_code_pin(self):
        identity = {"id": "colbert-small", "repository": "owner/model", "revision": "a" * 40,
                    "representation": "multi_vector"}
        output = {"representation": "multi_vector", "rows": 1, "columns": 96, "values": [1.0] * 96}
        result = reference.document(identity, {}, {}, "smoke", "text", 3, output, "versions")
        self.assertEqual(result["provenance"]["source_revision"], identity["revision"])
        self.assertIn(reference.COLBERT_REVISION, result["provenance"]["producer"])
        self.assertEqual(result["expected"], output)


if __name__ == "__main__":
    unittest.main()
