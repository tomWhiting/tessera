"""Network-free public-contract coverage for the Tessera extension."""

import pytest

import tessera


EMBEDDER_CLASSES = (
    tessera.TesseraDense,
    tessera.TesseraMultiVector,
    tessera.TesseraSparse,
    tessera.TesseraVision,
)

CATALOG_ONLY_MODELS = (
    (tessera.TesseraDense, "jina-embeddings-v3"),
    (tessera.TesseraMultiVector, "gte-modern-colbert"),
    (tessera.TesseraSparse, "minicoil-v1"),
    (tessera.TesseraVision, "colpali-v1.3-hf"),
)


def test_module_exports_supported_embedders():
    assert tessera.__version__
    assert callable(tessera.ResourcePolicy)
    assert all(callable(embedder_class) for embedder_class in EMBEDDER_CLASSES)
    assert not hasattr(tessera, "TesseraTimeSeries")


def test_resource_policy_defaults_and_immutable_updates():
    default_policy = tessera.ResourcePolicy()

    assert default_policy.max_sequence_tokens == 512
    assert default_policy.max_batch_items == 16
    assert default_policy.max_batch_tokens == 2_048
    assert default_policy.max_model_bytes == 2 * 1_024 * 1_024 * 1_024
    assert default_policy.max_input_bytes_per_sequence == 1_024 * 1_024
    assert default_policy.max_attention_cells == 1_048_576

    long_context = default_policy.with_max_sequence_tokens(
        8_192
    ).with_max_attention_cells(8_192**2)
    vision_budget = (
        long_context.with_max_batch_items(4)
        .with_max_batch_tokens(32_768)
        .with_max_model_bytes(12_000_000_000)
        .with_max_input_bytes_per_sequence(2 * 1_024 * 1_024)
    )
    assert default_policy.max_sequence_tokens == 512
    assert long_context.max_sequence_tokens == 8_192
    assert long_context.max_attention_cells == 8_192**2
    assert vision_budget.max_batch_items == 4
    assert vision_budget.max_batch_tokens == 32_768
    assert vision_budget.max_model_bytes == 12_000_000_000
    assert vision_budget.max_input_bytes_per_sequence == 2 * 1_024 * 1_024
    assert "max_sequence_tokens=8192" in repr(vision_budget)
    with pytest.raises(AttributeError):
        setattr(default_policy, "max_sequence_tokens", 8_192)


@pytest.mark.parametrize("embedder_class", EMBEDDER_CLASSES)
def test_resource_policy_is_keyword_only(embedder_class):
    with pytest.raises(TypeError):
        embedder_class("not-a-real-tessera-model", tessera.ResourcePolicy())


@pytest.mark.parametrize("embedder_class", EMBEDDER_CLASSES)
def test_resource_policy_reaches_builder_preflight(embedder_class):
    policy = tessera.ResourcePolicy(max_sequence_tokens=1_000_000)
    model_ids = {
        tessera.TesseraDense: "bge-base-en-v1.5",
        tessera.TesseraMultiVector: "colbert-small",
        tessera.TesseraSparse: "splade-pp-en-v1",
        tessera.TesseraVision: "colpali-v1.2",
    }

    with pytest.raises(ValueError, match="sequence token limit"):
        embedder_class(model_ids[embedder_class], resource_policy=policy)


@pytest.mark.parametrize("embedder_class", EMBEDDER_CLASSES)
def test_unknown_model_fails_without_network(embedder_class):
    with pytest.raises(RuntimeError, match="not found in registry"):
        embedder_class("not-a-real-tessera-model")


@pytest.mark.parametrize(("embedder_class", "model_id"), CATALOG_ONLY_MODELS)
def test_catalog_only_model_fails_without_network(embedder_class, model_id):
    with pytest.raises(ValueError, match="catalog-only"):
        embedder_class(model_id)


def test_dense_rejects_wrong_model_paradigm_before_loading():
    with pytest.raises(ValueError, match="not a dense model"):
        tessera.TesseraDense("colbert-v2")


def test_sparse_rejects_wrong_model_paradigm_before_loading():
    with pytest.raises(ValueError, match="not Sparse"):
        tessera.TesseraSparse("bge-base-en-v1.5")


def test_large_vision_model_requires_explicit_resource_budget():
    with pytest.raises(ValueError, match="model parameter bytes"):
        tessera.TesseraVision("colpali-v1.2")
