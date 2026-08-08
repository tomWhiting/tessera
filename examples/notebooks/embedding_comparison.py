import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(
        """
    # Embedding Paradigms with Tessera

    Explore dense, multi-vector, or sparse retrieval over the same small corpus.
    The safe default loads **one selected model**. A separate, conspicuous
    high-memory opt-in enables the side-by-side three-model comparison.

    The registered model adapters used here are experimental. This notebook is
    an API exploration tool, not a quality or throughput benchmark.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from embedding_comparison_data import COLORS_MAP, DATASET
    from plotly.subplots import make_subplots
    from scipy.sparse import csr_matrix
    from tessera import TesseraDense, TesseraMultiVector, TesseraSparse
    from umap import UMAP

    return (
        COLORS_MAP,
        DATASET,
        TesseraDense,
        TesseraMultiVector,
        TesseraSparse,
        UMAP,
        csr_matrix,
        go,
        make_subplots,
        mo,
        np,
        pd,
    )


@app.cell
def _(COLORS_MAP, DATASET, mo):
    colors_map = COLORS_MAP
    dataset = DATASET
    texts = [text for documents in dataset.values() for text in documents]
    labels = [category for category, documents in dataset.items() for _ in documents]
    colors = [colors_map[label] for label in labels]

    mo.md(f"**Dataset:** {len(texts)} documents across {len(dataset)} categories")
    return colors, colors_map, labels, texts


@app.cell
def _(mo):
    paradigm_specs = {
        "dense": {
            "choice": "Dense — BGE base",
            "label": "Dense (BGE)",
            "model_id": "bge-base-en-v1.5",
            "plot_title": "Dense (BGE): single vector",
        },
        "colbert": {
            "choice": "Multi-vector — ColBERT v2",
            "label": "ColBERT",
            "model_id": "colbert-v2",
            "plot_title": "ColBERT: token vectors",
        },
        "sparse": {
            "choice": "Sparse — SPLADE PP v1",
            "label": "SPLADE",
            "model_id": "splade-pp-en-v1",
            "plot_title": "SPLADE: vocabulary weights",
        },
    }
    choice_to_paradigm = {
        _spec["choice"]: _paradigm for _paradigm, _spec in paradigm_specs.items()
    }
    model_choice = mo.ui.dropdown(
        options=list(choice_to_paradigm),
        value=paradigm_specs["dense"]["choice"],
        label="Single model to load (safe default)",
    )
    high_memory_opt_in = mo.ui.checkbox(
        value=False,
        label=(
            "I explicitly opt in to HIGH MEMORY mode: load Dense + ColBERT + "
            "SPLADE simultaneously"
        ),
    )

    mo.vstack(
        [
            mo.md(
                """
                ## Model loading mode

                **Default:** exactly one model is constructed. The checkbox
                below is the only path that constructs all three together.

                > ⚠️ **HIGH MEMORY:** all-three mode retains three model objects,
                > three corpus embedding sets, and three UMAP reducers. Per-model
                > resource guards do not provide an aggregate process-memory cap.
                """
            ),
            model_choice,
            high_memory_opt_in,
        ]
    )
    return choice_to_paradigm, high_memory_opt_in, model_choice, paradigm_specs


@app.cell
def _(choice_to_paradigm, high_memory_opt_in, model_choice, mo, paradigm_specs):
    if high_memory_opt_in.value:
        active_paradigms = ("dense", "colbert", "sparse")
        _mode_description = "⚠️ HIGH MEMORY opt-in active: loading all three models"
    else:
        active_paradigms = (choice_to_paradigm[model_choice.value],)
        _selected_spec = paradigm_specs[active_paradigms[0]]
        _mode_description = (
            f"Safe single-model mode: loading {_selected_spec['label']} "
            f"(`{_selected_spec['model_id']}`)"
        )

    mo.md(f"**Current mode:** {_mode_description}")
    return (active_paradigms,)


@app.cell
def _(
    TesseraDense,
    TesseraMultiVector,
    TesseraSparse,
    active_paradigms,
    mo,
    paradigm_specs,
):
    models = {}
    with mo.status.spinner(title="Loading selected model(s)..."):
        for _paradigm in active_paradigms:
            _model_id = paradigm_specs[_paradigm]["model_id"]
            if _paradigm == "dense":
                models[_paradigm] = TesseraDense(_model_id)
            elif _paradigm == "colbert":
                models[_paradigm] = TesseraMultiVector(_model_id)
            else:
                models[_paradigm] = TesseraSparse(_model_id)

    _loaded = ", ".join(paradigm_specs[_name]["label"] for _name in models)
    mo.md(f"✓ Loaded: **{_loaded}**")
    return (models,)


@app.cell
def _(active_paradigms, mo, models, np, paradigm_specs, texts):
    document_embeddings = {}
    _details = []
    with mo.status.spinner(title="Encoding corpus with selected model(s)..."):
        for _paradigm in active_paradigms:
            _model = models[_paradigm]
            if _paradigm == "dense":
                _embeddings = np.asarray([_model.encode(_text) for _text in texts])
                _details.append(
                    f"- {paradigm_specs[_paradigm]['label']}: {_embeddings.shape}"
                )
            else:
                _embeddings = [_model.encode(_text) for _text in texts]
                _details.append(
                    f"- {paradigm_specs[_paradigm]['label']}: "
                    f"{len(_embeddings)} document embeddings"
                )
            document_embeddings[_paradigm] = _embeddings

    mo.md("## Corpus embeddings\n\n" + "\n".join(_details))
    return (document_embeddings,)


@app.cell
def _(
    UMAP,
    active_paradigms,
    csr_matrix,
    document_embeddings,
    mo,
    np,
):
    projection_state = {}
    with mo.status.spinner(title="Fitting selected UMAP projection(s)..."):
        for _paradigm in active_paradigms:
            _embeddings = document_embeddings[_paradigm]
            _reducer = UMAP(
                n_neighbors=15,
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            )
            if _paradigm == "dense":
                _projection_input = _embeddings
            elif _paradigm == "colbert":
                _projection_input = np.asarray(
                    [_embedding.mean(axis=0) for _embedding in _embeddings]
                )
            else:
                _sparse_indices = []
                _sparse_values = []
                _sparse_indptr = [0]
                for _indices, _values in _embeddings:
                    _sparse_indices.extend(_indices)
                    _sparse_values.extend(_values)
                    _sparse_indptr.append(len(_sparse_indices))
                _vocab_width = max(_sparse_indices, default=0) + 1
                _projection_input = csr_matrix(
                    (_sparse_values, _sparse_indices, _sparse_indptr),
                    shape=(len(_embeddings), _vocab_width),
                ).toarray()

            projection_state[_paradigm] = {
                "points": _reducer.fit_transform(_projection_input),
                "reducer": _reducer,
                "input_width": _projection_input.shape[1],
            }

    mo.md("✓ Selected UMAP projection(s) computed")
    return (projection_state,)


@app.cell
def _(mo):
    query_input = mo.ui.text(
        value="How do neural networks learn from data?",
        placeholder="Enter your search query...",
        label="Search query",
        full_width=True,
    )
    mo.vstack([mo.md("## Interactive query search"), query_input])
    return (query_input,)


@app.cell
def _(active_paradigms, document_embeddings, mo, models, np, query_input, texts):
    def _cosine(_left, _right):
        _left_array = np.asarray(_left)
        _right_array = np.asarray(_right)
        _denominator = np.linalg.norm(_left_array) * np.linalg.norm(_right_array)
        return (
            0.0
            if _denominator == 0
            else float(np.dot(_left_array, _right_array) / _denominator)
        )

    def _max_sim(_query_embedding, _document_embedding):
        _query_array = np.asarray(_query_embedding)
        _document_array = np.asarray(_document_embedding)
        _query_norms = np.linalg.norm(_query_array, axis=1, keepdims=True)
        _document_norms = np.linalg.norm(_document_array, axis=1, keepdims=True)
        _query_unit = np.divide(
            _query_array,
            _query_norms,
            out=np.zeros_like(_query_array),
            where=_query_norms != 0,
        )
        _document_unit = np.divide(
            _document_array,
            _document_norms,
            out=np.zeros_like(_document_array),
            where=_document_norms != 0,
        )
        return float((_query_unit @ _document_unit.T).max(axis=1).sum())

    def _sparse_dot(_query_embedding, _document_embedding):
        _query_indices, _query_values = _query_embedding
        _document_indices, _document_values = _document_embedding
        _query_lookup = dict(zip(_query_indices, _query_values))
        return float(
            sum(
                _query_lookup.get(_index, 0.0) * _value
                for _index, _value in zip(_document_indices, _document_values)
            )
        )

    query_embeddings = {}
    scores = {}
    if query_input.value:
        with mo.status.spinner(title="Computing selected similarities..."):
            for _paradigm in active_paradigms:
                _query_embedding = models[_paradigm].encode(query_input.value)
                _documents = document_embeddings[_paradigm]
                if _paradigm == "dense":
                    _paradigm_scores = [
                        _cosine(_query_embedding, _document) for _document in _documents
                    ]
                elif _paradigm == "colbert":
                    _paradigm_scores = [
                        _max_sim(_query_embedding, _document)
                        for _document in _documents
                    ]
                else:
                    _paradigm_scores = [
                        _sparse_dot(_query_embedding, _document)
                        for _document in _documents
                    ]
                query_embeddings[_paradigm] = _query_embedding
                scores[_paradigm] = np.asarray(_paradigm_scores)
    else:
        query_embeddings = dict.fromkeys(active_paradigms)
        scores = {_name: np.zeros(len(texts)) for _name in active_paradigms}

    top_k = 5
    top_indices = {
        _name: np.argsort(_values)[::-1][:top_k] if query_input.value else []
        for _name, _values in scores.items()
    }
    top_results = {_name: set(_indices) for _name, _indices in top_indices.items()}
    mo.md(f"**Query:** {query_input.value or '_enter a query_'}")
    return query_embeddings, scores, top_indices, top_k, top_results


@app.cell
def _(
    active_paradigms,
    colors,
    go,
    labels,
    make_subplots,
    mo,
    np,
    paradigm_specs,
    projection_state,
    query_embeddings,
    query_input,
    texts,
    top_results,
):
    _column_count = len(active_paradigms)
    _figure = make_subplots(
        rows=1,
        cols=_column_count,
        subplot_titles=tuple(
            paradigm_specs[_name]["plot_title"] for _name in active_paradigms
        ),
        horizontal_spacing=0.08 if _column_count > 1 else 0.0,
    )

    for _column, _paradigm in enumerate(active_paradigms, 1):
        _points = projection_state[_paradigm]["points"]
        _selected_indices = top_results[_paradigm]
        for _index, _text in enumerate(texts):
            _is_top_result = _index in _selected_indices
            _hover_text = _text[:100] + "..." if len(_text) > 100 else _text
            _figure.add_trace(
                go.Scatter(
                    x=[_points[_index, 0]],
                    y=[_points[_index, 1]],
                    mode="markers",
                    marker={
                        "size": 12 if _is_top_result else 8,
                        "color": colors[_index],
                        "symbol": "star" if _is_top_result else "circle",
                        "line": {
                            "width": 2 if _is_top_result else 0,
                            "color": "black",
                        },
                    },
                    customdata=[_hover_text],
                    hovertemplate=(
                        f"<b>%{{customdata}}</b><br>Category: {labels[_index]}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1,
                col=_column,
            )

        if query_input.value:
            _query_embedding = query_embeddings[_paradigm]
            if _paradigm == "colbert":
                _query_projection_input = np.asarray(_query_embedding).mean(axis=0)
            elif _paradigm == "sparse":
                _query_projection_input = np.zeros(
                    projection_state[_paradigm]["input_width"]
                )
                _query_indices, _query_values = _query_embedding
                for _index, _value in zip(_query_indices, _query_values):
                    if _index < len(_query_projection_input):
                        _query_projection_input[_index] = _value
            else:
                _query_projection_input = _query_embedding

            _query_point = projection_state[_paradigm]["reducer"].transform(
                [_query_projection_input]
            )[0]
            _figure.add_trace(
                go.Scatter(
                    x=[_query_point[0]],
                    y=[_query_point[1]],
                    mode="markers",
                    marker={
                        "size": 20,
                        "color": "red",
                        "symbol": "x",
                        "line": {"width": 3, "color": "darkred"},
                    },
                    customdata=[f"QUERY: {query_input.value}"],
                    hovertemplate="<b>%{customdata}</b><extra></extra>",
                    showlegend=False,
                ),
                row=1,
                col=_column,
            )

        _figure.update_xaxes(title_text="UMAP 1", row=1, col=_column)

    _figure.update_yaxes(title_text="UMAP 2", row=1, col=1)
    _figure.update_layout(
        height=500,
        title_text="Selected embedding space (top five marked with stars)",
        showlegend=False,
    )
    mo.ui.plotly(_figure)
    return


@app.cell
def _(
    active_paradigms, labels, mo, paradigm_specs, pd, scores, texts, top_indices, top_k
):
    if not all(len(top_indices[_name]) >= top_k for _name in active_paradigms):
        _result_display = mo.md("Enter a query to populate the top-five table.")
    else:
        _comparison_rows = []
        for _rank in range(top_k):
            _row = {"Rank": _rank + 1}
            for _paradigm in active_paradigms:
                _document_index = int(top_indices[_paradigm][_rank])
                _label = paradigm_specs[_paradigm]["label"]
                _row[f"{_label} document"] = texts[_document_index][:60] + "..."
                _row[f"{_label} score"] = f"{scores[_paradigm][_document_index]:.4f}"
                _row[f"{_label} category"] = labels[_document_index]
            _comparison_rows.append(_row)
        _comparison_frame = pd.DataFrame(_comparison_rows)
        _result_display = mo.vstack(
            [
                mo.md("## Top-five results"),
                mo.ui.table(_comparison_frame, selection=None),
            ]
        )

    _result_display
    return


@app.cell
def _(colors_map, mo):
    mo.md(
        f"""
    ## Interpretation notes

    - **Dense:** one pooled vector per document, scored here with cosine similarity.
    - **ColBERT:** token vectors, scored with the sum of each query token's best
      document-token match (MaxSim).
    - **SPLADE:** learned vocabulary weights, scored with a sparse dot product.

    Raw scores are not calibrated across paradigms. Each UMAP reducer is fitted
    independently, and this synthetic corpus is not a retrieval-quality evaluation.

    **Category colors:**
    {chr(10).join(f"- **{category.title()}**: {color}" for category, color in colors_map.items())}
    """
    )
    return


if __name__ == "__main__":
    app.run()
