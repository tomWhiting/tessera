import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(
        """
    # Embedding Paradigm Comparison with Tessera

    This notebook demonstrates three different embedding paradigms for semantic search:

    - **Dense Embeddings** (BGE): Single vector per document, fast cosine similarity
    - **Multi-Vector Embeddings** (ColBERT): Token-level embeddings, fine-grained matching
    - **Sparse Embeddings** (SPLADE): Learned sparse vectors, interpretable keywords

    We'll visualize how each paradigm clusters documents and responds to queries using UMAP projections.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from tessera import TesseraDense, TesseraMultiVector, TesseraSparse
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from umap import UMAP
    from scipy.sparse import csr_matrix
    from sklearn.preprocessing import normalize
    import pandas as pd
    return (
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
def _(mo):
    dataset = {
        'technology': [
            "Machine learning models require large datasets for training",
            "Neural networks consist of layers of interconnected nodes",
            "Cloud computing provides scalable infrastructure on demand",
            "Quantum computers use qubits instead of classical bits",
            "Blockchain technology enables decentralized ledger systems",
            "Edge computing processes data closer to the source",
            "Natural language processing helps computers understand human text",
            "Computer vision algorithms can identify objects in images",
            "Cybersecurity protects systems from digital attacks",
            "5G networks offer faster wireless communication speeds",
            "Artificial intelligence is transforming various industries",
            "Deep learning uses neural networks with many layers",
            "GPU acceleration speeds up parallel computations",
            "Containerization simplifies application deployment",
            "Microservices architecture breaks apps into smaller services"
        ],
        'sports': [
            "Basketball requires dribbling, passing, and shooting skills",
            "Soccer is the world's most popular team sport",
            "Tennis matches are played on grass, clay, or hard courts",
            "Swimming builds cardiovascular endurance and muscle strength",
            "Marathon running tests both physical and mental stamina",
            "Baseball combines pitching, batting, and fielding",
            "Golf requires precision and strategic course management",
            "Rock climbing develops strength, balance, and problem-solving",
            "Cycling can be recreational or competitive racing",
            "Volleyball teams rotate positions after winning serve",
            "Boxing demands quick reflexes and tactical thinking",
            "Gymnastics showcases flexibility, strength, and artistry",
            "Track and field includes sprints, jumps, and throws",
            "Rugby is a physical contact sport with continuous play",
            "Surfing combines balance with wave reading skills"
        ],
        'cooking': [
            "Sautéing vegetables in olive oil brings out their natural sweetness",
            "Baking bread requires proper kneading and proofing time",
            "Caramelizing onions develops deep, complex flavors",
            "Marinating meat tenderizes it and adds flavor",
            "Blanching vegetables preserves their bright color",
            "Deglazing a pan creates a flavorful sauce base",
            "Tempering chocolate ensures a smooth, glossy finish",
            "Braising tough cuts of meat makes them tender",
            "Emulsifying creates stable mixtures like mayonnaise",
            "Reducing sauces concentrates their flavors",
            "Grilling adds smoky char to meats and vegetables",
            "Poaching gently cooks delicate foods in liquid",
            "Roasting caramelizes surfaces at high heat",
            "Steaming preserves nutrients in vegetables",
            "Fermenting transforms ingredients into complex flavors"
        ],
        'science': [
            "Photosynthesis converts sunlight into chemical energy",
            "DNA contains the genetic instructions for living organisms",
            "Gravity is the force that attracts objects with mass",
            "Chemical reactions involve breaking and forming bonds",
            "Evolution explains the diversity of life through natural selection",
            "Atoms consist of protons, neutrons, and electrons",
            "The speed of light in vacuum is constant",
            "Cells are the basic building blocks of life",
            "The water cycle circulates water through evaporation and precipitation",
            "Ecosystems involve complex interactions between organisms",
            "Plate tectonics explains continental drift and earthquakes",
            "Neurons transmit electrical signals in the nervous system",
            "Antibodies help the immune system fight pathogens",
            "Mitochondria are the powerhouses of cells",
            "Black holes have gravity so strong light cannot escape"
        ],
        'arts': [
            "Impressionist paintings capture light and momentary effects",
            "Sculpture can be carved, modeled, or assembled",
            "Jazz improvisation creates spontaneous musical expression",
            "Watercolor painting uses transparent pigments on paper",
            "Classical ballet requires years of rigorous training",
            "Photography freezes moments in time through light",
            "Poetry uses language rhythmically to evoke emotion",
            "Opera combines singing, acting, and orchestral music",
            "Abstract art emphasizes form and color over representation",
            "Renaissance art revived classical themes and techniques",
            "Film editing shapes narrative through shot selection",
            "Contemporary dance breaks traditional movement rules",
            "Ceramic pottery is shaped and fired in kilns",
            "Graphic design communicates through visual composition",
            "Street art transforms urban spaces into galleries"
        ],
        'medicine': [
            "Antibiotics fight bacterial infections but not viruses",
            "Vaccination trains the immune system to recognize diseases",
            "MRI scans use magnetic fields to image soft tissues",
            "Physical therapy helps restore mobility after injuries",
            "Chemotherapy targets rapidly dividing cancer cells",
            "Anesthesia blocks pain signals during surgery",
            "Blood pressure measures force against artery walls",
            "Diabetes involves problems regulating blood sugar levels",
            "X-rays use radiation to image bones and dense tissues",
            "Organ transplants can replace failing kidneys or hearts",
            "Gene therapy aims to treat diseases by modifying DNA",
            "Prosthetics replace missing limbs with artificial devices",
            "Cardiac surgery repairs or replaces heart structures",
            "Dermatology treats skin, hair, and nail conditions",
            "Neurosurgery operates on the brain and nervous system"
        ]
    }

    texts = []
    labels = []
    colors_map = {
        'technology': '#1f77b4',
        'sports': '#ff7f0e',
        'cooking': '#2ca02c',
        'science': '#d62728',
        'arts': '#9467bd',
        'medicine': '#8c564b'
    }

    for category, docs in dataset.items():
        texts.extend(docs)
        labels.extend([category] * len(docs))

    colors = [colors_map[label] for label in labels]

    mo.md(f"**Dataset**: {len(texts)} documents across {len(dataset)} categories")
    return colors, colors_map, labels, texts


@app.cell
def _(mo):
    mo.md(
        """
    ## Loading Embedding Models

    Initializing three different embedding models from Tessera:
    """
    )
    return


@app.cell
def _(TesseraDense, TesseraMultiVector, TesseraSparse, mo):
    with mo.status.spinner(title="Loading embedding models...") as _spinner:
        dense = TesseraDense("bge-base-en-v1.5")
        colbert = TesseraMultiVector("colbert-v2")
        sparse = TesseraSparse("splade-pp-en-v1")

    mo.md("✓ All models loaded successfully")
    return colbert, dense, sparse


@app.cell
def _(colbert, dense, mo, np, sparse, texts):
    mo.md("## Encoding Documents")

    with mo.status.spinner(title="Encoding with Dense (BGE)..."):
        dense_embs = np.array([dense.encode(t) for t in texts])

    with mo.status.spinner(title="Encoding with Multi-Vector (ColBERT)..."):
        colbert_embs = [colbert.encode(t) for t in texts]

    with mo.status.spinner(title="Encoding with Sparse (SPLADE)..."):
        sparse_embs = [sparse.encode(t) for t in texts]

    mo.md(f"""
    **Embeddings generated:**
    - Dense: {dense_embs.shape}
    - Multi-Vector: {len(colbert_embs)} documents (variable token embeddings)
    - Sparse: {len(sparse_embs)} sparse vectors
    """)
    return colbert_embs, dense_embs, sparse_embs


@app.cell
def _(UMAP, colbert_embs, csr_matrix, dense_embs, mo, np, sparse_embs):
    mo.md("## Dimensionality Reduction with UMAP")

    # Create separate UMAP reducers for each paradigm (so we can transform queries later)
    dense_umap = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    colbert_umap = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    sparse_umap = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)

    with mo.status.spinner(title="UMAP: Dense embeddings..."):
        dense_2d = dense_umap.fit_transform(dense_embs)

    with mo.status.spinner(title="UMAP: Multi-Vector embeddings (averaged)..."):
        colbert_avg = np.array([e.mean(axis=0) for e in colbert_embs])
        colbert_2d = colbert_umap.fit_transform(colbert_avg)

    with mo.status.spinner(title="UMAP: Sparse embeddings..."):
        sparse_indices = []
        sparse_values = []
        sparse_indptr = [0]

        for emb in sparse_embs:
            indices, values = emb  # Unpack the tuple (indices, values)
            sparse_indices.extend(indices)
            sparse_values.extend(values)
            sparse_indptr.append(len(sparse_indices))

        vocab_size = max(sparse_indices) + 1
        sparse_matrix = csr_matrix(
            (sparse_values, sparse_indices, sparse_indptr),
            shape=(len(sparse_embs), vocab_size)
        )
        sparse_dense = sparse_matrix.toarray()
        sparse_2d = sparse_umap.fit_transform(sparse_dense)

    mo.md("✓ UMAP projections computed")
    return (
        colbert_2d,
        colbert_umap,
        dense_2d,
        dense_umap,
        sparse_2d,
        sparse_umap,
    )


@app.cell
def _(mo):
    mo.md(
        """
    ## Interactive Query Search

    Enter a query to see how each embedding paradigm retrieves documents:
    """
    )
    return


@app.cell
def _(mo):
    query_input = mo.ui.text(
        value="How do neural networks learn from data?",
        placeholder="Enter your search query...",
        label="Search Query",
        full_width=True
    )
    query_input
    return (query_input,)


@app.cell
def _(
    colbert,
    colbert_embs,
    dense,
    dense_embs,
    mo,
    np,
    query_input,
    sparse,
    sparse_embs,
    texts,
):
    if query_input.value:
        query = query_input.value

        with mo.status.spinner(title="Computing similarities..."):
            dense_query_emb = dense.encode(query)
            dense_scores = np.array([
                np.dot(dense_query_emb, doc_emb) /
                (np.linalg.norm(dense_query_emb) * np.linalg.norm(doc_emb))
                for doc_emb in dense_embs
            ])

            colbert_query_emb = colbert.encode(query)
            colbert_scores = np.array([
                np.max([
                    np.dot(q_token, d_token) /
                    (np.linalg.norm(q_token) * np.linalg.norm(d_token))
                    for q_token in colbert_query_emb
                    for d_token in doc_emb
                ])
                for doc_emb in colbert_embs
            ])

            sparse_query_emb = sparse.encode(query)
            sq_indices, sq_values = sparse_query_emb  # Unpack tuple
            sparse_scores = np.array([
                sum(
                    sq_values[i] * doc_values[j]
                    for i, qi in enumerate(sq_indices)
                    for j, dj in enumerate(doc_indices)
                    if qi == dj
                )
                for doc_indices, doc_values in sparse_embs  # Unpack each doc tuple
            ])

        top_k = 5
        top_dense_indices = np.argsort(dense_scores)[::-1][:top_k]
        top_colbert_indices = np.argsort(colbert_scores)[::-1][:top_k]
        top_sparse_indices = np.argsort(sparse_scores)[::-1][:top_k]

        top_dense = set(top_dense_indices)
        top_colbert = set(top_colbert_indices)
        top_sparse = set(top_sparse_indices)
    else:
        dense_scores = np.zeros(len(texts))
        colbert_scores = np.zeros(len(texts))
        sparse_scores = np.zeros(len(texts))
        top_k = 5
        top_dense = set()
        top_colbert = set()
        top_sparse = set()
        top_dense_indices = []
        top_colbert_indices = []
        top_sparse_indices = []
        dense_query_emb = None
        colbert_query_emb = None
        sparse_query_emb = None

    mo.md(f"**Query:** '{query_input.value}'")
    return (
        colbert_query_emb,
        colbert_scores,
        dense_query_emb,
        dense_scores,
        sparse_query_emb,
        sparse_scores,
        top_colbert,
        top_colbert_indices,
        top_dense,
        top_dense_indices,
        top_k,
        top_sparse,
        top_sparse_indices,
    )


@app.cell
def _(
    colbert_2d,
    colbert_query_emb,
    colbert_umap,
    colors,
    dense_2d,
    dense_query_emb,
    dense_umap,
    go,
    labels,
    make_subplots,
    mo,
    np,
    query_input,
    sparse_2d,
    sparse_query_emb,
    sparse_umap,
    texts,
    top_colbert,
    top_dense,
    top_sparse,
):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            'Dense (BGE): Single Vector',
            'Multi-Vector (ColBERT): Token Embeddings',
            'Sparse (SPLADE): Learned Keywords'
        ),
        horizontal_spacing=0.08
    )

    for idx in range(len(texts)):
        is_top_dense = idx in top_dense
        is_top_colbert = idx in top_colbert
        is_top_sparse = idx in top_sparse

        marker_size = 12 if is_top_dense else 8
        marker_symbol = 'star' if is_top_dense else 'circle'
        # Truncate text for hover display
        hover_text = texts[idx][:100] + "..." if len(texts[idx]) > 100 else texts[idx]
        fig.add_trace(
            go.Scatter(
                x=[dense_2d[idx, 0]],
                y=[dense_2d[idx, 1]],
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=colors[idx],
                    symbol=marker_symbol,
                    line=dict(width=2, color='black') if is_top_dense else dict(width=0)
                ),
                customdata=[hover_text],
                hovertemplate='<b>%{customdata}</b><br>Category: ' + labels[idx] + '<extra></extra>',
                showlegend=False,
                name=labels[idx]
            ),
            row=1, col=1
        )

        marker_size = 12 if is_top_colbert else 8
        marker_symbol = 'star' if is_top_colbert else 'circle'
        hover_text = texts[idx][:100] + "..." if len(texts[idx]) > 100 else texts[idx]
        fig.add_trace(
            go.Scatter(
                x=[colbert_2d[idx, 0]],
                y=[colbert_2d[idx, 1]],
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=colors[idx],
                    symbol=marker_symbol,
                    line=dict(width=2, color='black') if is_top_colbert else dict(width=0)
                ),
                customdata=[hover_text],
                hovertemplate='<b>%{customdata}</b><br>Category: ' + labels[idx] + '<extra></extra>',
                showlegend=False,
                name=labels[idx]
            ),
            row=1, col=2
        )

        marker_size = 12 if is_top_sparse else 8
        marker_symbol = 'star' if is_top_sparse else 'circle'
        hover_text = texts[idx][:100] + "..." if len(texts[idx]) > 100 else texts[idx]
        fig.add_trace(
            go.Scatter(
                x=[sparse_2d[idx, 0]],
                y=[sparse_2d[idx, 1]],
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=colors[idx],
                    symbol=marker_symbol,
                    line=dict(width=2, color='black') if is_top_sparse else dict(width=0)
                ),
                customdata=[hover_text],
                hovertemplate='<b>%{customdata}</b><br>Category: ' + labels[idx] + '<extra></extra>',
                showlegend=False,
                name=labels[idx]
            ),
            row=1, col=3
        )

    # Add query point if a query was entered
    if query_input.value and dense_query_emb is not None:
        # Transform query embeddings to UMAP space
        dense_query_2d = dense_umap.transform([dense_query_emb])[0]
        colbert_query_avg = colbert_query_emb.mean(axis=0)
        colbert_query_2d = colbert_umap.transform([colbert_query_avg])[0]

        # Convert sparse query to dense for UMAP
        sq_idx, sq_val = sparse_query_emb
        bert_vocab = 30522  # BERT vocab size
        sparse_query_dense = np.zeros(bert_vocab)
        for idx, val in zip(sq_idx, sq_val):
            if idx < bert_vocab:
                sparse_query_dense[idx] = val
        sparse_query_2d = sparse_umap.transform([sparse_query_dense])[0]

        # Add query markers to each plot (large red X)
        for col, (query_2d, title) in enumerate([
            (dense_query_2d, "Dense"),
            (colbert_query_2d, "ColBERT"),
            (sparse_query_2d, "Sparse")
        ], 1):
            fig.add_trace(
                go.Scatter(
                    x=[query_2d[0]],
                    y=[query_2d[1]],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color='red',
                        symbol='x',
                        line=dict(width=3, color='darkred')
                    ),
                    customdata=[f"QUERY: {query_input.value}"],
                    hovertemplate='<b>%{customdata}</b><extra></extra>',
                    showlegend=False,
                    name='Query'
                ),
                row=1, col=col
            )

    fig.update_xaxes(title_text="UMAP 1", row=1, col=1)
    fig.update_xaxes(title_text="UMAP 1", row=1, col=2)
    fig.update_xaxes(title_text="UMAP 1", row=1, col=3)
    fig.update_yaxes(title_text="UMAP 2", row=1, col=1)

    fig.update_layout(
        height=500,
        title_text="Embedding Space Visualization (Top-5 results marked with stars)",
        showlegend=False
    )

    mo.ui.plotly(fig)
    return


@app.cell
def _(
    colbert_scores,
    dense_scores,
    labels,
    mo,
    pd,
    sparse_scores,
    texts,
    top_colbert_indices,
    top_dense_indices,
    top_k,
    top_sparse_indices,
):
    comparison_data = []
    for rank in range(top_k):
        dense_idx = top_dense_indices[rank]
        colbert_idx = top_colbert_indices[rank]
        sparse_idx = top_sparse_indices[rank]

        comparison_data.append({
            'Rank': rank + 1,
            'Dense (BGE)': texts[dense_idx][:60] + '...',
            'Dense Score': f'{dense_scores[dense_idx]:.4f}',
            'Dense Cat': labels[dense_idx],
            'ColBERT': texts[colbert_idx][:60] + '...',
            'ColBERT Score': f'{colbert_scores[colbert_idx]:.4f}',
            'ColBERT Cat': labels[colbert_idx],
            'SPLADE': texts[sparse_idx][:60] + '...',
            'SPLADE Score': f'{sparse_scores[sparse_idx]:.4f}',
            'SPLADE Cat': labels[sparse_idx],
        })

    comparison_df = pd.DataFrame(comparison_data)

    mo.md("## Top-5 Results Comparison")
    mo.ui.table(comparison_df, selection=None)
    return


@app.cell
def _(colors_map, mo):
    mo.md(
        f"""
    ## Understanding the Paradigms

    ### Dense Embeddings (BGE)
    - **Representation**: Single 768-dimensional vector per document
    - **Similarity**: Fast cosine similarity (dot product)
    - **Strengths**: Efficient retrieval, captures semantic meaning
    - **Weaknesses**: Cannot distinguish importance of different terms

    ### Multi-Vector Embeddings (ColBERT)
    - **Representation**: One vector per token (variable length)
    - **Similarity**: Maximum similarity across token pairs (MaxSim)
    - **Strengths**: Fine-grained matching, handles multi-aspect queries
    - **Weaknesses**: Slower computation, larger storage

    ### Sparse Embeddings (SPLADE)
    - **Representation**: Learned sparse vector (expanded vocabulary)
    - **Similarity**: Dot product of sparse vectors
    - **Strengths**: Interpretable (keyword-like), combines lexical + semantic
    - **Weaknesses**: Can be less efficient than dense for large vocabularies

    ---

    **Category Colors:**
    {chr(10).join([f'- **{cat.title()}**: {color}' for cat, color in colors_map.items()])}
    """
    )
    return


if __name__ == "__main__":
    app.run()
