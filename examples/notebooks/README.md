# Tessera Marimo Notebooks

Interactive data science notebooks showcasing Tessera's Python bindings using [Marimo](https://marimo.io).

## 📚 Available Notebooks

### 1. Embedding Paradigm Comparison (`embedding_comparison.py`)

Compare three embedding paradigms with interactive UMAP visualizations:

- **Dense Embeddings** (BGE): Single vector per document
- **Multi-Vector Embeddings** (ColBERT): Token-level embeddings
- **Sparse Embeddings** (SPLADE): Learned sparse representations

**Features:**
- 90 documents across 6 categories
- Interactive query search
- Side-by-side UMAP projections
- Top-5 comparison table
- Real-time similarity scoring

### 2. Probabilistic Time Series Forecasting (`timeseries_forecasting.py`)

Zero-shot time series forecasting with Chronos Bolt:

**Features:**
- 4 synthetic datasets with different patterns
- Interactive dataset and context length selection
- Beautiful uncertainty band visualizations
- 9 quantile levels (10%, 20%, ..., 90%)
- Forecast statistics tables
- Educational explanations

## 🚀 Quick Start

### Prerequisites

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### Running the Notebooks

```bash
# Embedding comparison
uv run marimo edit examples/notebooks/embedding_comparison.py

# Time series forecasting
uv run marimo edit examples/notebooks/timeseries_forecasting.py
```

The notebooks will open in your browser with an interactive interface!

## 🎓 Learning Path

**New to Tessera?** Start with the embedding comparison notebook to understand different embedding paradigms.

**Interested in forecasting?** The time series notebook showcases probabilistic predictions with uncertainty quantification.

## 🛠️ What is Marimo?

[Marimo](https://marimo.io) is a reactive Python notebook that's:

- **Reactive**: Cells update automatically when dependencies change
- **Reproducible**: No hidden state, deterministic execution
- **Interactive**: Built-in UI elements (sliders, dropdowns, etc.)
- **Git-friendly**: Notebooks are pure Python files

Unlike Jupyter, Marimo notebooks are reactive and prevent common bugs from out-of-order execution.

## 📊 Notebook Structure

Both notebooks follow best practices:

```python
import marimo

app = marimo.App(width="full")

@app.cell
def __():
    # Imports
    import marimo as mo
    import numpy as np
    return mo, np

@app.cell
def __(mo):
    # Interactive widgets
    slider = mo.ui.slider(0, 100, value=50)
    slider
    return slider,

@app.cell
def __(slider):
    # Reactive cell - updates when slider changes
    value = slider.value
    return value,
```

Each `@app.cell` decorator defines a reactive cell that automatically re-runs when its dependencies change.

## 🎨 Visualizations

Both notebooks use **Plotly** for interactive visualizations:

- Hover to see details
- Zoom and pan
- Select regions
- Export as images

## 🔧 Customization

Feel free to modify the notebooks:

- **Change datasets**: Edit the `dataset` dictionary
- **Adjust parameters**: Modify sliders and dropdowns
- **Add models**: Load different Tessera models
- **Extend visualizations**: Add new Plotly traces

## 📝 Tips

1. **Reactive updates**: Changes to UI elements automatically trigger downstream cells
2. **No variable redefinition**: Each variable can only be defined once across all cells
3. **Display values**: The last expression in a cell is automatically displayed
4. **UI element values**: Access with `.value` attribute (e.g., `slider.value`)

## 🐛 Troubleshooting

**Notebook won't open?**
```bash
# Make sure marimo is installed
uv run marimo --version
```

**Import errors?**
```bash
# Sync dependencies
uv sync
```

**Model download issues?**
- Models download automatically on first use
- Check your internet connection
- Large models (ColBERT, Chronos) may take several minutes

## 📚 Additional Resources

- [Tessera Documentation](../README.md)
- [Marimo Documentation](https://docs.marimo.io)
- [Plotly Documentation](https://plotly.com/python/)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)

## 🤝 Contributing

Have ideas for new notebooks? Contributions welcome!

Possible topics:
- Vision-language embeddings (ColPali)
- Hybrid search (combining paradigms)
- Real-world datasets (e.g., arXiv papers)
- Custom similarity metrics
- Batch processing examples

## 📄 License

Same as Tessera project.
