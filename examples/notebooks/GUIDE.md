# Tessera Marimo Notebooks: Complete Guide

## 🎯 Overview

This guide covers everything you need to know about using the Tessera Marimo notebooks for interactive embedding exploration and time series forecasting.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Notebook 1: Embedding Comparison](#notebook-1-embedding-paradigm-comparison)
4. [Notebook 2: Time Series Forecasting](#notebook-2-probabilistic-time-series-forecasting)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

## Prerequisites

- Python 3.9 or higher
- `uv` package manager ([installation instructions](https://github.com/astral-sh/uv))
- 4GB+ RAM (8GB+ recommended for larger models)
- Internet connection (for model downloads)

## Installation

### 1. Install UV (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Navigate to the project root

```bash
cd /path/to/hypiler
```

### 3. Install dependencies

```bash
uv sync
```

This installs:
- Tessera (the embedding library)
- Marimo (reactive notebooks)
- Plotly (visualizations)
- UMAP (dimensionality reduction)
- NumPy, Pandas, SciPy, scikit-learn

## Notebook 1: Embedding Paradigm Comparison

### Launch

```bash
uv run marimo edit examples/notebooks/embedding_comparison.py

# Or use the launcher script
./examples/notebooks/run.sh embedding
```

### What It Does

Compares three embedding paradigms using a dataset of 90 documents across 6 categories:

1. **Dense Embeddings (BGE)**: Single 768-dimensional vector per document
2. **Multi-Vector Embeddings (ColBERT)**: Token-level embeddings with MaxSim scoring
3. **Sparse Embeddings (SPLADE)**: Learned sparse representations

### Key Features

#### Interactive Query Search
- Enter any search query in the text box
- See top-5 results for each paradigm
- Results highlighted on UMAP plots with stars

#### UMAP Visualizations
- Three side-by-side plots (one per paradigm)
- Color-coded by category
- Hover to see document text
- Top results marked with stars

#### Comparison Table
- Shows top-5 results for each paradigm
- Includes similarity scores
- Easy to see where paradigms agree/disagree

### Example Queries

Try these to see different behaviors:

```
"How do neural networks learn from data?"
→ Technology documents should rank high

"Training athletes for peak performance"
→ Sports documents should rank high

"Making bread from scratch"
→ Cooking documents should rank high

"Understanding photosynthesis in plants"
→ Science documents should rank high
```

### Understanding the Results

**Dense (BGE):**
- Best for: General semantic similarity
- Fast retrieval
- May miss fine-grained details

**Multi-Vector (ColBERT):**
- Best for: Complex queries with multiple aspects
- Token-level matching
- Slower but more nuanced

**Sparse (SPLADE):**
- Best for: Keyword-based + semantic search
- Interpretable (can see which keywords activated)
- Good balance of lexical and semantic

### Customization

Edit the `dataset` dictionary to add your own documents:

```python
dataset = {
    'your_category': [
        "Your first document...",
        "Your second document...",
    ]
}
```

## Notebook 2: Probabilistic Time Series Forecasting

### Launch

```bash
uv run marimo edit examples/notebooks/timeseries_forecasting.py

# Or use the launcher script
./examples/notebooks/run.sh timeseries
```

### What It Does

Demonstrates zero-shot probabilistic forecasting using Chronos Bolt:

- **Zero-shot**: No training needed, works on any time series
- **Probabilistic**: Full forecast distributions, not just point estimates
- **9 quantile levels**: 10%, 20%, 30%, ..., 90%
- **64-step horizon**: Forecasts 64 steps ahead

### Key Features

#### Interactive Dataset Selector
Choose from 4 synthetic time series:

1. **Sales (Weekly + Monthly)**: Regular seasonal patterns
2. **Stock Price**: Random walk with regime changes
3. **Temperature**: Daily and yearly cycles
4. **Energy Consumption**: Multi-scale patterns with spikes

#### Context Length Slider
- Adjust from 512 to 2048 timesteps
- See how more history affects forecast quality
- Model uses last N points as context

#### Uncertainty Visualization
- **Top plot**: Historical data + point forecast
- **Bottom plot**: Median with uncertainty bands
- Toggle uncertainty bands on/off
- Toggle individual quantile lines

#### Statistics Tables
- **Key timesteps**: Forecast stats at 5 key points
- **Full quantile table**: First 10 steps, all 9 quantiles

### Understanding Probabilistic Forecasts

#### What are Quantiles?

- **Q10**: 10% of scenarios fall below this
- **Q50 (Median)**: Middle of the distribution
- **Q90**: 90% of scenarios fall below this

#### Prediction Intervals

- **80% interval (Q10-Q90)**: Captures most likely outcomes
- **Narrower bands**: Higher confidence
- **Wider bands**: Higher uncertainty
- **Growing bands**: Uncertainty increases with time

#### When to Use Each Quantile

- **Risk management**: Use Q10/Q90 for worst/best cases
- **Central tendency**: Use Q50 (median) for most likely outcome
- **Anomaly detection**: Values outside Q10-Q90 are unusual
- **Inventory planning**: Use Q75/Q90 to avoid stockouts

### Example Use Cases

#### Sales Forecasting
```
Dataset: Sales (Weekly + Monthly)
Context: 2048 timesteps
→ Regular patterns, narrow uncertainty bands
→ Use Q75 for inventory planning
```

#### Financial Risk
```
Dataset: Stock Price
Context: 2048 timesteps
→ High uncertainty, wide bands
→ Use Q10 for risk assessment
```

#### Climate Modeling
```
Dataset: Temperature
Context: 2048 timesteps
→ Multiple seasonal cycles
→ Uncertainty grows with forecast horizon
```

### Advanced: Using Your Own Data

Replace the synthetic data generation:

```python
# Load your time series
my_data = np.loadtxt('my_timeseries.csv')

# Add to datasets
datasets['My Custom Series'] = my_data
```

Requirements:
- NumPy array of floats
- At least 512 timesteps (2048+ recommended)
- No missing values (NaN)

## Advanced Usage

### Running Multiple Notebooks

```bash
# Open both notebooks in separate tabs
uv run marimo edit \
    examples/notebooks/embedding_comparison.py \
    examples/notebooks/timeseries_forecasting.py
```

### Exporting Results

#### Export as Static HTML

```bash
uv run marimo export html \
    examples/notebooks/embedding_comparison.py \
    -o embedding_comparison.html
```

#### Export as Python Script

```bash
uv run marimo export script \
    examples/notebooks/embedding_comparison.py \
    -o embedding_comparison_script.py
```

### Programmatic Access

Use Tessera directly in Python:

```python
from tessera import TesseraDense, TesseraMultiVector, TesseraSparse, TesseraTimeSeries
import numpy as np

# Dense embeddings
dense = TesseraDense("bge-base-en-v1.5")
embedding = dense.encode("Your text here")

# Multi-vector embeddings
colbert = TesseraMultiVector("colbert-v2")
token_embeddings = colbert.encode("Your text here")

# Sparse embeddings
sparse = TesseraSparse("splade-pp-en-v1")
sparse_embedding = sparse.encode("Your text here")

# Time series forecasting
forecaster = TesseraTimeSeries("chronos-bolt-small")
context = np.random.randn(1, 2048).astype(np.float32)
forecast = forecaster.forecast(context)
quantiles = forecaster.forecast_quantiles(context)
```

## Troubleshooting

### Model Download Issues

**Problem**: Models fail to download

**Solution**:
```bash
# Check internet connection
ping huggingface.co

# Try downloading manually
from tessera import TesseraDense
model = TesseraDense("bge-base-en-v1.5")  # Forces download
```

### Memory Issues

**Problem**: Out of memory errors

**Solutions**:
- Use smaller context length (512 instead of 2048)
- Close other applications
- Use smaller models:
  - `colbert-v2` → Try a smaller variant
  - `chronos-bolt-small` → Already smallest version

### Slow Performance

**Problem**: Notebook runs slowly

**Solutions**:
- **First run**: Models download (wait 1-5 minutes)
- **Subsequent runs**: Should be fast (models cached)
- **Large datasets**: Reduce number of documents
- **UMAP**: First projection is slow, then cached

### Import Errors

**Problem**: `ModuleNotFoundError`

**Solution**:
```bash
# Reinstall dependencies
uv sync

# Verify installation
uv run python -c "import marimo; import tessera; print('OK')"
```

### Marimo Not Found

**Problem**: `marimo: command not found`

**Solution**:
```bash
# Always use uv run
uv run marimo edit examples/notebooks/embedding_comparison.py
```

## Best Practices

### For Embedding Comparison

1. **Start with simple queries**: Test with single-concept queries first
2. **Compare paradigms**: Note when different paradigms agree/disagree
3. **Understand tradeoffs**: Dense (fast), ColBERT (accurate), SPLADE (interpretable)
4. **Check clusters**: Well-separated clusters indicate good embeddings
5. **Add your data**: Replace synthetic data with real documents

### For Time Series Forecasting

1. **Use sufficient context**: 2048 timesteps recommended
2. **Check uncertainty**: Wide bands = low confidence
3. **Compare quantiles**: Look at full distribution, not just median
4. **Validate**: If you have ground truth, compare forecast accuracy
5. **Experiment**: Try different context lengths to see impact

### General Marimo Tips

1. **Reactive cells**: Changes propagate automatically
2. **No redefinition**: Each variable defined once
3. **UI values**: Access with `.value` (e.g., `slider.value`)
4. **Execution order**: Determined by dependencies, not cell order
5. **Debug**: Use `mo.md()` to display intermediate values

## Performance Benchmarks

### Embedding Comparison Notebook

| Operation | Time | Notes |
|-----------|------|-------|
| Model loading | 5-30s | First time only (downloads models) |
| Encoding 90 docs (Dense) | 2-5s | BGE model |
| Encoding 90 docs (ColBERT) | 10-20s | Token embeddings |
| Encoding 90 docs (SPLADE) | 5-10s | Sparse vectors |
| UMAP projection | 1-3s | Per paradigm |
| Query search | <100ms | All three paradigms |

### Time Series Forecasting Notebook

| Operation | Time | Notes |
|-----------|------|-------|
| Model loading | 10-60s | First time only (downloads) |
| Point forecast | 50-200ms | Context 2048, horizon 64 |
| Quantile forecast | 100-500ms | 9 quantiles |
| Visualization | <100ms | Plotly rendering |

*Benchmarks on M2 MacBook Pro. Your results may vary.*

## Additional Resources

- **Tessera Documentation**: See main project README
- **Marimo Docs**: https://docs.marimo.io
- **Plotly Docs**: https://plotly.com/python/
- **UMAP Tutorial**: https://umap-learn.readthedocs.io/
- **Chronos Paper**: https://arxiv.org/abs/2403.07815

## Next Steps

1. **Explore the notebooks**: Try different queries and datasets
2. **Customize**: Add your own data
3. **Experiment**: Modify visualizations and parameters
4. **Build**: Use Tessera in your own projects
5. **Contribute**: Share your notebooks with the community!

## Support

Having issues? Check:
1. This guide's troubleshooting section
2. Project README
3. GitHub issues
4. Marimo community forum

Happy exploring! 🚀
