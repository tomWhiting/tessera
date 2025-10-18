# Tessera Marimo Notebooks - Quick Reference

## 🚀 Launch Commands

```bash
# Embedding Comparison
uv run marimo edit examples/notebooks/embedding_comparison.py

# Time Series Forecasting
uv run marimo edit examples/notebooks/timeseries_forecasting.py

# Both at once
uv run marimo edit examples/notebooks/*.py

# Using launcher script
./examples/notebooks/run.sh embedding
./examples/notebooks/run.sh timeseries
./examples/notebooks/run.sh both
```

## 📊 Embedding Comparison Notebook

### Models
| Type | Model | Speed | Use Case |
|------|-------|-------|----------|
| Dense | bge-base-en-v1.5 | Fast | General semantic search |
| Multi-Vector | colbert-v2 | Slow | Fine-grained matching |
| Sparse | splade-pp-en-v1 | Medium | Keyword + semantic |

### UI Elements
- **Query Input**: Text box for search queries
- **UMAP Plots**: 3 side-by-side visualizations
- **Comparison Table**: Top-5 results per paradigm

### Example Queries
```
"machine learning neural networks"  → Technology
"basketball training exercises"     → Sports
"baking bread techniques"           → Cooking
"photosynthesis biology"            → Science
```

### Key Shortcuts
- Hover over points: See document text
- Stars: Top-5 results
- Colors: Categories

## 📈 Time Series Forecasting Notebook

### Model
- **Chronos Bolt Small**: Zero-shot probabilistic forecasting
- **Context**: Up to 2048 timesteps
- **Horizon**: 64 steps ahead
- **Quantiles**: 9 levels (10%-90%)

### Datasets
| Name | Pattern | Predictability |
|------|---------|----------------|
| Sales | Weekly + Monthly | High |
| Stock Price | Random Walk | Low |
| Temperature | Daily + Yearly | Medium |
| Energy | Multi-scale + Spikes | Medium |

### UI Elements
- **Dataset Selector**: Choose time series
- **Context Slider**: 512-2048 timesteps
- **Uncertainty Toggle**: Show/hide bands
- **Quantile Toggle**: Show/hide lines

### Quantile Interpretation
| Quantile | Meaning | Use Case |
|----------|---------|----------|
| Q10 | 10th percentile | Pessimistic scenario |
| Q50 | Median | Most likely outcome |
| Q90 | 90th percentile | Optimistic scenario |
| Q10-Q90 | 80% interval | Risk management |

## 🔧 Common Tasks

### Change Number of Documents (Embedding)
```python
# Edit the dataset dictionary
dataset = {
    'category': ["doc1", "doc2", ...]
}
```

### Add Custom Time Series (Forecasting)
```python
# Add to datasets
my_data = np.loadtxt('my_data.csv')
datasets['My Series'] = my_data
```

### Export Notebook as HTML
```bash
uv run marimo export html notebook.py -o output.html
```

### Run as Read-Only App
```bash
uv run marimo run notebook.py
```

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Module not found | `uv sync` |
| Model download fails | Check internet, retry |
| Out of memory | Use smaller context (512) |
| Slow first run | Models downloading (wait 1-5 min) |
| marimo not found | Use `uv run marimo` |

## 💡 Tips & Tricks

### Embedding Comparison
1. Try contrasting queries (tech vs. sports)
2. Note when paradigms disagree
3. Check if clusters are well-separated
4. Test multi-aspect queries for ColBERT

### Time Series Forecasting
1. Use max context (2048) for best results
2. Wide uncertainty = low confidence
3. Compare quantiles to assess risk
4. Try different datasets to see patterns

## 📐 Performance Reference

### Embedding Comparison
- Model load: 5-30s (first time)
- Encode 90 docs: 2-20s (depends on model)
- UMAP: 1-3s per paradigm
- Query: <100ms

### Time Series Forecasting
- Model load: 10-60s (first time)
- Point forecast: 50-200ms
- Quantile forecast: 100-500ms
- Visualization: <100ms

## 🎨 Customization Ideas

### Embedding Comparison
- Add more categories
- Use real datasets (Wikipedia, arXiv)
- Try different UMAP parameters
- Add clustering overlays

### Time Series Forecasting
- Load real data (stocks, weather)
- Adjust quantile levels
- Add evaluation metrics (MAE, RMSE)
- Compare different context lengths

## 📚 Documentation Index

- **README.md**: Quick start
- **GUIDE.md**: Comprehensive guide
- **SUMMARY.md**: Implementation details
- **QUICKREF.md**: This file

## 🔗 External Resources

- [Marimo Docs](https://docs.marimo.io)
- [Tessera Repo](https://github.com/tessera-embeddings/tessera)
- [Plotly Gallery](https://plotly.com/python/)
- [UMAP Docs](https://umap-learn.readthedocs.io/)
- [Chronos Paper](https://arxiv.org/abs/2403.07815)

## 🎯 Quick Checklist

Before running:
- [ ] `uv` installed
- [ ] `uv sync` completed
- [ ] Internet connection (for models)
- [ ] 4GB+ RAM available

First time:
- [ ] Models will download (1-5 min)
- [ ] Be patient during first run
- [ ] Subsequent runs are fast

When using:
- [ ] Enter queries in embedding notebook
- [ ] Adjust sliders for different results
- [ ] Hover over plots for details
- [ ] Check comparison tables

## 💻 Keyboard Shortcuts

In Marimo:
- `Ctrl+Enter`: Run cell
- `Ctrl+S`: Save notebook
- `Ctrl+D`: Delete cell
- `Ctrl+M`: Toggle markdown
- `Ctrl+K`: Command palette

## 🎓 Learning Path

1. **Start**: Run embedding notebook, try basic queries
2. **Explore**: Change query, observe differences
3. **Understand**: Read paradigm explanations
4. **Experiment**: Add your own documents
5. **Advanced**: Customize visualizations

Then:

6. **Forecasting**: Run time series notebook
7. **Interact**: Change dataset and context
8. **Interpret**: Study uncertainty bands
9. **Apply**: Use with your own data
10. **Extend**: Build on top of notebooks

---

**Quick Help**: See GUIDE.md for detailed instructions
**Issues**: Check troubleshooting section in GUIDE.md
**Examples**: See notebooks for working code
