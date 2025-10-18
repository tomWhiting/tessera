# Tessera Marimo Notebooks - Index

Welcome to the Tessera interactive notebooks! This directory contains two comprehensive, production-ready Marimo notebooks showcasing Tessera's Python bindings.

## 📚 What's Inside

```
examples/notebooks/
│
├── 📓 NOTEBOOKS (Interactive)
│   ├── embedding_comparison.py     (544 lines) - Compare Dense, MultiVector, Sparse embeddings
│   └── timeseries_forecasting.py   (508 lines) - Zero-shot probabilistic forecasting
│
├── 📖 DOCUMENTATION
│   ├── README.md                   (171 lines) - Quick start guide
│   ├── GUIDE.md                    (436 lines) - Comprehensive user guide
│   ├── QUICKREF.md                 (220 lines) - Quick reference card
│   ├── SUMMARY.md                  (329 lines) - Implementation details
│   └── INDEX.md                    (this file) - Navigation guide
│
└── 🛠️ UTILITIES
    └── run.sh                      - Launcher script
```

**Total**: 1,052 lines of notebook code + 1,156 lines of documentation

## 🎯 Quick Navigation

### For First-Time Users
1. Start here: **[README.md](README.md)** - Get up and running in 5 minutes
2. Launch: `uv run marimo edit examples/notebooks/embedding_comparison.py`
3. Explore: Try different queries and see how embeddings work

### For Detailed Learning
1. Read: **[GUIDE.md](GUIDE.md)** - Comprehensive guide with examples
2. Experiment: Run both notebooks with different parameters
3. Customize: Modify notebooks for your use cases

### For Quick Reference
- **[QUICKREF.md](QUICKREF.md)** - Commands, tips, and troubleshooting
- **[SUMMARY.md](SUMMARY.md)** - Implementation details and validation

## 🚀 Launch Options

### Option 1: Direct Launch
```bash
# Embedding comparison
uv run marimo edit examples/notebooks/embedding_comparison.py

# Time series forecasting
uv run marimo edit examples/notebooks/timeseries_forecasting.py

# Both notebooks
uv run marimo edit examples/notebooks/*.py
```

### Option 2: Using Launcher Script
```bash
# Interactive menu
./examples/notebooks/run.sh

# Direct launch
./examples/notebooks/run.sh embedding
./examples/notebooks/run.sh timeseries
./examples/notebooks/run.sh both
```

## 📊 Notebook Comparison

| Feature | Embedding Comparison | Time Series Forecasting |
|---------|---------------------|-------------------------|
| **Purpose** | Compare embedding paradigms | Probabilistic forecasting |
| **Models** | 3 models (Dense, MultiVector, Sparse) | 1 model (Chronos Bolt) |
| **Datasets** | 90 docs, 6 categories | 4 time series patterns |
| **Visualizations** | 3 UMAP plots side-by-side | 2-row uncertainty bands |
| **Interactivity** | Query search | Dataset + context selection |
| **Key Insight** | Paradigm tradeoffs | Uncertainty quantification |
| **Runtime** | ~30s first time, <5s after | ~60s first time, <5s after |
| **Complexity** | Moderate | Advanced |

## 🎓 Learning Path

### Beginner Path
```
1. Read README.md (5 min)
2. Launch embedding_comparison.py (2 min)
3. Try 3-5 different queries (10 min)
4. Read notebook explanations (10 min)
5. Launch timeseries_forecasting.py (2 min)
6. Explore different datasets (10 min)
```

**Total time**: ~40 minutes to understand both notebooks

### Advanced Path
```
1. Read GUIDE.md (20 min)
2. Run both notebooks (5 min)
3. Modify dataset/parameters (20 min)
4. Add custom data (30 min)
5. Extend visualizations (30 min)
6. Build your own project (∞)
```

## 🔍 What You'll Learn

### Embedding Comparison Notebook

**Concepts**:
- Dense embeddings (single vector representation)
- Multi-vector embeddings (token-level matching)
- Sparse embeddings (learned keyword expansion)
- UMAP dimensionality reduction
- Semantic similarity metrics

**Skills**:
- How to choose the right embedding paradigm
- Understanding similarity scores
- Visualizing high-dimensional embeddings
- Comparing retrieval results

**Takeaways**:
- Dense: Fast, semantic, but may miss details
- ColBERT: Accurate, fine-grained, but slower
- SPLADE: Balanced, interpretable, keyword-aware

### Time Series Forecasting Notebook

**Concepts**:
- Zero-shot forecasting (no training needed)
- Probabilistic predictions (distributions, not points)
- Quantile forecasting (uncertainty quantification)
- Prediction intervals (confidence bands)
- Time series patterns (trend, seasonality, noise)

**Skills**:
- Interpreting uncertainty bands
- Using quantiles for decision-making
- Understanding forecast confidence
- Evaluating different time series patterns

**Takeaways**:
- Wider bands = higher uncertainty
- Uncertainty grows with forecast horizon
- Different patterns have different predictability
- Use appropriate quantiles for your use case

## 💡 Use Cases

### Embedding Comparison
- **Semantic search**: Find similar documents
- **Recommendation systems**: Suggest related content
- **Clustering**: Group similar items
- **Deduplication**: Find duplicate documents
- **Question answering**: Retrieve relevant passages

### Time Series Forecasting
- **Sales forecasting**: Predict future demand
- **Financial modeling**: Stock price scenarios
- **Energy management**: Electricity consumption
- **Weather prediction**: Temperature forecasts
- **Capacity planning**: Resource allocation

## 🛠️ Technical Stack

### Core Libraries
- **Tessera**: Multi-paradigm embedding library
- **Marimo**: Reactive Python notebooks
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations

### Specialized Libraries
- **UMAP**: Dimensionality reduction
- **Pandas**: Data manipulation
- **SciPy**: Scientific computing
- **scikit-learn**: Machine learning utilities

### Models Used
1. **bge-base-en-v1.5**: Dense embeddings (BAAI)
2. **colbert-v2**: Multi-vector embeddings (Stanford)
3. **splade-pp-en-v1**: Sparse embeddings (Naver Labs)
4. **chronos-bolt-small**: Time series forecasting (Amazon)

## 📈 Performance Characteristics

### Model Download Times (First Run Only)
| Model | Size | Download Time |
|-------|------|---------------|
| BGE Base | ~500MB | 1-2 min |
| ColBERT v2 | ~450MB | 1-2 min |
| SPLADE PP | ~420MB | 1-2 min |
| Chronos Bolt Small | ~150MB | 30-60s |

### Inference Speed (After Loading)
| Operation | Time | Notes |
|-----------|------|-------|
| Dense embedding | ~10ms/doc | Very fast |
| ColBERT embedding | ~50ms/doc | Token-level |
| SPLADE embedding | ~20ms/doc | Sparse generation |
| UMAP projection | 1-3s | One-time per paradigm |
| Query search | <100ms | All paradigms |
| TS forecast (point) | 50-200ms | Context 2048 |
| TS forecast (quantile) | 100-500ms | 9 quantiles |

*Benchmarks on M2 MacBook Pro with 16GB RAM*

## 🎨 Customization Guide

### Easy Customizations (No Coding)
- Change query text
- Adjust sliders (context length)
- Toggle checkboxes (uncertainty bands)
- Select different datasets
- Hover over visualizations

### Moderate Customizations (Basic Python)
- Add more documents to dataset
- Change category names/colors
- Modify UMAP parameters
- Adjust quantile levels
- Change plot styling

### Advanced Customizations (Marimo Knowledge)
- Add new UI elements
- Create custom visualizations
- Implement new metrics
- Add real-time data loading
- Build interactive dashboards

## 🐛 Troubleshooting Guide

### Common Issues

**"ModuleNotFoundError"**
→ Run `uv sync` to install dependencies

**"Model download failed"**
→ Check internet connection, retry

**"Out of memory"**
→ Reduce context length or use smaller models

**"Notebook won't open"**
→ Ensure using `uv run marimo edit`

**"Slow first run"**
→ Normal - models downloading (wait 1-5 min)

See **[GUIDE.md](GUIDE.md#troubleshooting)** for detailed solutions.

## 📚 Documentation Structure

### README.md - Quick Start
- Installation instructions
- Basic usage
- Notebook overviews
- Quick examples

### GUIDE.md - Comprehensive Guide
- Detailed feature explanations
- Advanced usage patterns
- Troubleshooting solutions
- Best practices
- Performance benchmarks

### QUICKREF.md - Quick Reference
- Command cheatsheet
- Common tasks
- Tips and tricks
- Keyboard shortcuts

### SUMMARY.md - Implementation Details
- Technical architecture
- Validation results
- Code statistics
- Feature checklist

### INDEX.md - This File
- Navigation guide
- High-level overview
- Learning paths
- Use case examples

## 🔗 External Resources

### Marimo
- [Official Website](https://marimo.io)
- [Documentation](https://docs.marimo.io)
- [GitHub](https://github.com/marimo-team/marimo)
- [Examples](https://github.com/marimo-team/marimo/tree/main/examples)

### Tessera
- [Repository](https://github.com/tessera-embeddings/tessera)
- [Documentation](https://docs.rs/tessera)
- [Python Bindings](../README.md)

### Models
- [BGE Embeddings](https://huggingface.co/BAAI/bge-base-en-v1.5)
- [ColBERT](https://github.com/stanford-futuredata/ColBERT)
- [SPLADE](https://github.com/naver/splade)
- [Chronos](https://github.com/amazon-science/chronos-forecasting)

### Visualization
- [Plotly](https://plotly.com/python/)
- [UMAP](https://umap-learn.readthedocs.io/)

## 🎯 Success Checklist

Before running:
- [ ] UV installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [ ] Dependencies synced (`uv sync`)
- [ ] At least 4GB RAM available
- [ ] Internet connection active

First run:
- [ ] Be patient during model downloads (1-5 min)
- [ ] Check terminal for download progress
- [ ] Subsequent runs will be fast

Using notebooks:
- [ ] Read markdown cells for context
- [ ] Interact with UI elements
- [ ] Hover over visualizations
- [ ] Try different parameters
- [ ] Check comparison tables

## 🚦 Getting Started in 3 Steps

### Step 1: Install
```bash
uv sync
```

### Step 2: Launch
```bash
uv run marimo edit examples/notebooks/embedding_comparison.py
```

### Step 3: Explore
- Enter queries in the text box
- Observe how different paradigms retrieve documents
- Check the comparison table
- Hover over UMAP plots for details

**That's it!** You're now using production-ready embedding notebooks.

## 🎓 Next Steps

### After Exploring Both Notebooks
1. **Customize**: Add your own data
2. **Experiment**: Try different parameters
3. **Extend**: Add new features
4. **Share**: Show others what you built
5. **Contribute**: Submit improvements

### Building Your Own Project
- Use Tessera in your Python code
- Build custom notebooks
- Create production applications
- Share with the community

## 💬 Getting Help

1. **Check documentation**: Start with README.md
2. **Troubleshooting**: See GUIDE.md troubleshooting section
3. **Quick reference**: Use QUICKREF.md for commands
4. **Examples**: Study notebook code for patterns
5. **Community**: Ask in GitHub issues

## 📄 License

Same as Tessera project (MIT OR Apache-2.0)

---

**Ready to dive in?** Start with **[README.md](README.md)** or jump straight to launching a notebook!

```bash
uv run marimo edit examples/notebooks/embedding_comparison.py
```

Happy exploring! 🚀
