# Manual example fixtures

This directory retains assets for manual vision-language experiments. No
current automated example consumes these files, so the directory is excluded
from the published crate package.

## `attention_is_all_you_need.pdf`

- Source: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Title: “Attention Is All You Need” by Vaswani et al. (2017)
- Checked-in size: 2,215,244 bytes
- Pages: 15
- Intended use: manual document-rendering and ColPali experiments

The public `TesseraVision::encode_document` façade currently accepts image
paths, not PDF paths. The default `pdf` feature provides lower-level rendering
plumbing; it does not make this fixture an active end-to-end example.

Before promoting a fixture into an automated example, verify its redistribution
terms, keep the test opt-in, and apply the same image and model-memory limits as
the public API.
