# miniCOIL test tables

`tables.json` is an excerpt of two files from the Hugging Face repository
`Qdrant/minicoil-v1` at revision `4a7b05822a7a246d25778508593fff58fe574dfe`,
licensed Apache-2.0:

- `minicoil.triplet.model.vocab`: `vocab_size` is the full vocabulary length plus
  one for the unknown id. `words` holds, by id, every vocabulary word that a word
  in the eight fixtures under `certification/fixtures/minicoil/`, or that word's
  Snowball English stem (`py_rust_stemmers` 0.1.8), matches directly or through
  the stem mapping. `stem_mapping` holds every stem-mapping entry whose key is
  such a word or stem.
- `stopwords.txt`: the whole list, 153 words, in file order.

Entries no fixture word reaches are left out, so these tables resolve the
fixture texts exactly as the full files do and say nothing about other text.
