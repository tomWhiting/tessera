# Legacy time-series runtime

The Chronos Bolt implementation is retained as revival source, but it is not
part of Tessera's active runtime on stock Candle 0.11.

The old implementation depended on Candle fork revision `b014c086`. In
particular, it imported the otherwise-private `T5Stack`, called its private
`load` constructor, and used the fork-only `forward_with_embeddings` method to
feed Chronos patch embeddings into T5. Stock Candle's public T5 wrappers accept
token IDs and therefore cannot replace that path without changing the model's
semantics.

The source remains under these unreferenced paths:

- `src/timeseries/`
- `src/api/builder/timeseries.rs`
- `src/api/embedder/timeseries.rs`
- `src/bindings/python/timeseries.rs`

The `timeseries` Cargo feature currently exposes only the generic time-series
embedding types. Catalog entries for Chronos and TimesFM are discoverable, but
the unified factory returns an actionable unavailable-runtime error for them.

Before reactivating Chronos, either land a narrow upstream continuous-embedding
T5 API or maintain a local adapter with independent conformance tests. Also fix
the decoder cache lifetime and add memory-soak coverage before exposing it as a
supported runtime. TimesFM needs its own adapter; it must not be routed through
Chronos.
