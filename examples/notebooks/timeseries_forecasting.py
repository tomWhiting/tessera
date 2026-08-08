import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Legacy Chronos forecasting notebook

        This notebook is intentionally non-executable in the revived Tessera
        runtime.

        The original demo imported `TesseraTimeSeries` and presented Chronos
        Bolt as an active Python model. That façade is no longer exported:
        Tessera now builds against stock Candle 0.11, while the retained
        Chronos implementation requires hidden-state T5 APIs from an old
        Candle fork.

        Chronos and TimesFM remain `CatalogOnly` entries so their metadata is
        discoverable, but neither has a runnable adapter. The old forecasting
        source is retained for reference rather than silently routed through a
        model with different semantics.

        See [`docs/legacy/TIMESERIES.md`](../../docs/legacy/TIMESERIES.md) for:

        - the exact fork-only APIs used by the legacy implementation;
        - the source paths retained for future work; and
        - the validation required before forecasting can be reactivated.

        The current revival is focused on dense, sparse, multi-vector, and
        vision-language retrieval with bounded resource use.
        """
    )
    return


if __name__ == "__main__":
    app.run()
