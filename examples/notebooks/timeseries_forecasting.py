import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(
        """
    # Probabilistic Time Series Forecasting with Chronos Bolt

    This notebook demonstrates **zero-shot probabilistic forecasting** using Chronos Bolt,
    a state-of-the-art time series foundation model.

    ## Key Features
    - **Zero-shot**: No training required, works on any time series
    - **Probabilistic**: Generates full forecast distributions, not just point predictions
    - **Quantile forecasts**: 9 quantile levels (10%, 20%, ..., 90%)
    - **Long context**: Uses up to 2048 historical timesteps
    - **Fast inference**: Optimized for production deployments

    We'll explore three different synthetic time series with distinct characteristics.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from tessera import TesseraTimeSeries
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    return TesseraTimeSeries, go, make_subplots, mo, np, pd


@app.cell
def _(mo, np):
    mo.md("## Generating Synthetic Time Series")

    np.random.seed(42)

    def generate_sales_data(n=2048):
        t = np.arange(n)
        trend = 0.05 * t
        weekly = 50 * np.sin(2 * np.pi * t / 7)
        monthly = 30 * np.sin(2 * np.pi * t / 30)
        noise = np.random.normal(0, 10, n)
        return 100 + trend + weekly + monthly + noise

    def generate_stock_data(n=2048):
        returns = np.random.normal(0.0005, 0.02, n)
        returns[500:600] += 0.001
        returns[1200:1300] -= 0.0015
        price = 100 * np.exp(np.cumsum(returns))
        return price

    def generate_temperature_data(n=2048):
        t = np.arange(n)
        yearly = 20 * np.sin(2 * np.pi * t / 365.25)
        daily = 8 * np.sin(2 * np.pi * t / 1)
        trend = 0.002 * t
        noise = np.random.normal(0, 2, n)
        return 15 + yearly + daily + trend + noise

    def generate_energy_data(n=2048):
        t = np.arange(n)
        weekly = 200 * np.sin(2 * np.pi * t / 7)
        daily = 150 * (1 + 0.3 * np.sin(2 * np.pi * t / 1))
        base = 500
        spikes = np.zeros(n)
        spike_times = np.random.choice(n, size=20, replace=False)
        spikes[spike_times] = np.random.uniform(200, 400, size=20)
        noise = np.random.normal(0, 30, n)
        return base + weekly + daily + spikes + noise

    datasets = {
        'Sales (Weekly + Monthly Seasonality)': generate_sales_data(),
        'Stock Price (Random Walk with Regime Changes)': generate_stock_data(),
        'Temperature (Daily + Yearly Cycles)': generate_temperature_data(),
        'Energy Consumption (Multi-scale + Spikes)': generate_energy_data()
    }

    mo.md(f"Generated {len(datasets)} synthetic time series, each with {len(next(iter(datasets.values())))} timesteps")
    return (datasets,)


@app.cell
def _(TesseraTimeSeries, mo):
    mo.md("## Loading Chronos Bolt Model")

    with mo.status.spinner(title="Loading chronos-bolt-small...") as _spinner:
        forecaster = TesseraTimeSeries("chronos-bolt-small")

    pred_len = forecaster.prediction_length()
    ctx_len = forecaster.context_length()

    mo.md(f"""
    ✓ Model loaded successfully

    **Model Configuration:**
    - Maximum context length: {ctx_len} timesteps
    - Prediction horizon: {pred_len} steps ahead
    - Quantile levels: {len(forecaster.quantiles())} quantiles
    """)
    return forecaster, pred_len


@app.cell
def _(datasets, mo):
    mo.md("## Interactive Controls")

    dataset_selector = mo.ui.dropdown(
        options=list(datasets.keys()),
        value=list(datasets.keys())[0],
        label="Select Time Series Dataset"
    )

    context_slider = mo.ui.slider(
        start=512,
        stop=2048,
        step=256,
        value=2048,
        label=f"Context Length (Historical Data to Use)"
    )

    show_uncertainty = mo.ui.checkbox(
        label="Show uncertainty bands",
        value=True
    )

    show_individual_quantiles = mo.ui.checkbox(
        label="Show individual quantile lines",
        value=False
    )

    mo.hstack([
        mo.vstack([dataset_selector, context_slider]),
        mo.vstack([show_uncertainty, show_individual_quantiles])
    ])
    return (
        context_slider,
        dataset_selector,
        show_individual_quantiles,
        show_uncertainty,
    )


@app.cell
def _(context_slider, dataset_selector, datasets, forecaster, mo, np):
    mo.md(f"## Generating Forecast for: {dataset_selector.value}")

    selected_data = datasets[dataset_selector.value]
    context_len_selected = context_slider.value

    context_data = selected_data[-context_len_selected:]
    context = context_data.reshape(1, -1).astype(np.float32)

    with mo.status.spinner(title="Computing point forecast..."):
        point_forecast = forecaster.forecast(context)

    with mo.status.spinner(title="Computing quantile forecasts..."):
        quantile_forecast = forecaster.forecast_quantiles(context)

    quantile_levels = forecaster.quantiles()
    q_dict = {
        'q10': quantile_forecast[0, :, 0],
        'q20': quantile_forecast[0, :, 1],
        'q30': quantile_forecast[0, :, 2],
        'q40': quantile_forecast[0, :, 3],
        'q50': quantile_forecast[0, :, 4],
        'q60': quantile_forecast[0, :, 5],
        'q70': quantile_forecast[0, :, 6],
        'q80': quantile_forecast[0, :, 7],
        'q90': quantile_forecast[0, :, 8],
    }

    mo.md(f"""
    ✓ Forecast complete

    - Context used: {context_len_selected} timesteps
    - Forecast horizon: {point_forecast.shape[1]} steps
    - Point forecast shape: {point_forecast.shape}
    - Quantile forecast shape: {quantile_forecast.shape}
    """)
    return (
        context_data,
        context_len_selected,
        point_forecast,
        q_dict,
        quantile_forecast,
        quantile_levels,
    )


@app.cell
def _(
    context_data,
    go,
    make_subplots,
    mo,
    np,
    point_forecast,
    pred_len,
    q_dict,
    show_individual_quantiles,
    show_uncertainty,
):
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            'Historical Data + Point Forecast',
            'Uncertainty Quantification (Prediction Intervals)'
        ),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5]
    )

    hist_x = np.arange(len(context_data))
    forecast_x = np.arange(len(context_data), len(context_data) + pred_len)

    fig.add_trace(
        go.Scatter(
            x=hist_x,
            y=context_data,
            name='Historical Data',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='Time: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_x,
            y=point_forecast[0],
            name='Point Forecast',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            hovertemplate='Time: %{x}<br>Forecast: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_x,
            y=q_dict['q50'],
            name='Median (Q50)',
            line=dict(color='#d62728', width=3),
            hovertemplate='Time: %{x}<br>Median: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )

    if show_uncertainty.value:
        fig.add_trace(
            go.Scatter(
                x=forecast_x,
                y=q_dict['q90'],
                name='90th Percentile',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_x,
                y=q_dict['q10'],
                name='10-90% Interval',
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.15)',
                line=dict(width=0),
                hovertemplate='Time: %{x}<br>10th: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_x,
                y=q_dict['q80'],
                name='80th Percentile',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_x,
                y=q_dict['q20'],
                name='20-80% Interval',
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.25)',
                line=dict(width=0),
                hovertemplate='Time: %{x}<br>20th: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_x,
                y=q_dict['q70'],
                name='70th Percentile',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_x,
                y=q_dict['q30'],
                name='30-70% Interval',
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.35)',
                line=dict(width=0),
                hovertemplate='Time: %{x}<br>30th: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_x,
                y=q_dict['q60'],
                name='60th Percentile',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_x,
                y=q_dict['q40'],
                name='40-60% Interval',
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.45)',
                line=dict(width=0),
                hovertemplate='Time: %{x}<br>40th: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )

    if show_individual_quantiles.value:
        quantile_colors = {
            'q10': '#e7298a',
            'q20': '#d95f02',
            'q30': '#7570b3',
            'q40': '#66a61e',
            'q60': '#66a61e',
            'q70': '#7570b3',
            'q80': '#d95f02',
            'q90': '#e7298a'
        }

        for q_name in ['q10', 'q20', 'q30', 'q40', 'q60', 'q70', 'q80', 'q90']:
            fig.add_trace(
                go.Scatter(
                    x=forecast_x,
                    y=q_dict[q_name],
                    name=q_name.upper(),
                    line=dict(color=quantile_colors[q_name], width=1, dash='dot'),
                    hovertemplate=f'{q_name.upper()}: %{{y:.2f}}<extra></extra>'
                ),
                row=2, col=1
            )

    fig.update_xaxes(title_text="Time Step", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)

    fig.update_layout(
        height=800,
        title_text="Chronos Bolt: Probabilistic Time Series Forecast",
        showlegend=True,
        hovermode='x unified'
    )

    mo.ui.plotly(fig)
    return


@app.cell
def _(mo, pd, point_forecast, pred_len, q_dict):
    mo.md("## Forecast Statistics at Key Timesteps")

    stats_data = []
    key_steps = [0, pred_len // 4, pred_len // 2, 3 * pred_len // 4, pred_len - 1]

    for step in key_steps:
        stats_data.append({
            'Timestep': f't+{step + 1}',
            'Point Forecast': f'{point_forecast[0, step]:.2f}',
            'Q10': f'{q_dict["q10"][step]:.2f}',
            'Q25 (Q20+Q30)/2': f'{(q_dict["q20"][step] + q_dict["q30"][step])/2:.2f}',
            'Median (Q50)': f'{q_dict["q50"][step]:.2f}',
            'Q75 (Q70+Q80)/2': f'{(q_dict["q70"][step] + q_dict["q80"][step])/2:.2f}',
            'Q90': f'{q_dict["q90"][step]:.2f}',
            'IQR': f'{(q_dict["q70"][step] + q_dict["q80"][step])/2 - (q_dict["q20"][step] + q_dict["q30"][step])/2:.2f}',
            '80% PI Width': f'{q_dict["q90"][step] - q_dict["q10"][step]:.2f}'
        })

    stats_df = pd.DataFrame(stats_data)
    mo.ui.table(stats_df, selection=None)
    return


@app.cell
def _(mo, pd, quantile_forecast, quantile_levels):
    mo.md("## Full Quantile Table (First 10 Forecast Steps)")

    quantile_table_data = []
    for forecast_step in range(min(10, quantile_forecast.shape[1])):
        row = {'Step': f't+{forecast_step + 1}'}
        for i, q_level in enumerate(quantile_levels):
            row[f'Q{int(q_level * 100)}'] = f'{quantile_forecast[0, forecast_step, i]:.2f}'
        quantile_table_data.append(row)

    quantile_df = pd.DataFrame(quantile_table_data)
    mo.ui.table(quantile_df, selection=None)
    return


@app.cell
def _(context_len_selected, mo, pred_len):
    mo.md(
        f"""
    ## Understanding Probabilistic Forecasting

    ### What are Quantiles?

    Quantiles represent percentiles of the forecast distribution:

    - **Q10 (10th percentile)**: 10% of possible futures fall below this value
    - **Q50 (50th percentile/Median)**: The middle of the distribution
    - **Q90 (90th percentile)**: 90% of possible futures fall below this value

    The **80% prediction interval** (Q10 to Q90) captures the middle 80% of likely outcomes.

    ### Why Probabilistic Forecasts?

    1. **Uncertainty quantification**: Know when predictions are more/less certain
    2. **Risk management**: Make better decisions with confidence intervals
    3. **Anomaly detection**: Identify values outside prediction intervals
    4. **Scenario planning**: Explore best-case and worst-case scenarios

    ### Model Details

    - **Architecture**: Chronos Bolt (transformer-based foundation model)
    - **Training**: Pre-trained on diverse time series datasets
    - **Zero-shot**: No fine-tuning required for new time series
    - **Context window**: {context_len_selected} timesteps used
    - **Forecast horizon**: {pred_len} steps ahead
    - **Quantile levels**: 9 quantiles (10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%)

    ### Interpreting the Visualizations

    **Top plot**: Shows historical data (blue) and point forecast (orange dashed line)

    **Bottom plot**: Shows the median forecast (red) with uncertainty bands:
    - Darkest band: 40-60% interval (most likely outcomes)
    - Medium bands: 30-70% and 20-80% intervals
    - Lightest band: 10-90% interval (captures most scenarios)

    Wider bands indicate higher uncertainty. Notice how uncertainty typically grows further into the future!
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Try Different Time Series!

    Use the dropdown above to explore different patterns:

    - **Sales**: Regular weekly and monthly cycles (predictable)
    - **Stock Price**: Random walk with regime changes (high uncertainty)
    - **Temperature**: Multiple seasonal cycles (daily + yearly)
    - **Energy**: Complex multi-scale patterns with sudden spikes

    Adjust the context length to see how more historical data affects forecast quality.
    """
    )
    return


if __name__ == "__main__":
    app.run()
