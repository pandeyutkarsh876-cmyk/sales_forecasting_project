# Sales Forecasting Project

A complete, runnable sales forecasting pipeline: synthetic data generation,
exploratory analysis, multiple forecasting models with backtesting, and a
90-day forward forecast — all in plain Python (pandas, scikit-learn,
matplotlib).

## Project structure

```
sales_forecasting_project/
├── README.md
├── requirements.txt
├── main.py                      # Runs the full pipeline end to end
├── data/
│   └── sales_data.csv           # 3 years of daily sales (generated)
├── src/
│   ├── generate_data.py         # Synthetic data generator
│   ├── eda.py                   # Exploratory data analysis + charts
│   └── forecast.py              # Model training, backtesting, forecasting
└── outputs/
    ├── charts/                  # All PNG charts
    └── reports/                 # CSV/TXT summaries and forecast results
```

## Setup

```bash
cd sales_forecasting_project
python -m venv venv               # optional but recommended
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run everything in one go:

```bash
python main.py
```

Or run each stage individually:

```bash
python src/generate_data.py   # creates data/sales_data.csv
python src/eda.py             # creates outputs/charts/01–04 + eda_summary.txt
python src/forecast.py        # creates outputs/charts/05–06 + model results
```

## The dataset

`data/sales_data.csv` is synthetic but realistic: 3 years (2022–2024) of
daily sales across 3 product lines (Electronics, Home & Garden, Apparel)
and 3 regions (North, South, West). It includes:

- An upward long-term trend
- Yearly seasonality with a Black Friday / holiday spike in Nov–Dec
- Weekend lift on Saturdays/Sundays
- Random promotion events with a revenue bump
- Realistic noise

Columns: `date, region, product_line, units_sold, unit_price, revenue, on_promotion`

**Using your own data:** replace `data/sales_data.csv` with your own file
using the same columns (or adjust `src/eda.py` / `src/forecast.py` to match
your schema — at minimum they need a `date` and a `revenue`-like numeric
column).

## Exploratory analysis (`src/eda.py`)

Produces:
1. **Daily revenue trend** with a 30-day moving average
2. **Monthly revenue by product line**
3. **Revenue by day of week** (weekly seasonality)
4. **Revenue share by region** (pie chart)
5. A text summary (`outputs/reports/eda_summary.txt`) with totals, averages,
   and the measured promotion lift

## Forecasting models (`src/forecast.py`)

Five models are trained and compared using a time-based holdout
(last 90 days as the test set — no shuffling, since this is a time series):

| Model | Description |
|---|---|
| Naive | Carries the last observed value forward |
| Seasonal Naive (7d) | Repeats the value from 7 days prior |
| Holt-Winters | Hand-implemented additive triple exponential smoothing (level + trend + weekly seasonality) |
| Linear Regression | Trend + Fourier (sine/cosine) terms for weekly & yearly seasonality |
| Random Forest | Calendar features + lag features (1, 7, 14 days), forecast recursively |

> **Note on dependencies:** `statsmodels` and `prophet` were not available
> in this environment, so Holt-Winters was implemented by hand and
> Random Forest / Linear Regression (scikit-learn) are used as the
> ML-based models. If you have `statsmodels` or `prophet` available in
> your own environment, they're straightforward drop-in additions —
> see "Extending this project" below.

**Evaluation metrics:** MAE, RMSE, and MAPE, all computed on the 90-day
holdout. Results are saved to `outputs/reports/model_comparison.csv` and
printed to the console, sorted by MAPE (best first).

The best-performing model on the backtest is automatically selected to
produce the final **90-day forward forecast**, saved to
`outputs/reports/forecast_next_90_days.csv` and charted in
`outputs/charts/06_final_90_day_forecast.png`.

On the synthetic dataset included here, **Random Forest** typically wins
(≈8% MAPE), correctly picking up the weekly cycle and trend.

## Extending this project

- **Swap in your real sales data** — just match the expected columns, or
  adjust the `load_daily_series()` function in `src/forecast.py`.
- **Add statsmodels/Prophet** if available in your environment:
  `pip install statsmodels prophet`, then add an ARIMA/SARIMA or Prophet
  model alongside the existing ones in `forecast.py` following the same
  `(train, steps) -> np.ndarray` pattern.
- **Forecast by segment** — the raw data already has `region` and
  `product_line` columns; group by those before calling the forecasting
  functions to get per-segment forecasts instead of one company-wide total.
- **Add confidence intervals** — for Random Forest, use the spread across
  individual trees' predictions; for Linear Regression, use
  `statsmodels`' prediction interval support if you add it.
- **Automate retraining** — wrap `main.py` in a scheduled job (cron,
  Airflow, etc.) to refresh the forecast as new sales data comes in.

## Requirements

- Python 3.9+
- pandas, numpy, matplotlib, scikit-learn (see `requirements.txt`)
