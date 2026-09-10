"""
forecast.py
-----------
Builds and evaluates several forecasting models on total daily revenue,
then produces a 90-day forward forecast using the best performer.

Models included:
  1. Naive (last value carried forward)
  2. Seasonal Naive (value from 7 days prior)
  3. Holt-Winters triple exponential smoothing (hand-implemented, additive)
  4. Linear Regression with trend + weekly + yearly seasonal features
  5. Random Forest with lag + calendar features

Evaluation uses a time-based holdout: last 90 days as test set.
Metrics: MAE, RMSE, MAPE.

Run:
    python src/forecast.py
Outputs:
    outputs/reports/model_comparison.csv
    outputs/reports/forecast_next_90_days.csv
    outputs/charts/05_model_backtest_comparison.png
    outputs/charts/06_final_90_day_forecast.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sales_data.csv"
CHART_DIR = ROOT / "outputs" / "charts"
REPORT_DIR = ROOT / "outputs" / "reports"

TEST_HORIZON = 90       # days held out for backtesting
FORECAST_HORIZON = 90   # days to forecast into the future


# --------------------------------------------------------------------------
# Data prep
# --------------------------------------------------------------------------

def load_daily_series() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    daily = df.groupby("date", as_index=False)["revenue"].sum().sort_values("date")
    daily = daily.set_index("date").asfreq("D")
    daily["revenue"] = daily["revenue"].interpolate()
    return daily


def make_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    feats = pd.DataFrame(index=index)
    feats["day_of_week"] = index.dayofweek
    feats["day_of_year"] = index.dayofyear
    feats["month"] = index.month
    feats["is_weekend"] = (index.dayofweek >= 5).astype(int)
    feats["t"] = np.arange(len(index))  # linear time trend index
    # Fourier terms for yearly seasonality
    feats["sin_year"] = np.sin(2 * np.pi * feats["day_of_year"] / 365.25)
    feats["cos_year"] = np.cos(2 * np.pi * feats["day_of_year"] / 365.25)
    # Fourier terms for weekly seasonality
    feats["sin_week"] = np.sin(2 * np.pi * feats["day_of_week"] / 7)
    feats["cos_week"] = np.cos(2 * np.pi * feats["day_of_week"] / 7)
    return feats


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def naive_forecast(train: pd.Series, steps: int) -> np.ndarray:
    return np.repeat(train.iloc[-1], steps)


def seasonal_naive_forecast(train: pd.Series, steps: int, season: int = 7) -> np.ndarray:
    last_season = train.iloc[-season:].values
    reps = int(np.ceil(steps / season))
    return np.tile(last_season, reps)[:steps]


def holt_winters_additive(train: pd.Series, steps: int, season_len: int = 7,
                           alpha: float = 0.2, beta: float = 0.05, gamma: float = 0.3):
    """
    Minimal hand-rolled additive Holt-Winters (level + trend + seasonal),
    since statsmodels is unavailable in this environment.
    """
    y = train.values.astype(float)
    n = len(y)

    # Initialize
    level = np.mean(y[:season_len])
    trend = (np.mean(y[season_len:2 * season_len]) - np.mean(y[:season_len])) / season_len
    seasonal = [y[i] - level for i in range(season_len)]

    levels, trends, seasonals = [level], [trend], list(seasonal)

    for i in range(n):
        s_idx = i % season_len
        val = y[i]
        last_level = levels[-1]
        last_trend = trends[-1]
        last_seasonal = seasonals[s_idx]

        new_level = alpha * (val - last_seasonal) + (1 - alpha) * (last_level + last_trend)
        new_trend = beta * (new_level - last_level) + (1 - beta) * last_trend
        new_seasonal = gamma * (val - new_level) + (1 - gamma) * last_seasonal

        levels.append(new_level)
        trends.append(new_trend)
        seasonals[s_idx] = new_seasonal

    final_level = levels[-1]
    final_trend = trends[-1]

    forecast = []
    for h in range(1, steps + 1):
        s_idx = (n + h - 1) % season_len
        forecast.append(final_level + h * final_trend + seasonals[s_idx])

    return np.array(forecast)


def linear_regression_forecast(train: pd.Series, train_feats: pd.DataFrame,
                                future_feats: pd.DataFrame):
    model = LinearRegression()
    model.fit(train_feats, train.values)
    return model.predict(future_feats), model


def random_forest_forecast(train: pd.Series, train_feats: pd.DataFrame,
                            future_feats: pd.DataFrame, lags=(1, 7, 14)):
    """
    Random Forest using calendar features + lag features.
    Future lag features are filled iteratively (recursive forecasting)
    since true future lags aren't known ahead of time.
    """
    full_series = train.copy()
    feat_cols = list(train_feats.columns) + [f"lag_{l}" for l in lags]

    def build_lag_df(series: pd.Series, feats: pd.DataFrame) -> pd.DataFrame:
        df = feats.copy()
        for l in lags:
            df[f"lag_{l}"] = series.shift(l).reindex(feats.index)
        return df

    train_lagged = build_lag_df(full_series, train_feats).dropna()
    y_train = full_series.loc[train_lagged.index]

    model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(train_lagged[feat_cols], y_train)

    # Recursive forecasting
    history = full_series.copy()
    preds = []
    for date in future_feats.index:
        row = future_feats.loc[[date]].copy()
        for l in lags:
            lag_date = date - pd.Timedelta(days=l)
            row[f"lag_{l}"] = history.get(lag_date, history.iloc[-1])
        pred = model.predict(row[feat_cols])[0]
        preds.append(pred)
        history.loc[date] = pred  # feed prediction back in for recursive lags

    return np.array(preds), model


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def run_backtest(daily: pd.DataFrame):
    train = daily["revenue"].iloc[:-TEST_HORIZON]
    test = daily["revenue"].iloc[-TEST_HORIZON:]

    all_feats = make_calendar_features(daily.index)
    train_feats = all_feats.loc[train.index]
    test_feats = all_feats.loc[test.index]

    results = {}
    predictions = {}

    predictions["Naive"] = naive_forecast(train, len(test))
    predictions["Seasonal Naive (7d)"] = seasonal_naive_forecast(train, len(test))
    predictions["Holt-Winters"] = holt_winters_additive(train, len(test))
    lr_pred, _ = linear_regression_forecast(train, train_feats, test_feats)
    predictions["Linear Regression"] = lr_pred
    rf_pred, _ = random_forest_forecast(train, train_feats, test_feats)
    predictions["Random Forest"] = rf_pred

    for name, pred in predictions.items():
        results[name] = evaluate(test.values, pred)

    results_df = pd.DataFrame(results).T.sort_values("MAPE")
    return results_df, predictions, train, test


def plot_backtest(train, test, predictions, results_df):
    plt.figure(figsize=(13, 6))
    recent_train = train.iloc[-60:]
    plt.plot(recent_train.index, recent_train.values, label="Train (recent)", color="gray")
    plt.plot(test.index, test.values, label="Actual", color="black", linewidth=2)

    colors = ["#e07a5f", "#3d5a80", "#81b29a", "#f2cc8f", "#9b5de5"]
    for (name, pred), color in zip(predictions.items(), colors):
        plt.plot(test.index, pred, label=name, linestyle="--", color=color, alpha=0.85)

    best_model = results_df.index[0]
    plt.title(f"Backtest on Last {TEST_HORIZON} Days — Best: {best_model} "
              f"(MAPE {results_df.loc[best_model, 'MAPE']:.2f}%)")
    plt.xlabel("Date")
    plt.ylabel("Revenue ($)")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "05_model_backtest_comparison.png", dpi=150)
    plt.close()


def run_final_forecast(daily: pd.DataFrame, best_model_name: str):
    full_series = daily["revenue"]
    future_dates = pd.date_range(full_series.index[-1] + pd.Timedelta(days=1),
                                  periods=FORECAST_HORIZON, freq="D")

    all_index = full_series.index.append(future_dates)
    all_feats = make_calendar_features(all_index)
    train_feats = all_feats.loc[full_series.index]
    future_feats = all_feats.loc[future_dates]

    if best_model_name == "Random Forest":
        forecast_vals, _ = random_forest_forecast(full_series, train_feats, future_feats)
    elif best_model_name == "Linear Regression":
        forecast_vals, _ = linear_regression_forecast(full_series, train_feats, future_feats)
    elif best_model_name == "Holt-Winters":
        forecast_vals = holt_winters_additive(full_series, FORECAST_HORIZON)
    elif best_model_name == "Seasonal Naive (7d)":
        forecast_vals = seasonal_naive_forecast(full_series, FORECAST_HORIZON)
    else:
        forecast_vals = naive_forecast(full_series, FORECAST_HORIZON)

    forecast_df = pd.DataFrame({"date": future_dates, "forecast_revenue": forecast_vals})
    return forecast_df


def plot_final_forecast(daily: pd.DataFrame, forecast_df: pd.DataFrame, best_model_name: str):
    plt.figure(figsize=(13, 6))
    history = daily["revenue"].iloc[-180:]
    plt.plot(history.index, history.values, label="Historical revenue", color="#1f5fbf")
    plt.plot(forecast_df["date"], forecast_df["forecast_revenue"],
              label=f"{best_model_name} forecast (next {FORECAST_HORIZON}d)",
              color="#e07a5f", linewidth=2)
    plt.axvline(history.index[-1], color="gray", linestyle=":")
    plt.title(f"90-Day Forward Sales Forecast ({best_model_name})")
    plt.xlabel("Date")
    plt.ylabel("Revenue ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_DIR / "06_final_90_day_forecast.png", dpi=150)
    plt.close()


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    daily = load_daily_series()

    print(f"Backtesting {len(daily) - TEST_HORIZON} training days, "
          f"{TEST_HORIZON} test days...\n")
    results_df, predictions, train, test = run_backtest(daily)
    results_df.to_csv(REPORT_DIR / "model_comparison.csv")
    print("MODEL COMPARISON (sorted by MAPE, lower is better):")
    print(results_df.round(2).to_string())

    plot_backtest(train, test, predictions, results_df)

    best_model_name = results_df.index[0]
    print(f"\nBest model: {best_model_name}")

    forecast_df = run_final_forecast(daily, best_model_name)
    forecast_df.to_csv(REPORT_DIR / "forecast_next_90_days.csv", index=False)
    plot_final_forecast(daily, forecast_df, best_model_name)

    print(f"\nForecast for next {FORECAST_HORIZON} days saved to "
          f"outputs/reports/forecast_next_90_days.csv")
    print(f"Charts saved to: {CHART_DIR}")
    preview = forecast_df.head(10).copy()
    preview["forecast_revenue"] = preview["forecast_revenue"].round(2)
    print(f"\nForecast preview:")
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
