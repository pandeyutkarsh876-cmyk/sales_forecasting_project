"""
generate_data.py
-----------------
Creates a realistic synthetic daily sales dataset for a retail-style business,
covering 3 full years. Includes:
  - Long-term upward trend
  - Yearly seasonality (holiday peaks in Nov/Dec, summer dip)
  - Weekly seasonality (weekend lift)
  - Random promotion spikes
  - Store/product-line segmentation
  - Realistic random noise

Run:
    python src/generate_data.py
Output:
    data/sales_data.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

START_DATE = "2022-01-01"
END_DATE = "2024-12-31"
PRODUCT_LINES = ["Electronics", "Home & Garden", "Apparel"]
REGIONS = ["North", "South", "West"]

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"


def build_calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame({"date": dates})
    df["day_of_year"] = df["date"].dt.dayofyear
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Mon
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def simulate_series(dates: pd.DatetimeIndex, base_level: float, trend_per_day: float,
                     yearly_amplitude: float, weekend_lift: float, noise_std: float,
                     promo_prob: float, promo_lift: float) -> np.ndarray:
    n = len(dates)
    t = np.arange(n)

    # Trend
    trend = base_level + trend_per_day * t

    # Yearly seasonality (peak around late Nov, dip in summer)
    day_of_year = dates.dayofyear.values
    yearly = yearly_amplitude * np.sin(2 * np.pi * (day_of_year - 80) / 365.0)
    holiday_boost = yearly_amplitude * 1.8 * np.exp(-0.5 * ((day_of_year - 330) / 12) ** 2)  # Black Friday/Xmas bump

    # Weekly seasonality
    weekday = dates.dayofweek.values
    weekly = np.where(np.isin(weekday, [5, 6]), weekend_lift, 0)

    # Random promotions (short spikes)
    promo_flags = np.random.rand(n) < promo_prob
    promo_effect = promo_flags * promo_lift * np.random.uniform(0.7, 1.3, size=n)

    # Noise
    noise = np.random.normal(0, noise_std, size=n)

    values = trend + yearly + holiday_boost + weekly + promo_effect + noise
    values = np.clip(values, a_min=0, a_max=None)
    return values, promo_flags.astype(int)


def main():
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    records = []

    # Different base characteristics per product line x region combo
    config = {
        "Electronics":   dict(base=800, trend=0.35, yearly_amp=180, weekend_lift=60,  noise=45, promo_p=0.04, promo_lift=350),
        "Home & Garden": dict(base=500, trend=0.15, yearly_amp=140, weekend_lift=90,  noise=35, promo_p=0.05, promo_lift=220),
        "Apparel":       dict(base=650, trend=0.25, yearly_amp=220, weekend_lift=120, noise=40, promo_p=0.06, promo_lift=300),
    }
    region_multiplier = {"North": 1.15, "South": 0.90, "West": 1.0}

    for product in PRODUCT_LINES:
        cfg = config[product]
        for region in REGIONS:
            mult = region_multiplier[region]
            values, promo_flags = simulate_series(
                dates,
                base_level=cfg["base"] * mult,
                trend_per_day=cfg["trend"] * mult,
                yearly_amplitude=cfg["yearly_amp"] * mult,
                weekend_lift=cfg["weekend_lift"],
                noise_std=cfg["noise"],
                promo_prob=cfg["promo_p"],
                promo_lift=cfg["promo_lift"],
            )
            units = np.round(values / np.random.uniform(18, 25))  # rough unit price divisor
            avg_price = np.round(values / np.maximum(units, 1), 2)

            for i, d in enumerate(dates):
                records.append({
                    "date": d.strftime("%Y-%m-%d"),
                    "region": region,
                    "product_line": product,
                    "units_sold": int(max(units[i], 0)),
                    "unit_price": float(avg_price[i]) if units[i] > 0 else 0.0,
                    "revenue": round(float(values[i]), 2),
                    "on_promotion": int(promo_flags[i]),
                })

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["date", "region", "product_line"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Generated {len(df):,} rows across {df['date'].nunique()} days.")
    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
