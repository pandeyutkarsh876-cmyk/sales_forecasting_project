"""
eda.py
------
Exploratory data analysis on the sales dataset. Produces summary stats and
saves charts to outputs/charts/:
  - total daily revenue trend + 30-day moving average
  - monthly revenue by product line
  - weekday seasonality (avg revenue by day of week)
  - revenue share by region

Run:
    python src/eda.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sales_data.csv"
CHART_DIR = ROOT / "outputs" / "charts"
REPORT_DIR = ROOT / "outputs" / "reports"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df


def daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("date", as_index=False)["revenue"].sum()


def plot_trend(daily: pd.DataFrame):
    daily = daily.sort_values("date").copy()
    daily["ma_30"] = daily["revenue"].rolling(30).mean()

    plt.figure(figsize=(12, 5))
    plt.plot(daily["date"], daily["revenue"], color="#a8c6f0", linewidth=0.8, label="Daily revenue")
    plt.plot(daily["date"], daily["ma_30"], color="#1f5fbf", linewidth=2, label="30-day moving average")
    plt.title("Total Daily Revenue (2022–2024)")
    plt.xlabel("Date")
    plt.ylabel("Revenue ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_DIR / "01_daily_revenue_trend.png", dpi=150)
    plt.close()


def plot_monthly_by_product(df: pd.DataFrame):
    monthly = df.copy()
    monthly["month"] = monthly["date"].values.astype("datetime64[M]")
    monthly = monthly.groupby(["month", "product_line"], as_index=False)["revenue"].sum()
    pivot = monthly.pivot(index="month", columns="product_line", values="revenue")

    pivot.plot(figsize=(12, 5), linewidth=1.8)
    plt.title("Monthly Revenue by Product Line")
    plt.xlabel("Month")
    plt.ylabel("Revenue ($)")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "02_monthly_revenue_by_product.png", dpi=150)
    plt.close()


def plot_weekday_seasonality(df: pd.DataFrame):
    d = df.copy()
    d["day_of_week"] = d["date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    avg = d.groupby("day_of_week")["revenue"].sum().reindex(order)

    plt.figure(figsize=(8, 5))
    avg.plot(kind="bar", color="#2e8b57")
    plt.title("Total Revenue by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Revenue ($)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "03_weekday_seasonality.png", dpi=150)
    plt.close()


def plot_region_share(df: pd.DataFrame):
    region_totals = df.groupby("region")["revenue"].sum().sort_values(ascending=False)

    plt.figure(figsize=(6, 6))
    plt.pie(region_totals, labels=region_totals.index, autopct="%1.1f%%",
            colors=["#4c72b0", "#dd8452", "#55a868"])
    plt.title("Revenue Share by Region")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "04_region_share.png", dpi=150)
    plt.close()


def write_summary(df: pd.DataFrame, daily: pd.DataFrame):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("SALES DATA — EDA SUMMARY")
    lines.append("=" * 40)
    lines.append(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    lines.append(f"Total rows: {len(df):,}")
    lines.append(f"Total revenue: ${df['revenue'].sum():,.2f}")
    lines.append(f"Avg daily revenue: ${daily['revenue'].mean():,.2f}")
    lines.append(f"Daily revenue std dev: ${daily['revenue'].std():,.2f}")
    lines.append("")
    lines.append("Revenue by product line:")
    for name, val in df.groupby("product_line")["revenue"].sum().sort_values(ascending=False).items():
        lines.append(f"  {name:<15} ${val:,.2f}")
    lines.append("")
    lines.append("Revenue by region:")
    for name, val in df.groupby("region")["revenue"].sum().sort_values(ascending=False).items():
        lines.append(f"  {name:<15} ${val:,.2f}")
    lines.append("")
    promo_lift = df[df["on_promotion"] == 1]["revenue"].mean() - df[df["on_promotion"] == 0]["revenue"].mean()
    lines.append(f"Avg revenue lift on promotion days (per row): ${promo_lift:,.2f}")

    text = "\n".join(lines)
    (REPORT_DIR / "eda_summary.txt").write_text(text)
    print(text)


def main():
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    daily = daily_totals(df)

    plot_trend(daily)
    plot_monthly_by_product(df)
    plot_weekday_seasonality(df)
    plot_region_share(df)
    write_summary(df, daily)

    print(f"\nCharts saved to: {CHART_DIR}")


if __name__ == "__main__":
    main()
