"""
main.py
-------
Runs the full sales forecasting pipeline end to end:
  1. Generate synthetic sales data (skips if data already exists)
  2. Run exploratory data analysis and save charts
  3. Train/backtest forecasting models and produce a 90-day forecast

Run:
    python main.py
"""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sales_data.csv"


def run(script: str):
    print(f"\n{'=' * 60}\nRunning {script}\n{'=' * 60}")
    subprocess.run([sys.executable, str(ROOT / script)], check=True)


def main():
    if not DATA_PATH.exists():
        run("src/generate_data.py")
    else:
        print(f"Data already exists at {DATA_PATH}, skipping generation.")
        print("(Delete data/sales_data.csv to regenerate.)")

    run("src/eda.py")
    run("src/forecast.py")

    print(f"\n{'=' * 60}")
    print("Pipeline complete. See outputs/charts/ and outputs/reports/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
