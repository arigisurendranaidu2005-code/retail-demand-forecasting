"""Generate a realistic SYNTHETIC retail daily-sales dataset (stores x categories).

No real company data is used. Patterns included: trend, weekly + yearly seasonality,
festive-season spikes, promotions, and noise.
"""
from pathlib import Path

import numpy as np
import pandas as pd

STORES = [f"STORE_{i:02d}" for i in range(1, 9)]
CATEGORIES = ["Apparel", "Footwear", "Accessories", "Home", "Kids"]
FESTIVE_WINDOWS = [("2023-10-15", "2023-11-15"), ("2024-10-15", "2024-11-15")]


def generate(seed: int = 42, start="2023-01-01", end="2024-12-31") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    festive = np.zeros(len(dates), dtype=bool)
    for s, e in FESTIVE_WINDOWS:
        festive |= (dates >= s) & (dates <= e)

    rows = []
    for s_idx, store in enumerate(STORES):
        store_scale = rng.uniform(0.8, 1.6)
        for c_idx, cat in enumerate(CATEGORIES):
            base = rng.uniform(25, 70) * store_scale
            trend = np.linspace(0, rng.uniform(0.05, 0.25), len(dates))
            weekly = 1 + 0.25 * (dates.dayofweek >= 5) + 0.05 * (dates.dayofweek == 4)
            yearly = 1 + 0.15 * np.sin(2 * np.pi * (dates.dayofyear / 365.25) + c_idx)
            promo = rng.random(len(dates)) < 0.12
            level = base * (1 + trend) * weekly * yearly
            level = level * np.where(festive, 1.6, 1.0) * np.where(promo, 1.25, 1.0)
            units = rng.poisson(np.clip(level, 1, None))
            price = rng.uniform(300, 1500) * (1 - 0.15 * promo)
            rows.append(pd.DataFrame({
                "date": dates, "store": store, "category": cat,
                "units_sold": units, "promo": promo.astype(int),
                "festive": festive.astype(int), "avg_price": np.round(price, 2),
            }))
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data"
    out.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(out / "retail_sales.csv", index=False)
    print(f"Saved data/retail_sales.csv  rows={len(df):,}  "
          f"{df.date.min().date()} -> {df.date.max().date()}")
