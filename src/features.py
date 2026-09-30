"""Feature engineering for 7-day-ahead demand forecasting (no target leakage)."""
import pandas as pd

HORIZON = 7  # forecast horizon in days; all lag features are >= HORIZON days old
FEATURES = [
    "store_code", "category_code", "dayofweek", "month", "weekofyear", "is_weekend",
    "promo", "festive", "avg_price",
    "lag_7", "lag_14", "lag_28", "roll_mean_7", "roll_mean_28", "roll_std_28",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store", "category", "date"]).reset_index(drop=True)
    key = [df["store"], df["category"]]
    g = df.groupby(["store", "category"])["units_sold"]
    for lag in (7, 14, 28):
        df[f"lag_{lag}"] = g.shift(lag)
    shifted = g.shift(HORIZON)          # everything below uses data >= 7 days old
    grp = shifted.groupby(key)
    df["roll_mean_7"] = grp.transform(lambda s: s.rolling(7).mean())
    df["roll_mean_28"] = grp.transform(lambda s: s.rolling(28).mean())
    df["roll_std_28"] = grp.transform(lambda s: s.rolling(28).std())
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["store_code"] = df["store"].astype("category").cat.codes
    df["category_code"] = df["category"].astype("category").cat.codes
    return df.dropna(subset=["lag_28", "roll_mean_28", "roll_std_28"]).reset_index(drop=True)
