"""Train + evaluate the demand forecasting model and compute inventory recommendations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor

from features import FEATURES, HORIZON, build_features
from generate_data import generate
from metrics import mae, rmse, wape

ROOT = Path(__file__).resolve().parent.parent   # works from any directory
OUT = ROOT / "outputs"
DATA = ROOT / "data"
TEST_START = "2024-10-01"   # last quarter (includes festive season) is held out
LEAD_TIME_DAYS = 7
SERVICE_LEVEL = 0.95


def main():
    OUT.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(DATA / "retail_sales.csv", index=False)
    data = build_features(df)

    train = data[data["date"] < TEST_START]
    test = data[data["date"] >= TEST_START].copy()

    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        categorical_features=[FEATURES.index("store_code"), FEATURES.index("category_code")],
        random_state=42,
    )
    model.fit(train[FEATURES], train["units_sold"])
    test["forecast"] = np.clip(model.predict(test[FEATURES]), 0, None)

    # Baselines: same weekday last week, and 28-day moving average
    test["baseline_naive"] = test["lag_7"]
    test["baseline_ma28"] = test["roll_mean_28"]

    results = {}
    for name, col in [("gradient_boosting", "forecast"),
                      ("seasonal_naive_lag7", "baseline_naive"),
                      ("moving_average_28d", "baseline_ma28")]:
        results[name] = {"WAPE": round(wape(test.units_sold, test[col]), 4),
                         "MAE": round(mae(test.units_sold, test[col]), 3),
                         "RMSE": round(rmse(test.units_sold, test[col]), 3)}
    best_base = min(results["seasonal_naive_lag7"]["WAPE"], results["moving_average_28d"]["WAPE"])
    results["wape_improvement_vs_best_baseline_pct"] = round(
        100 * (best_base - results["gradient_boosting"]["WAPE"]) / best_base, 1)
    results["horizon_days"] = HORIZON
    results["train_rows"], results["test_rows"] = int(len(train)), int(len(test))
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))

    # Inventory recommendation: reorder point = lead-time demand + safety stock
    test["error"] = test["units_sold"] - test["forecast"]
    inv = test.groupby(["store", "category"]).agg(
        avg_daily_forecast=("forecast", "mean"), error_std=("error", "std")).reset_index()
    z = norm.ppf(SERVICE_LEVEL)
    inv["safety_stock"] = np.ceil(z * inv["error_std"] * np.sqrt(LEAD_TIME_DAYS))
    inv["reorder_point"] = np.ceil(inv["avg_daily_forecast"] * LEAD_TIME_DAYS + inv["safety_stock"])
    inv.round(2).to_csv(OUT / "inventory_recommendations.csv", index=False)

    pred = test[["date", "store", "category", "units_sold", "forecast",
                 "baseline_naive", "baseline_ma28"]].copy()
    pred[["forecast", "baseline_ma28"]] = pred[["forecast", "baseline_ma28"]].round(2)
    pred.to_csv(OUT / "predictions.csv", index=False)

    daily = test.groupby("date")[["units_sold", "forecast", "baseline_naive"]].sum()
    plt.figure(figsize=(11, 4))
    plt.plot(daily.index, daily.units_sold, label="Actual", lw=2)
    plt.plot(daily.index, daily.forecast, label="GB forecast", lw=2)
    plt.plot(daily.index, daily.baseline_naive, label="Naive (lag-7)", ls="--", alpha=.7)
    plt.title("Total daily units - held-out test period"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "forecast_vs_actual.png", dpi=130)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
