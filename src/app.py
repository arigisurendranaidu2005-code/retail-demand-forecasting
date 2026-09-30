"""Streamlit dashboard. Run:  streamlit run src/app.py   (after running src/train.py)"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

OUT = Path(__file__).resolve().parent.parent / "outputs"

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")
st.title("Retail Demand Forecasting & Inventory Planning")

if not (OUT / "predictions.csv").exists():
    st.error("Run `python src/train.py` first to generate the outputs.")
    st.stop()

pred = pd.read_csv(OUT / "predictions.csv", parse_dates=["date"])
inv = pd.read_csv(OUT / "inventory_recommendations.csv")
metrics = json.loads((OUT / "metrics.json").read_text())

best_base = min(metrics["seasonal_naive_lag7"]["WAPE"], metrics["moving_average_28d"]["WAPE"])
c1, c2, c3 = st.columns(3)
c1.metric("Model WAPE", f"{metrics['gradient_boosting']['WAPE']:.1%}")
c2.metric("Best baseline WAPE", f"{best_base:.1%}")
c3.metric("Improvement vs baseline", f"{metrics['wape_improvement_vs_best_baseline_pct']}%")

all_stores, all_cats = sorted(pred.store.unique()), sorted(pred.category.unique())
stores = st.sidebar.multiselect("Store", all_stores, default=all_stores)
cats = st.sidebar.multiselect("Category", all_cats, default=all_cats)
view = pred[pred.store.isin(stores) & pred.category.isin(cats)]

st.subheader("Actual vs forecast (daily units)")
st.line_chart(view.groupby("date")[["units_sold", "forecast", "baseline_naive"]].sum())

st.subheader("Reorder point & safety stock (95% service level, 7-day lead time)")
st.dataframe(inv[inv.store.isin(stores) & inv.category.isin(cats)])
