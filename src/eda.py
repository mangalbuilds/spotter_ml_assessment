from __future__ import annotations

import numpy as np
import pandas as pd


def main() -> None:
    train = pd.read_csv("data/train_test.csv")
    val = pd.read_csv("data/validation.csv")
    dec = pd.read_csv("data/december_chart_inputs.csv")

    train["date_dt"] = pd.to_datetime(train["date"])
    val["date_dt"] = pd.to_datetime(val["date"])
    train["rpm"] = train["posted_rate"] / train["distance"]

    print("=== SHAPES ===")
    print(f"train: {train.shape} | val: {val.shape} | dec: {dec.shape}")

    print("\n=== MISSING VALUES ===")
    print("Train nulls:\n", train.isna().sum()[train.isna().sum() > 0])
    print("Val nulls:\n", val.isna().sum()[val.isna().sum() > 0])

    print("\n=== WEIGHT ANOMALIES ===")
    print("Train weight <= 0:", (train["weight"] <= 0).sum(), "| min:", train["weight"].min(), "| max:", train["weight"].max())
    print("Val weight   <= 0:", (val["weight"] <= 0).sum(), "| min:", val["weight"].min(), "| max:", val["weight"].max())
    print("Train abs(weight) min/max:", train["weight"].abs().min(), train["weight"].abs().max())

    print("\n=== RPM DISTRIBUTION & OUTLIERS ===")
    qs = [0.001, 0.005, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.995, 0.999]
    print(train["rpm"].quantile(qs).round(3))

    print("\n=== EQUIPMENT STATS (TRAIN) ===")
    print(train.groupby("equipment")[["posted_rate", "distance", "rpm"]].agg(["count", "mean", "median", "std"]).round(3))

    print("\n=== DAY OF WEEK STATS (TRAIN) ===")
    train["dow"] = train["date_dt"].dt.day_name()
    print(train.groupby("dow")[["posted_rate", "rpm", "market_index", "quote_signal"]].mean().round(3))

    print("\n=== QUOTE_SIGNAL & MARKET_INDEX BY EQUIPMENT / LANE ===")
    for col in ["market_index", "quote_signal"]:
        print(f"{col} corr with rpm by equipment:")
        for eq, grp in train.groupby("equipment"):
            c = grp[col].corr(grp["rpm"])
            print(f"  {eq:10s}: {c:+.4f}")

    print("\n=== LEXINGTON -> FORT WAYNE HISTORY ===")
    lane_tr = train[(train["pickup"] == "Lexington") & (train["delivery"] == "Fort Wayne")]
    lane_va = val[(val["pickup"] == "Lexington") & (val["delivery"] == "Fort Wayne")]
    print(f"Train Lexington->FW rows: {len(lane_tr)} | Val rows: {len(lane_va)}")
    if len(lane_tr) > 0:
        print(lane_tr[["date", "equipment", "distance", "weight", "market_index", "quote_signal", "posted_rate", "rpm"]].head(12).to_string(index=False))

    print("\n=== CHECK IF MARKET_INDEX IS PURELY DAILY OR REGIONAL ===")
    daily_mkt_std = pd.concat([train[["date", "market_index"]], val[["date", "market_index"]]]).groupby("date")["market_index"].std().describe()
    print("Within-day std of market_index:\n", daily_mkt_std.round(4))
    daily_qs_std = pd.concat([train[["date", "quote_signal"]], val[["date", "quote_signal"]]]).groupby("date")["quote_signal"].std().describe()
    print("Within-day std of quote_signal:\n", daily_qs_std.round(4))

    print("\n=== CHECK OUTLIER PATTERN (RPM < 0.6 or RPM > 7.5) ===")
    low = train[train["rpm"] < 0.6]
    high = train[train["rpm"] > 7.5]
    print(f"Low RPM (<0.6) count: {len(low)} | High RPM (>7.5) count: {len(high)}")
    print("Low sample:\n", low[["load_id", "equipment", "distance", "weight", "market_index", "quote_signal", "posted_rate", "rpm"]].head(8).to_string(index=False))
    print("High sample:\n", high[["load_id", "equipment", "distance", "weight", "market_index", "quote_signal", "posted_rate", "rpm"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
