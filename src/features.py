from __future__ import annotations

import numpy as np
import pandas as pd


US_HOLIDAYS_2025 = pd.to_datetime([
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-10-13",
    "2025-11-11", "2025-11-27", "2025-12-25",
])


def haversine_miles(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> np.ndarray:
    """Compute great-circle distance in miles between origin and destination coordinates."""
    r = 3958.8
    p1 = np.radians(lat1.to_numpy(dtype=float))
    l1 = np.radians(lon1.to_numpy(dtype=float))
    p2 = np.radians(lat2.to_numpy(dtype=float))
    l2 = np.radians(lon2.to_numpy(dtype=float))
    dlat = p2 - p1
    dlon = l2 - l1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def build_city_coord_lookup(frames: list[pd.DataFrame]) -> dict[str, tuple[float, float]]:
    """Build deterministic city -> (lat, lon) mapping from datasets containing coordinates."""
    coords: dict[str, tuple[float, float]] = {}
    for df in frames:
        if {"pickup", "pickup_lat", "pickup_lon"}.issubset(df.columns):
            sub = df[["pickup", "pickup_lat", "pickup_lon"]].dropna().drop_duplicates("pickup")
            for row in sub.itertuples(index=False):
                coords[str(row.pickup)] = (float(row.pickup_lat), float(row.pickup_lon))
        if {"delivery", "delivery_lat", "delivery_lon"}.issubset(df.columns):
            sub = df[["delivery", "delivery_lat", "delivery_lon"]].dropna().drop_duplicates("delivery")
            for row in sub.itertuples(index=False):
                coords[str(row.delivery)] = (float(row.delivery_lat), float(row.delivery_lon))
    return coords


def build_daily_market_lookup(frames: list[pd.DataFrame]) -> dict[str, float]:
    """Build date -> median market_index lookup across available datasets."""
    parts = [df[["date", "market_index"]] for df in frames if "market_index" in df.columns]
    if not parts:
        return {}
    combined = pd.concat(parts, ignore_index=True).dropna(subset=["market_index"])
    return combined.groupby("date")["market_index"].median().to_dict()


def clean_and_engineer_features(
    df: pd.DataFrame,
    weight_medians_by_eq: dict[str, float],
    global_weight_median: float,
    daily_market_lookup: dict[str, float],
    global_market_median: float,
    city_coords: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Clean anomalies and create domain-engineered trucking rate features (excluding poisoned quote_signal)."""
    out = pd.DataFrame(index=df.index)

    # 1. Populate missing coordinates (e.g. for december_chart_inputs.csv)
    if "pickup_lat" in df.columns:
        p_lat = df["pickup_lat"].astype(float)
        p_lon = df["pickup_lon"].astype(float)
        d_lat = df["delivery_lat"].astype(float)
        d_lon = df["delivery_lon"].astype(float)
    else:
        p_lat = df["pickup"].map(lambda c: city_coords[c][0]).astype(float)
        p_lon = df["pickup"].map(lambda c: city_coords[c][1]).astype(float)
        d_lat = df["delivery"].map(lambda c: city_coords[c][0]).astype(float)
        d_lon = df["delivery"].map(lambda c: city_coords[c][1]).astype(float)

    # 2. Clean weight: fix negative sign-flip corruption & impute missing by equipment median
    raw_weight = pd.to_numeric(df["weight"], errors="coerce")
    out["is_weight_missing"] = raw_weight.isna().astype(np.int8)
    out["is_weight_negative"] = (raw_weight <= 0).fillna(False).astype(np.int8)
    abs_weight = raw_weight.abs()
    fallback_w = df["equipment"].map(weight_medians_by_eq).fillna(global_weight_median)
    weight_clean = abs_weight.fillna(fallback_w)
    out["weight_clean"] = weight_clean.astype(float)
    out["weight_k"] = out["weight_clean"] / 1000.0
    out["is_heavy_load"] = (out["weight_clean"] >= 40000.0).astype(np.int8)

    # 3. Clean market_index: impute missing via same-day median, then global median
    if "market_index" in df.columns:
        raw_mkt = pd.to_numeric(df["market_index"], errors="coerce")
        out["is_market_missing"] = raw_mkt.isna().astype(np.int8)
        date_fallback = df["date"].map(daily_market_lookup).fillna(global_market_median)
        out["market_index_clean"] = raw_mkt.fillna(date_fallback).astype(float)
    else:
        out["is_market_missing"] = np.int8(1)
        out["market_index_clean"] = df["date"].map(daily_market_lookup).fillna(global_market_median).astype(float)

    # 4. Distance & Spatial Geometry
    dist = pd.to_numeric(df["distance"], errors="coerce").astype(float)
    hav = haversine_miles(p_lat, p_lon, d_lat, d_lon)
    out["distance"] = dist
    out["log_distance"] = np.log1p(dist)
    out["sqrt_distance"] = np.sqrt(dist)
    out["inv_distance"] = 1.0 / np.maximum(dist, 1.0)
    out["haversine_miles"] = hav
    out["circuity_ratio"] = dist / np.maximum(hav, 1.0)
    out["detour_miles"] = dist - hav

    out["pickup_lat"] = p_lat
    out["pickup_lon"] = p_lon
    out["delivery_lat"] = d_lat
    out["delivery_lon"] = d_lon
    out["lat_diff"] = d_lat - p_lat
    out["lon_diff"] = d_lon - p_lon
    out["abs_lat_diff"] = np.abs(out["lat_diff"])
    out["abs_lon_diff"] = np.abs(out["lon_diff"])
    bearing = np.arctan2(out["lon_diff"], out["lat_diff"])
    out["bearing_sin"] = np.sin(bearing)
    out["bearing_cos"] = np.cos(bearing)

    # 5. Interactions
    out["market_x_dist"] = out["market_index_clean"] * dist
    out["weight_x_dist"] = out["weight_k"] * dist

    # 6. Calendar & Seasonality
    dt = pd.to_datetime(df["date"])
    out["month"] = dt.dt.month.astype(np.int8)
    out["day"] = dt.dt.day.astype(np.int8)
    out["dayofweek"] = dt.dt.dayofweek.astype(np.int8)
    out["dayofyear"] = dt.dt.dayofyear.astype(np.int16)
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(np.int8)
    out["is_month_end"] = dt.dt.is_month_end.astype(np.int8)
    out["dow_sin"] = np.sin(2.0 * np.pi * out["dayofweek"] / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * out["dayofweek"] / 7.0)
    out["doy_sin"] = np.sin(2.0 * np.pi * out["dayofyear"] / 365.25)
    out["doy_cos"] = np.cos(2.0 * np.pi * out["dayofyear"] / 365.25)
    out["is_holiday"] = dt.isin(US_HOLIDAYS_2025).astype(np.int8)

    # Days to nearest US holiday (captures pre/post holiday freight tightening)
    dt_vals = dt.to_numpy(dtype="datetime64[D]")
    hol_vals = US_HOLIDAYS_2025.to_numpy(dtype="datetime64[D]")
    diff_days = np.abs((dt_vals[:, None] - hol_vals[None, :]).astype(int))
    out["days_to_holiday"] = np.minimum(diff_days.min(axis=1), 30).astype(np.int16)

    # 7. Categoricals (encoded cleanly later)
    out["equipment"] = df["equipment"].astype(str)
    out["pickup"] = df["pickup"].astype(str)
    out["delivery"] = df["delivery"].astype(str)
    out["lane"] = out["pickup"] + " -> " + out["delivery"]

    return out
