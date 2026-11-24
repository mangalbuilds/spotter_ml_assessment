from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from features import (
    build_city_coord_lookup,
    build_daily_market_lookup,
    clean_and_engineer_features,
)


CAT_COLS = ["equipment", "pickup", "delivery"]


def add_target_encodings(
    X_tr: pd.DataFrame,
    y_tr_rpm: np.ndarray,
    X_ev: pd.DataFrame,
    smoothing: float = 25.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit smoothed historical RPM encodings strictly on training fold to prevent leakage."""
    X_tr = X_tr.copy()
    X_ev = X_ev.copy()
    global_mean = float(np.mean(y_tr_rpm))

    tmp = X_tr[["equipment", "pickup", "delivery", "lane"]].copy()
    tmp["rpm"] = y_tr_rpm

    for col in ["pickup", "delivery", "lane"]:
        stats = tmp.groupby(col)["rpm"].agg(["count", "mean"])
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        mapping = smooth.to_dict()
        X_tr[f"{col}_te_rpm"] = X_tr[col].map(mapping).fillna(global_mean)
        X_ev[f"{col}_te_rpm"] = X_ev[col].map(mapping).fillna(global_mean)

    # Equipment x distance band average RPM
    for df in (X_tr, X_ev):
        df["dist_band"] = pd.cut(df["distance"], bins=[0, 250, 500, 1000, 1800, 5000], labels=False).fillna(2)
    tmp["dist_band"] = X_tr["dist_band"]
    eq_band = tmp.groupby(["equipment", "dist_band"])["rpm"].mean().to_dict()
    X_tr["eq_band_te_rpm"] = [eq_band.get((e, b), global_mean) for e, b in zip(X_tr["equipment"], X_tr["dist_band"])]
    X_ev["eq_band_te_rpm"] = [eq_band.get((e, b), global_mean) for e, b in zip(X_ev["equipment"], X_ev["dist_band"])]

    X_tr = X_tr.drop(columns=["dist_band", "lane"])
    X_ev = X_ev.drop(columns=["dist_band", "lane"])
    return X_tr, X_ev


def encode_categoricals(X_tr: pd.DataFrame, X_ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ordinal-encode categoricals cleanly for HistGradientBoosting & tree models."""
    X_tr = X_tr.copy()
    X_ev = X_ev.copy()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_tr[CAT_COLS] = enc.fit_transform(X_tr[CAT_COLS].astype(str))
    X_ev[CAT_COLS] = enc.transform(X_ev[CAT_COLS].astype(str))
    return X_tr, X_ev


def evaluate_preds(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred) * 100.0),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    train_raw = pd.read_csv("data/train_test.csv")
    val_raw = pd.read_csv("data/validation.csv")

    city_coords = build_city_coord_lookup([train_raw, val_raw])

    # Strict Temporal Holdout Split:
    # Train = Jan-Aug 2025 | Holdout = Sep-Oct 2025
    dt_all = pd.to_datetime(train_raw["date"])
    df_tr = train_raw[dt_all < "2025-09-01"].copy()
    df_ev = train_raw[dt_all >= "2025-09-01"].copy()

    w_clean_tr = df_tr["weight"].abs()
    w_medians = w_clean_tr.groupby(df_tr["equipment"]).median().to_dict()
    w_global = float(w_clean_tr.median())
    mkt_global = float(df_tr["market_index"].median())
    dm_tr = build_daily_market_lookup([df_tr])
    dm_ev = build_daily_market_lookup([df_tr, df_ev])

    X_tr_base = clean_and_engineer_features(df_tr, w_medians, w_global, dm_tr, mkt_global, city_coords)
    X_ev_base = clean_and_engineer_features(df_ev, w_medians, w_global, dm_ev, mkt_global, city_coords)

    y_tr = df_tr["posted_rate"].to_numpy(dtype=float)
    y_ev = df_ev["posted_rate"].to_numpy(dtype=float)
    y_tr_rpm = y_tr / df_tr["distance"].to_numpy(dtype=float)
    y_ev_rpm = y_ev / df_ev["distance"].to_numpy(dtype=float)

    clean_mask = (y_tr_rpm >= 1.0) & (y_tr_rpm <= 4.5)
    ev_clean_mask = (y_ev_rpm >= 1.0) & (y_ev_rpm <= 4.5)

    print(f"Temporal Split -> Train (Jan-Aug): {len(df_tr):,} rows | Holdout (Sep-Oct): {len(df_ev):,} rows")
    print(f"Target outliers filtered in train fold: {(~clean_mask).sum()} ({(~clean_mask).mean()*100:.2f}%)")

    # 1. Poisoned feature demonstration (with quote_signal)
    X_tr_lk, X_ev_lk = encode_categoricals(X_tr_base.drop(columns=["lane"]), X_ev_base.drop(columns=["lane"]))
    X_tr_lk["quote_signal"] = df_tr["quote_signal"].to_numpy()
    X_ev_lk["quote_signal"] = df_ev["quote_signal"].to_numpy()
    m_leaky = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=63, categorical_features=CAT_COLS, random_state=42)
    m_leaky.fit(X_tr_lk, y_tr)
    p_leaky = np.maximum(m_leaky.predict(X_ev_lk), 50.0)
    r_lk_all = evaluate_preds(y_ev, p_leaky)
    r_lk_cln = evaluate_preds(y_ev[ev_clean_mask], p_leaky[ev_clean_mask])
    print(f"\n[TRAP DEMO] Naive HistGB WITH poisoned quote_signal -> ALL MAE=${r_lk_all['MAE']:.2f} RMSE=${r_lk_all['RMSE']:.2f} | CLEAN-98.6% MAE=${r_lk_cln['MAE']:.2f}")

    # Clean features (quote_signal excluded)
    X_tr_clean_raw, X_ev_clean_raw = add_target_encodings(
        X_tr_base[clean_mask], y_tr_rpm[clean_mask], X_ev_base, smoothing=25.0
    )
    X_tr_full_raw, X_ev_full_raw = add_target_encodings(X_tr_base, y_tr_rpm, X_ev_base, smoothing=25.0)

    y_tr_clean_rpm = y_tr_rpm[clean_mask]
    y_tr_clean_rate = y_tr[clean_mask]
    dist_ev = df_ev["distance"].to_numpy(dtype=float)

    num_cols = [c for c in X_tr_clean_raw.columns if c not in CAT_COLS]
    results = []

    # Model 1: Ridge Baseline
    pre_ridge = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["equipment", "pickup", "delivery"]),
        ("num", "passthrough", num_cols),
    ])
    ridge = Pipeline([("pre", pre_ridge), ("reg", Ridge(alpha=10.0))])
    ridge.fit(X_tr_clean_raw, y_tr_clean_rate)
    p_ridge = np.maximum(ridge.predict(X_ev_clean_raw), 50.0)
    results.append(("1. Ridge Baseline (Clean Rate)", evaluate_preds(y_ev, p_ridge), evaluate_preds(y_ev[ev_clean_mask], p_ridge[ev_clean_mask])))

    X_tr_c, X_ev_c = encode_categoricals(X_tr_clean_raw, X_ev_clean_raw)
    X_tr_f, X_ev_f = encode_categoricals(X_tr_full_raw, X_ev_full_raw)

    # Model 2: Random Forest (Clean RPM * dist)
    rf = RandomForestRegressor(n_estimators=150, max_depth=16, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(X_tr_c, y_tr_clean_rpm)
    p_rf = np.maximum(rf.predict(X_ev_c) * dist_ev, 50.0)
    results.append(("2. Random Forest (Clean RPM * dist)", evaluate_preds(y_ev, p_rf), evaluate_preds(y_ev[ev_clean_mask], p_rf[ev_clean_mask])))

    # Model 3: Extra Trees (Clean RPM * dist)
    et = ExtraTreesRegressor(n_estimators=150, max_depth=18, min_samples_leaf=5, n_jobs=-1, random_state=42)
    et.fit(X_tr_c, y_tr_clean_rpm)
    p_et = np.maximum(et.predict(X_ev_c) * dist_ev, 50.0)
    results.append(("3. Extra Trees (Clean RPM * dist)", evaluate_preds(y_ev, p_et), evaluate_preds(y_ev[ev_clean_mask], p_et[ev_clean_mask])))

    # Model 4: HistGB L1 (All Train, Direct Rate)
    hgb_all = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=600, learning_rate=0.04, max_leaf_nodes=63,
        min_samples_leaf=25, l2_regularization=1.0, categorical_features=CAT_COLS, random_state=42
    )
    hgb_all.fit(X_tr_f, y_tr)
    p_hgb_all = np.maximum(hgb_all.predict(X_ev_f), 50.0)
    results.append(("4. HistGB L1 (All Train, Direct)", evaluate_preds(y_ev, p_hgb_all), evaluate_preds(y_ev[ev_clean_mask], p_hgb_all[ev_clean_mask])))

    # Model 5: HistGB L1 (Clean Direct Rate)
    hgb_dir = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=600, learning_rate=0.04, max_leaf_nodes=63,
        min_samples_leaf=25, l2_regularization=1.0, categorical_features=CAT_COLS, random_state=42
    )
    hgb_dir.fit(X_tr_c, y_tr_clean_rate)
    p_hgb_dir = np.maximum(hgb_dir.predict(X_ev_c), 50.0)
    results.append(("5. HistGB L1 (Clean Direct Rate)", evaluate_preds(y_ev, p_hgb_dir), evaluate_preds(y_ev[ev_clean_mask], p_hgb_dir[ev_clean_mask])))

    # Model 6: HistGB L1 (Clean RPM * dist)
    hgb_rpm_l1 = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=700, learning_rate=0.035, max_leaf_nodes=63,
        min_samples_leaf=25, l2_regularization=1.5, categorical_features=CAT_COLS, random_state=42
    )
    hgb_rpm_l1.fit(X_tr_c, y_tr_clean_rpm)
    p_hgb_rpm_l1 = np.maximum(hgb_rpm_l1.predict(X_ev_c) * dist_ev, 50.0)
    results.append(("6. HistGB L1 (Clean RPM * dist)", evaluate_preds(y_ev, p_hgb_rpm_l1), evaluate_preds(y_ev[ev_clean_mask], p_hgb_rpm_l1[ev_clean_mask])))

    # Model 7: HistGB L2 (Clean RPM * dist)
    hgb_rpm_l2 = HistGradientBoostingRegressor(
        loss="squared_error", max_iter=700, learning_rate=0.035, max_leaf_nodes=63,
        min_samples_leaf=25, l2_regularization=1.5, categorical_features=CAT_COLS, random_state=42
    )
    hgb_rpm_l2.fit(X_tr_c, y_tr_clean_rpm)
    p_hgb_rpm_l2 = np.maximum(hgb_rpm_l2.predict(X_ev_c) * dist_ev, 50.0)
    results.append(("7. HistGB L2 (Clean RPM * dist)", evaluate_preds(y_ev, p_hgb_rpm_l2), evaluate_preds(y_ev[ev_clean_mask], p_hgb_rpm_l2[ev_clean_mask])))

    # Model 8: Ensemble Blend (55% HistGB L1 + 25% HistGB L2 + 20% RF)
    p_ens = 0.55 * p_hgb_rpm_l1 + 0.25 * p_hgb_rpm_l2 + 0.20 * p_rf
    results.append(("8. Blend (HistGB L1+L2 + RF RPM)", evaluate_preds(y_ev, p_ens), evaluate_preds(y_ev[ev_clean_mask], p_ens[ev_clean_mask])))

    print("\n=== TEMPORAL HOLDOUT TOURNAMENT (SEP-OCT 2025) ===")
    print(f"{'Model':36s} | {'ALL MAE':>8s} {'ALL RMSE':>9s} {'ALL R2':>7s} | {'CLEAN MAE':>9s} {'CLEAN RMSE':>10s} {'CLEAN MAPE':>10s} {'CLEAN R2':>8s}")
    print("-" * 116)
    for name, m_all, m_cln in results:
        print(f"{name:36s} | ${m_all['MAE']:7.2f} ${m_all['RMSE']:8.2f} {m_all['R2']:7.4f} | ${m_cln['MAE']:8.2f} ${m_cln['RMSE']:9.2f} {m_cln['MAPE']:9.2f}% {m_cln['R2']:8.4f}")

    # Permutation importance on clean holdout
    perm = permutation_importance(hgb_rpm_l1, X_ev_c[ev_clean_mask], y_ev_rpm[ev_clean_mask], n_repeats=5, random_state=42, n_jobs=-1)
    imp = pd.Series(perm.importances_mean, index=X_ev_c.columns).sort_values(ascending=False)
    print("\n=== TOP 12 PERMUTATION IMPORTANCES (HistGB L1 RPM) ===")
    print(imp.head(12).round(5))


if __name__ == "__main__":
    main()
