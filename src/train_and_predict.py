from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.preprocessing import OrdinalEncoder

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
    """Fit smoothed historical RPM encodings strictly on training data to prevent leakage."""
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

    for df in (X_tr, X_ev):
        df["dist_band"] = pd.cut(df["distance"], bins=[0, 250, 500, 1000, 1800, 5000], labels=False).fillna(2)
    tmp["dist_band"] = X_tr["dist_band"]
    eq_band = tmp.groupby(["equipment", "dist_band"])["rpm"].mean().to_dict()
    X_tr["eq_band_te_rpm"] = [eq_band.get((e, b), global_mean) for e, b in zip(X_tr["equipment"], X_tr["dist_band"])]
    X_ev["eq_band_te_rpm"] = [eq_band.get((e, b), global_mean) for e, b in zip(X_ev["equipment"], X_ev["dist_band"])]

    X_tr = X_tr.drop(columns=["dist_band", "lane"])
    X_ev = X_ev.drop(columns=["dist_band", "lane"])
    return X_tr, X_ev


def fit_and_predict_ensemble(
    X_tr: pd.DataFrame,
    y_tr_rpm: np.ndarray,
    X_te_list: list[pd.DataFrame],
) -> list[np.ndarray]:
    """Train diverse RPM-target tree ensemble and return RPM predictions for each test frame."""
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_tr_enc = X_tr.copy()
    X_tr_enc[CAT_COLS] = enc.fit_transform(X_tr_enc[CAT_COLS].astype(str))

    X_te_encs = []
    for X_te in X_te_list:
        df_e = X_te.copy()
        df_e[CAT_COLS] = enc.transform(df_e[CAT_COLS].astype(str))
        X_te_encs.append(df_e)

    models = [
        (0.45, HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=800, learning_rate=0.03, max_leaf_nodes=63,
            min_samples_leaf=25, l2_regularization=1.5, categorical_features=CAT_COLS, random_state=42
        )),
        (0.25, HistGradientBoostingRegressor(
            loss="squared_error", max_iter=800, learning_rate=0.03, max_leaf_nodes=63,
            min_samples_leaf=25, l2_regularization=1.5, categorical_features=CAT_COLS, random_state=42
        )),
        (0.15, ExtraTreesRegressor(
            n_estimators=160, max_depth=18, min_samples_leaf=4, n_jobs=-1, random_state=42
        )),
        (0.15, RandomForestRegressor(
            n_estimators=160, max_depth=16, min_samples_leaf=4, n_jobs=-1, random_state=42
        )),
    ]

    preds = [np.zeros(len(x), dtype=float) for x in X_te_encs]
    for weight, model in models:
        model.fit(X_tr_enc, y_tr_rpm)
        for i, x_te in enumerate(X_te_encs):
            preds[i] += weight * model.predict(x_te)
    return preds


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "RMSE": round(float(root_mean_squared_error(y_true, y_pred)), 2),
        "MAPE": round(float(mean_absolute_percentage_error(y_true, y_pred) * 100.0), 2),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def save_diagnostic_plots(df_holdout: pd.DataFrame, p_holdout: np.ndarray, tr_raw: pd.DataFrame, va_raw: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: quote_signal poisoning comparison across months
    tr_c = tr_raw.copy()
    tr_c["rpm"] = tr_c["posted_rate"] / tr_c["distance"]
    tr_c["m"] = tr_c["date"].str[:7]
    corrs = []
    months = sorted(tr_c["m"].unique())
    for m in months:
        g = tr_c[tr_c["m"] == m]
        corrs.append(g["quote_signal"].corr(g["rpm"]))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=180)
    colors = ["#064A56" if c > 0.2 else ("#D9534F" if c < -0.2 else "#888888") for c in corrs]
    axes[0].bar(months, corrs, color=colors)
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_title("Upstream Trap: quote_signal Corr with RPM by Month", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Pearson Correlation")
    axes[0].tick_params(axis="x", rotation=40)
    axes[0].grid(axis="y", alpha=0.3)

    # Plot 2: Holdout Predicted vs Actual (Clean Sep-Oct)
    rpm_true = df_holdout["posted_rate"].to_numpy() / df_holdout["distance"].to_numpy()
    cln = (rpm_true >= 1.2) & (rpm_true <= 3.5)
    yt_cln = df_holdout["posted_rate"].to_numpy()[cln]
    yp_cln = p_holdout[cln]
    axes[1].scatter(yt_cln[::4], yp_cln[::4], s=8, alpha=0.35, color="#064A56")
    lim = max(yt_cln.max(), yp_cln.max())
    axes[1].plot([0, lim], [0, lim], "--", color="#D9534F", linewidth=1.4, label="Ideal y = x")
    axes[1].set_title("Temporal Holdout (Sep-Oct 2025): Actual vs Predicted", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Actual Posted Rate ($)")
    axes[1].set_ylabel("Predicted Rate ($)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "eda_and_validation_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    tr_raw = pd.read_csv("data/train_test.csv")
    va_raw = pd.read_csv("data/validation.csv")
    dec_raw = pd.read_csv("data/december_chart_inputs.csv")
    tmpl = pd.read_csv("data/validation_predictions_template.csv")

    city_coords = build_city_coord_lookup([tr_raw, va_raw])
    daily_mkt_all = build_daily_market_lookup([tr_raw, va_raw])

    # 1. Run 3-Fold Expanding Window Temporal CV + Sep-Oct Holdout
    dt = pd.to_datetime(tr_raw["date"])
    cv_metrics = []
    for cutoff in ["2025-08-01", "2025-09-01", "2025-10-01"]:
        end = (pd.to_datetime(cutoff) + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
        df_t = tr_raw[dt < cutoff].copy()
        df_e = tr_raw[(dt >= cutoff) & (dt < end)].copy()

        wm = df_t["weight"].abs().groupby(df_t["equipment"]).median().to_dict()
        wg = float(df_t["weight"].abs().median())
        mg = float(df_t["market_index"].median())
        dm_t = build_daily_market_lookup([df_t])
        dm_e = build_daily_market_lookup([df_t, df_e])

        Xt = clean_and_engineer_features(df_t, wm, wg, dm_t, mg, city_coords)
        Xe = clean_and_engineer_features(df_e, wm, wg, dm_e, mg, city_coords)
        yt_rpm = df_t["posted_rate"].to_numpy() / df_t["distance"].to_numpy()
        ye = df_e["posted_rate"].to_numpy()
        ye_rpm = ye / df_e["distance"].to_numpy()

        cm = (yt_rpm >= 1.2) & (yt_rpm <= 3.5)
        ecm = (ye_rpm >= 1.2) & (ye_rpm <= 3.5)

        Xt_c, Xe_c = add_target_encodings(Xt[cm], yt_rpm[cm], Xe, smoothing=25.0)
        [p_rpm] = fit_and_predict_ensemble(Xt_c, yt_rpm[cm], [Xe_c])
        p_rate = np.maximum(p_rpm * df_e["distance"].to_numpy(), 50.0)

        m_all = eval_metrics(ye, p_rate)
        m_cln = eval_metrics(ye[ecm], p_rate[ecm])
        cv_metrics.append({"fold": cutoff[:7], "n_train": len(df_t), "n_val": len(df_e), "all": m_all, "clean": m_cln})
        print(f"Expanding CV Fold {cutoff[:7]} -> ALL MAE=${m_all['MAE']:.2f} R2={m_all['R2']:.4f} | CLEAN MAE=${m_cln['MAE']:.2f} MAPE={m_cln['MAPE']:.2f}% R2={m_cln['R2']:.4f}")

    # Dedicated Sep-Oct 2-month holdout plot & summary
    df_t_ho = tr_raw[dt < "2025-09-01"].copy()
    df_e_ho = tr_raw[dt >= "2025-09-01"].copy()
    wm_h = df_t_ho["weight"].abs().groupby(df_t_ho["equipment"]).median().to_dict()
    wg_h = float(df_t_ho["weight"].abs().median())
    mg_h = float(df_t_ho["market_index"].median())
    dm_th = build_daily_market_lookup([df_t_ho])
    dm_eh = build_daily_market_lookup([df_t_ho, df_e_ho])
    Xt_h = clean_and_engineer_features(df_t_ho, wm_h, wg_h, dm_th, mg_h, city_coords)
    Xe_h = clean_and_engineer_features(df_e_ho, wm_h, wg_h, dm_eh, mg_h, city_coords)
    yt_h_rpm = df_t_ho["posted_rate"].to_numpy() / df_t_ho["distance"].to_numpy()
    cm_h = (yt_h_rpm >= 1.2) & (yt_h_rpm <= 3.5)
    Xt_hc, Xe_hc = add_target_encodings(Xt_h[cm_h], yt_h_rpm[cm_h], Xe_h, smoothing=25.0)
    [p_ho_rpm] = fit_and_predict_ensemble(Xt_hc, yt_h_rpm[cm_h], [Xe_hc])
    p_ho_rate = np.maximum(p_ho_rpm * df_e_ho["distance"].to_numpy(), 50.0)

    ye_ho = df_e_ho["posted_rate"].to_numpy()
    ye_ho_rpm = ye_ho / df_e_ho["distance"].to_numpy()
    ecm_ho = (ye_ho_rpm >= 1.2) & (ye_ho_rpm <= 3.5)
    ho_all = eval_metrics(ye_ho, p_ho_rate)
    ho_cln = eval_metrics(ye_ho[ecm_ho], p_ho_rate[ecm_ho])
    print(f"\nSep-Oct Holdout Summary -> ALL MAE=${ho_all['MAE']:.2f} RMSE=${ho_all['RMSE']:.2f} R2={ho_all['R2']:.4f} | CLEAN MAE=${ho_cln['MAE']:.2f} RMSE=${ho_cln['RMSE']:.2f} MAPE={ho_cln['MAPE']:.2f}% R2={ho_cln['R2']:.4f}")

    save_diagnostic_plots(df_e_ho, p_ho_rate, tr_raw, va_raw, Path("reports"))

    # 2. Final Retrain on Full Cleaned train_test.csv -> Predict validation.csv & december_chart_inputs.csv
    wm_f = tr_raw["weight"].abs().groupby(tr_raw["equipment"]).median().to_dict()
    wg_f = float(tr_raw["weight"].abs().median())
    mg_f = float(tr_raw["market_index"].median())
    daily_mkt_tr = build_daily_market_lookup([tr_raw])

    X_full = clean_and_engineer_features(tr_raw, wm_f, wg_f, daily_mkt_tr, mg_f, city_coords)
    X_val = clean_and_engineer_features(va_raw, wm_f, wg_f, daily_mkt_all, mg_f, city_coords)
    X_dec = clean_and_engineer_features(dec_raw, wm_f, wg_f, daily_mkt_all, mg_f, city_coords)

    y_full_rpm = tr_raw["posted_rate"].to_numpy() / tr_raw["distance"].to_numpy()
    cm_f = (y_full_rpm >= 1.2) & (y_full_rpm <= 3.5)

    # Encode targets using full clean training set
    X_full_c, X_val_c = add_target_encodings(X_full[cm_f], y_full_rpm[cm_f], X_val, smoothing=25.0)
    _, X_dec_c = add_target_encodings(X_full[cm_f], y_full_rpm[cm_f], X_dec, smoothing=25.0)

    p_val_rpm, p_dec_rpm = fit_and_predict_ensemble(X_full_c, y_full_rpm[cm_f], [X_val_c, X_dec_c])

    val_preds = np.round(np.maximum(p_val_rpm * va_raw["distance"].to_numpy(), 50.0), 2)
    dec_preds = np.round(np.maximum(p_dec_rpm * dec_raw["distance"].to_numpy(), 50.0), 2)

    # Align strictly to template load_id order
    pred_map = dict(zip(va_raw["load_id"], val_preds))
    out_val = tmpl.copy()
    out_val["predicted_rate"] = out_val["load_id"].map(pred_map)
    out_val.to_csv("validation_predictions.csv", index=False)

    out_dec = dec_raw.copy()
    out_dec["predicted_rate"] = dec_preds
    out_dec.to_csv("data/december_chart_inputs.csv", index=False)

    summary = {
        "cv_folds": cv_metrics,
        "sep_oct_holdout": {"all": ho_all, "clean": ho_cln},
        "validation_pred_stats": {
            "count": int(len(out_val)),
            "min": float(out_val["predicted_rate"].min()),
            "median": float(out_val["predicted_rate"].median()),
            "mean": float(round(out_val["predicted_rate"].mean(), 2)),
            "max": float(out_val["predicted_rate"].max()),
        },
        "december_chart_stats": {
            "count": int(len(out_dec)),
            "min": float(out_dec["predicted_rate"].min()),
            "mean": float(round(out_dec["predicted_rate"].mean(), 2)),
            "max": float(out_dec["predicted_rate"].max()),
            "dec25_christmas": float(out_dec.loc[out_dec["date"] == "2025-12-25", "predicted_rate"].iloc[0]),
        },
    }
    Path("reports/metrics_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSaved validation_predictions.csv, data/december_chart_inputs.csv, and reports/metrics_summary.json")


if __name__ == "__main__":
    main()
