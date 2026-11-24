# Freight Rate Prediction Challenge — Spotter AI ML Assessment

End-to-end ML pipeline predicting US truckload freight rates (`posted_rate`) across 48,000 historical development loads (Jan–Oct 2025) and 12,000 future validation loads (Nov–Dec 2025), plus a 31-day December 2025 lane forecast.

## Key Engineering & Data Findings

1. **`quote_signal` Upstream Poisoning Trap:** Monthly correlation analysis revealed three unstable regimes in `quote_signal`: strong positive proxy in Jan/Feb/Mar/Jun/Sep (`+0.45` to `+0.49`), **sign-inverted** in Apr/May/Jul/Oct (`-0.43` to `-0.47`), and **pure uncorrelated white noise** in August (`-0.01`) and all of `validation.csv` Nov–Dec 2025 (`corr` with distance `~0.00`, Dry Van = Reefer = `2.05`). Additionally, `december_chart_inputs.csv` omits `quote_signal`. Excluding `quote_signal` improved out-of-time Sep–Oct holdout MAE from **`$156.51` down to `$114.62`**.
2. **Negative & Missing Load Weights:** `292` training rows and `145` validation rows had corrupted negative weights (`-47,500 lb` min), alongside `300` train / `165` val missing values. Restored via `abs(weight)` + corruption indicator flags + equipment-median imputation.
3. **Target Outliers & Missing Market Index:** `~1.38%` of training loads had severe rate corruption (`<$1.20/mi` or `>$3.50/mi`). Missing `market_index` values (`374` train / `249` val) and December chart inputs were imputed via same-day median `market_index`.
4. **RPM Formulation & Temporal Validation:** Using strict time-based holdout (`Jan–Aug` -> `Sep–Oct`) and 3-fold expanding-window CV (`Aug`, `Sep`, `Oct`), predicting `rate_per_mile = posted_rate / distance` with a 4-model tree ensemble (**45% HistGB L1 + 25% HistGB L2 + 15% ExtraTrees + 15% RandomForest**) achieved **Clean Holdout MAE = `$58.84`**, **MAPE = `2.47%`**, and **R² = `0.9963`**.

## Project Structure

```text
spotter_ml_assessment/
├── data/
│   ├── train_test.csv
│   ├── validation.csv
│   ├── validation_predictions_template.csv
│   └── december_chart_inputs.csv
├── src/
│   ├── eda.py                  # Exploratory & anomaly audit script
│   ├── features.py             # Deterministic cleaning & domain feature pipeline
│   ├── train_cv.py             # Temporal holdout model tournament
│   ├── train_and_predict.py    # Expanding CV, diagnostics, final ensemble & inference
│   └── build_report.py         # Generates PDF report embedding scorer chart
├── reports/
│   ├── eda_and_validation_diagnostics.png
│   ├── metrics_summary.json
│   └── Spotter_ML_Assessment_Report_Mangal_Soren.pdf
├── scorer_results/
│   └── candidate_december.png
├── validation_predictions.csv
├── score.py
└── requirements.txt
```

## Setup & Reproduction

```bash
python -m pip install -r requirements.txt

# 1. Run exploratory audit
PYTHONPATH=src python src/eda.py

# 2. Run model tournament (temporal holdout)
PYTHONPATH=src python src/train_cv.py

# 3. Train final ensemble, generate predictions & diagnostics
PYTHONPATH=src python src/train_and_predict.py

# 4. Run Spotter AI official scorer
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv

# 5. Build PDF report
python src/build_report.py
```
