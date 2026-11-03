"""
ML Algorithm Benchmarking and Model Evaluation Module.
Evaluates Ridge, Random Forest, and LightGBM across TimeSeriesSplit cross-validation
incorporating RapidMiner Z-score normalized risk features and PasokMataram live data.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.feature_pipeline import FeaturePipeline
from pipeline.sync_pasokmataram import load_integrated_training_dataset

FEATURE_COLS = [
    "lag_1",
    "lag_2",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_30",
    "price_ratio_lag1",
    "volatility_ratio_7d",
    "sin_day",
    "cos_day",
    "month",
    "is_weekend",
    "rainfall_zscore",
    "demand_shock_zscore",
    "composite_risk_index",
]


def evaluate_algorithms():
    df = load_integrated_training_dataset()

    processed_dfs = []
    for _, group in df.groupby("commodity_code"):
        feat = FeaturePipeline.extract_time_series_features(group)
        feat["target_h1"] = feat["price"].shift(-1)
        processed_dfs.append(feat.dropna(subset=FEATURE_COLS + ["target_h1"]))

    full_df = pd.concat(processed_dfs, ignore_index=True).sort_values("date").reset_index(drop=True)
    X = full_df[FEATURE_COLS]
    y = full_df["target_h1"]

    algorithms = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=15,
            random_state=42,
            verbosity=-1,
        ),
    }

    tscv = TimeSeriesSplit(n_splits=5)
    results = []

    print(f"\n{'='*70}")
    print(f"{'ALGORITHM BENCHMARK WITH RISK-ADJUSTED NORMALIZATION':^70}")
    print(f"{'='*70}")
    print(f"{'Algorithm':<16} | {'MAE (Rp)':<14} | {'RMSE (Rp)':<14} | {'MAPE (%)':<10}")
    print(f"{'-'*70}")

    best_algo = None
    best_mape = float("inf")

    for name, model in algorithms.items():
        mae_list, rmse_list, mape_list = [], [], []
        for train_idx, test_idx in tscv.split(X):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)

            mae_list.append(mean_absolute_error(y_te, preds))
            rmse_list.append(root_mean_squared_error(y_te, preds))
            mape_list.append(mean_absolute_percentage_error(y_te, preds) * 100)

        avg_mae = np.mean(mae_list)
        avg_rmse = np.mean(rmse_list)
        avg_mape = np.mean(mape_list)

        results.append({"name": name, "mae": avg_mae, "rmse": avg_rmse, "mape": avg_mape})
        print(f"{name:<16} | Rp {avg_mae:>10,.0f} | Rp {avg_rmse:>10,.0f} | {avg_mape:>8.2f}%")

        if avg_mape < best_mape:
            best_mape = avg_mape
            best_algo = name

    print(f"{'='*70}")
    print(f"Algorithm Terpilih: {best_algo} (MAPE: {best_mape:.2f}%)\n")
    return results


if __name__ == "__main__":
    evaluate_algorithms()
