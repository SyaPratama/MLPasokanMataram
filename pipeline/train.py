"""
Model Retraining Pipeline Script.
Trains 7 Direct Multi-Horizon LightGBM Regressors (H+1 to H+7)
using risk-adjusted features, RapidMiner Z-scores, and live PasokMataram market data.
"""

import os
import sys
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

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


def train_multi_horizon_models(
    data_df: Optional[pd.DataFrame] = None,
    model_output_dir: str = "models"
) -> Dict[str, Any]:
    """
    Melatih 7 model LightGBM horizon langsung (H+1 s/d H+7)
    dan menyimpan artefak model terkompresi.
    """
    os.makedirs(model_output_dir, exist_ok=True)

    if data_df is None:
        data_df = load_integrated_training_dataset()

    trained_models = {}
    metrics_summary = {}

    for h in range(1, 8):
        featured_dfs = []
        if "commodity_code" in data_df.columns:
            for _, group in data_df.groupby("commodity_code"):
                f_grp = FeaturePipeline.extract_time_series_features(group)
                f_grp[f"target_h{h}"] = f_grp["price"].shift(-h)
                featured_dfs.append(f_grp)
            featured = pd.concat(featured_dfs, ignore_index=True).sort_values("date").reset_index(drop=True)
        else:
            featured = FeaturePipeline.extract_time_series_features(data_df)
            featured[f"target_h{h}"] = featured["price"].shift(-h)

        clean_df = featured.dropna(subset=FEATURE_COLS + [f"target_h{h}"])

        X = clean_df[FEATURE_COLS]
        y = clean_df[f"target_h{h}"]

        if len(X) < 10:
            print(f"[SKIP] Sampel tidak mencukupi untuk Horizon H+{h}")
            continue

        model = lgb.LGBMRegressor(
            n_estimators=120,
            learning_rate=0.04,
            num_leaves=20,
            min_child_samples=5,
            random_state=42,
            verbosity=-1,
        )
        model.fit(X, y)

        preds = model.predict(X)
        residuals = y - preds
        mape = float(np.mean(np.abs(residuals / y)) * 100)
        p10_res = float(np.percentile(residuals, 10))
        p90_res = float(np.percentile(residuals, 90))

        print(f"[TRAINED] Horizon H+{h} -> MAPE: {mape:.2f}% | Residual P10: Rp {p10_res:,.0f}, P90: Rp {p90_res:,.0f}")

        model_path = os.path.join(model_output_dir, f"lgbm_horizon_{h}.pkl")
        joblib.dump(model, model_path)
        trained_models[f"h{h}"] = model
        metrics_summary[f"h{h}"] = {
            "mape": mape,
            "p10": p10_res,
            "p90": p90_res,
        }

    # Simpan metadata residual & metrik
    meta_path = os.path.join(model_output_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "algorithm": "LightGBM Direct Multi-Horizon v1.3 (Risk-Adjusted)",
                "features": FEATURE_COLS,
                "metrics": metrics_summary,
                "trained_at": pd.Timestamp.now().isoformat(),
            },
            f,
            indent=2,
        )

    print("[SUCCESS] Pelatihan 7 model multi-horizon selesai.")
    return trained_models


if __name__ == "__main__":
    train_multi_horizon_models()
