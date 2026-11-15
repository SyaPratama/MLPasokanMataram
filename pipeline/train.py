"""Train one LightGBM model per commodity and forecast horizon."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from app.services.forecast import CORE_COMMODITIES
from pipeline.feature_pipeline import FeaturePipeline

FEATURE_COLS = [
    "current_price",
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


HOLDOUT_FRACTION = 0.20


def train_multi_horizon_models(
    data_df: Optional[pd.DataFrame] = None,
    model_output_dir: str = "models",
) -> Dict[str, Any]:
    source_name = "provided_dataframe"
    if data_df is None:
        source_path = ROOT_DIR / "pipeline" / "data_training.csv"
        if not source_path.exists():
            raise FileNotFoundError(f"Real training history is unavailable: {source_path}")
        data_df = pd.read_csv(source_path)
        source_name = "pipeline/data_training.csv"
    else:
        data_df = data_df.copy()

    required_columns = {
        "date",
        "commodity_code",
        "price",
        "rainfall_mm",
        "demand_shock_score",
        "price_source",
        "rainfall_source",
    }
    missing_columns = required_columns.difference(data_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Training data is missing required columns: {missing}")

    data_df["date"] = pd.to_datetime(data_df["date"], errors="raise")
    for column in ("price", "rainfall_mm", "demand_shock_score"):
        data_df[column] = pd.to_numeric(data_df[column], errors="coerce")
    data_df = data_df.dropna(subset=list(required_columns)).sort_values(["commodity_code", "date"])
    data_df = data_df.loc[data_df["price"] > 0].reset_index(drop=True)

    expected_codes = {commodity["code"] for commodity in CORE_COMMODITIES}
    missing_codes = expected_codes.difference(data_df["commodity_code"].unique())
    if missing_codes:
        missing = ", ".join(sorted(missing_codes))
        raise ValueError(f"Training data is missing runtime commodities: {missing}")

    model_params = {
        "n_estimators": 120,
        "learning_rate": 0.04,
        "num_leaves": 20,
        "min_child_samples": 5,
        "random_state": 42,
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": 1,
        "verbosity": -1,
    }
    trained_models: Dict[str, Any] = {}
    metrics_summary: Dict[str, Dict[str, Any]] = {}

    for commodity_code in sorted(expected_codes):
        commodity_data = data_df.loc[data_df["commodity_code"] == commodity_code].copy()
        if len(commodity_data) < 15:
            raise ValueError(
                f"At least 15 real observations are required for {commodity_code}; found {len(commodity_data)}"
            )

        featured = FeaturePipeline.extract_time_series_features(commodity_data)
        metrics_summary[commodity_code] = {}

        for horizon in range(1, 8):
            target_column = f"target_h{horizon}"
            target_date_column = f"target_date_h{horizon}"
            horizon_data = featured.copy()
            horizon_data[target_column] = horizon_data["price"].shift(-horizon)
            horizon_data[target_date_column] = horizon_data["date"].shift(-horizon)
            clean_data = horizon_data.dropna(
                subset=FEATURE_COLS + [target_column, target_date_column]
            ).reset_index(drop=True)

            holdout_size = max(1, int(np.ceil(len(clean_data) * HOLDOUT_FRACTION)))
            holdout_data = clean_data.iloc[-holdout_size:]
            holdout_start = pd.Timestamp(holdout_data["date"].min())
            training_data = clean_data.iloc[:-holdout_size]
            training_data = training_data.loc[
                training_data[target_date_column] < holdout_start
            ]
            if len(training_data) < 10:
                raise ValueError(f"Insufficient pre-holdout training rows for {commodity_code} H+{horizon}")

            evaluation_model = lgb.LGBMRegressor(**model_params)
            evaluation_model.fit(training_data[FEATURE_COLS], training_data[target_column])
            holdout_predictions = evaluation_model.predict(holdout_data[FEATURE_COLS])
            holdout_actual = holdout_data[target_column].to_numpy()
            residuals = holdout_actual - holdout_predictions
            holdout_mape = float(np.mean(np.abs(residuals / holdout_actual)) * 100.0)
            residual_p10 = float(np.percentile(residuals, 10))
            residual_p90 = float(np.percentile(residuals, 90))
            interval_coverage = float(
                np.mean((residuals >= residual_p10) & (residuals <= residual_p90))
            )

            runtime_model = lgb.LGBMRegressor(**model_params)
            runtime_model.fit(clean_data[FEATURE_COLS], clean_data[target_column])
            model_key = f"{commodity_code}:h{horizon}"
            trained_models[model_key] = runtime_model
            metrics_summary[commodity_code][f"h{horizon}"] = {
                "holdout_mape": holdout_mape,
                "residual_p10": residual_p10,
                "residual_p90": residual_p90,
                "interval_coverage": interval_coverage,
                "training_observations": int(len(training_data)),
                "holdout_observations": int(len(holdout_data)),
                "training_target_end": pd.Timestamp(training_data[target_date_column].max()).strftime("%Y-%m-%d"),
                "holdout_start": holdout_start.strftime("%Y-%m-%d"),
                "holdout_end": pd.Timestamp(holdout_data[target_date_column].max()).strftime("%Y-%m-%d"),
            }
            print(
                f"[EVALUATED] {commodity_code} H+{horizon} holdout MAPE: "
                f"{holdout_mape:.2f}% ({len(holdout_data)} rows)"
            )

    output_dir = Path(model_output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_artifact in output_dir.glob("*_horizon_*.pkl"):
        stale_artifact.unlink()
    for model_key, model in trained_models.items():
        commodity_code, horizon = model_key.rsplit(":h", 1)
        joblib.dump(model, output_dir / f"{commodity_code}_horizon_{horizon}.pkl")

    metadata = {
        "algorithm": "LightGBM per-commodity direct multi-horizon",
        "model_contract": "one_model_per_commodity_per_horizon",
        "artifact_pattern": "{commodity_code}_horizon_{horizon}.pkl",
        "model_count": len(trained_models),
        "features": FEATURE_COLS,
        "evaluation": {
            "method": "time_ordered_holdout",
            "holdout_fraction": HOLDOUT_FRACTION,
            "horizon_purge": True,
            "artifact_fit": "all_available_rows_after_holdout_evaluation",
        },
        "data_source": source_name,
        "price_sources": sorted(data_df["price_source"].unique().tolist()),
        "rainfall_sources": sorted(data_df["rainfall_source"].unique().tolist()),
        "history_start": data_df["date"].min().strftime("%Y-%m-%d"),
        "history_end": data_df["date"].max().strftime("%Y-%m-%d"),
        "metrics": metrics_summary,
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    with (output_dir / "model_metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print(f"[SUCCESS] Trained {len(trained_models)} per-commodity horizon models.")
    return trained_models


if __name__ == "__main__":
    train_multi_horizon_models()
