"""
Forecast Service Module.
Handles LightGBM inference, multi-horizon projection synthesis, risk assessment,
and synchronization with Next.js REST API in PasokMataram DIY.
"""

import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from app.schemas import (
    CommodityForecastResponse,
    PricePoint,
    BuyWindowRecommendation,
    WeatherImpact,
    ModelMetadata,
)

from pipeline.feature_pipeline import FeaturePipeline

ROOT_DIR = Path(__file__).resolve().parents[2]

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

CORE_COMMODITIES: List[Dict[str, Any]] = [
    {
        "id": "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
        "code": "CABAI_RAWIT_MERAH",
        "name": "Cabai Rawit Merah Samas",
        "category": "Hasil Tani Segar",
        "origin_region": "Kulon Progo & Bantul",
        "unit": "kg",
    },
    {
        "id": "b1eebc99-9c0b-4ef8-bb6d-6bb9bd380a22",
        "code": "BAWANG_MERAH",
        "name": "Bawang Merah Srikayangan",
        "category": "Hasil Tani Segar",
        "origin_region": "Kulon Progo & Bantul",
        "unit": "kg",
    },
    {
        "id": "c2eebc99-9c0b-4ef8-bb6d-6bb9bd380a33",
        "code": "CABAI_MERAH_KERITING",
        "name": "Cabai Merah Keriting",
        "category": "Hasil Tani Segar",
        "origin_region": "Sleman & Bantul",
        "unit": "kg",
    },
    {
        "id": "d3eebc99-9c0b-4ef8-bb6d-6bb9bd380a44",
        "code": "BERAS_PREMIUM_SLEMAN",
        "name": "Beras Pandan Wangi Sleman",
        "category": "Hasil Tani Segar",
        "origin_region": "Sleman",
        "unit": "kg",
    },
    {
        "id": "e4eebc99-9c0b-4ef8-bb6d-6bb9bd380a55",
        "code": "BAWANG_PUTIH_HONAN",
        "name": "Bawang Putih Kating",
        "category": "Hasil Tani Segar",
        "origin_region": "Nasional / Impor",
        "unit": "kg",
    },
    {
        "id": "f5eebc99-9c0b-4ef8-bb6d-6bb9bd380a66",
        "code": "DAGING_AYAM_RAS",
        "name": "Daging Ayam Broiler Turi",
        "category": "Unggas & Telur",
        "origin_region": "Bantul & Sleman",
        "unit": "kg",
    },
    {
        "id": "a11ebc99-9c0b-4ef8-bb6d-6bb9bd380a77",
        "code": "TELUR_AYAM_RAS",
        "name": "Telur Ayam Ras",
        "category": "Unggas & Telur",
        "origin_region": "Sleman & Kulon Progo",
        "unit": "kg",
    },
    {
        "id": "b22ebc99-9c0b-4ef8-bb6d-6bb9bd380a88",
        "code": "DAGING_SAPI_SEGAR",
        "name": "Daging Sapi Gandik Segoroyoso",
        "category": "Daging Sapi Segar",
        "origin_region": "Bantul (Segoroyoso)",
        "unit": "kg",
    },
    {
        "id": "c33ebc99-9c0b-4ef8-bb6d-6bb9bd380a99",
        "code": "SUSU_SAPI_MURNI",
        "name": "Susu Sapi Murni Kaliurang",
        "category": "Susu Segar & Olahan",
        "origin_region": "Sleman (Kaliurang)",
        "unit": "liter",
    },
    {
        "id": "d44ebc99-9c0b-4ef8-bb6d-6bb9bd380b00",
        "code": "TEMPE_KRAPYAK",
        "name": "Tempe Tradisional Krapyak",
        "category": "Hasil Tani Segar",
        "origin_region": "Bantul (Krapyak)",
        "unit": "papan",
    },
]


class ForecastUnavailableError(RuntimeError):
    pass


class UnsupportedScenarioError(ValueError):
    pass


class ForecastService:
    def __init__(
        self,
        models: Optional[Dict[str, Any]] = None,
        models_dir: str = "models",
        history_path: str = "pipeline/data_training.csv",
    ):
        self.models = dict(models or {})

        model_dir = Path(models_dir)
        if not model_dir.is_absolute():
            model_dir = ROOT_DIR / model_dir

        metadata_path = model_dir / "model_metadata.json"
        if not metadata_path.exists():
            raise ForecastUnavailableError(f"Model metadata is unavailable: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            self.metadata = json.load(metadata_file)

        if self.metadata.get("evaluation", {}).get("method") != "time_ordered_holdout":
            raise ForecastUnavailableError("Model metadata does not contain time-ordered holdout metrics")

        for commodity in CORE_COMMODITIES:
            for horizon in range(1, 8):
                model_key = f"{commodity['code']}:h{horizon}"
                if model_key in self.models:
                    continue
                model_path = model_dir / f"{commodity['code']}_horizon_{horizon}.pkl"
                if model_path.exists():
                    self.models[model_key] = joblib.load(model_path)

        history_file = Path(history_path)
        if not history_file.is_absolute():
            history_file = ROOT_DIR / history_file
        if not history_file.exists():
            raise ForecastUnavailableError(f"Market history is unavailable: {history_file}")

        history = pd.read_csv(history_file)
        required_columns = {
            "date",
            "commodity_code",
            "price",
            "rainfall_mm",
            "demand_shock_score",
            "price_source",
            "rainfall_source",
        }
        missing_columns = required_columns.difference(history.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ForecastUnavailableError(f"Market history is missing required columns: {missing}")

        history["date"] = pd.to_datetime(history["date"], errors="raise")
        for column in ("price", "rainfall_mm", "demand_shock_score"):
            history[column] = pd.to_numeric(history[column], errors="coerce")
        self.history = (
            history.dropna(subset=list(required_columns))
            .loc[lambda frame: frame["price"] > 0]
            .sort_values(["commodity_code", "date"])
            .reset_index(drop=True)
        )

    def get_all_forecasts(self) -> List[CommodityForecastResponse]:
        results: List[CommodityForecastResponse] = []
        for commodity in CORE_COMMODITIES:
            forecast = self.generate_single_forecast(commodity["code"])
            if forecast is not None:
                results.append(forecast)
        return results

    def generate_single_forecast(
        self,
        commodity_code: str,
        current_price: Optional[float] = None,
        rainfall_anomaly_percent: float = 0.0,
        fuel_price_hike_percent: float = 0.0,
    ) -> Optional[CommodityForecastResponse]:
        commodity = next((item for item in CORE_COMMODITIES if item["code"] == commodity_code), None)
        if commodity is None:
            return None

        if fuel_price_hike_percent != 0:
            raise UnsupportedScenarioError(
                "fuel_price_hike_percent is unsupported because fuel price is not a trained model feature"
            )
        if not np.isfinite(rainfall_anomaly_percent) or rainfall_anomaly_percent < -100:
            raise UnsupportedScenarioError("rainfall_anomaly_percent must be finite and at least -100")

        history = self.history.loc[self.history["commodity_code"] == commodity_code].copy()
        if len(history) < 15:
            raise ForecastUnavailableError(
                f"At least 15 real observations are required for {commodity_code}; found {len(history)}"
            )

        latest_index = history.index[-1]
        if current_price is not None:
            if not np.isfinite(current_price) or current_price <= 0:
                raise ValueError("current_price must be finite and greater than zero")
            history.loc[latest_index, "price"] = float(current_price)

        latest_rainfall = float(history.loc[latest_index, "rainfall_mm"])
        scenario_rainfall = latest_rainfall * (1.0 + rainfall_anomaly_percent / 100.0)
        history.loc[latest_index, "rainfall_mm"] = scenario_rainfall

        featured = FeaturePipeline.extract_time_series_features(history)
        latest = featured.iloc[-1]
        feature_row = featured.iloc[[-1]][FEATURE_COLS]
        forecast_origin = pd.Timestamp(latest["date"])
        observed_price = float(latest["price"])

        curve = [
            PricePoint(
                date=pd.Timestamp(row.date).strftime("%Y-%m-%d"),
                price=float(row.price),
                is_projected=False,
            )
            for row in history.tail(15).itertuples()
        ]

        commodity_metrics = self.metadata.get("metrics", {}).get(commodity_code)
        if not isinstance(commodity_metrics, dict):
            raise ForecastUnavailableError(f"Holdout metrics are unavailable for {commodity_code}")

        predicted_prices: List[float] = []
        for horizon in range(1, 8):
            model_key = f"{commodity_code}:h{horizon}"
            model = self.models.get(model_key)
            metric = commodity_metrics.get(f"h{horizon}")
            if model is None or not hasattr(model, "predict"):
                raise ForecastUnavailableError(f"Model artifact is unavailable for {commodity_code} H+{horizon}")
            if not isinstance(metric, dict) or not {
                "holdout_mape",
                "residual_p10",
                "residual_p90",
                "interval_coverage",
                "holdout_observations",
            }.issubset(metric):
                raise ForecastUnavailableError(f"Holdout metrics are incomplete for {commodity_code} H+{horizon}")

            predicted = float(model.predict(feature_row)[0])
            if not np.isfinite(predicted) or predicted <= 0:
                raise ForecastUnavailableError(f"Model returned an invalid forecast for {commodity_code} H+{horizon}")

            predicted = round(predicted, 0)
            lower = round(max(1.0, predicted + float(metric["residual_p10"])), 0)
            upper = round(max(lower, predicted + float(metric["residual_p90"])), 0)
            predicted_prices.append(predicted)
            curve.append(
                PricePoint(
                    date=(forecast_origin + timedelta(days=horizon)).strftime("%Y-%m-%d"),
                    price=predicted,
                    is_projected=True,
                    confidence_lower=lower,
                    confidence_upper=upper,
                )
            )

        final_prediction = predicted_prices[-1]
        trend_percentage = round(((final_prediction - observed_price) / observed_price) * 100, 1)
        trend = "UP" if trend_percentage > 1.5 else ("DOWN" if trend_percentage < -1.5 else "STABLE")

        rainfall_zscore = float(latest["rainfall_zscore"])
        composite_risk = float(latest["composite_risk_index"])
        if composite_risk > 1.2:
            risk_level = "TINGGI"
        elif composite_risk > 0.4:
            risk_level = "SEDANG"
        else:
            risk_level = "RENDAH"

        recommended_action = (
            "KUNCI_KONTRAK"
            if trend == "UP"
            else ("BELI_SEKARANG" if trend == "STABLE" else "TUNGGU_PANEN_RAYA")
        )
        h7_metrics = commodity_metrics["h7"]

        return CommodityForecastResponse(
            commodity_id=commodity["id"],
            commodity_code=commodity["code"],
            name=commodity["name"],
            category=commodity["category"],
            origin_region=commodity["origin_region"],
            unit=commodity["unit"],
            current_price=observed_price,
            predicted_prices_7d=predicted_prices,
            predicted_price_7d=final_prediction,
            trend=trend,
            trend_percentage=trend_percentage,
            confidence_score=round(float(h7_metrics["interval_coverage"]) * 100.0, 1),
            optimal_buy_window=BuyWindowRecommendation(
                start_date=forecast_origin.strftime("%Y-%m-%d"),
                end_date=(forecast_origin + timedelta(days=4)).strftime("%Y-%m-%d"),
                target_price=observed_price,
                recommended_action=recommended_action,
                explanation=(
                    f"Proyeksi tren {trend} ({trend_percentage:+.1f}% H+7). "
                    f"Strategi pasokan lelang {commodity['origin_region']}."
                ),
            ),
            weather_impact=WeatherImpact(
                risk_level=risk_level,
                description=f"Indeks anomali cuaca historis Z={rainfall_zscore:+.2f}.",
                rainfall_forecast_mm=round(scenario_rainfall, 1),
                bmkg_source="Open-Meteo historical precipitation",
            ),
            model_metadata=ModelMetadata(
                algorithm=self.metadata["algorithm"],
                mape_accuracy=round(float(h7_metrics["holdout_mape"]), 2),
                evaluation_method=self.metadata["evaluation"]["method"],
                holdout_observations=int(h7_metrics["holdout_observations"]),
                last_trained=self.metadata["trained_at"],
                horizon_days=7,
            ),
            curve=curve,
        )
