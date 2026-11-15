"""Regression tests for real per-commodity forecasts and holdout metadata."""

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app, ml_models
from app.services.forecast import CORE_COMMODITIES, ForecastService
from pipeline.collect_data import collect_jogjakota_realtime_dataset
from pipeline.train import train_multi_horizon_models


@pytest.fixture(scope="module")
def forecast_service():
    return ForecastService()


def test_forecast_all_core_commodities(forecast_service):
    forecasts = forecast_service.get_all_forecasts()
    assert len(forecasts) == len(CORE_COMMODITIES)

    categories = {forecast.category for forecast in forecasts}
    assert categories == {
        "Unggas & Telur",
        "Daging Sapi Segar",
        "Susu Segar & Olahan",
        "Hasil Tani Segar",
    }


def test_served_history_comes_from_repository_data(forecast_service):
    commodity_code = "DAGING_AYAM_RAS"
    forecast = forecast_service.generate_single_forecast(commodity_code)
    source = pd.read_csv("pipeline/data_training.csv")
    expected = source.loc[source["commodity_code"] == commodity_code].tail(15)
    expected_points = list(zip(expected["date"], expected["price"].astype(float)))
    observed_points = [
        (point.date, point.price)
        for point in forecast.curve
        if not point.is_projected
    ]

    assert observed_points == expected_points
    assert forecast.current_price == expected_points[-1][1]


def test_same_current_price_uses_distinct_commodity_models(forecast_service):
    current_price = 30000.0
    rawit = forecast_service.generate_single_forecast(
        "CABAI_RAWIT_MERAH",
        current_price=current_price,
    )
    rice = forecast_service.generate_single_forecast(
        "BERAS_PREMIUM_SLEMAN",
        current_price=current_price,
    )

    assert rawit.current_price == rice.current_price == current_price
    assert rawit.predicted_prices_7d != rice.predicted_prices_7d
    assert [point.price for point in rawit.curve[:14]] != [
        point.price for point in rice.curve[:14]
    ]

def test_current_price_changes_forecast(forecast_service):
    baseline = forecast_service.generate_single_forecast("CABAI_RAWIT_MERAH")
    changed = forecast_service.generate_single_forecast(
        "CABAI_RAWIT_MERAH",
        current_price=baseline.current_price * 1.5,
    )

    assert changed.current_price == baseline.current_price * 1.5
    assert changed.predicted_prices_7d != baseline.predicted_prices_7d


def test_metadata_contains_only_time_ordered_holdout_mape(forecast_service):
    metadata = json.loads(Path("models/model_metadata.json").read_text(encoding="utf-8"))
    assert metadata["model_contract"] == "one_model_per_commodity_per_horizon"
    assert metadata["evaluation"]["method"] == "time_ordered_holdout"
    assert metadata["model_count"] == len(CORE_COMMODITIES) * 7
    assert metadata["price_sources"] == ["https://hargapangan.jogjakota.go.id"]
    assert metadata["rainfall_sources"] == ["https://archive-api.open-meteo.com"]

    for commodity in CORE_COMMODITIES:
        commodity_metrics = metadata["metrics"][commodity["code"]]
        assert set(commodity_metrics) == {f"h{horizon}" for horizon in range(1, 8)}
        for metric in commodity_metrics.values():
            assert "mape" not in metric
            assert metric["holdout_mape"] >= 0
            assert metric["holdout_observations"] > 0
            assert metric["training_target_end"] < metric["holdout_start"]

    forecast = forecast_service.generate_single_forecast("CABAI_RAWIT_MERAH")
    expected_h7 = metadata["metrics"]["CABAI_RAWIT_MERAH"]["h7"]
    assert forecast.model_metadata.mape_accuracy == round(expected_h7["holdout_mape"], 2)
    assert forecast.model_metadata.evaluation_method == "time_ordered_holdout"
    assert forecast.model_metadata.holdout_observations == expected_h7["holdout_observations"]


def test_training_creates_every_runtime_artifact(tmp_path):
    trained_models = train_multi_horizon_models(model_output_dir=str(tmp_path))
    expected_keys = {
        f"{commodity['code']}:h{horizon}"
        for commodity in CORE_COMMODITIES
        for horizon in range(1, 8)
    }
    expected_files = {
        f"{commodity['code']}_horizon_{horizon}.pkl"
        for commodity in CORE_COMMODITIES
        for horizon in range(1, 8)
    }

    assert set(trained_models) == expected_keys
    assert {path.name for path in tmp_path.glob("*_horizon_*.pkl")} == expected_files
    generated_metadata = json.loads((tmp_path / "model_metadata.json").read_text(encoding="utf-8"))
    assert generated_metadata["model_count"] == len(expected_files)


def test_missing_real_history_returns_service_unavailable(tmp_path):
    source = pd.read_csv("pipeline/data_training.csv")
    short_history = source.loc[source["commodity_code"] == "CABAI_RAWIT_MERAH"].head(14)
    history_path = tmp_path / "short_history.csv"
    short_history.to_csv(history_path, index=False)
    ml_models["service"] = ForecastService(history_path=str(history_path))

    try:
        response = TestClient(app).get("/api/v1/forecast/CABAI_RAWIT_MERAH")
    finally:
        ml_models.clear()

    assert response.status_code == 503
    assert "At least 15 real observations" in response.json()["detail"]


def test_fastapi_rest_endpoints(forecast_service):
    ml_models["service"] = forecast_service
    client = TestClient(app)
    try:
        health = client.get("/")
        forecasts = client.get("/api/v1/forecast/commodities")
        single = client.get("/api/v1/forecast/CABAI_RAWIT_MERAH")
        simulation = client.post(
            "/api/v1/forecast/simulate",
            json={
                "commodity_code": "CABAI_RAWIT_MERAH",
                "current_price": 48500.0,
                "rainfall_anomaly_percent": 25.0,
            },
        )
        unsupported_fuel = client.post(
            "/api/v1/forecast/simulate",
            json={
                "commodity_code": "CABAI_RAWIT_MERAH",
                "current_price": 48500.0,
                "fuel_price_hike_percent": 10.0,
            },
        )
    finally:
        ml_models.clear()

    assert health.status_code == 200
    assert health.json()["status"] == "online"
    assert forecasts.status_code == 200
    assert forecasts.json()["count"] == len(CORE_COMMODITIES)
    assert single.status_code == 200
    assert single.json()["commodity_code"] == "CABAI_RAWIT_MERAH"
    assert single.json()["model_metadata"]["evaluation_method"] == "time_ordered_holdout"
    assert simulation.status_code == 200
    assert simulation.json()["predicted_price_7d"] > 0
    assert unsupported_fuel.status_code == 422


def test_collection_refuses_synthetic_fallback(monkeypatch, tmp_path):
    class EmptyResponse:
        def json(self):
            return {"data": []}

    class EmptyClient:
        def get(self, *_args, **_kwargs):
            return EmptyResponse()

    monkeypatch.setattr("pipeline.collect_data.httpx.Client", lambda **_kwargs: EmptyClient())

    with pytest.raises(RuntimeError, match="At least 15 real price observations"):
        collect_jogjakota_realtime_dataset(
            output_path=str(tmp_path / "training.csv"),
            days_history=30,
        )
