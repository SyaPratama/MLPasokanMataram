"""
Unit and Integration tests for PasokMataram Machine Learning Multi-Horizon Forecast Service.
Verifies all CORE_COMMODITIES forecasting across the expanded food ecosystem and REST API endpoints.
"""

import json
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.services.forecast import ForecastService, CORE_COMMODITIES


def test_forecast_all_core_commodities():
    service = ForecastService()
    forecasts = service.get_all_forecasts()
    assert len(forecasts) == len(CORE_COMMODITIES)
    assert len(forecasts) >= 8

    categories = {f.category for f in forecasts}
    assert "Unggas & Telur" in categories
    assert "Daging Sapi Segar" in categories
    assert "Susu Segar & Olahan" in categories
    assert "Hasil Tani Segar" in categories


def test_single_commodity_forecast():
    service = ForecastService()
    forecast = service.generate_single_forecast("DAGING_AYAM_RAS")
    assert forecast is not None
    assert forecast.commodity_code == "DAGING_AYAM_RAS"
    assert forecast.current_price == 36000.0
    assert len(forecast.predicted_prices_7d) == 7
    assert len(forecast.curve) >= 15
    assert forecast.optimal_buy_window.recommended_action in [
        "BELI_SEKARANG",
        "KUNCI_KONTRAK",
        "TUNGGU_PANEN_RAYA",
    ]


def test_simulation_anomaly_impact():
    service = ForecastService()
    baseline = service.generate_single_forecast("CABAI_RAWIT_MERAH", rainfall_anomaly_percent=0.0)
    anomalous = service.generate_single_forecast("CABAI_RAWIT_MERAH", rainfall_anomaly_percent=50.0)

    assert anomalous.predicted_price_7d >= baseline.predicted_price_7d
    assert anomalous.weather_impact.risk_level in ["SEDANG", "TINGGI"]


def test_model_accuracy_high_threshold():
    """Memvalidasi seluruh model LightGBM H+1 s/d H+7 memiliki akurasi tinggi (MAPE < 3.0%)."""
    meta_path = Path("models/model_metadata.json")
    assert meta_path.exists(), "Berkas model_metadata.json wajib tersedia"

    with open(meta_path, "r") as f:
        meta = json.load(f)

    metrics = meta.get("metrics", {})
    assert len(metrics) == 7, "Harus terdapat metrik untuk 7 horizon proyeksi"

    for h, data in metrics.items():
        mape = data.get("mape", 100.0)
        assert mape < 3.0, f"MAPE horizon {h} ({mape:.2f}%) melampaui ambang batas 3.0%"


def test_fastapi_rest_endpoints():
    """Memvalidasi responsivitas seluruh REST API endpoint FastAPI."""
    client = TestClient(app)

    # 1. Healthcheck
    r_health = client.get("/")
    assert r_health.status_code == 200
    assert r_health.json().get("status") == "online"

    # 2. List commodities
    r_list = client.get("/api/v1/forecast/commodities")
    assert r_list.status_code == 200
    data_list = r_list.json().get("data", [])
    assert len(data_list) == len(CORE_COMMODITIES)

    # 3. Single commodity
    r_single = client.get("/api/v1/forecast/CABAI_RAWIT_MERAH")
    assert r_single.status_code == 200
    assert r_single.json().get("commodity_code") == "CABAI_RAWIT_MERAH"

    # 4. Scenario simulation
    r_sim = client.post(
        "/api/v1/forecast/simulate",
        json={
            "commodity_code": "CABAI_RAWIT_MERAH",
            "current_price": 48500.0,
            "rainfall_anomaly_percent": 25.0,
        },
    )
    assert r_sim.status_code == 200
    assert r_sim.json().get("predicted_price_7d") > 0
