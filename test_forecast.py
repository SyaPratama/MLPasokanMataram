"""
Unit tests for PasokMataram Machine Learning Multi-Horizon Forecast Service.
Verifies all CORE_COMMODITIES forecasting across the expanded food ecosystem.
"""

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
