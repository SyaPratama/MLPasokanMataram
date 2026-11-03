"""
Pydantic Schemas for PasokMataram Machine Learning Forecasting API.
Defines strict validation contracts for multi-horizon food price predictions.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class PricePoint(BaseModel):
    date: str = Field(..., description="Tanggal ISO format YYYY-MM-DD")
    price: float = Field(..., gt=0, description="Nilai harga riil atau hasil proyeksi")
    is_projected: bool = Field(False, description="Penanda apakah titik merupakan hasil prediksi")
    confidence_lower: Optional[float] = Field(None, description="Batas bawah interval keyakinan P10")
    confidence_upper: Optional[float] = Field(None, description="Batas atas interval keyakinan P90")


class BuyWindowRecommendation(BaseModel):
    start_date: str
    end_date: str
    target_price: float
    recommended_action: Literal["BELI_SEKARANG", "TUNGGU_PANEN_RAYA", "KUNCI_KONTRAK"]
    explanation: str


class WeatherImpact(BaseModel):
    risk_level: Literal["RENDAH", "SEDANG", "TINGGI"]
    description: str
    rainfall_forecast_mm: float
    bmkg_source: str


class ModelMetadata(BaseModel):
    algorithm: str
    mape_accuracy: float
    last_trained: str
    horizon_days: int = 7


class CommodityForecastResponse(BaseModel):
    commodity_id: str
    commodity_code: str
    name: str
    category: str
    origin_region: str
    unit: str = "kg"
    current_price: float
    predicted_prices_7d: List[float] = Field(..., description="Proyeksi H+1 s/d H+7")
    predicted_price_7d: float
    trend: Literal["UP", "DOWN", "STABLE"]
    trend_percentage: float
    confidence_score: float
    optimal_buy_window: BuyWindowRecommendation
    weather_impact: WeatherImpact
    model_metadata: ModelMetadata
    curve: List[PricePoint]


class ForecastListResponse(BaseModel):
    success: bool = True
    count: int
    data: List[CommodityForecastResponse]


class SimulationRequest(BaseModel):
    commodity_code: str
    current_price: float
    rainfall_anomaly_percent: float = Field(0.0, description="Simulasi kenaikan/penurunan curah hujan (%)")
    fuel_price_hike_percent: float = Field(0.0, description="Simulasi kenaikan biaya logistik/solar (%)")
