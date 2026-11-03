"""
Main FastAPI Application Entrypoint.
Production-ready Multi-Horizon Food Price Forecasting Service for PasokMataram DIY.
Complies with Model-as-a-Service (MaaS) Lifespan Pattern and Zero-Cost FOSS Mandate.
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

try:
    from dotenv import load_dotenv
    root_dir = Path(__file__).resolve().parent.parent
    env_file = os.getenv("ENV_FILE", ".env")
    env_path = root_dir / env_file
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv(root_dir / ".env")
except ImportError:
    pass

from app.schemas import (
    ForecastListResponse,
    CommodityForecastResponse,
    SimulationRequest,
)
from app.services.forecast import ForecastService

ml_models: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["status"] = "loaded"
    ml_models["service"] = ForecastService(models=ml_models)
    yield
    ml_models.clear()


app = FastAPI(
    title="PasokMataram ML Forecasting API",
    description="Layanan Prediksi Harga Pangan Multi-Horizon (H+1 s/d H+7) Berbasis LightGBM & BMKG Terbuka D.I. Yogyakarta",
    version="1.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "PasokMataram ML Price Forecasting Service",
        "status": "online",
        "engine": "FastAPI + LightGBM Direct Multi-Horizon",
        "docs": "/docs",
    }


@app.get("/api/v1/forecast/commodities", response_model=ForecastListResponse, tags=["Forecast"])
async def list_commodity_forecasts(
    category: str = Query(None, description="Filter kategori (Hortikultura / Pangan Pokok)"),
):
    service: ForecastService = ml_models.get("service")
    if not service:
        service = ForecastService()

    forecasts = service.get_all_forecasts()
    if category:
        forecasts = [f for f in forecasts if f.category.lower() == category.lower()]

    return ForecastListResponse(
        success=True,
        count=len(forecasts),
        data=forecasts,
    )


@app.get("/api/v1/forecast/{commodity_code}", response_model=CommodityForecastResponse, tags=["Forecast"])
async def get_single_forecast(commodity_code: str):
    service: ForecastService = ml_models.get("service")
    if not service:
        service = ForecastService()

    forecast = service.generate_single_forecast(commodity_code.upper())
    if not forecast:
        raise HTTPException(
            status_code=404,
            detail=f"Komoditas dengan kode '{commodity_code}' tidak ditemukan",
        )
    return forecast


@app.post("/api/v1/forecast/simulate", response_model=CommodityForecastResponse, tags=["Simulation"])
async def simulate_scenario(req: SimulationRequest):
    service: ForecastService = ml_models.get("service")
    if not service:
        service = ForecastService()

    forecast = service.generate_single_forecast(
        commodity_code=req.commodity_code.upper(),
        current_price=req.current_price,
        rainfall_anomaly_percent=req.rainfall_anomaly_percent,
        fuel_price_hike_percent=req.fuel_price_hike_percent,
    )
    if not forecast:
        raise HTTPException(
            status_code=404,
            detail=f"Komoditas dengan kode '{req.commodity_code}' tidak ditemukan",
        )
    return forecast
