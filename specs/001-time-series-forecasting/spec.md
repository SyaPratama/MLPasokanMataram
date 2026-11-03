# Feature Specification: Direct Multi-Horizon Time Series Price Forecasting

**Branch**: `001-time-series-forecasting` | **Date**: 2026-10-13  
**Status**: Production Ready

## 1. Objective
Menyediakan estimasi harga komoditas pangan pokok dan hortikultura D.I. Yogyakarta 7 hari ke depan (H+1 sampai H+7) untuk membantu restoran HoReKa merencanakan anggaran belanja bahan baku dan kelompok tani merencanakan waktu petik panen.

## 2. Requirements
- Model: LightGBM Regressor (7 direct horizon models).
- Fitur: Lags (1, 2, 7, 14), Moving averages (7d, 30d), Standard deviation, Sine/Cosine calendar seasonality, Curah hujan BMKG (mm).
- Latensi Serving: < 10ms melalui pre-computed cache.
- Validasi Input: Pydantic v2.
