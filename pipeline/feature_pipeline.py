"""
Time Series Feature Pipeline Module.
Implements risk-adjusted Polars/Pandas transformations for lag extraction,
rolling volatility, RapidMiner Z-score exogenous integration, and asymmetric risk boundaries.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


class FeaturePipeline:
    @staticmethod
    def extract_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Menghasilkan fitur lag deret waktu, rasio volatilitas, dan mengintegrasikan
        fitur normalisasi Z-score RapidMiner sebagai indikator risiko pasar.
        """
        df = df.sort_values("date").copy()

        # Lag harga historis
        df["lag_1"] = df["price"].shift(1)
        df["lag_2"] = df["price"].shift(2)
        df["lag_7"] = df["price"].shift(7)
        df["lag_14"] = df["price"].shift(14)

        # Statistik bergulir (rolling mean & volatility)
        df["rolling_mean_7"] = df["price"].shift(1).rolling(window=7, min_periods=1).mean()
        df["rolling_std_7"] = df["price"].shift(1).rolling(window=7, min_periods=1).std().fillna(0.0)
        df["rolling_mean_30"] = df["price"].shift(1).rolling(window=30, min_periods=1).mean()

        # Rasio momentum harga (bebas skala, stabil terhadap inflasi komoditas)
        df["price_ratio_lag1"] = (df["lag_1"] / (df["lag_2"] + 1e-5)).clip(0.5, 2.0)
        df["volatility_ratio_7d"] = (df["rolling_std_7"] / (df["rolling_mean_7"] + 1e-5)).clip(0.0, 1.0)

        # Fitur kalender & musiman DIY
        date_col = pd.to_datetime(df["date"])
        df["day_of_week"] = date_col.dt.dayofweek
        df["month"] = date_col.dt.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        df["sin_day"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["cos_day"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Integrasi fitur normalisasi Z-score eksogen (Curah Hujan & Demand Shock)
        if "rainfall_mm" in df.columns:
            # Jika belum dinormalisasi, lakukan Z-score standardisasi
            r_val = df["rainfall_mm"].astype(float)
            if r_val.std() > 0:
                rainfall_z = (r_val - r_val.mean()) / r_val.std()
            else:
                rainfall_z = r_val * 0.0
            df["rainfall_zscore"] = rainfall_z
            df["rolling_rainfall_3d"] = r_val.rolling(window=3, min_periods=1).sum()
        else:
            df["rainfall_zscore"] = 0.0
            df["rolling_rainfall_3d"] = 0.0

        if "demand_shock_score" in df.columns:
            d_val = df["demand_shock_score"].astype(float)
            if d_val.std() > 0:
                demand_z = (d_val - d_val.mean()) / d_val.std()
            else:
                demand_z = d_val * 0.0
            df["demand_shock_zscore"] = demand_z
        else:
            df["demand_shock_zscore"] = 0.0

        # Composite Risk Index: menggabungkan anomali cuaca dan lonjakan permintaan wisatawan/mahasiswa
        df["composite_risk_index"] = np.maximum(
            0.0,
            0.6 * df["rainfall_zscore"] + 0.4 * df["demand_shock_zscore"]
        ).clip(0.0, 3.0)

        # Imputasi protektif maju (ffill) dan nol untuk lag awal
        return df.ffill().fillna(0)
