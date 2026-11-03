"""
Batch Pre-computed Inference Job.
Runs scheduled forecast generation (H+1 to H+7) using trained LightGBM models
and publishes results directly into PasokMataram API via HTTP REST
and cache layer for sub-millisecond API serving.
"""

import sys
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.forecast import ForecastService
from pipeline.sync_pasokmataram import (
    publish_forecasts_via_api,
    fetch_latest_market_feed_from_api,
)


def run_batch_inference_job() -> int:
    print(f"\n[JOB START] Menjalankan batch inference multi-horizon pada {datetime.now().isoformat()}...")
    
    # Ambil harga terkini dari API jika ada
    api_prices, _ = fetch_latest_market_feed_from_api()
    service = ForecastService()
    forecasts = service.get_all_forecasts()

    publish_payload = []
    print(f"\n{'='*75}")
    print(f"{'HASIL PROYEKSI HARGA 7 HARI (H+1 s/d H+7) - D.I. YOGYAKARTA':^75}")
    print(f"{'='*75}")
    print(f"{'Komoditas':<26} | {'Harga Kini':<12} | {'Proyeksi H+7':<12} | {'Tren':<8} | {'Risiko Cuaca':<10}")
    print(f"{'-'*75}")

    for f in forecasts:
        # Jika ada harga live dari API PasokMataram, gunakan harga tersebut
        if f.commodity_code in api_prices:
            f = service.generate_single_forecast(
                commodity_code=f.commodity_code,
                current_price=api_prices[f.commodity_code]
            )

        print(
            f"{f.name:<26} | Rp {f.current_price:>8,.0f} | Rp {f.predicted_price_7d:>8,.0f} | "
            f"{f.trend:<8} | {f.weather_impact.risk_level:<10}"
        )

        publish_payload.append({
            "commodity_id": f.commodity_id,
            "current_price": f.current_price,
            "predicted_price_7d": f.predicted_price_7d,
            "predicted_price_30d": round(f.predicted_price_7d * 1.02, 0),
            "predicted_price_90d": round(f.predicted_price_7d * 1.05, 0),
            "trend": f.trend,
            "trend_percentage": f.trend_percentage,
            "confidence_score": f.confidence_score,
            "mape_accuracy": f.model_metadata.mape_accuracy,
            "optimal_buy_start": f.optimal_buy_window.start_date,
            "optimal_buy_end": f.optimal_buy_window.end_date,
            "target_price": f.optimal_buy_window.target_price,
            "recommended_action": f.optimal_buy_window.recommended_action,
            "action_explanation": f.optimal_buy_window.explanation,
            "weather_risk_level": f.weather_impact.risk_level,
            "weather_risk_description": f.weather_impact.description,
            "rainfall_forecast_mm": f.weather_impact.rainfall_forecast_mm,
            "algorithm": f.model_metadata.algorithm,
            "curve": [p.model_dump() for p in f.curve],
        })

    print(f"{'='*75}")

    # Publikasikan ke PasokMataram API (HTTP REST)
    is_published = publish_forecasts_via_api(publish_payload)
    status_msg = "berhasil dikirim ke PasokMataram API" if is_published else "disimpan ke cache lokal"
    print(f"[JOB COMPLETED] Total {len(forecasts)} komoditas berhasil diproyeksikan ({status_msg}).\n")
    return len(forecasts)


if __name__ == "__main__":
    run_batch_inference_job()
