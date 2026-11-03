"""
PasokMataram API Integration Client Module.
Strictly uses HTTP REST API (http://localhost:3000/api/v1/forecast) to exchange
live commodity market prices, historical series, and forecast publications.
ZERO direct database connections: complies strictly with API-first separation of concerns.
"""

import os
import sys
import json
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PASOK_API_BASE_URL = os.getenv("PASOK_API_URL", "http://localhost:3000/api/v1/forecast")

COMMODITY_METADATA: Dict[str, Dict[str, Any]] = {
    "CABAI_RAWIT_MERAH": {
        "id": "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
        "name": "Cabai Rawit Merah Samas",
        "category": "Hasil Tani Segar",
        "origin_region": "Kulon Progo & Bantul",
        "unit": "kg",
        "base_price": 48500.0,
    },
    "BAWANG_MERAH": {
        "id": "b1eebc99-9c0b-4ef8-bb6d-6bb9bd380a22",
        "name": "Bawang Merah Srikayangan",
        "category": "Hasil Tani Segar",
        "origin_region": "Kulon Progo & Bantul",
        "unit": "kg",
        "base_price": 36000.0,
    },
    "CABAI_MERAH_KERITING": {
        "id": "c2eebc99-9c0b-4ef8-bb6d-6bb9bd380a33",
        "name": "Cabai Merah Keriting",
        "category": "Hasil Tani Segar",
        "origin_region": "Sleman & Bantul",
        "unit": "kg",
        "base_price": 32000.0,
    },
    "BERAS_PREMIUM_SLEMAN": {
        "id": "d3eebc99-9c0b-4ef8-bb6d-6bb9bd380a44",
        "name": "Beras Pandan Wangi Sleman",
        "category": "Hasil Tani Segar",
        "origin_region": "Sleman",
        "unit": "kg",
        "base_price": 15500.0,
    },
    "BAWANG_PUTIH_HONAN": {
        "id": "e4eebc99-9c0b-4ef8-bb6d-6bb9bd380a55",
        "name": "Bawang Putih Kating",
        "category": "Hasil Tani Segar",
        "origin_region": "Nasional / Impor",
        "unit": "kg",
        "base_price": 40000.0,
    },
    "DAGING_AYAM_RAS": {
        "id": "f5eebc99-9c0b-4ef8-bb6d-6bb9bd380a66",
        "name": "Daging Ayam Broiler Turi",
        "category": "Unggas & Telur",
        "origin_region": "Bantul & Sleman",
        "unit": "kg",
        "base_price": 36000.0,
    },
    "TELUR_AYAM_RAS": {
        "id": "a11ebc99-9c0b-4ef8-bb6d-6bb9bd380a77",
        "name": "Telur Ayam Ras",
        "category": "Unggas & Telur",
        "origin_region": "Sleman & Kulon Progo",
        "unit": "kg",
        "base_price": 28500.0,
    },
    "DAGING_SAPI_SEGAR": {
        "id": "b22ebc99-9c0b-4ef8-bb6d-6bb9bd380a88",
        "name": "Daging Sapi Gandik Segoroyoso",
        "category": "Daging Sapi Segar",
        "origin_region": "Bantul (Segoroyoso)",
        "unit": "kg",
        "base_price": 135000.0,
    },
    "SUSU_SAPI_MURNI": {
        "id": "c33ebc99-9c0b-4ef8-bb6d-6bb9bd380a99",
        "name": "Susu Sapi Murni Kaliurang",
        "category": "Susu Segar & Olahan",
        "origin_region": "Sleman (Kaliurang)",
        "unit": "liter",
        "base_price": 18000.0,
    },
    "TEMPE_KRAPYAK": {
        "id": "d44ebc99-9c0b-4ef8-bb6d-6bb9bd380b00",
        "name": "Tempe Tradisional Krapyak",
        "category": "Hasil Tani Segar",
        "origin_region": "Bantul (Krapyak)",
        "unit": "papan",
        "base_price": 6000.0,
    },
}


def fetch_latest_market_feed_from_api() -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Mengambil data pasar terkini dan deret waktu historis secara eksklusif
    melalui REST API PasokMataram (GET /api/v1/forecast).
    Tidak ada koneksi langsung ke basis data.
    """
    latest_prices: Dict[str, float] = {}
    items: List[Dict[str, Any]] = []

    try:
        req = urllib.request.Request(
            PASOK_API_BASE_URL,
            headers={"User-Agent": "PasokMataramML-Client/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            items = payload.get("data", [])

            for item in items:
                name = item.get("name", "").lower()
                price = item.get("currentPrice") or item.get("current_price")
                for code, meta in COMMODITY_METADATA.items():
                    if meta["name"].lower() in name or code.lower() in name:
                        if price is not None:
                            latest_prices[code] = float(price)

            print(f"[API CLIENT] Sukses mengambil {len(items)} feed komoditas dari {PASOK_API_BASE_URL}.")
    except Exception as exc:
        print(f"[API CLIENT WARN] Tidak dapat menghubungi API PasokMataram ({exc}). Menggunakan cache lokal.")

    return latest_prices, items


def load_integrated_training_dataset() -> pd.DataFrame:
    """
    Mengambil dataset pelatihan dengan memadukan data pra-pemrosesan
    dan pembaruan harga live dari API PasokMataram.
    """
    preprocessed_path = Path("pipeline/data_preprocessed.csv")
    training_path = Path("pipeline/data_training.csv")

    if preprocessed_path.exists():
        df = pd.read_csv(preprocessed_path, sep=";")
    elif training_path.exists():
        df = pd.read_csv(training_path)
    else:
        from pipeline.collect_data import collect_jogjakota_realtime_dataset
        df = collect_jogjakota_realtime_dataset()

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce").fillna(0.0)

    # Sinkronisasi dengan harga live dari API PasokMataram
    api_prices, _ = fetch_latest_market_feed_from_api()
    if api_prices:
        today_str = datetime.now().strftime("%Y-%m-%d")
        for code, latest_p in api_prices.items():
            mask = (df["commodity_code"] == code) & (df["date"].str.startswith(today_str))
            if mask.any():
                df.loc[mask, "price"] = latest_p

    return df


def publish_forecasts_via_api(forecasts_data: List[Dict[str, Any]]) -> bool:
    """
    Mempublikasikan hasil prakiraan ML langsung ke API PasokMataram via HTTP POST.
    Sepenuhnya berbasis HTTP REST tanpa akses database langsung.
    """
    payload = {"forecasts": forecasts_data}
    json_bytes = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        PASOK_API_BASE_URL,
        data=json_bytes,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "PasokMataramML-Publisher/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            print(f"[API PUBLISH SUCCESS] Respon API: {res_data.get('message', 'Sukses mempublikasikan')}")
            return True
    except urllib.error.HTTPError as http_err:
        print(f"[API PUBLISH ERROR] HTTP {http_err.code}: {http_err.read().decode('utf-8', errors='ignore')}")
        return False
    except Exception as exc:
        print(f"[API PUBLISH FAILED] Gagal menghubungi {PASOK_API_BASE_URL}: {exc}")
        return False


if __name__ == "__main__":
    prices, items = fetch_latest_market_feed_from_api()
    print("Latest live prices from API:", prices)
