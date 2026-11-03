# PasokMataram ML Price Forecasting Service (D.I. Yogyakarta)

Service Machine Learning mandiri berbasis **FastAPI + LightGBM Direct Multi-Horizon (H+1 s/d H+7) + Docker** untuk memproyeksikan harga bahan pangan komersial D.I. Yogyakarta: Daging Ayam Broiler Turi, Telur Ayam Ras, Daging Sapi Gandik Segoroyoso, Susu Sapi Murni Kaliurang, Cabai Rawit Merah Samas, Bawang Merah Srikayangan, Beras Pandan Wangi Sleman, dan Tempe Tradisional Krapyak.

---

## 1. Arsitektur & Pola Desain (Architectural Pattern)

Sistem mengadopsi 2 pola arsitektur utama:
1. **Pre-computed Batch Inference Pattern**:
   - Prediksi harian dijalankan otomatis via job scheduler (`pipeline/inference_job.py`).
   - Hasil disimpan ke Redis / PostgreSQL agar endpoint API menyajikan data sub-milidetik tanpa komputasi berulang.
2. **Model-as-a-Service (MaaS) Lifespan Pattern**:
   - Model tersimpan di memori saat aplikasi FastAPI boot melalui `@asynccontextmanager lifespan`.
   - Melayani simulasi skenario interaktif (`POST /api/v1/forecast/simulate`) seketika.

---

## 2. Struktur Repositori

```text
D:/MLPasokanMataram/
├── app/
│   ├── main.py              # Entrypoint FastAPI & REST routing
│   ├── schemas.py           # Skema Pydantic (Request, Response, PricePoint)
│   └── services/
│       └── forecast.py      # Logika pemanggilan prediksi, kurva harga & rekomendasi beli
├── pipeline/
│   ├── feature_pipeline.py  # Ekstraksi fitur Polars/Pandas (lag 1, 2, 7, 14, rolling, kalender, BMKG)
│   ├── train.py             # Script retrain model Direct Multi-Horizon (H+1 s/d H+7)
│   ├── inference_job.py     # Batch job harian penghasil proyeksi 7 hari
│   ├── collect_data.py      # Koleksi data harga pangan & cuaca terbuka DIY
│   ├── evaluate_models.py   # Evaluasi metrik akurasi (MAPE, RMSE, MAE)
│   └── sync_pasokmataram.py # Sinkronisasi hasil prediksi ke database utama
├── models/                  # Direktori artefak model LightGBM terkompresi
├── Dockerfile               # Konfigurasi container image Python 3.11-slim
├── docker-compose.yml       # Orkestrasi container API & Redis cache
├── requirements.txt         # Daftar pustaka FOSS terbuka
└── README.md                # Dokumentasi arsitektur dan operasional
```

---

## 3. Menjalankan Layanan Secara Lokal

### Menggunakan Python Virtualenv:
```bash
python -m venv .venv
source .venv/bin/activate  # Di Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Akses OpenAPI Documentation (Swagger UI): `http://localhost:8000/docs`

### Menggunakan Docker Compose:
```bash
docker compose up -d --build
```

---

## 4. Endpoints API

- `GET /` — Healthcheck & metadata service
- `GET /api/v1/forecast/commodities` — Daftar proyeksi 7 hari seluruh bahan pangan aktif DIY
- `GET /api/v1/forecast/{commodity_code}` — Rincian proyeksi & kurva titik harga satu bahan pangan
- `POST /api/v1/forecast/simulate` — Simulasi interaktif pengaruh anomali curah hujan BMKG (%) atau BBM (%) terhadap harga
