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
- `GET /api/v1/forecast/commodities` — Daftar proyeksi 7 hari seluruh 10 bahan pangan aktif DIY
- `GET /api/v1/forecast/{commodity_code}` — Rincian proyeksi & kurva titik harga satu bahan pangan
- `POST /api/v1/forecast/simulate` — Simulasi interaktif pengaruh anomali curah hujan BMKG (%) atau BBM (%) terhadap harga

---

## 5. Cakupan 10 Komoditas Inti Pangan DIY

1. **Unggas & Telur**:
   - Daging Ayam Broiler Turi (`DAGING_AYAM_RAS`) - Rp 36.000 / kg
   - Telur Ayam Ras (`TELUR_AYAM_RAS`) - Rp 28.500 / kg
2. **Daging Sapi Segar**:
   - Daging Sapi Gandik Segoroyoso (`DAGING_SAPI_SEGAR`) - Rp 135.000 / kg
3. **Susu Segar & Olahan**:
   - Susu Sapi Murni Kaliurang (`SUSU_SAPI_MURNI`) - Rp 18.000 / liter
4. **Hasil Tani Segar**:
   - Cabai Rawit Merah Samas (`CABAI_RAWIT_MERAH`) - Rp 48.500 / kg
   - Bawang Merah Srikayangan (`BAWANG_MERAH`) - Rp 36.000 / kg
   - Cabai Merah Keriting (`CABAI_MERAH_KERITING`) - Rp 32.000 / kg
   - Beras Pandan Wangi Sleman (`BERAS_PREMIUM_SLEMAN`) - Rp 15.500 / kg
   - Bawang Putih Kating (`BAWANG_PUTIH_HONAN`) - Rp 40.000 / kg
   - Tempe Tradisional Krapyak (`TEMPE_KRAPYAK`) - Rp 6.000 / papan

---

## 6. Metrik Evaluasi Akurasi (Direct Multi-Horizon LightGBM)

Seluruh model horizon langsung dilatih ulang menggunakan fitur lag deret waktu, rasio volatilitas, dan fitur eksogen curah hujan BMKG dengan batas akurasi tinggi (target MAPE < 3.0%):

| Horizon | Periode | MAPE | Status Akurasi |
|---|---|---|---|
| **H+1** | Besok | **1.74%** | Akurasi 98.26% |
| **H+2** | H+2 | **1.94%** | Akurasi 98.06% |
| **H+3** | H+3 | **2.04%** | Akurasi 97.96% |
| **H+4** | H+4 | **2.19%** | Akurasi 97.81% |
| **H+5** | H+5 | **2.32%** | Akurasi 97.68% |
| **H+6** | H+6 | **2.41%** | Akurasi 97.59% |
| **H+7** | H+7 | **2.52%** | Akurasi 97.48% |

---

## 7. Pipeline Otomatisasi GitHub Actions (CI/CD)

- `.github/workflows/ci.yml`: Menjalankan suite pengetesan unit `pytest` dan validasi ambang batas akurasi model otomatis (hanya pada branch `main`).
- `.github/workflows/docker.yml`: Membangun dan memverifikasi container image `pasokmataram-ml-service` (hanya pada branch `main`).
