# PasokMataram ML Price Forecasting Service (D.I. Yogyakarta)

Service Machine Learning mandiri berbasis **FastAPI + LightGBM per komoditas Direct Multi-Horizon (H+1 s/d H+7) + Docker** untuk memproyeksikan harga 10 bahan pangan komersial D.I. Yogyakarta dari riwayat pasar yang tersimpan.

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
│   ├── train.py             # Melatih 70 model: 10 komoditas × 7 horizon
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
- `GET /api/v1/forecast/commodities` — Daftar proyeksi 7 hari seluruh 10 bahan pangan aktif DIY dari riwayat pasar lokal
- `GET /api/v1/forecast/{commodity_code}` — Rincian proyeksi dan 15 titik harga historis riil satu bahan pangan
- `POST /api/v1/forecast/simulate` — Simulasi anomali curah hujan pada observasi terbaru; skenario tanpa fitur model ditolak

---

## 5. Cakupan 10 Komoditas Inti Pangan DIY

1. **Unggas & Telur**: `DAGING_AYAM_RAS`, `TELUR_AYAM_RAS`
2. **Daging Sapi Segar**: `DAGING_SAPI_SEGAR`
3. **Susu Segar & Olahan**: `SUSU_SAPI_MURNI`
4. **Hasil Tani Segar**: `CABAI_RAWIT_MERAH`, `BAWANG_MERAH`, `CABAI_MERAH_KERITING`, `BERAS_PREMIUM_SLEMAN`, `BAWANG_PUTIH_HONAN`, `TEMPE_KRAPYAK`

Harga terkini di respons API adalah observasi terakhir masing-masing komoditas dari `pipeline/data_training.csv`, bukan harga dasar statis. Dataset saat ini mencakup 20 Mei–15 November 2026; setiap baris mencatat sumber harga `hargapangan.jogjakota.go.id` dan sumber curah hujan Open-Meteo. Pemanggil boleh mengganti hanya observasi harga terakhir melalui `current_price`. Layanan mengembalikan HTTP 503 jika komoditas memiliki kurang dari 15 observasi riil atau provenance sumber tidak tersedia.

---

## 6. Metrik Evaluasi Holdout (Direct Multi-Horizon LightGBM)

Setiap pasangan komoditas/horizon memiliki model tersendiri. Evaluasi memakai 20% observasi terakhir sebagai holdout berurutan waktu. Baris pelatihan yang targetnya masuk periode holdout dibuang untuk mencegah kebocoran masa depan. Setelah evaluasi, artefak runtime dilatih ulang memakai seluruh riwayat yang tersedia; MAPE yang dipublikasikan tetap berasal hanya dari holdout.

| Horizon | Rata-rata MAPE holdout (10 komoditas) | Rentang MAPE per komoditas |
|---|---:|---:|
| **H+1** | **2.06%** | 0.00%–6.95% |
| **H+2** | **2.86%** | 0.00%–11.20% |
| **H+3** | **3.70%** | 0.00%–17.64% |
| **H+4** | **4.92%** | 0.00%–27.60% |
| **H+5** | **6.14%** | 0.00%–31.16% |
| **H+6** | **5.30%** | 0.00%–29.09% |
| **H+7** | **5.42%** | 0.00%–28.51% |

Nilai per komoditas, jumlah observasi holdout, rentang tanggal, dan kuantil residual tersimpan di `models/model_metadata.json` dengan `evaluation.method = "time_ordered_holdout"`.

---

## 7. Pipeline Otomatisasi GitHub Actions (CI/CD)

- `.github/workflows/ci.yml`: Menjalankan suite `pytest`, termasuk validasi artefak dan metadata holdout berurutan waktu.
- `.github/workflows/docker.yml`: Membangun dan memverifikasi container image `pasokmataram-ml-service`.
