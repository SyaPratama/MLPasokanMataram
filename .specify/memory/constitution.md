# PasokMataram ML Service Constitution

Prinsip fundamental pengembangan layanan Machine Learning PasokMataram:
1. **Zero-Cost FOSS Mandate**: Seluruh pustaka wajib 100% open-source (FastAPI, LightGBM, Polars, Docker). Bebas API berbayar proprietary.
2. **Batch Pre-computed First**: Prediksi H+1 s/d H+7 dihitung malam hari via scheduled job dan disimpan ke cache. Dilarang komputasi berulang saat GET request publik.
3. **Pydantic Validation Boundary**: Seluruh request dan response wajib divalidasi skema Pydantic tanpa tipe `Any` bebas.
4. **Resilient Fallback**: Kegagalan koneksi data BMKG atau scraper tidak boleh menyebabkan API crash; gunakan moving average imputation.
5. **Spec-Driven Development (SDD)**: Setiap penambahan horizon prediksi atau algoritma baru wajib diawali dengan spesifikasi formal di `specs/`.
