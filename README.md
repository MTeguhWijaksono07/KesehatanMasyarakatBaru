# KesehatanMasyarakat_RAG

Sistem RAG + LLM untuk edukasi kesehatan masyarakat.

## Cara Menjalankan

1. Buat virtual environment

```bash
python -m venv venv
```

2. Aktifkan environment

Windows:
```bash
venv\Scripts\activate
```

3. Install dependency

```bash
pip install -r requirements.txt
```

4. Konfigurasi Environment Variables

Salin file `.env.example` menjadi `.env` dan isi API Key serta URI database Anda:
```bash
cp .env.example .env
```

5. Jalankan notebook di folder notebooks/

6. Jalankan streamlit

```bash
streamlit run app.py
```
