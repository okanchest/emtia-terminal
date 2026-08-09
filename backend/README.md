# Emtia Intelligence — Gerçek Veri Kurulumu

## 1. Backend'i çalıştır

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Terminalde `Uvicorn running on http://127.0.0.1:8000` gördüğünde backend hazır.
Tarayıcıdan `http://localhost:8000/api/instrument/altin` adresine gidip JSON döndüğünü kontrol edebilirsin.

## 2. Frontend'i aç

`emtia_intelligence.html` dosyasını doğrudan tarayıcıda aç (çift tıkla ya da
tarayıcıya sürükle). Sayfa otomatik olarak `http://localhost:8000` adresine
istek atacak. Üst çubuktaki rozet:

- **● CANLI VERİ** (yeşil) — backend'den gerçek veri geldi
- **● DEMO VERİ** (kırmızı) — backend'e ulaşılamadı, demo veri gösteriliyor

## 3. Veri kaynakları ve sınırlamalar

| Katman | Kaynak | Not |
|---|---|---|
| Fiyat, RSI, hacim | yfinance (COMEX/CBOT vadeli işlemleri) | Anahtar gerekmez, ~15 dk gecikmeli olabilir |
| DXY, 10Y getiri | yfinance | Piyasa kapalıyken son kapanış gösterilir |
| Enflasyon (CPI) | FRED CSV (anahtarsız) | Aylık veri, gecikmeli açıklanır |
| COT pozisyonlama | CFTC Socrata API | **Haftalık** yayınlanır (Cuma), anahtar gerekmez |

**Dikkat:** CFTC API'sindeki `market_and_exchange_names` alanı bazen
beklenenden farklı yazılabiliyor (örn. "GOLD" yerine "COMEX GOLD" gibi).
`backend/main.py` içindeki `cot_hint` değerleri ilk denemede eşleşmezse,
`https://publicreporting.cftc.gov/resource/6dca-aqww.json?$limit=5` adresini
tarayıcıda açıp gerçek alan adlarını kontrol et ve `INSTRUMENTS` sözlüğündeki
`cot_hint` değerlerini buna göre düzelt.

## 4. Yayına alma (opsiyonel)

Bu backend'i sürekli çalışır tutmak istersen (örn. bilgisayarını
kapattığında da erişilebilir olsun diye), Render.com, Railway, veya bir
VPS üzerine `uvicorn` ile deploy edip `emtia_intelligence.html` içindeki
`API_BASE` değişkenini o adrese güncellemen yeterli.
