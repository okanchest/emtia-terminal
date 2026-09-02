"""
Emtia Intelligence — canlı veri backend'i
==========================================
FastAPI ile: fiyat + teknik veriyi yfinance'ten, enflasyonu FRED'in
anahtar gerektirmeyen CSV uç noktasından, COT (spekülatör pozisyonlama)
verisini CFTC'nin herkese açık Socrata API'sinden çeker; basit ve
şeffaf kurallarla RISK-ON / NEUTRAL / RISK-OFF üretir.

ÖNEMLİ: Bu kurallar demonstrasyon amaçlıdır, yatırım tavsiyesi değildir.
Kendi strategine göre eşik değerlerini (SMA periyotları, RSI sınırları,
COT percentile eşikleri) değiştirmen beklenir.

Kurulum:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Sonra frontend'deki API_BASE değerini http://localhost:8000 olarak
bırakabilirsin (zaten öyle ayarlı).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import math
from datetime import datetime

app = FastAPI(title="Emtia Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Enstrüman tanımları — yfinance ticker sembolleri
# ------------------------------------------------------------------
INSTRUMENTS = {
    "altin":    {"label": "Altın (XAU/USD)",    "yf": "GC=F", "cot_hint": "GOLD"},
    "gumus":    {"label": "Gümüş (XAG/USD)",    "yf": "SI=F", "cot_hint": "SILVER"},
    "platin":   {"label": "Platin (XPT/USD)",   "yf": "PL=F", "cot_hint": "PLATINUM"},
    "paladyum": {"label": "Paladyum (XPD/USD)", "yf": "PA=F", "cot_hint": "PALLADIUM"},
    "bakir":    {"label": "Bakır (HG1!)",       "yf": "HG=F", "cot_hint": "COPPER"},
    "bugday":   {"label": "Buğday (ZW1!)",      "yf": "ZW=F", "cot_hint": "WHEAT"},
}

# Basit bellek-içi cache (API'leri gereksiz yormamak için)
_cache: dict = {}
CACHE_TTL_PRICE = 8 * 60       # 8 dakika
CACHE_TTL_MACRO = 60 * 60      # 1 saat
CACHE_TTL_COT = 12 * 60 * 60   # 12 saat (CFTC verisi haftalık yayınlanır)


def cached(key, ttl, fn):
    now = time.time()
    if key in _cache and now - _cache[key][0] < ttl:
        return _cache[key][1]
    try:
        val = fn()
        _cache[key] = (now, val)
        return val
    except Exception:
        # Taze veri alınamadı (örn. Yahoo Finance geçici rate limit
        # uyguluyor) — elimizde eski ama çalışan bir veri varsa onu
        # döndür, hiç veri yoksa hatayı yukarı fırlat.
        if key in _cache:
            return _cache[key][1]
        raise


def cache_timestamp(key):
    """Bu cache anahtarındaki verinin gerçekten ne zaman (unix timestamp)
    çekildiğini döndürür — kartların 'ne kadar taze' olduğunu göstermek
    için kullanılır. Hiç cache'lenmemişse None döner."""
    if key in _cache:
        return _cache[key][0]
    return None


def sanitize_json(obj):
    """Yanıtı JSON'a göndermeden önce temizler — NaN/Infinity gibi
    JSON-uyumsuz float değerlerini None'a çevirir. Herhangi bir
    hesaplamada (örn. yetersiz veri, sıfıra bölme) beklenmedik bir NaN
    sızarsa, API çökmek yerine o alanı sessizce null döndürür."""
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def verdict_from_score(score):
    if score > 0:
        return "on"
    if score < 0:
        return "off"
    return "neutral"


# ------------------------------------------------------------------
# Fiyat + teknik göstergeler (yfinance)
# ------------------------------------------------------------------
def fetch_price_history(ticker):
    def _do():
        hist = yf.Ticker(ticker).history(period="1y", interval="1d")
        if hist.empty:
            raise HTTPException(502, f"{ticker} için fiyat verisi alınamadı")
        return hist
    return cached(f"hist:{ticker}", CACHE_TTL_PRICE, _do)


def fetch_1h_history(ticker):
    def _do():
        hourly = yf.Ticker(ticker).history(period="60d", interval="60m")
        if hourly.empty:
            raise HTTPException(502, f"{ticker} için 1H veri alınamadı")
        return hourly
    return cached(f"1h:{ticker}", CACHE_TTL_PRICE, _do)


def fetch_4h_history(ticker):
    def _do():
        # Yahoo Finance doğrudan 4H mum vermiyor — 1 saatlik veriyi
        # (zaten cache'lenmiş, Scalp sinyaliyle paylaşılan) çekip 4'erli
        # gruplar halinde birleştiriyoruz (resample).
        hourly = fetch_1h_history(ticker)
        h4 = hourly.resample("4h").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna()
        if h4.empty:
            raise HTTPException(502, f"{ticker} için 4H veri hesaplanamadı")
        return h4
    return cached(f"4h:{ticker}", CACHE_TTL_PRICE, _do)


def compute_scalp_signals(hourly: pd.DataFrame):
    """1 saatlik ham veriden 3 hızlı gösterge üretir — Scalp sinyalinin
    TEK dayanağı. Day Trade'in (4H) daha da kısa vadeli bir versiyonu."""
    close = hourly["Close"]

    rsi_val = compute_rsi(close)
    if rsi_val >= 70:
        rsi_v, rsi_note = "off", f"1H RSI {rsi_val:.0f} — aşırı alım bölgesinde."
    elif rsi_val <= 30:
        rsi_v, rsi_note = "on", f"1H RSI {rsi_val:.0f} — aşırı satım bölgesinde."
    else:
        rsi_v, rsi_note = "neutral", f"1H RSI {rsi_val:.0f} — nötr bölgede."

    macd_now, macd_prev = compute_macd(close)
    if macd_now > 0 and macd_prev <= 0:
        macd_v, macd_note = "on", "1H MACD sinyal çizgisini yukarı kesti."
    elif macd_now < 0 and macd_prev >= 0:
        macd_v, macd_note = "off", "1H MACD sinyal çizgisini aşağı kesti."
    elif macd_now > 0:
        macd_v, macd_note = "on", "1H MACD pozitif bölgede."
    elif macd_now < 0:
        macd_v, macd_note = "off", "1H MACD negatif bölgede."
    else:
        macd_v, macd_note = "neutral", "1H MACD sıfıra yakın."

    ema5 = close.ewm(span=5, adjust=False).mean()
    ema13 = close.ewm(span=13, adjust=False).mean()
    price_now = float(close.iloc[-1])
    ema5_now = float(ema5.iloc[-1])
    ema13_now = float(ema13.iloc[-1])
    if ema5_now > ema13_now and price_now > ema5_now:
        ema_v, ema_note = "on", "1H EMA5, EMA13'ün üstünde ve fiyat EMA5'in üstünde — kısa vadeli momentum yukarı."
    elif ema5_now < ema13_now and price_now < ema5_now:
        ema_v, ema_note = "off", "1H EMA5, EMA13'ün altında ve fiyat EMA5'in altında — kısa vadeli momentum aşağı."
    else:
        ema_v, ema_note = "neutral", "1H EMA5/EMA13 net bir yön vermiyor."

    return [
        {"name": "1H RSI", "verdict": rsi_v, "note": rsi_note},
        {"name": "1H MACD (12/26/9)", "verdict": macd_v, "note": macd_note},
        {"name": "1H EMA5/13 Kesişimi", "verdict": ema_v, "note": ema_note},
    ]


def compute_4h_reversal(h4: pd.DataFrame):
    close = h4["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi_series = 100 - (100 / (1 + rs))

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()

    rsi_now = float(rsi_series.iloc[-1])
    price_now = float(close.iloc[-1])
    ema9_now = float(ema9.iloc[-1])
    ema21_now = float(ema21.iloc[-1])

    # Son 6 barda (yaklaşık son 1 gün) RSI aşırı bölgeye değdi mi diye bak
    recent_rsi = rsi_series.tail(6)
    was_overbought = bool((recent_rsi > 70).any())
    was_oversold = bool((recent_rsi < 30).any())

    if was_overbought and rsi_now < 70 and price_now < ema9_now:
        v = "off"
        note = f"4H RSI aşırı alımdan döndü ({rsi_now:.0f}) ve fiyat 4H EMA9'un altına indi — kısa vadeli tepe/dönüş sinyali."
    elif was_oversold and rsi_now > 30 and price_now > ema9_now:
        v = "on"
        note = f"4H RSI aşırı satımdan döndü ({rsi_now:.0f}) ve fiyat 4H EMA9'un üstüne çıktı — kısa vadeli dip/dönüş sinyali."
    elif ema9_now > ema21_now and price_now > ema9_now:
        v = "on"
        note = f"4H trend yukarı yönlü (EMA9>EMA21), henüz dönüş sinyali yok — mevcut yön sürüyor."
    elif ema9_now < ema21_now and price_now < ema9_now:
        v = "off"
        note = f"4H trend aşağı yönlü (EMA9<EMA21), henüz dönüş sinyali yok — mevcut yön sürüyor."
    else:
        v = "neutral"
        note = f"4H RSI {rsi_now:.0f}, net bir dönüş sinyali yok."

    return v, note


def compute_weekly_trend(hist: pd.DataFrame):
    """Günlük veriyi haftalık mumlara çevirip (resample) basit bir
    haftalık trend teyidi üretir — Haftalık Trade sinyaline gerçek bir
    fiyat/teknik bileşeni katmak için. Yeni bir API çağrısı gerektirmez,
    zaten çekilen günlük veriden hesaplanır."""
    weekly = hist["Close"].resample("W").last().dropna()
    if len(weekly) < 9:
        return "neutral", "Haftalık trend için yeterli geçmiş veri yok."

    sma4w = weekly.rolling(4).mean()
    sma8w = weekly.rolling(8).mean()
    price_now = float(weekly.iloc[-1])
    sma4w_now = float(sma4w.iloc[-1])
    sma8w_now = float(sma8w.iloc[-1])
    recent_change = float((weekly.iloc[-1] / weekly.iloc[-5] - 1) * 100) if len(weekly) >= 5 else 0.0

    if price_now > sma4w_now > sma8w_now:
        v = "on"
        note = f"Haftalık mumlar yukarı yönlü diziliyor (4 ve 8 haftalık ortalamanın üstünde), son 4 haftada %{recent_change:+.1f}."
    elif price_now < sma4w_now < sma8w_now:
        v = "off"
        note = f"Haftalık mumlar aşağı yönlü diziliyor (4 ve 8 haftalık ortalamanın altında), son 4 haftada %{recent_change:+.1f}."
    else:
        v = "neutral"
        note = f"Haftalık trend karışık/net değil, son 4 haftada %{recent_change:+.1f}."
    return v, note


def compute_rsi(close: pd.Series, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def compute_weekly_rsi(hist: pd.DataFrame, period=14):
    """Günlük veriyi haftalık mumlara çevirip RSI hesaplar — çoklu zaman
    dilimi (günlük + haftalık) RSI teyidi için. Yeterli veri yoksa None
    döner."""
    weekly_close = hist["Close"].resample("W").last().dropna()
    if len(weekly_close) < period + 1:
        return None
    return compute_rsi(weekly_close, period)


def find_pivots(series: pd.Series, window=5):
    """Yerel tepe ve dip noktalarının index konumlarını bulur — bir
    nokta, kendi window bar solundaki ve sağındaki tüm noktalardan
    yüksekse tepe, düşükse diptir."""
    highs, lows = [], []
    n = len(series)
    for i in range(window, n - window):
        seg = series.iloc[i - window:i + window + 1]
        if series.iloc[i] == seg.max():
            highs.append(i)
        if series.iloc[i] == seg.min():
            lows.append(i)
    return highs, lows


def compute_rsi_series(close: pd.Series, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def compute_liquidity_levels(hist: pd.DataFrame, window=5, lookback=60):
    """ICT (Inner Circle Trader) konseptine göre likidite seviyeleri:
    - BSL (Buy-Side Liquidity): en yakın önemli swing high'ın üstü —
      short pozisyonların stop-loss'larının kümelendiği, fiyatın
      genelde çekildiği bölge.
    - SSL (Sell-Side Liquidity): en yakın önemli swing low'un altı —
      long pozisyonların stop-loss'larının kümelendiği bölge.
    - Premium/Discount: fiyatın son aralığın üst yarısında (Premium,
      ICT'ye göre satış için 'pahalı' bölge) mı alt yarısında
      (Discount, alış için 'ucuz' bölge) mı olduğu."""
    recent = hist.tail(lookback + window * 2)
    close, high, low = recent["Close"], recent["High"], recent["Low"]
    price_now = float(close.iloc[-1])

    highs_idx, lows_idx = find_pivots(close, window)

    bsl_level = None
    for idx in reversed(highs_idx):
        level = float(high.iloc[idx])
        if level > price_now:
            bsl_level = level
            break
    if bsl_level is None and highs_idx:
        bsl_level = float(high.iloc[highs_idx[-1]])

    ssl_level = None
    for idx in reversed(lows_idx):
        level = float(low.iloc[idx])
        if level < price_now:
            ssl_level = level
            break
    if ssl_level is None and lows_idx:
        ssl_level = float(low.iloc[lows_idx[-1]])

    range_high = float(high.tail(lookback).max())
    range_low = float(low.tail(lookback).min())
    midpoint = (range_high + range_low) / 2

    if price_now > midpoint:
        zone, v = "Premium", "off"
        zone_note = "fiyat son aralığın Premium (üst) bölgesinde — ICT'ye göre satış için daha 'pahalı' bir bölge."
    else:
        zone, v = "Discount", "on"
        zone_note = "fiyat son aralığın Discount (alt) bölgesinde — ICT'ye göre alış için daha 'ucuz' bir bölge."

    bsl_txt = f"{bsl_level:.2f}" if bsl_level is not None else "N/A"
    ssl_txt = f"{ssl_level:.2f}" if ssl_level is not None else "N/A"
    note = f"BSL {bsl_txt} · SSL {ssl_txt} — {zone_note}"

    levels = {
        "bsl": round(bsl_level, 2) if bsl_level is not None else None,
        "ssl": round(ssl_level, 2) if ssl_level is not None else None,
        "midpoint": round(midpoint, 2),
        "zone": zone,
    }
    return levels, v, note


def compute_divergence(hist: pd.DataFrame, lookback=90, window=5):
    """Fiyat ile RSI arasındaki uyuşmazlığı (divergence) tespit eder.
    Negatif uyuşmazlık: fiyat yeni bir zirve yapar ama RSI daha düşük
    bir zirve yapar (yükseliş gücü tükeniyor, tepe riski).
    Pozitif uyuşmazlık: fiyat yeni bir dip yapar ama RSI daha yüksek
    bir dip yapar (düşüş gücü tükeniyor, dip riski)."""
    recent = hist.tail(lookback + window * 2)
    close = recent["Close"]
    rsi_series = compute_rsi_series(close)

    price_highs, price_lows = find_pivots(close, window)

    bearish = False
    bullish = False

    if len(price_highs) >= 2:
        i1, i2 = price_highs[-2], price_highs[-1]
        if close.iloc[i2] > close.iloc[i1] and rsi_series.iloc[i2] < rsi_series.iloc[i1]:
            bearish = True

    if len(price_lows) >= 2:
        j1, j2 = price_lows[-2], price_lows[-1]
        if close.iloc[j2] < close.iloc[j1] and rsi_series.iloc[j2] > rsi_series.iloc[j1]:
            bullish = True

    if bearish:
        v, note = "off", "Fiyat yeni bir zirve yaptı ama RSI daha düşük bir zirve yaptı — negatif uyuşmazlık, yükseliş momentumu zayıflıyor."
    elif bullish:
        v, note = "on", "Fiyat yeni bir dip yaptı ama RSI daha yüksek bir dip yaptı — pozitif uyuşmazlık, düşüş momentumu zayıflıyor."
    else:
        v, note = "neutral", "Şu an belirgin bir fiyat-RSI uyuşmazlığı yok."

    return v, note, bearish, bullish


def compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return float(hist.iloc[-1]), float(hist.iloc[-2])


def compute_atr(hist: pd.DataFrame, period=14):
    high, low, close = hist["High"], hist["Low"], hist["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_supertrend(hist: pd.DataFrame, period=10, multiplier=3.0):
    """Klasik Supertrend göstergesi (ATR bazlı) — hem trend yönünü teyit
    eder hem de bir 'trailing stop' referans seviyesi verir. Standart
    parametreler: period=10, multiplier=3."""
    high, low, close = hist["High"], hist["Low"], hist["Close"]
    atr = compute_atr(hist, period)
    hl2 = (high + low) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    n = len(hist)

    # ATR ilk `period` barda NaN olur (rolling ortalama dolana kadar).
    # Hesaplamaya İLK GEÇERLİ noktadan başlıyoruz — 0. index'ten
    # başlarsak NaN, karşılaştırmalarda hep "False" ürettiği için
    # sonsuza kadar zincirleme yayılıp tüm seriyi bozuyor (ve JSON'a
    # NaN sızdırıp API'yi çökertiyor).
    valid_idx = atr.first_valid_index()
    if valid_idx is None or n < 2:
        return None, 1, float(close.iloc[-1]) if n else None
    start = hist.index.get_loc(valid_idx)
    if start >= n - 1:
        return None, 1, float(close.iloc[-1])

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    for i in range(start + 1, n):
        if basic_upper.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if basic_lower.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

    trend = [1] * n
    supertrend_line = [0.0] * n
    supertrend_line[start] = float(final_lower.iloc[start])

    for i in range(start + 1, n):
        if trend[i - 1] == 1:
            if close.iloc[i] < final_lower.iloc[i]:
                trend[i] = -1
                supertrend_line[i] = float(final_upper.iloc[i])
            else:
                trend[i] = 1
                supertrend_line[i] = float(final_lower.iloc[i])
        else:
            if close.iloc[i] > final_upper.iloc[i]:
                trend[i] = 1
                supertrend_line[i] = float(final_lower.iloc[i])
            else:
                trend[i] = -1
                supertrend_line[i] = float(final_upper.iloc[i])

    return supertrend_line[-1], trend[-1], float(close.iloc[-1])


def compute_adx(hist: pd.DataFrame, period=14):
    high, low, close = hist["High"], hist["Low"], hist["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=hist.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=hist.index)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return float(adx.iloc[-1]), float(plus_di.iloc[-1]), float(minus_di.iloc[-1])


def compute_bollinger(close: pd.Series, period=20, num_std=2):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return float(upper.iloc[-1]), float(lower.iloc[-1]), float(sma.iloc[-1])


def compute_stochastic(hist: pd.DataFrame, period=14, smooth=3):
    low_min = hist["Low"].rolling(period).min()
    high_max = hist["High"].rolling(period).max()
    k = 100 * (hist["Close"] - low_min) / (high_max - low_min)
    d = k.rolling(smooth).mean()
    return float(k.iloc[-1]), float(d.iloc[-1])


def fetch_seasonal_history(ticker):
    def _do():
        hist = yf.Ticker(ticker).history(period="15y", interval="1mo")
        if hist.empty:
            raise HTTPException(502, f"{ticker} için mevsimsel veri alınamadı")
        return hist
    return cached(f"seasonal:{ticker}", CACHE_TTL_COT, _do)


def compute_seasonality(hist_monthly: pd.DataFrame):
    df = hist_monthly.copy()
    df["month"] = df.index.month
    df["ret"] = df["Close"].pct_change() * 100
    avg_by_month = df.groupby("month")["ret"].mean()
    current_month = datetime.now().month
    current_avg = float(avg_by_month.get(current_month, 0.0))
    return current_avg, current_month


MONTH_NAMES_TR = ["", "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
                   "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]


def technical_agents(hist: pd.DataFrame):
    close = hist["Close"]
    price = float(close.iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else sma50

    # 1) Trend / Momentum
    if price > sma50 > sma200:
        trend_v, trend_note = "on", f"Fiyat ({price:.2f}) 50G ve 200G ortalamanın üstünde — yükseliş trendi."
    elif price < sma50 < sma200:
        trend_v, trend_note = "off", f"Fiyat ({price:.2f}) 50G ve 200G ortalamanın altında — düşüş trendi."
    else:
        trend_v, trend_note = "neutral", f"Fiyat ({price:.2f}) ortalamalara göre karışık sinyal veriyor."

    # 2) RSI
    rsi = compute_rsi(close)
    if rsi >= 70:
        rsi_v, rsi_note = "off", f"RSI {rsi:.0f} — aşırı alım bölgesinde."
    elif rsi <= 30:
        rsi_v, rsi_note = "on", f"RSI {rsi:.0f} — aşırı satım bölgesinde, tepki potansiyeli."
    else:
        rsi_v, rsi_note = "neutral", f"RSI {rsi:.0f} — nötr bölgede."

    # 3) Hacim + fiyat yönü
    vol = hist["Volume"]
    vol_avg20 = float(vol.rolling(20).mean().iloc[-1]) if vol.rolling(20).mean().iloc[-1] == vol.rolling(20).mean().iloc[-1] else 0
    last_vol = float(vol.iloc[-1])
    price_chg = price - float(close.iloc[-2])
    if vol_avg20 and last_vol > vol_avg20 * 1.1 and price_chg > 0:
        vol_v, vol_note = "on", "Yükseliş ortalamanın üstünde hacimle teyit ediliyor."
    elif vol_avg20 and last_vol > vol_avg20 * 1.1 and price_chg < 0:
        vol_v, vol_note = "off", "Düşüş ortalamanın üstünde hacimle teyit ediliyor."
    else:
        vol_v, vol_note = "neutral", "Hacimde belirgin bir teyit sinyali yok."

    # 4) MACD
    macd_hist_now, macd_hist_prev = compute_macd(close)
    if macd_hist_now > 0 and macd_hist_prev <= 0:
        macd_v, macd_note = "on", "MACD sinyal çizgisini yukarı kesti — momentum pozitife dönüyor."
    elif macd_hist_now < 0 and macd_hist_prev >= 0:
        macd_v, macd_note = "off", "MACD sinyal çizgisini aşağı kesti — momentum negatife dönüyor."
    elif macd_hist_now > 0:
        macd_v, macd_note = "on", "MACD pozitif bölgede, yükseliş momentumu sürüyor."
    elif macd_hist_now < 0:
        macd_v, macd_note = "off", "MACD negatif bölgede, düşüş momentumu sürüyor."
    else:
        macd_v, macd_note = "neutral", "MACD sıfıra yakın, net yön yok."

    # 5) Destek / Direnç (son 20 gün)
    recent20 = hist.tail(20)
    high20 = float(recent20["High"].max())
    low20 = float(recent20["Low"].min())
    dist_to_high = (high20 - price) / price * 100
    dist_to_low = (price - low20) / price * 100
    if dist_to_high < 1:
        sr_v, sr_note = "off", f"Fiyat 20 günlük direnç seviyesine ({high20:.2f}) çok yakın — satış baskısı gelebilir."
    elif dist_to_low < 1:
        sr_v, sr_note = "on", f"Fiyat 20 günlük destek seviyesine ({low20:.2f}) çok yakın — tepki alımı gelebilir."
    else:
        sr_v, sr_note = "neutral", f"Fiyat destek ({low20:.2f}) ile direnç ({high20:.2f}) arasında, aralığın ortasında."

    # 6) ADX (Trend Gücü)
    adx_val, plus_di, minus_di = compute_adx(hist)
    if adx_val >= 25 and plus_di > minus_di:
        adx_v, adx_note = "on", f"ADX {adx_val:.0f} — güçlü ve yukarı yönlü bir trend var."
    elif adx_val >= 25 and minus_di > plus_di:
        adx_v, adx_note = "off", f"ADX {adx_val:.0f} — güçlü ve aşağı yönlü bir trend var."
    else:
        adx_v, adx_note = "neutral", f"ADX {adx_val:.0f} — trend zayıf/yatay, sinyaller güvenilir olmayabilir."

    # 7) Bollinger Bantları
    bb_upper, bb_lower, bb_mid = compute_bollinger(close)
    band_width_pct = (bb_upper - bb_lower) / bb_mid * 100
    if price >= bb_upper:
        bb_v, bb_note = "off", f"Fiyat üst Bollinger bandına ({bb_upper:.2f}) çok yakın/üstünde — aşırı alım, geri çekilme riski."
    elif price <= bb_lower:
        bb_v, bb_note = "on", f"Fiyat alt Bollinger bandına ({bb_lower:.2f}) çok yakın/altında — aşırı satım, tepki potansiyeli."
    elif band_width_pct < 5:
        bb_v, bb_note = "neutral", f"Bantlar daralmış (sıkışma, %{band_width_pct:.1f} genişlik) — büyük bir hareket öncesi olabilir."
    else:
        bb_v, bb_note = "neutral", f"Fiyat bantlar arasında, normal aralıkta (%{band_width_pct:.1f} genişlik)."

    # 8) Stochastic Osilatör
    stoch_k, stoch_d = compute_stochastic(hist)
    if stoch_k >= 80:
        stoch_v, stoch_note = "off", f"Stochastic %K {stoch_k:.0f} — aşırı alım bölgesinde."
    elif stoch_k <= 20:
        stoch_v, stoch_note = "on", f"Stochastic %K {stoch_k:.0f} — aşırı satım bölgesinde."
    else:
        stoch_v, stoch_note = "neutral", f"Stochastic %K {stoch_k:.0f} — nötr bölgede."

    return [
        {"name": "Trend / Momentum (SMA50/200)", "verdict": trend_v, "note": trend_note},
        {"name": "RSI (14)", "verdict": rsi_v, "note": rsi_note},
        {"name": "Hacim Teyidi", "verdict": vol_v, "note": vol_note},
        {"name": "MACD (12/26/9)", "verdict": macd_v, "note": macd_note},
        {"name": "Destek / Direnç (20G)", "verdict": sr_v, "note": sr_note},
        {"name": "ADX (Trend Gücü)", "verdict": adx_v, "note": adx_note},
        {"name": "Bollinger Bantları", "verdict": bb_v, "note": bb_note},
        {"name": "Stochastic Osilatör", "verdict": stoch_v, "note": stoch_note},
    ], price, float((close.iloc[-1] / close.iloc[-2] - 1) * 100)


# ------------------------------------------------------------------
# Makro göstergeler (tüm enstrümanlar için ortak)
# ------------------------------------------------------------------
def macro_agents():
    def _do():
        agents = []

        # Dolar Endeksi (ICE DXY)
        try:
            dxy = yf.Ticker("DX-Y.NYB").history(period="6mo")
            price = float(dxy["Close"].iloc[-1])
            sma50 = float(dxy["Close"].rolling(50).mean().iloc[-1])
            if price < sma50 * 0.995:
                v, note = "on", f"DXY ({price:.2f}) 50G ortalamanın altında — dolar zayıflıyor, emtia için destekleyici."
            elif price > sma50 * 1.005:
                v, note = "off", f"DXY ({price:.2f}) 50G ortalamanın üstünde — güçlü dolar emtiaya baskı yapıyor."
            else:
                v, note = "neutral", f"DXY ({price:.2f}) ortalamaya yakın, net yön yok."
        except Exception:
            v, note = "neutral", "DXY verisi şu an alınamadı."
        agents.append({"name": "Dolar Endeksi (DXY)", "verdict": v, "note": note})

        # ABD 10Y Reel Getiri (TIPS)
        try:
            tips = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10")
            tips.columns = ["date", "value"]
            tips["value"] = pd.to_numeric(tips["value"], errors="coerce")
            tips = tips.dropna()
            recent = tips["value"].iloc[-10:].mean()
            older = tips["value"].iloc[-30:-10].mean()
            if recent < older - 0.05:
                v, note = "on", f"Reel getiri geriliyor (%{recent:.2f}) — altın için fırsat maliyeti azalıyor."
            elif recent > older + 0.05:
                v, note = "off", f"Reel getiri yükseliyor (%{recent:.2f}) — altın için fırsat maliyeti artıyor."
            else:
                v, note = "neutral", f"Reel getiri yatay seyrediyor (~%{recent:.2f})."
        except Exception:
            v, note = "neutral", "Reel getiri (TIPS) verisi şu an alınamadı."
        agents.append({"name": "ABD 10Y Reel Getiri (TIPS)", "verdict": v, "note": note})

        # Faiz İndirim Olasılığı (CME 30-Günlük Fed Funds Vadeli İşlemleri)
        # Not: Bu, gerçek vadeli işlem fiyatından ima edilen faiz patikasını
        # kullanır (FedWatch'ın mantığına yakın), ama FOMC toplantı tarihine
        # göre gün-ağırlıklandırma yapmıyor — basitleştirilmiş bir yaklaşım.
        try:
            zq = yf.Ticker("ZQ=F").history(period="5d")
            futures_price = float(zq["Close"].iloc[-1])
            implied_rate = 100 - futures_price

            fed_funds2 = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF")
            fed_funds2.columns = ["date", "value"]
            fed_funds2["value"] = pd.to_numeric(fed_funds2["value"], errors="coerce")
            fed_funds2 = fed_funds2.dropna()
            current_ff2 = fed_funds2["value"].iloc[-1]

            implied_change_bps = (implied_rate - current_ff2) * 100

            if implied_change_bps <= -20:
                v, note = "on", f"Vadeli işlemler yaklaşık tam bir 25bp indirimi fiyatlıyor ({implied_change_bps:.0f} baz puan) — altın için destekleyici bir arka plan."
            elif implied_change_bps <= -5:
                v, note = "on", f"Vadeli işlemler kısmi bir faiz indirimi fiyatlıyor ({implied_change_bps:.0f} baz puan)."
            elif implied_change_bps >= 5:
                v, note = "off", f"Vadeli işlemler faiz artışı ya da 'daha uzun süre yüksek' senaryosu fiyatlıyor (+{implied_change_bps:.0f} baz puan)."
            else:
                v, note = "neutral", f"Vadeli işlemler önemli bir faiz değişikliği fiyatlamıyor ({implied_change_bps:.0f} baz puan)."
        except Exception:
            v, note = "neutral", "Fed funds vadeli işlem verisi şu an alınamadı — ZQ kontrat sembolü kaynakta değişmiş olabilir."
        agents.append({"name": "Faiz İndirim Olasılığı (Fed Funds Futures)", "verdict": v, "note": note})

        # Jeopolitik Risk Endeksi (Caldara & Iacoviello GPR)
        try:
            gpr = pd.read_excel("https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls")
            gpr.columns = [str(c).strip() for c in gpr.columns]
            value_col = "GPRD" if "GPRD" in gpr.columns else gpr.columns[1]
            gpr[value_col] = pd.to_numeric(gpr[value_col], errors="coerce")
            gpr = gpr.dropna(subset=[value_col])
            recent = gpr[value_col].iloc[-5:].mean()
            baseline = gpr[value_col].iloc[-90:].mean()
            if recent > baseline * 1.2:
                v, note = "on", f"GPR endeksi ortalamanın belirgin üstünde ({recent:.0f}) — jeopolitik gerginlik yüksek, güvenli liman talebi artabilir."
            elif recent < baseline * 0.8:
                v, note = "neutral", f"GPR endeksi ortalamanın altında ({recent:.0f}) — jeopolitik gerginlik düşük."
            else:
                v, note = "neutral", f"GPR endeksi normal aralıkta (~{recent:.0f})."
        except Exception:
            v, note = "neutral", "Jeopolitik risk (GPR) verisi şu an alınamadı — kaynak formatı değişmiş olabilir."
        agents.append({"name": "Jeopolitik Risk Endeksi (GPR)", "verdict": v, "note": note})

        # Enflasyon (FRED CPI, anahtar gerektirmez — CPIAUCNS: mevsimsel
        # düzeltilmemiş, resmi BLS manşet YoY rakamıyla eşleşen seri)
        try:
            cpi = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS")
            cpi.columns = ["date", "value"]
            cpi["value"] = pd.to_numeric(cpi["value"], errors="coerce")
            cpi = cpi.dropna()
            latest = cpi["value"].iloc[-1]
            yoy_now = (latest / cpi["value"].iloc[-13] - 1) * 100
            yoy_prev = (cpi["value"].iloc[-2] / cpi["value"].iloc[-14] - 1) * 100
            if yoy_now < yoy_prev - 0.05:
                v, note = "neutral", f"CPI yıllık %{yoy_now:.1f} — enflasyon yavaşlıyor."
            elif yoy_now > yoy_prev + 0.05:
                v, note = "on", f"CPI yıllık %{yoy_now:.1f} — enflasyon hızlanıyor, reel varlık talebi artabilir."
            else:
                v, note = "neutral", f"CPI yıllık %{yoy_now:.1f} — önceki aya yakın."
        except Exception:
            v, note = "neutral", "CPI verisi şu an alınamadı."
        agents.append({"name": "Enflasyon (CPI, YoY)", "verdict": v, "note": note})

        # Altın/Gümüş Oranı (klasik kıymetli metal göreceli değer göstergesi)
        try:
            gold_hist = yf.Ticker("GC=F").history(period="5y")
            silver_hist = yf.Ticker("SI=F").history(period="5y")
            ratio_df = pd.DataFrame({"gold": gold_hist["Close"], "silver": silver_hist["Close"]}).dropna()
            ratio_df["ratio"] = ratio_df["gold"] / ratio_df["silver"]
            current_ratio = float(ratio_df["ratio"].iloc[-1])
            pct_rank = (ratio_df["ratio"] < current_ratio).mean() * 100
            if pct_rank >= 80:
                v, note = "on", f"Altın/Gümüş oranı ({current_ratio:.1f}) 5 yıllık aralığın üst dilimlerinde — gümüş altına göre tarihsel olarak ucuz."
            elif pct_rank <= 20:
                v, note = "off", f"Altın/Gümüş oranı ({current_ratio:.1f}) 5 yıllık aralığın alt dilimlerinde — altın gümüşe göre tarihsel olarak ucuz."
            else:
                v, note = "neutral", f"Altın/Gümüş oranı ({current_ratio:.1f}) normal aralıkta (~%{pct_rank:.0f}. dilim)."
        except Exception:
            v, note = "neutral", "Altın/Gümüş oranı şu an hesaplanamadı."
        agents.append({"name": "Altın/Gümüş Oranı", "verdict": v, "note": note})

        # VIX (CBOE Piyasa Korku Endeksi — çapraz varlık risk iştahı göstergesi)
        try:
            vix = yf.Ticker("^VIX").history(period="1mo")
            vix_now = float(vix["Close"].iloc[-1])
            if vix_now >= 22:
                v, note = "on", f"VIX {vix_now:.1f} — piyasa geneli tedirgin, güvenli liman talebi artabilir."
            elif vix_now <= 14:
                v, note = "off", f"VIX {vix_now:.1f} — piyasa sakin/rahat, güvenli liman talebi zayıf."
            else:
                v, note = "neutral", f"VIX {vix_now:.1f} — normal aralıkta."
        except Exception:
            v, note = "neutral", "VIX verisi şu an alınamadı."
        agents.append({"name": "VIX (Piyasa Korku Endeksi)", "verdict": v, "note": note})

        return agents
    return cached("macro", CACHE_TTL_MACRO, _do)


# ------------------------------------------------------------------
# COT (CFTC Commitments of Traders) — spekülatör pozisyonlama
# ------------------------------------------------------------------
def positioning_agents(cot_hint: str):
    def _do():
        try:
            url = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
            params = {
                "$limit": 5000,
                "$order": "report_date_as_yyyy_mm_dd DESC",
                "$where": f"upper(market_and_exchange_names) like '%{cot_hint}%'",
            }
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            if not data:
                raise ValueError("boş sonuç")
            df = pd.DataFrame(data)
            df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
            df = df.sort_values("report_date_as_yyyy_mm_dd")
            for col in ["noncomm_positions_long_all", "noncomm_positions_short_all",
                        "comm_positions_long_all", "comm_positions_short_all", "open_interest_all"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["net"] = df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]
            df["net_pct_oi"] = df["net"] / df["open_interest_all"]
            df["comm_net"] = df["comm_positions_long_all"] - df["comm_positions_short_all"]
            df["comm_net_pct_oi"] = df["comm_net"] / df["open_interest_all"]

            latest = df.iloc[-1]
            hist_window = df.tail(156)  # ~3 yıl haftalık
            pct_rank = (hist_window["net_pct_oi"] < latest["net_pct_oi"]).mean() * 100

            # Kurumsal (spekülatif) Long/Short oranı — ham yüzdeler
            long_total = float(latest["noncomm_positions_long_all"])
            short_total = float(latest["noncomm_positions_short_all"])
            total_ls = long_total + short_total
            long_pct = round(long_total / total_ls * 100, 1) if total_ls else 50.0
            short_pct = round(100 - long_pct, 1)

            if pct_rank >= 80:
                v1, n1 = "off", f"Net spekülatif pozisyon 3 yıllık aralığın %{pct_rank:.0f}. dilimi — aşırı long, düzeltme riski."
            elif pct_rank <= 20:
                v1, n1 = "on", f"Net spekülatif pozisyon 3 yıllık aralığın %{pct_rank:.0f}. dilimi — aşırı short/düşük, yukarı risk."
            else:
                v1, n1 = "neutral", f"Net spekülatif pozisyon 3 yıllık aralığın %{pct_rank:.0f}. diliminde — aşırılık yok."

            change = latest["net"] - df.iloc[-2]["net"]
            if change > 0:
                v2, n2 = "on", "Son haftada net long pozisyon arttı."
            elif change < 0:
                v2, n2 = "off", "Son haftada net long pozisyon azaldı."
            else:
                v2, n2 = "neutral", "Son haftada pozisyonlamada belirgin değişim yok."

            oi_change = latest["open_interest_all"] - df.iloc[-5]["open_interest_all"]
            if oi_change > 0:
                v3, n3 = "on", "Açık pozisyon (OI) son bir ayda artıyor — piyasaya yeni para giriyor."
            else:
                v3, n3 = "off", "Açık pozisyon (OI) son bir ayda azalıyor — ilgi zayıflıyor."

            # Ticari (Commercial/Hedger) pozisyon — spekülatörlerin genelde
            # tersi yönde hareket eder, kendi tarihine göre aşırı uçlar
            # kontraryan bir sinyal sayılır ("akıllı para" yaklaşımı)
            comm_pct_rank = (hist_window["comm_net_pct_oi"] < latest["comm_net_pct_oi"]).mean() * 100
            if comm_pct_rank >= 80:
                v4, n4 = "on", f"Ticari (hedger) net pozisyon 3 yıllık aralığın %{comm_pct_rank:.0f}. dilimi — ticari taraf tarihsel olarak en az short/en çok long, güven verici."
            elif comm_pct_rank <= 20:
                v4, n4 = "off", f"Ticari (hedger) net pozisyon 3 yıllık aralığın %{comm_pct_rank:.0f}. dilimi — ticari taraf tarihsel olarak en çok short, temkinli sinyal."
            else:
                v4, n4 = "neutral", f"Ticari (hedger) net pozisyon 3 yıllık aralığın %{comm_pct_rank:.0f}. diliminde — aşırılık yok."

            # Klasik "COT Index" (Larry Williams min-max yöntemi) — çoğu
            # profesyonel COT servisinin kullandığı endüstri standardı
            # formül, percentile'dan farklı olarak min-max normalizasyonu
            # kullanır: (şimdiki - min) / (max - min) × 100
            min_comm = hist_window["comm_net_pct_oi"].min()
            max_comm = hist_window["comm_net_pct_oi"].max()
            comm_range = max_comm - min_comm
            cot_index = ((latest["comm_net_pct_oi"] - min_comm) / comm_range * 100) if comm_range != 0 else 50.0
            if cot_index >= 80:
                v5, n5 = "on", f"COT Index (Ticari, min-max) {cot_index:.0f} — ticari taraf 3 yıllık aralığın üst ucunda, güven verici."
            elif cot_index <= 20:
                v5, n5 = "off", f"COT Index (Ticari, min-max) {cot_index:.0f} — ticari taraf 3 yıllık aralığın alt ucunda, temkinli sinyal."
            else:
                v5, n5 = "neutral", f"COT Index (Ticari, min-max) {cot_index:.0f} — aralığın ortasında, aşırılık yok."

            agents = [
                {"name": "COT Net Pozisyon (3Y Percentile)", "verdict": v1, "note": n1},
                {"name": "COT Haftalık Değişim", "verdict": v2, "note": n2},
                {"name": "Açık Pozisyon (OI) Trendi", "verdict": v3, "note": n3},
                {"name": "Ticari (Commercial) Pozisyon", "verdict": v4, "note": n4},
                {"name": "COT Index (Min-Max, Klasik)", "verdict": v5, "note": n5},
            ]
            return agents, {"long_pct": long_pct, "short_pct": short_pct}
        except Exception as e:
            note = f"COT verisi alınamadı ({cot_hint}) — CFTC market adı eşleşmemiş olabilir."
            agents = [
                {"name": "COT Net Pozisyon (3Y Percentile)", "verdict": "neutral", "note": note},
                {"name": "COT Haftalık Değişim", "verdict": "neutral", "note": "Veri yok."},
                {"name": "Açık Pozisyon (OI) Trendi", "verdict": "neutral", "note": "Veri yok."},
                {"name": "Ticari (Commercial) Pozisyon", "verdict": "neutral", "note": "Veri yok."},
                {"name": "COT Index (Min-Max, Klasik)", "verdict": "neutral", "note": "Veri yok."},
            ]
            return agents, {"long_pct": None, "short_pct": None}
    return cached(f"cot:{cot_hint}", CACHE_TTL_COT, _do)


def desk_verdict(agents):
    score = sum(1 if a["verdict"] == "on" else -1 if a["verdict"] == "off" else 0 for a in agents)
    return verdict_from_score(score)


def compute_trade_signals(macro, teknik, pozisyon):
    """Day trade ve haftalık trade için mekanik, şeffaf bir AL/SAT/BEKLE
    sinyali üretir. Bu bir backtest edilmiş strateji DEĞİLDİR — sadece
    zaten var olan göstergeleri belirli ağırlıklarla toplayan basit bir
    kural setidir. Yatırım tavsiyesi değildir."""

    def score(agents):
        return sum(1 if a["verdict"] == "on" else -1 if a["verdict"] == "off" else 0 for a in agents)

    # --- Scalp: SADECE gerçekten 1H bazlı 3 gösterge — Day Trade'den
    # (4H) daha kısa vadeli. ---
    scalp_names = ["1H RSI", "1H MACD (12/26/9)", "1H EMA5/13 Kesişimi"]
    scalp_agents = [a for a in teknik if a["name"] in scalp_names]
    scalp_score = score(scalp_agents)

    if scalp_score >= 2:
        scalp_signal = "AL"
    elif scalp_score <= -2:
        scalp_signal = "SAT"
    else:
        scalp_signal = "BEKLE"

    scalp_note = (
        f"3 adet 1H göstergesinin (RSI, MACD, EMA5/13) toplamı: {scalp_score:+d}. "
        f"(Eşik: ≥+2 AL, ≤-2 SAT, arası BEKLE)"
    )

    # --- Day Trade: SADECE gerçekten 4H bazlı 3 gösterge ---
    # (Mevsimsellik, SMA50/200, 20G Destek/Direnç gibi günlük/aylık
    # göstergeler day-trade skorunu sulandırmasın diye dahil edilmiyor.)
    h4_names = [
        "4H RSI", "4H MACD (12/26/9)", "4H Dönüş (Reversal) Sinyali",
        "4H Supertrend", "4H ADX (Trend Gücü)", "4H VWAP",
    ]
    h4_agents = [a for a in teknik if a["name"] in h4_names]
    day_score = score(h4_agents)

    if day_score >= 4:
        day_signal = "AL"
    elif day_score <= -4:
        day_signal = "SAT"
    else:
        day_signal = "BEKLE"

    day_note = (
        f"6 adet 4H göstergesinin (RSI, MACD, Dönüş, Supertrend, ADX, VWAP) toplamı: {day_score:+d}. "
        f"(Eşik: ≥+4 AL, ≤-4 SAT, arası BEKLE)"
    )

    # --- Haftalık Trade: Macro + Pozisyonlama + Haftalık Mum Trendi ---
    macro_score = score(macro)
    pozisyon_score = score(pozisyon)
    weekly_trend_agent = next((a for a in teknik if a["name"] == "Haftalık Mum Trendi"), None)
    weekly_trend_score = 0
    if weekly_trend_agent:
        if weekly_trend_agent["verdict"] == "on":
            weekly_trend_score = 2
        elif weekly_trend_agent["verdict"] == "off":
            weekly_trend_score = -2
    weekly_score = macro_score + pozisyon_score + weekly_trend_score

    if weekly_score >= 5:
        weekly_signal = "AL"
    elif weekly_score <= -5:
        weekly_signal = "SAT"
    else:
        weekly_signal = "BEKLE"

    weekly_note = (
        f"Macro {macro_score:+d}, Pozisyonlama {pozisyon_score:+d}, Haftalık Mum Trendi {weekly_trend_score:+d} "
        f"→ toplam {weekly_score:+d}. (Eşik: ≥+5 AL, ≤-5 SAT, arası BEKLE)"
    )

    return {
        "scalp": {"signal": scalp_signal, "score": scalp_score, "note": scalp_note},
        "day": {"signal": day_signal, "score": day_score, "note": day_note},
        "weekly": {"signal": weekly_signal, "score": weekly_score, "note": weekly_note},
    }


@app.get("/api/instrument/{key}")
def get_instrument(key: str):
    if key not in INSTRUMENTS:
        raise HTTPException(404, "Bilinmeyen enstrüman")
    cfg = INSTRUMENTS[key]

    hist = fetch_price_history(cfg["yf"])
    teknik, price, change_pct = technical_agents(hist)
    macro = macro_agents()
    pozisyon, cot_ratio = positioning_agents(cfg["cot_hint"])

    # Mevsimsellik (15 yıllık aylık ortalama getiri, ayrı bir kaynak
    # gerektirdiği için hataya dayanıklı şekilde ayrıca ekleniyor)
    try:
        seasonal_hist = fetch_seasonal_history(cfg["yf"])
        seasonal_avg, seasonal_month = compute_seasonality(seasonal_hist)
        month_name = MONTH_NAMES_TR[seasonal_month]
        if seasonal_avg > 1:
            seasonal_v = "on"
            seasonal_note = f"{month_name} ayı, son 15 yılda ortalama %{seasonal_avg:.1f} pozitif getiri gösteriyor — mevsimsel eğilim yukarı yönlü."
        elif seasonal_avg < -1:
            seasonal_v = "off"
            seasonal_note = f"{month_name} ayı, son 15 yılda ortalama %{abs(seasonal_avg):.1f} negatif getiri gösteriyor — mevsimsel eğilim aşağı yönlü (örn. hasat baskısı olabilir)."
        else:
            seasonal_v = "neutral"
            seasonal_note = f"{month_name} ayı için belirgin bir mevsimsel eğilim yok (~%{seasonal_avg:.1f})."
    except Exception:
        seasonal_v, seasonal_note = "neutral", "Mevsimsellik verisi şu an hesaplanamadı."
    teknik.append({"name": "Mevsimsellik (15Y Ortalama)", "verdict": seasonal_v, "note": seasonal_note})

    # Haftalık Mum Trendi — zaten çekilen günlük veriden (hist) resample
    # edilir, yeni bir API çağrısı gerekmez. Haftalık Trade sinyaline
    # gerçek bir fiyat/teknik teyidi katmak için eklendi.
    try:
        weekly_trend_v, weekly_trend_note = compute_weekly_trend(hist)
    except Exception:
        weekly_trend_v, weekly_trend_note = "neutral", "Haftalık trend hesaplanamadı."
    teknik.append({"name": "Haftalık Mum Trendi", "verdict": weekly_trend_v, "note": weekly_trend_note})

    # Çoklu Zaman Dilimi RSI Uyarısı (TEPE/DİP) — günlük VE haftalık RSI
    # aynı anda aşırı uçta olduğunda, tek zaman diliminin yanılabildiği
    # durumlara karşı ekstra güçlü bir uyarı üretir. Bu, normal
    # AL/SAT/BEKLE puanlamasının DIŞINDA, bağımsız bir alarm bayrağıdır.
    tepe_reasons = []
    dip_reasons = []

    try:
        daily_rsi_val = compute_rsi(hist["Close"])
        weekly_rsi_val = compute_weekly_rsi(hist)

        if weekly_rsi_val is not None:
            if daily_rsi_val > 85 and weekly_rsi_val > 85:
                tepe_reasons.append(f"Günlük RSI ({daily_rsi_val:.0f}) ve Haftalık RSI ({weekly_rsi_val:.0f}) aynı anda 85 üstünde")
                mtf_v = "off"
            elif daily_rsi_val < 15 and weekly_rsi_val < 15:
                dip_reasons.append(f"Günlük RSI ({daily_rsi_val:.0f}) ve Haftalık RSI ({weekly_rsi_val:.0f}) aynı anda 15 altında")
                mtf_v = "on"
            else:
                mtf_v = "neutral"
            mtf_note = f"Günlük RSI {daily_rsi_val:.0f}, Haftalık RSI {weekly_rsi_val:.0f}."
        else:
            mtf_v, mtf_note = "neutral", "Haftalık RSI için yeterli veri yok."
    except Exception:
        mtf_v, mtf_note = "neutral", "Çoklu zaman dilimi RSI verisi şu an hesaplanamadı."

    teknik.append({"name": "Çoklu Zaman Dilimi RSI Uyarısı", "verdict": mtf_v, "note": mtf_note})

    # RSI Uyuşmazlığı (Divergence) — mutlak RSI seviyesinden bağımsız,
    # fiyat ile RSI'ın YÖN olarak birbirini teyit etmediği durumları
    # yakalar. Erken bir tepe/dip uyarısı olarak TEPE/DİP banner'ına
    # da katkı sağlar.
    try:
        div_v, div_note, bearish_div, bullish_div = compute_divergence(hist)
    except Exception:
        div_v, div_note, bearish_div, bullish_div = "neutral", "Uyuşmazlık verisi şu an hesaplanamadı.", False, False

    teknik.append({"name": "RSI Uyuşmazlığı (Divergence)", "verdict": div_v, "note": div_note})

    # ICT Likidite Seviyeleri (BSL/SSL/Premium-Discount) — zaten
    # çekilen günlük veriden hesaplanır, yeni bir API çağrısı gerekmez.
    try:
        liquidity_levels, liq_v, liq_note = compute_liquidity_levels(hist)
    except Exception:
        liquidity_levels, liq_v, liq_note = None, "neutral", "Likidite seviyeleri şu an hesaplanamadı."
    teknik.append({"name": "ICT Likidite Seviyeleri (BSL/SSL)", "verdict": liq_v, "note": liq_note})

    if bearish_div:
        tepe_reasons.append("Fiyat yeni zirve yaptı ama RSI daha düşük zirve yaptı (negatif uyuşmazlık)")
    if bullish_div:
        dip_reasons.append("Fiyat yeni dip yaptı ama RSI daha yüksek dip yaptı (pozitif uyuşmazlık)")

    peak_dip_type = None
    peak_dip_note = None
    if tepe_reasons:
        peak_dip_type = "TEPE"
        peak_dip_note = ". ".join(tepe_reasons) + "."
    elif dip_reasons:
        peak_dip_type = "DIP"
        peak_dip_note = ". ".join(dip_reasons) + "."

    # Scalp göstergeleri (1H RSI, MACD, EMA kesişimi) — Day Trade'in
    # (4H) daha kısa vadeli versiyonu. 4H verisiyle aynı ham 1H veriyi
    # paylaşır, ekstra bir API çağrısı gerektirmez.
    try:
        hourly = fetch_1h_history(cfg["yf"])
        scalp_cards = compute_scalp_signals(hourly)
    except Exception:
        scalp_cards = [
            {"name": "1H RSI", "verdict": "neutral", "note": "1H verisi şu an alınamadı."},
            {"name": "1H MACD (12/26/9)", "verdict": "neutral", "note": "1H verisi şu an alınamadı."},
            {"name": "1H EMA5/13 Kesişimi", "verdict": "neutral", "note": "1H verisi şu an alınamadı."},
        ]
        hourly = None
    teknik.extend(scalp_cards)

    # 4H göstergeleri (RSI, MACD, Dönüş sinyali) — ayrı bir veri çekimi
    # (1H veriden resample) gerektirdiği için hataya dayanıklı şekilde
    # ekleniyor. Bu üçü, Day Trade sinyalinin TEK dayanağı olacak
    # şekilde ayrı kartlara çıkarıldı (günlük/aylık göstergeler
    # day-trade skorunu sulandırmasın diye).
    try:
        h4 = fetch_4h_history(cfg["yf"])
        reversal_v, reversal_note = compute_4h_reversal(h4)

        h4_rsi_val = compute_rsi(h4["Close"])
        if h4_rsi_val >= 70:
            h4_rsi_v, h4_rsi_note = "off", f"4H RSI {h4_rsi_val:.0f} — aşırı alım bölgesinde."
        elif h4_rsi_val <= 30:
            h4_rsi_v, h4_rsi_note = "on", f"4H RSI {h4_rsi_val:.0f} — aşırı satım bölgesinde."
        else:
            h4_rsi_v, h4_rsi_note = "neutral", f"4H RSI {h4_rsi_val:.0f} — nötr bölgede."

        h4_macd_now, h4_macd_prev = compute_macd(h4["Close"])
        if h4_macd_now > 0 and h4_macd_prev <= 0:
            h4_macd_v, h4_macd_note = "on", "4H MACD sinyal çizgisini yukarı kesti — kısa vadeli momentum pozitife dönüyor."
        elif h4_macd_now < 0 and h4_macd_prev >= 0:
            h4_macd_v, h4_macd_note = "off", "4H MACD sinyal çizgisini aşağı kesti — kısa vadeli momentum negatife dönüyor."
        elif h4_macd_now > 0:
            h4_macd_v, h4_macd_note = "on", "4H MACD pozitif bölgede."
        elif h4_macd_now < 0:
            h4_macd_v, h4_macd_note = "off", "4H MACD negatif bölgede."
        else:
            h4_macd_v, h4_macd_note = "neutral", "4H MACD sıfıra yakın."

        st_level, st_trend, st_price = compute_supertrend(h4)
        if st_level is None:
            st_v, st_note = "neutral", "4H Supertrend için henüz yeterli veri yok."
        elif st_trend == 1:
            st_v = "on"
            st_note = f"4H Supertrend yukarı yönlü — fiyat ({st_price:.2f}) çizginin ({st_level:.2f}) üstünde. Trailing stop: {st_level:.2f}."
        else:
            st_v = "off"
            st_note = f"4H Supertrend aşağı yönlü — fiyat ({st_price:.2f}) çizginin ({st_level:.2f}) altında. Trailing stop: {st_level:.2f}."

        h4_adx_val, h4_plus_di, h4_minus_di = compute_adx(h4)
        if h4_adx_val >= 25 and h4_plus_di > h4_minus_di:
            h4_adx_v, h4_adx_note = "on", f"4H ADX {h4_adx_val:.0f} — güçlü ve yukarı yönlü bir gün içi trend var."
        elif h4_adx_val >= 25 and h4_minus_di > h4_plus_di:
            h4_adx_v, h4_adx_note = "off", f"4H ADX {h4_adx_val:.0f} — güçlü ve aşağı yönlü bir gün içi trend var."
        else:
            h4_adx_v, h4_adx_note = "neutral", f"4H ADX {h4_adx_val:.0f} — gün içi trend zayıf/yatay, sinyaller güvenilir olmayabilir."

        vwap_window = h4.tail(18)  # ~son 3 gün (4H×6/gün)
        typical_price = (vwap_window["High"] + vwap_window["Low"] + vwap_window["Close"]) / 3
        vwap_val = float((typical_price * vwap_window["Volume"]).sum() / vwap_window["Volume"].sum())
        vwap_price_now = float(vwap_window["Close"].iloc[-1])
        if vwap_price_now > vwap_val * 1.001:
            vwap_v, vwap_note = "on", f"Fiyat ({vwap_price_now:.2f}) 3 günlük VWAP'ın ({vwap_val:.2f}) üstünde — alıcılar ortalama maliyetin üzerinde."
        elif vwap_price_now < vwap_val * 0.999:
            vwap_v, vwap_note = "off", f"Fiyat ({vwap_price_now:.2f}) 3 günlük VWAP'ın ({vwap_val:.2f}) altında — satıcılar baskın."
        else:
            vwap_v, vwap_note = "neutral", f"Fiyat ({vwap_price_now:.2f}) VWAP'a ({vwap_val:.2f}) çok yakın, net bir yön yok."
    except Exception:
        reversal_v, reversal_note = "neutral", "4H dönüş verisi şu an alınamadı."
        h4_rsi_v, h4_rsi_note = "neutral", "4H RSI verisi şu an alınamadı."
        h4_macd_v, h4_macd_note = "neutral", "4H MACD verisi şu an alınamadı."
        st_v, st_note, st_level = "neutral", "4H Supertrend verisi şu an alınamadı.", None
        h4_adx_v, h4_adx_note = "neutral", "4H ADX verisi şu an alınamadı."
        vwap_v, vwap_note = "neutral", "4H VWAP verisi şu an alınamadı."
        h4 = None

    teknik.append({"name": "4H RSI", "verdict": h4_rsi_v, "note": h4_rsi_note})
    teknik.append({"name": "4H MACD (12/26/9)", "verdict": h4_macd_v, "note": h4_macd_note})
    teknik.append({"name": "4H Dönüş (Reversal) Sinyali", "verdict": reversal_v, "note": reversal_note})
    teknik.append({"name": "4H Supertrend", "verdict": st_v, "note": st_note})
    teknik.append({"name": "4H ADX (Trend Gücü)", "verdict": h4_adx_v, "note": h4_adx_note})
    teknik.append({"name": "4H VWAP", "verdict": vwap_v, "note": vwap_note})

    # Scalp (1H) için, Day Trade (4H) için ve Haftalık Trade için
    # destek/direnç seviyeleri — sinyal kutularına somut fiyat referansı
    # eklemek için.
    scalp_levels = None
    if hourly is not None and len(hourly) >= 10:
        try:
            recent_1h = hourly.tail(24)  # ~son 1 gün (24 saat)
            scalp_levels = {
                "resistance": round(float(recent_1h["High"].max()), 2),
                "support": round(float(recent_1h["Low"].min()), 2),
            }
        except Exception:
            scalp_levels = None

    day_levels = None
    if h4 is not None and len(h4) >= 10:
        try:
            recent_h4 = h4.tail(30)  # ~son 5 gün (4H×6/gün)
            day_levels = {
                "resistance": round(float(recent_h4["High"].max()), 2),
                "support": round(float(recent_h4["Low"].min()), 2),
            }
            if st_level is not None:
                day_levels["supertrend"] = round(float(st_level), 2)
        except Exception:
            day_levels = None

    weekly_levels = None
    try:
        weekly_ohlc = hist.resample("W").agg({"High": "max", "Low": "min"}).dropna().tail(10)
        weekly_levels = {
            "resistance": round(float(weekly_ohlc["High"].max()), 2),
            "support": round(float(weekly_ohlc["Low"].min()), 2),
        }
    except Exception:
        weekly_levels = None

    mv, tv, pv = desk_verdict(macro), desk_verdict(teknik), desk_verdict(pozisyon)
    bias_map = {"on": "RISK-ON", "off": "RISK-OFF", "neutral": "NEUTRAL"}

    trade_signals = compute_trade_signals(macro, teknik, pozisyon)
    trade_signals["scalp"]["levels"] = scalp_levels
    trade_signals["day"]["levels"] = day_levels
    trade_signals["weekly"]["levels"] = weekly_levels

    note = (
        f"Macro desk {bias_map[mv]}, Teknik desk {bias_map[tv]}, "
        f"Pozisyonlama desk {bias_map[pv]} sinyali veriyor. "
        "Bu otomatik bir sentezdir, yatırım tavsiyesi değildir."
    )

    # Her desk'in dayandığı verinin gerçekte ne zaman çekildiğini
    # (saniye cinsinden yaş) hesapla — kartların ne kadar taze olduğunu
    # frontend'de göstermek için.
    now = time.time()

    def age_seconds(cache_key):
        ts = cache_timestamp(cache_key)
        return round(now - ts) if ts is not None else None

    freshness = {
        "macro_age_sec": age_seconds("macro"),
        "teknik_age_sec": age_seconds(f"hist:{cfg['yf']}"),
        "pozisyon_age_sec": age_seconds(f"cot:{cfg['cot_hint']}"),
    }

    return sanitize_json({
        "label": cfg["label"],
        "price": round(price, 2),
        "changePct": round(change_pct, 2),
        "macro": macro,
        "teknik": teknik,
        "pozisyon": pozisyon,
        "cotRatio": cot_ratio,
        "bias": {"htf": bias_map[mv], "mtf": bias_map[tv], "ltf": bias_map[pv]},
        "note": note,
        "freshness": freshness,
        "tradeSignals": trade_signals,
        "peakDipAlert": {"type": peak_dip_type, "note": peak_dip_note},
        "liquidityLevels": liquidity_levels,
    })


@app.get("/api/all")
def get_all():
    out = {}
    for key, cfg in INSTRUMENTS.items():
        try:
            hist = fetch_price_history(cfg["yf"])
            price = float(hist["Close"].iloc[-1])
            change_pct = float((hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1) * 100)
            out[key] = {"label": cfg["label"], "price": round(price, 2), "changePct": round(change_pct, 2)}
        except Exception:
            out[key] = {"label": cfg["label"], "price": None, "changePct": None}
    return sanitize_json(out)


@app.get("/api/correlation")
def get_correlation():
    try:
        closes = {}
        for key, cfg in INSTRUMENTS.items():
            hist = fetch_price_history(cfg["yf"])
            short_label = cfg["label"].split(" ")[0]
            closes[short_label] = hist["Close"].tail(90)

        try:
            dxy_hist = yf.Ticker("DX-Y.NYB").history(period="6mo")
            closes["DXY"] = dxy_hist["Close"].tail(90)
        except Exception:
            pass

        df = pd.DataFrame(closes).dropna()
        returns = df.pct_change().dropna()
        corr = returns.corr().round(2)

        labels = list(corr.columns)
        matrix = [[None if pd.isna(v) else float(v) for v in row] for row in corr.values]
        return sanitize_json({"labels": labels, "matrix": matrix})
    except Exception as e:
        raise HTTPException(502, f"Korelasyon hesaplanamadı: {str(e)}")
