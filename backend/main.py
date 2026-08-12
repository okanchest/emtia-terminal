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
import requests
import time
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
CACHE_TTL_PRICE = 5 * 60       # 5 dakika
CACHE_TTL_MACRO = 60 * 60      # 1 saat
CACHE_TTL_COT = 12 * 60 * 60   # 12 saat (CFTC verisi haftalık yayınlanır)


def cached(key, ttl, fn):
    now = time.time()
    if key in _cache and now - _cache[key][0] < ttl:
        return _cache[key][1]
    val = fn()
    _cache[key] = (now, val)
    return val


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


def compute_rsi(close: pd.Series, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


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

    return [
        {"name": "Trend / Momentum (SMA50/200)", "verdict": trend_v, "note": trend_note},
        {"name": "RSI (14)", "verdict": rsi_v, "note": rsi_note},
        {"name": "Hacim Teyidi", "verdict": vol_v, "note": vol_note},
        {"name": "MACD (12/26/9)", "verdict": macd_v, "note": macd_note},
        {"name": "Destek / Direnç (20G)", "verdict": sr_v, "note": sr_note},
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

        # Faiz İndirim Beklentisi (2Y getiri vs Fed Funds — piyasanın
        # fiyatladığı faiz patikasının basit ama güvenilir bir proxy'si)
        try:
            fed_funds = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF")
            fed_funds.columns = ["date", "value"]
            fed_funds["value"] = pd.to_numeric(fed_funds["value"], errors="coerce")
            fed_funds = fed_funds.dropna()
            current_ff = fed_funds["value"].iloc[-1]

            dgs2 = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2")
            dgs2.columns = ["date", "value"]
            dgs2["value"] = pd.to_numeric(dgs2["value"], errors="coerce")
            dgs2 = dgs2.dropna()
            current_2y = dgs2["value"].iloc[-1]

            spread = current_2y - current_ff  # negatif = piyasa indirim fiyatlıyor
            if spread < -0.30:
                v, note = "on", f"2Y getiri Fed faizinin {abs(spread):.2f} puan altında — piyasa belirgin faiz indirimi fiyatlıyor, değerli metaller için destekleyici."
            elif spread < -0.10:
                v, note = "neutral", f"2Y getiri Fed faizinin {abs(spread):.2f} puan altında — hafif indirim beklentisi var."
            elif spread > 0.10:
                v, note = "off", f"2Y getiri Fed faizinin {spread:.2f} puan üstünde — piyasa faiz artışı ya da 'daha uzun süre yüksek' senaryosu fiyatlıyor."
            else:
                v, note = "neutral", f"2Y getiri Fed faizine yakın (%{current_2y:.2f} vs %{current_ff:.2f}) — net bir beklenti yok."
        except Exception:
            v, note = "neutral", "Faiz beklentisi verisi şu an alınamadı."
        agents.append({"name": "Faiz İndirim Beklentisi (Fed vs 2Y)", "verdict": v, "note": note})

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
            for col in ["noncomm_positions_long_all", "noncomm_positions_short_all", "open_interest_all"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["net"] = df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]
            df["net_pct_oi"] = df["net"] / df["open_interest_all"]

            latest = df.iloc[-1]
            hist_window = df.tail(156)  # ~3 yıl haftalık
            pct_rank = (hist_window["net_pct_oi"] < latest["net_pct_oi"]).mean() * 100

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

            return [
                {"name": "COT Net Pozisyon (3Y Percentile)", "verdict": v1, "note": n1},
                {"name": "COT Haftalık Değişim", "verdict": v2, "note": n2},
                {"name": "Açık Pozisyon (OI) Trendi", "verdict": v3, "note": n3},
            ]
        except Exception as e:
            note = f"COT verisi alınamadı ({cot_hint}) — CFTC market adı eşleşmemiş olabilir."
            return [
                {"name": "COT Net Pozisyon (3Y Percentile)", "verdict": "neutral", "note": note},
                {"name": "COT Haftalık Değişim", "verdict": "neutral", "note": "Veri yok."},
                {"name": "Açık Pozisyon (OI) Trendi", "verdict": "neutral", "note": "Veri yok."},
            ]
    return cached(f"cot:{cot_hint}", CACHE_TTL_COT, _do)


def desk_verdict(agents):
    score = sum(1 if a["verdict"] == "on" else -1 if a["verdict"] == "off" else 0 for a in agents)
    return verdict_from_score(score)


@app.get("/api/instrument/{key}")
def get_instrument(key: str):
    if key not in INSTRUMENTS:
        raise HTTPException(404, "Bilinmeyen enstrüman")
    cfg = INSTRUMENTS[key]

    hist = fetch_price_history(cfg["yf"])
    teknik, price, change_pct = technical_agents(hist)
    macro = macro_agents()
    pozisyon = positioning_agents(cfg["cot_hint"])

    mv, tv, pv = desk_verdict(macro), desk_verdict(teknik), desk_verdict(pozisyon)
    bias_map = {"on": "RISK-ON", "off": "RISK-OFF", "neutral": "NEUTRAL"}

    note = (
        f"Macro desk {bias_map[mv]}, Teknik desk {bias_map[tv]}, "
        f"Pozisyonlama desk {bias_map[pv]} sinyali veriyor. "
        "Bu otomatik bir sentezdir, yatırım tavsiyesi değildir."
    )

    return {
        "label": cfg["label"],
        "price": round(price, 2),
        "changePct": round(change_pct, 2),
        "macro": macro,
        "teknik": teknik,
        "pozisyon": pozisyon,
        "bias": {"htf": bias_map[mv], "mtf": bias_map[tv], "ltf": bias_map[pv]},
        "note": note,
    }


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
    return out
