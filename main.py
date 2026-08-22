"""
AuraPlay Scanner — single-file build.

Everything (config, Binance client, indicators, signal engine, Telegram
alerts, FastAPI app, and the dashboard UI) lives in this one file so it's
easy to upload straight from a phone: just this file + requirements.txt.

Binance USDT-M Futures scanner: cycles through every perpetual pair,
computes RSI, EMA trend, SuperTrend, VWAP, OBV, liquidity sweeps, FVGs,
Break-of-Structure (BOS), Long/Short account ratio, and Open Interest —
serves it as a live dashboard and pushes Buy/Sell/BOS alerts to Telegram.
"""
import os
import time
import asyncio
import logging
from typing import Optional

import httpx
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("aura-scanner")

# ======================================================================
# CONFIG — override any of these with environment variables on Render/Railway
# ======================================================================
# ---- Binance Futures (USDT-M) public REST base ----
BINANCE_FAPI_BASE = "https://fapi.binance.com"

# ---- Telegram ----
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # user or channel/group id

# ---- Universe ----
QUOTE_ASSET = os.getenv("QUOTE_ASSET", "USDT")          # only USDT-margined pairs
ONLY_PERPETUAL = True
EXCLUDE_SYMBOLS = set(os.getenv("EXCLUDE_SYMBOLS", "").split(",")) if os.getenv("EXCLUDE_SYMBOLS") else set()

# ---- Timeframes used for indicator calc (mirrors the 4H/1H setup you trade) ----
SIGNAL_INTERVAL = os.getenv("SIGNAL_INTERVAL", "1h")     # candle interval for signal calc
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "150"))        # candles fetched per symbol
LS_RATIO_PERIOD = os.getenv("LS_RATIO_PERIOD", "1h")      # period for long/short ratio endpoints

# ---- Indicator params ----
EMA_FAST = int(os.getenv("EMA_FAST", "20"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "50"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "6"))            # matches your screenshot (RSI 6)
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "88"))  # RSI(6) pins high during real trend moves — 80 was excluding genuine breakouts
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "12"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "10"))
SUPERTREND_MULT = float(os.getenv("SUPERTREND_MULT", "3.0"))
# Stop-loss / take-profit are calculated as multiples of ATR (volatility-based,
# not a fixed %) — a common, signal-agnostic way to set risk levels. Default
# gives roughly a 1:2 risk:reward (1.5x ATR risk, 3x ATR target).
# Stop-loss / take-profit are primarily structure-based (the swept liquidity
# level / nearest swing point), not blind ATR multiples — ATR is used as a
# small buffer beyond structure, and as the fallback when no structural
# level is available. This ties risk levels to the actual reason the signal
# fired instead of an arbitrary distance from entry.
ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", "1.5"))       # fallback stop distance in ATR (only used when no structural stop level exists)
ATR_SL_BUFFER_MULT = float(os.getenv("ATR_SL_BUFFER_MULT", "0.25"))  # buffer beyond a structural stop level
MIN_STRUCTURE_RR = float(os.getenv("MIN_STRUCTURE_RR", "1.2"))  # min reward:risk to accept a nearby structural target
DEFAULT_RR = float(os.getenv("DEFAULT_RR", "2.0"))  # target RR used when no structural target clears MIN_STRUCTURE_RR — applied to the ACTUAL risk distance, not a fixed ATR amount, so the RR guarantee always holds
TP2_RR_MULT = float(os.getenv("TP2_RR_MULT", "1.8"))  # TP2 fallback = TP1 distance × this, when no second structural level exists
MAX_STOP_ATR_MULT = float(os.getenv("MAX_STOP_ATR_MULT", "2.5"))  # if the nearest structural stop level is farther than this many ATRs from entry, it's stale/irrelevant — use the ATR fallback stop instead
REQUIRE_BOS_CONFLUENCE = os.getenv("REQUIRE_BOS_CONFLUENCE", "true").lower() == "true"  # require a same-direction BOS on the current candle as part of entry — fewer, higher-conviction signals
PIVOT_LOOKBACK = int(os.getenv("PIVOT_LOOKBACK", "5"))    # bars each side for swing high/low
FVG_MIN_GAP_PCT = float(os.getenv("FVG_MIN_GAP_PCT", "0.05"))  # % gap to count as a real FVG

# ---- Scanner scheduling ----
# All symbols are cycled through in batches so we stay well under Binance's
# futures REST weight limit (2400/min). ~300 symbols, 5 calls each => spread
# across a few minutes rather than hammered in one burst.
SYMBOLS_PER_BATCH = int(os.getenv("SYMBOLS_PER_BATCH", "8"))
BATCH_INTERVAL_SECONDS = int(os.getenv("BATCH_INTERVAL_SECONDS", "20"))
FULL_UNIVERSE_REFRESH_SECONDS = int(os.getenv("FULL_UNIVERSE_REFRESH_SECONDS", "3600"))

# ---- Server ----
PORT = int(os.getenv("PORT", "8000"))

# ======================================================================
# BINANCE FUTURES CLIENT (public REST — no API key needed)
# ======================================================================

_client: Optional[httpx.AsyncClient] = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(base_url=BINANCE_FAPI_BASE, timeout=15.0)
    return _client


async def _get(path: str, params: dict | None = None):
    client = get_client()
    for attempt in range(3):
        try:
            resp = await client.get(path, params=params)
            if resp.status_code == 429 or resp.status_code == 418:
                # rate limited — back off
                wait = 2 ** (attempt + 1)
                log.warning(f"Rate limited on {path}, backing off {wait}s")
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            log.warning(f"Request failed ({path}): {e}, attempt {attempt + 1}/3")
            await asyncio.sleep(1.5 * (attempt + 1))
    return None


async def get_all_perpetual_symbols() -> list[str]:
    """All actively trading USDT-margined perpetual futures symbols."""
    data = await _get("/fapi/v1/exchangeInfo")
    if not data:
        return []
    symbols = []
    for s in data.get("symbols", []):
        if (
            s.get("status") == "TRADING"
            and s.get("quoteAsset") == QUOTE_ASSET
            and s.get("contractType") == "PERPETUAL"
            and s.get("symbol") not in EXCLUDE_SYMBOLS
        ):
            symbols.append(s["symbol"])
    return sorted(symbols)


async def get_24hr_tickers() -> dict:
    """Bulk 24hr stats for ALL symbols in a single call. Keyed by symbol."""
    data = await _get("/fapi/v1/ticker/24hr")
    if not data:
        return {}
    return {d["symbol"]: d for d in data}


async def get_klines(symbol: str, interval: str = None, limit: int = None):
    interval = interval or SIGNAL_INTERVAL
    limit = limit or KLINE_LIMIT
    data = await _get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    return data or []


async def get_open_interest(symbol: str):
    data = await _get("/fapi/v1/openInterest", {"symbol": symbol})
    return data


async def get_long_short_account_ratio(symbol: str, period: str = None):
    period = period or LS_RATIO_PERIOD
    data = await _get(
        "/futures/data/globalLongShortAccountRatio",
        {"symbol": symbol, "period": period, "limit": 1},
    )
    return data[0] if data else None


async def get_top_trader_position_ratio(symbol: str, period: str = None):
    period = period or LS_RATIO_PERIOD
    data = await _get(
        "/futures/data/topLongShortPositionRatio",
        {"symbol": symbol, "period": period, "limit": 1},
    )
    return data[0] if data else None

# ======================================================================
# TECHNICAL INDICATORS (pandas/numpy — EMA, RSI, ATR, SuperTrend, VWAP, OBV, pivots, FVG)
# ======================================================================
def klines_to_df(klines: list) -> pd.DataFrame:
    """Binance kline rows -> tidy OHLCV DataFrame."""
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(klines, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df[["open_time", "open", "high", "low", "close", "volume"]]


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def supertrend(df: pd.DataFrame, period: int, multiplier: float) -> pd.DataFrame:
    """Returns df with 'st' (line value) and 'st_trend' (1 bullish / -1 bearish)."""
    hl2 = (df["high"] + df["low"]) / 2
    a = atr(df, period)
    upper = hl2 + multiplier * a
    lower = hl2 - multiplier * a

    trend = np.ones(len(df))
    st = np.zeros(len(df))
    final_upper = upper.copy()
    final_lower = lower.copy()

    for i in range(1, len(df)):
        if df["close"].iloc[i - 1] > final_upper.iloc[i - 1]:
            trend[i] = 1
        elif df["close"].iloc[i - 1] < final_lower.iloc[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
            if trend[i] == 1 and lower.iloc[i] < final_lower.iloc[i - 1]:
                final_lower.iloc[i] = final_lower.iloc[i - 1]
            if trend[i] == -1 and upper.iloc[i] > final_upper.iloc[i - 1]:
                final_upper.iloc[i] = final_upper.iloc[i - 1]

        st[i] = final_lower.iloc[i] if trend[i] == 1 else final_upper.iloc[i]

    out = df.copy()
    out["st"] = st
    out["st_trend"] = trend
    return out


def vwap(df: pd.DataFrame) -> pd.Series:
    """Rolling session VWAP over the whole fetched window (resets not modeled
    since we only hold a rolling window of recent candles, not full sessions)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_vp = (typical * df["volume"]).cumsum()
    return cum_vp / cum_vol.replace(0, np.nan)


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def pivot_highs_lows(df: pd.DataFrame, lookback: int) -> tuple[pd.Series, pd.Series]:
    """Boolean series marking swing highs / swing lows using `lookback` bars
    on each side (classic fractal pivot)."""
    highs = df["high"]
    lows = df["low"]
    is_high = pd.Series(False, index=df.index)
    is_low = pd.Series(False, index=df.index)
    for i in range(lookback, len(df) - lookback):
        window_h = highs.iloc[i - lookback: i + lookback + 1]
        window_l = lows.iloc[i - lookback: i + lookback + 1]
        if highs.iloc[i] == window_h.max():
            is_high.iloc[i] = True
        if lows.iloc[i] == window_l.min():
            is_low.iloc[i] = True
    return is_high, is_low


def fair_value_gaps(df: pd.DataFrame, min_gap_pct: float) -> pd.DataFrame:
    """3-candle imbalance (FVG). Bullish FVG: candle[i-2].high < candle[i].low.
    Bearish FVG: candle[i-2].low > candle[i].high. Returns df with
    'bull_fvg' / 'bear_fvg' booleans on the middle candle's index."""
    bull = pd.Series(False, index=df.index)
    bear = pd.Series(False, index=df.index)
    for i in range(2, len(df)):
        c0, c2 = df.iloc[i - 2], df.iloc[i]
        mid_close = df["close"].iloc[i - 1]
        if c2["low"] > c0["high"] and (c2["low"] - c0["high"]) / mid_close * 100 >= min_gap_pct:
            bull.iloc[i - 1] = True
        if c2["high"] < c0["low"] and (c0["low"] - c2["high"]) / mid_close * 100 >= min_gap_pct:
            bear.iloc[i - 1] = True
    out = df.copy()
    out["bull_fvg"] = bull
    out["bear_fvg"] = bear
    return out


def liquidity_sweep(df: pd.DataFrame, is_high: pd.Series, is_low: pd.Series) -> dict:
    """Did the latest candle wick through the most recent swing high/low and
    close back inside it? (classic liquidity grab). Returns the actual swing
    price levels too, since those are the real invalidation points for a
    stop-loss — not just a boolean."""
    last = df.iloc[-1]
    recent_highs = df.loc[is_high, "high"]
    recent_lows = df.loc[is_low, "low"]

    swept_high = False
    swept_low = False
    swept_high_level = None
    swept_low_level = None
    if len(recent_highs) > 0:
        last_swing_high = float(recent_highs.iloc[-1])
        swept_high = bool(last["high"] > last_swing_high and last["close"] < last_swing_high)
        if swept_high:
            swept_high_level = last_swing_high
    if len(recent_lows) > 0:
        last_swing_low = float(recent_lows.iloc[-1])
        swept_low = bool(last["low"] < last_swing_low and last["close"] > last_swing_low)
        if swept_low:
            swept_low_level = last_swing_low

    return {
        "swept_low": swept_low, "swept_high": swept_high,
        "swept_low_level": swept_low_level, "swept_high_level": swept_high_level,
    }


def nearest_structure_level(df: pd.DataFrame, is_high: pd.Series, is_low: pd.Series,
                             price: float, side: str) -> float | None:
    """Nearest real swing level relevant to a trade:
    side='below' -> most recent confirmed swing low under current price (stop reference for a BUY)
    side='above' -> most recent confirmed swing high over current price (stop reference for a SELL)
    Used both for stop placement (when no liquidity sweep triggered the signal)
    and — read from the opposite side — for a take-profit target."""
    if side == "below":
        levels = df.loc[is_low, "low"]
        levels = levels[levels < price]
        return float(levels.iloc[-1]) if len(levels) else None
    else:
        levels = df.loc[is_high, "high"]
        levels = levels[levels > price]
        return float(levels.iloc[-1]) if len(levels) else None


def next_level_beyond(df: pd.DataFrame, is_high: pd.Series, is_low: pd.Series,
                       reference: float, side: str) -> float | None:
    """Nearest structural level that is genuinely further out than `reference`
    (used to pick TP2 as a level beyond TP1 — ordered by price, not by how
    recently the swing formed, so TP2 > TP1 always holds for a BUY, and
    TP2 < TP1 always holds for a SELL)."""
    if side == "above":
        levels = df.loc[is_high, "high"]
        levels = levels[levels > reference]
        return float(levels.min()) if len(levels) else None
    else:
        levels = df.loc[is_low, "low"]
        levels = levels[levels < reference]
        return float(levels.max()) if len(levels) else None


def break_of_structure(df: pd.DataFrame, is_high: pd.Series, is_low: pd.Series) -> str | None:
    """Bullish BOS: close breaks above the last confirmed swing high.
    Bearish BOS: close breaks below the last confirmed swing low.
    Returns 'bullish', 'bearish', or None."""
    last_close = df["close"].iloc[-1]
    recent_highs = df.loc[is_high, "high"]
    recent_lows = df.loc[is_low, "low"]

    bullish = len(recent_highs) > 0 and last_close > recent_highs.iloc[-1] and df["close"].iloc[-2] <= recent_highs.iloc[-1]
    bearish = len(recent_lows) > 0 and last_close < recent_lows.iloc[-1] and df["close"].iloc[-2] >= recent_lows.iloc[-1]

    if bullish:
        return "bullish"
    if bearish:
        return "bearish"
    return None

# ======================================================================
# SIGNAL ENGINE (combines indicators + L/S ratio + OI into BUY/SELL/BOS)
# ======================================================================


class SymbolSnapshot:
    __slots__ = (
        "symbol", "price", "pct_change_24h", "rsi", "ema_fast", "ema_slow",
        "supertrend_trend", "vwap", "atr", "obv", "open_interest", "long_short_account_ratio",
        "long_pct", "short_pct", "signal", "bos", "stop_loss", "take_profit", "take_profit2",
        "stop_ref_below", "stop_ref_above",
        "swept_low", "swept_high", "bull_fvg_recent", "bear_fvg_recent", "updated_at",
    )

    def to_dict(self):
        out = {}
        for k in self.__slots__:
            v = getattr(self, k)
            if isinstance(v, (np.bool_,)):
                v = bool(v)
            elif isinstance(v, np.integer):
                v = int(v)
            elif isinstance(v, np.floating):
                v = float(v)
            out[k] = v
        return out


def _compute_indicators(klines: list) -> dict:
    """CPU-heavy pandas/loop work — deliberately kept synchronous and run via
    asyncio.to_thread so it never blocks the event loop (Render's free tier
    only allocates ~0.1 vCPU, and this loop-heavy code can otherwise starve
    incoming HTTP requests long enough to time out)."""
    df = klines_to_df(klines)
    df["ema_fast"] = ema(df["close"], EMA_FAST)
    df["ema_slow"] = ema(df["close"], EMA_SLOW)
    df["rsi"] = rsi(df["close"], RSI_PERIOD)
    df["vwap"] = vwap(df)
    df["obv"] = obv(df)
    df["atr"] = atr(df, ATR_PERIOD)
    df = supertrend(df, ATR_PERIOD, SUPERTREND_MULT)
    df = fair_value_gaps(df, FVG_MIN_GAP_PCT)
    is_high, is_low = pivot_highs_lows(df, PIVOT_LOOKBACK)

    sweep = liquidity_sweep(df, is_high, is_low)
    bos = break_of_structure(df, is_high, is_low)

    last = df.iloc[-1]
    price = float(last["close"])

    # Stop reference: the level whose break would invalidate the setup.
    # Prefer the actual swept liquidity level (that's *why* the signal
    # fired); if the signal is FVG-based instead, fall back to the nearest
    # unswept swing level on the relevant side.
    stop_ref_below = sweep["swept_low_level"] or nearest_structure_level(df, is_high, is_low, price, "below")
    stop_ref_above = sweep["swept_high_level"] or nearest_structure_level(df, is_high, is_low, price, "above")

    # Target reference: the nearest real structure on the opposite side —
    # a swing high above price (for a long) or swing low below price (for
    # a short) — since that's the next place the market is actually likely
    # to react, rather than an arbitrary distance from entry.
    target_above = nearest_structure_level(df, is_high, is_low, price, "above")
    target_below = nearest_structure_level(df, is_high, is_low, price, "below")

    return {
        "df": df, "is_high": is_high, "is_low": is_low,  # kept for TP2 lookup after TP1 is known — not sent to JSON
        "price": price,
        "rsi": round(float(last["rsi"]), 2),
        "ema_fast": float(last["ema_fast"]),
        "ema_slow": float(last["ema_slow"]),
        "supertrend_trend": int(last["st_trend"]),
        "vwap": float(last["vwap"]) if pd.notna(last["vwap"]) else None,
        "atr": float(last["atr"]) if pd.notna(last["atr"]) else None,
        "obv": float(last["obv"]),
        "swept_low": sweep["swept_low"],
        "swept_high": sweep["swept_high"],
        "stop_ref_below": stop_ref_below,
        "stop_ref_above": stop_ref_above,
        "target_above": target_above,
        "target_below": target_below,
        "bull_fvg_recent": bool(df["bull_fvg"].tail(5).any()),
        "bear_fvg_recent": bool(df["bear_fvg"].tail(5).any()),
        "bos": bos,
    }


async def build_snapshot(symbol: str, ticker_24h: dict | None) -> SymbolSnapshot | None:
    klines = await get_klines(symbol)
    if not klines or len(klines) < PIVOT_LOOKBACK * 2 + 5:
        return None

    computed = await asyncio.to_thread(_compute_indicators, klines)

    # ---- long/short ratio + open interest (Binance futures data endpoints) ----
    ls = await get_long_short_account_ratio(symbol)
    oi = await get_open_interest(symbol)

    snap = SymbolSnapshot()
    snap.symbol = symbol
    snap.price = computed["price"]
    snap.pct_change_24h = float(ticker_24h["priceChangePercent"]) if ticker_24h else None
    snap.rsi = computed["rsi"]
    snap.ema_fast = computed["ema_fast"]
    snap.ema_slow = computed["ema_slow"]
    snap.supertrend_trend = computed["supertrend_trend"]
    snap.vwap = computed["vwap"]
    snap.obv = computed["obv"]
    snap.open_interest = float(oi["openInterest"]) if oi else None
    snap.long_short_account_ratio = float(ls["longShortRatio"]) if ls else None
    snap.long_pct = float(ls["longAccount"]) * 100 if ls else None
    snap.short_pct = float(ls["shortAccount"]) * 100 if ls else None
    snap.swept_low = computed["swept_low"]
    snap.swept_high = computed["swept_high"]
    snap.bull_fvg_recent = computed["bull_fvg_recent"]
    snap.bear_fvg_recent = computed["bear_fvg_recent"]
    snap.bos = computed["bos"]
    snap.atr = computed["atr"]
    snap.stop_ref_below = computed["stop_ref_below"]
    snap.stop_ref_above = computed["stop_ref_above"]
    snap.updated_at = time.time()

    # ---- confluence signal ----
    bullish_trend = snap.supertrend_trend == 1 and snap.ema_fast > snap.ema_slow
    bearish_trend = snap.supertrend_trend == -1 and snap.ema_fast < snap.ema_slow
    above_vwap = snap.vwap is not None and snap.price > snap.vwap
    below_vwap = snap.vwap is not None and snap.price < snap.vwap

    buy = (
        bullish_trend
        and above_vwap
        and snap.rsi < RSI_OVERBOUGHT
        and (computed["swept_low"] or computed["bull_fvg_recent"])
        and (not REQUIRE_BOS_CONFLUENCE or computed["bos"] == "bullish")
    )
    sell = (
        bearish_trend
        and below_vwap
        and snap.rsi > RSI_OVERSOLD
        and (computed["swept_high"] or computed["bear_fvg_recent"])
        and (not REQUIRE_BOS_CONFLUENCE or computed["bos"] == "bearish")
    )

    if buy:
        snap.signal = "BUY"
    elif sell:
        snap.signal = "SELL"
    else:
        snap.signal = "NEUTRAL"

    # ---- Stop-loss / take-profit: structure-based, ATR as buffer/fallback ----
    snap.stop_loss = None
    snap.take_profit = None
    snap.take_profit2 = None
    if snap.atr is not None and snap.signal in ("BUY", "SELL"):
        buffer = snap.atr * ATR_SL_BUFFER_MULT
        if snap.signal == "BUY":
            stop_ref = computed["stop_ref_below"]
            if stop_ref is not None and (snap.price - stop_ref) + buffer > snap.atr * MAX_STOP_ATR_MULT:
                stop_ref = None  # structural level too far away to be a usable stop — treat as stale
            snap.stop_loss = (stop_ref - buffer) if stop_ref is not None else (snap.price - snap.atr * ATR_SL_MULT)
            risk = snap.price - snap.stop_loss

            target = computed["target_above"]
            if target is not None and risk > 0 and (target - snap.price) >= MIN_STRUCTURE_RR * risk:
                snap.take_profit = target
            else:
                snap.take_profit = snap.price + risk * DEFAULT_RR

            target2 = next_level_beyond(computed["df"], computed["is_high"], computed["is_low"],
                                         snap.take_profit, "above")
            fallback_tp2 = snap.price + risk * DEFAULT_RR * TP2_RR_MULT
            # whichever path TP1 took, TP2 must land strictly beyond it —
            # a structural TP1 can sit further out than a fixed-multiple
            # fallback would reach, so guard with max() rather than trusting
            # either path in isolation
            snap.take_profit2 = max(target2 or fallback_tp2, fallback_tp2, snap.take_profit + risk * 0.1)
        else:
            stop_ref = computed["stop_ref_above"]
            if stop_ref is not None and (stop_ref - snap.price) + buffer > snap.atr * MAX_STOP_ATR_MULT:
                stop_ref = None
            snap.stop_loss = (stop_ref + buffer) if stop_ref is not None else (snap.price + snap.atr * ATR_SL_MULT)
            risk = snap.stop_loss - snap.price

            target = computed["target_below"]
            if target is not None and risk > 0 and (snap.price - target) >= MIN_STRUCTURE_RR * risk:
                snap.take_profit = target
            else:
                snap.take_profit = snap.price - risk * DEFAULT_RR

            target2 = next_level_beyond(computed["df"], computed["is_high"], computed["is_low"],
                                         snap.take_profit, "below")
            fallback_tp2 = snap.price - risk * DEFAULT_RR * TP2_RR_MULT
            snap.take_profit2 = min(target2 or fallback_tp2, fallback_tp2, snap.take_profit - risk * 0.1)

    return snap

# ======================================================================
# TELEGRAM ALERTS
# ======================================================================

_client: httpx.AsyncClient | None = None


def _get_client():
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=10.0)
    return _client


async def send_alert(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram not configured — skipping alert: %s", text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        client = _get_client()
        resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            log.warning("Telegram send failed: %s %s", resp.status_code, resp.text)
    except httpx.HTTPError as e:
        log.warning("Telegram send error: %s", e)


def _fmt_price(p: float) -> str:
    return f"{p:.6f}" if p < 1 else f"{p:.4f}"


def format_signal_alert(symbol: str, signal: str, price: float, rsi: float,
                         long_pct: float | None, oi: float | None,
                         stop_loss: float | None = None, take_profit: float | None = None,
                         take_profit2: float | None = None) -> str:
    emoji = "🟢" if signal == "BUY" else "🔴"
    lines = [
        f"{emoji} <b>{signal} — {symbol}</b>",
        f"Entry: {_fmt_price(price)}",
        f"RSI: {rsi}",
    ]
    if stop_loss is not None and take_profit is not None:
        risk = abs(price - stop_loss)
        reward = abs(take_profit - price)
        rr = f"{reward / risk:.2f}" if risk > 0 else "—"
        lines.append(f"Stop Loss: {_fmt_price(stop_loss)}")
        lines.append(f"TP1: {_fmt_price(take_profit)} (1:{rr})")
        if take_profit2 is not None:
            reward2 = abs(take_profit2 - price)
            rr2 = f"{reward2 / risk:.2f}" if risk > 0 else "—"
            lines.append(f"TP2: {_fmt_price(take_profit2)} (1:{rr2})")
    if long_pct is not None:
        lines.append(f"Long/Short: {long_pct:.1f}% / {100 - long_pct:.1f}%")
    if oi is not None:
        lines.append(f"Open Interest: {oi:,.0f}")
    lines.append("💡 Consider taking partial profit at TP1, trailing the rest toward TP2.")
    lines.append("⚠️ ATR/structure-based levels, not financial advice — confirm before trading.")
    return "\n".join(lines)


def format_bos_alert(symbol: str, direction: str, price: float) -> str:
    emoji = "🔺" if direction == "bullish" else "🔻"
    return f"{emoji} <b>BOS ({direction.upper()}) — {symbol}</b>\nPrice: {price:.6f}" if price < 1 else \
           f"{emoji} <b>BOS ({direction.upper()}) — {symbol}</b>\nPrice: {price:.4f}"


def format_trail_alert(symbol: str, direction: str, new_stop: float, price: float) -> str:
    return (f"🔧 <b>Trail stop — {symbol}</b>\n"
            f"New stop: {_fmt_price(new_stop)}\n"
            f"Current price: {_fmt_price(price)}\n"
            f"Structure has advanced in your favor — consider moving your stop up to lock in gains.")

# ======================================================================
# FASTAPI APP + SCANNER LOOP
# ======================================================================
from fastapi.responses import HTMLResponse


app = FastAPI(title="AuraPlay Scanner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- in-memory state ----
STATE: dict[str, dict] = {}          # symbol -> snapshot dict
LAST_SIGNAL: dict[str, str] = {}     # symbol -> last alerted signal (BUY/SELL/NEUTRAL)
LAST_BOS: dict[str, str] = {}        # symbol -> last alerted BOS direction
# symbol -> {"direction": "BUY"/"SELL", "trail_stop": float} — tracks an
# open setup so we can suggest tightening the stop as structure advances in
# your favor. Cleared once the signal itself flips away (trend invalidated).
ACTIVE_TRADES: dict[str, dict] = {}
UNIVERSE: list[str] = []
SCAN_META = {"last_full_cycle_started": None, "cycle_count": 0, "universe_size": 0}


async def refresh_universe():
    global UNIVERSE
    symbols = await get_all_perpetual_symbols()
    if symbols:
        UNIVERSE = symbols
        SCAN_META["universe_size"] = len(symbols)
        log.info(f"Universe refreshed: {len(symbols)} USDT perpetual pairs")


async def process_symbol(symbol: str, ticker_24h: dict | None):
    try:
        snap = await build_snapshot(symbol, ticker_24h)
    except Exception as e:
        log.warning(f"{symbol}: snapshot failed: {e}")
        return
    if snap is None:
        return

    STATE[symbol] = snap.to_dict()

    # ---- fire Telegram alerts only on state change (no spam every cycle) ----
    prev_signal = LAST_SIGNAL.get(symbol, "NEUTRAL")
    if snap.signal != prev_signal and snap.signal in ("BUY", "SELL"):
        text = format_signal_alert(symbol, snap.signal, snap.price, snap.rsi, snap.long_pct,
                                    snap.open_interest, snap.stop_loss, snap.take_profit, snap.take_profit2)
        await send_alert(text)
    LAST_SIGNAL[symbol] = snap.signal

    if snap.bos and LAST_BOS.get(symbol) != snap.bos:
        await send_alert(format_bos_alert(symbol, snap.bos, snap.price))
    if snap.bos:
        LAST_BOS[symbol] = snap.bos

    # ---- trailing-stop tracking ----
    # Stay "in the trade" as long as the broader trend holds (SuperTrend
    # direction), not just while the entry-trigger condition (a recent
    # sweep/FVG) is still fresh — that trigger is momentary by design and
    # will naturally lapse a few candles after entry even mid-trend.
    # Tighten the stop as new structure forms in your favor; drop tracking
    # on a trend flip or once price would have stopped you out.
    if snap.signal in ("BUY", "SELL") and snap.stop_loss is not None:
        ACTIVE_TRADES[symbol] = {"direction": snap.signal, "trail_stop": snap.stop_loss}
    elif symbol in ACTIVE_TRADES:
        trade = ACTIVE_TRADES[symbol]
        direction = trade["direction"]
        trend_ok = (
            (direction == "BUY" and snap.supertrend_trend == 1)
            or (direction == "SELL" and snap.supertrend_trend == -1)
        )
        stopped_out = (
            (direction == "BUY" and snap.price <= trade["trail_stop"])
            or (direction == "SELL" and snap.price >= trade["trail_stop"])
        )
        if not trend_ok or stopped_out:
            del ACTIVE_TRADES[symbol]
        elif snap.atr is not None:
            buffer = snap.atr * ATR_SL_BUFFER_MULT
            if direction == "BUY" and snap.stop_ref_below is not None:
                candidate = snap.stop_ref_below - buffer
                if candidate > trade["trail_stop"]:
                    trade["trail_stop"] = candidate
                    await send_alert(format_trail_alert(symbol, direction, candidate, snap.price))
            elif direction == "SELL" and snap.stop_ref_above is not None:
                candidate = snap.stop_ref_above + buffer
                if candidate < trade["trail_stop"]:
                    trade["trail_stop"] = candidate
                    await send_alert(format_trail_alert(symbol, direction, candidate, snap.price))


async def scanner_loop():
    await refresh_universe()
    last_universe_refresh = time.time()
    idx = 0

    while True:
        if not UNIVERSE:
            await asyncio.sleep(5)
            await refresh_universe()
            continue

        if time.time() - last_universe_refresh > FULL_UNIVERSE_REFRESH_SECONDS:
            await refresh_universe()
            last_universe_refresh = time.time()

        tickers = await get_24hr_tickers()

        batch = UNIVERSE[idx: idx + SYMBOLS_PER_BATCH]
        if not batch:
            idx = 0
            SCAN_META["cycle_count"] += 1
            SCAN_META["last_full_cycle_started"] = time.time()
            batch = UNIVERSE[idx: idx + SYMBOLS_PER_BATCH]

        log.info(f"Scanning batch [{idx}:{idx + len(batch)}] of {len(UNIVERSE)}: {batch}")
        await asyncio.gather(*[process_symbol(s, tickers.get(s)) for s in batch])

        idx += SYMBOLS_PER_BATCH
        await asyncio.sleep(BATCH_INTERVAL_SECONDS)


@app.on_event("startup")
async def startup():
    asyncio.create_task(scanner_loop())


# ---------------- REST API ----------------

@app.get("/api/scanner")
async def get_scanner(signal: str | None = None, sort: str = "symbol"):
    try:
        rows = list(STATE.values())
        if signal:
            rows = [r for r in rows if r["signal"] == signal.upper()]
        if sort == "rsi":
            rows.sort(key=lambda r: r["rsi"] or 0, reverse=True)
        elif sort == "change":
            rows.sort(key=lambda r: r["pct_change_24h"] or 0, reverse=True)
        elif sort == "oi":
            rows.sort(key=lambda r: r["open_interest"] or 0, reverse=True)
        else:
            rows.sort(key=lambda r: r["symbol"])
        return {
            "meta": SCAN_META,
            "tracked": len(STATE),
            "universe": len(UNIVERSE),
            "rows": rows,
        }
    except Exception as e:
        log.exception("get_scanner failed")
        return {"meta": SCAN_META, "tracked": len(STATE), "universe": len(UNIVERSE),
                "rows": [], "error": str(e)}


@app.get("/api/symbol/{symbol}")
async def get_symbol(symbol: str):
    return STATE.get(symbol.upper(), {"error": "not tracked yet"})


@app.get("/api/health")
async def health():
    return {"ok": True, "tracked": len(STATE), "universe": len(UNIVERSE)}


# ---------------- Frontend (embedded — see DASHBOARD_HTML at bottom of file) ----------------
@app.get("/")
async def index():
    return HTMLResponse(DASHBOARD_HTML)


# ======================================================================
# DASHBOARD UI (served at "/")
# ======================================================================
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AuraPlay Scanner</title>
<style>
  :root{
    --bg:#0a0d13;
    --panel:#11151d;
    --panel-2:#161b25;
    --line:#232a37;
    --text:#e7ebf3;
    --muted:#7c8798;
    --green:#20d78c;
    --red:#ff4d6a;
    --amber:#f5b942;
    --mono:"SF Mono","JetBrains Mono",ui-monospace,Menlo,Consolas,monospace;
  }
  *{box-sizing:border-box;}
  body{
    margin:0;
    background:var(--bg);
    color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    -webkit-font-smoothing:antialiased;
  }
  header{
    position:sticky;top:0;z-index:5;
    background:linear-gradient(180deg,rgba(10,13,19,0.98),rgba(10,13,19,0.9));
    backdrop-filter:blur(8px);
    border-bottom:1px solid var(--line);
    padding:14px 16px 10px;
  }
  .brand{display:flex;align-items:center;justify-content:space-between;}
  .brand h1{
    font-size:17px;margin:0;letter-spacing:0.02em;
    display:flex;align-items:center;gap:8px;
  }
  .brand h1 .dot{
    width:8px;height:8px;border-radius:50%;background:var(--green);
    box-shadow:0 0 8px var(--green);
    animation:pulse 2s infinite;
  }
  @keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.35;}}
  .meta{font-family:var(--mono);font-size:11px;color:var(--muted);}
  .controls{
    display:flex;gap:8px;margin-top:12px;overflow-x:auto;
    padding-bottom:2px;
  }
  .controls::-webkit-scrollbar{display:none;}
  .chip{
    flex:0 0 auto;
    padding:6px 14px;border-radius:20px;
    border:1px solid var(--line);
    background:var(--panel);
    color:var(--muted);
    font-size:12.5px;font-weight:600;
    cursor:pointer;white-space:nowrap;
    transition:all .15s ease;
  }
  .chip.active{
    color:#0a0d13;background:var(--text);border-color:var(--text);
  }
  .chip.buy.active{background:var(--green);border-color:var(--green);}
  .chip.sell.active{background:var(--red);border-color:var(--red);}
  .searchwrap{padding:10px 16px 0;}
  input[type=text]{
    width:100%;padding:9px 12px;border-radius:10px;
    background:var(--panel);border:1px solid var(--line);
    color:var(--text);font-size:14px;font-family:var(--mono);
  }
  input[type=text]:focus{outline:none;border-color:#3a4456;}
  main{padding:10px 8px 40px;}
  .row{
    display:grid;
    grid-template-columns: 1.4fr 1fr 0.7fr 0.9fr 0.9fr;
    gap:6px;
    padding:12px 10px;
    background:var(--panel);
    border:1px solid var(--line);
    border-radius:12px;
    margin-bottom:6px;
    align-items:center;
  }
  .row.head{
    background:transparent;border:none;padding:4px 10px;
    color:var(--muted);font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;
  }
  .sym{font-weight:700;font-size:14px;}
  .sym .sub{display:block;font-size:10.5px;color:var(--muted);font-weight:500;margin-top:2px;}
  .num{font-family:var(--mono);font-size:13px;}
  .price{font-family:var(--mono);font-size:14px;font-weight:600;}
  .chg-pos{color:var(--green);}
  .chg-neg{color:var(--red);}
  .badge{
    display:inline-flex;align-items:center;gap:4px;
    padding:4px 9px;border-radius:7px;font-size:11.5px;font-weight:700;
    font-family:var(--mono);letter-spacing:.02em;
  }
  .badge.BUY{background:rgba(32,215,140,0.14);color:var(--green);}
  .badge.SELL{background:rgba(255,77,106,0.14);color:var(--red);}
  .badge.NEUTRAL{background:rgba(124,135,152,0.14);color:var(--muted);}
  .bos-tag{
    display:inline-block;margin-top:3px;font-size:10px;font-weight:700;
    padding:1px 6px;border-radius:5px;font-family:var(--mono);
  }
  .bos-tag.bullish{color:var(--green);background:rgba(32,215,140,0.1);}
  .bos-tag.bearish{color:var(--red);background:rgba(255,77,106,0.1);}
  .ls-bar{
    height:5px;border-radius:3px;overflow:hidden;background:var(--red);
    display:flex;margin-top:5px;width:64px;
  }
  .ls-bar .l{background:var(--green);}
  .empty{
    text-align:center;color:var(--muted);padding:60px 20px;font-size:13px;
  }
  footer{
    text-align:center;color:var(--muted);font-size:11px;padding:20px;
    font-family:var(--mono);
  }
  .row:not(.head){cursor:pointer;}
  .row:not(.head):active{background:var(--panel-2);}
  .detail-overlay{
    position:fixed;inset:0;background:rgba(0,0,0,0.6);
    display:none;align-items:flex-end;z-index:50;
  }
  .detail-overlay.open{display:flex;}
  .detail-sheet{
    width:100%;max-height:82vh;overflow-y:auto;
    background:var(--panel);border-top:1px solid var(--line);
    border-radius:18px 18px 0 0;
    padding:18px 16px 28px;
    animation:slideup .18s ease-out;
  }
  @keyframes slideup{from{transform:translateY(24px);opacity:0;}to{transform:translateY(0);opacity:1;}}
  .detail-header{
    display:flex;justify-content:space-between;align-items:flex-start;
    margin-bottom:14px;
  }
  .detail-title{font-size:19px;font-weight:800;font-family:var(--mono);}
  .detail-close{
    font-size:26px;line-height:1;color:var(--muted);
    padding:2px 10px;cursor:pointer;
  }
  .detail-section{
    background:var(--panel-2);border:1px solid var(--line);
    border-radius:12px;padding:10px 14px;margin-bottom:10px;
  }
  .detail-row{
    display:flex;justify-content:space-between;align-items:center;
    padding:7px 0;font-family:var(--mono);font-size:13px;
    border-bottom:1px solid var(--line);
  }
  .detail-row:last-child{border-bottom:none;}
  .detail-row span:first-child{color:var(--muted);}
  .detail-disclaimer{
    color:var(--muted);font-size:10.5px;text-align:center;
    padding:6px 4px 0;line-height:1.5;
  }
  @media (max-width:420px){
    .row{grid-template-columns: 1.3fr 0.9fr 0.7fr 0.85fr;}
    .row .col-oi{display:none;}
    .row.head .col-oi{display:none;}
  }
</style>
</head>
<body>

<header>
  <div class="brand">
    <h1><span class="dot"></span> AuraPlay Scanner</h1>
    <span class="meta" id="metaTxt">connecting…</span>
  </div>
  <div class="controls" id="filterChips">
    <div class="chip active" data-filter="ALL">All</div>
    <div class="chip buy" data-filter="BUY">Buy</div>
    <div class="chip sell" data-filter="SELL">Sell</div>
    <div class="chip" data-filter="NEUTRAL">Neutral</div>
    <div class="chip" data-sort="change">Top movers</div>
    <div class="chip" data-sort="rsi">RSI</div>
    <div class="chip" data-sort="oi">Open interest</div>
  </div>
</header>

<div class="searchwrap">
  <input type="text" id="search" placeholder="Search symbol e.g. PNUT, BTC…">
</div>

<main>
  <div class="row head">
    <div>Symbol</div>
    <div>Price / 24h</div>
    <div>RSI</div>
    <div>Signal</div>
    <div class="col-oi">L/S · OI</div>
  </div>
  <div id="rows"></div>
</main>

<footer>Data: Binance USDT-M Futures · Signals are informational, not financial advice</footer>

<div id="detailOverlay" class="detail-overlay" onclick="if(event.target===this) closeDetail()">
  <div class="detail-sheet">
    <div class="detail-header">
      <div>
        <div class="detail-title" id="detailTitle"></div>
        <div id="detailBadge" style="margin-top:6px;"></div>
      </div>
      <div class="detail-close" onclick="closeDetail()">&times;</div>
    </div>
    <div id="detailBody"></div>
  </div>
</div>

<script>
const API_BASE = ""; // same origin (backend serves this file)
let currentFilter = "ALL";
let currentSort = "symbol";
let searchTerm = "";

document.querySelectorAll(".chip[data-filter]").forEach(chip => {
  chip.addEventListener("click", () => {
    document.querySelectorAll(".chip[data-filter]").forEach(c => c.classList.remove("active"));
    chip.classList.add("active");
    currentFilter = chip.dataset.filter;
    load();
  });
});
document.querySelectorAll(".chip[data-sort]").forEach(chip => {
  chip.addEventListener("click", () => {
    currentSort = chip.dataset.sort;
    load();
  });
});
document.getElementById("search").addEventListener("input", (e) => {
  searchTerm = e.target.value.trim().toUpperCase();
  render(lastRows);
});

let lastRows = [];

function timeAgo(ts){
  if (!ts) return "—";
  const secs = Math.max(0, Math.floor(Date.now()/1000 - ts));
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.floor(secs/60)}m ago`;
  return `${Math.floor(secs/3600)}h ago`;
}

function rrText(entry, stop, target){
  if (entry == null || stop == null || target == null) return "—";
  const risk = Math.abs(entry - stop);
  const reward = Math.abs(target - entry);
  if (risk <= 0) return "—";
  return `1:${(reward/risk).toFixed(2)}`;
}

function openDetail(symbol){
  const r = lastRows.find(x => x.symbol === symbol);
  if (!r) return;
  const longPct = r.long_pct ?? 50;
  const hasLevels = r.signal !== "NEUTRAL" && r.stop_loss != null && r.take_profit != null;

  document.getElementById("detailTitle").textContent = r.symbol;
  document.getElementById("detailBadge").innerHTML = `<span class="badge ${r.signal}">${r.signal}</span>`;

  let levelsHtml = "";
  if (hasLevels){
    levelsHtml = `
      <div class="detail-section">
        <div class="detail-row"><span>Entry</span><span>${fmtPrice(r.price)}</span></div>
        <div class="detail-row"><span>Stop Loss</span><span class="chg-neg">${fmtPrice(r.stop_loss)}</span></div>
        <div class="detail-row"><span>Take Profit 1</span><span class="chg-pos">${fmtPrice(r.take_profit)} (${rrText(r.price, r.stop_loss, r.take_profit)})</span></div>
        ${r.take_profit2 != null ? `<div class="detail-row"><span>Take Profit 2</span><span class="chg-pos">${fmtPrice(r.take_profit2)} (${rrText(r.price, r.stop_loss, r.take_profit2)})</span></div>` : ""}
      </div>`;
  }

  document.getElementById("detailBody").innerHTML = `
    ${levelsHtml}
    <div class="detail-section">
      <div class="detail-row"><span>Price</span><span>${fmtPrice(r.price)}</span></div>
      <div class="detail-row"><span>24h Change</span><span class="${(r.pct_change_24h ?? 0) >= 0 ? 'chg-pos' : 'chg-neg'}">${(r.pct_change_24h ?? 0).toFixed(2)}%</span></div>
      <div class="detail-row"><span>RSI</span><span>${r.rsi ?? "—"}</span></div>
      <div class="detail-row"><span>SuperTrend</span><span>${r.supertrend_trend === 1 ? "Bullish" : r.supertrend_trend === -1 ? "Bearish" : "—"}</span></div>
      <div class="detail-row"><span>VWAP</span><span>${r.vwap != null ? fmtPrice(r.vwap) : "—"}</span></div>
    </div>
    <div class="detail-section">
      <div class="detail-row"><span>Long / Short</span><span>${longPct.toFixed(1)}% / ${(100-longPct).toFixed(1)}%</span></div>
      <div class="detail-row"><span>Open Interest</span><span>${fmtCompact(r.open_interest)}</span></div>
      <div class="detail-row"><span>BOS</span><span>${r.bos ? r.bos.toUpperCase() : "—"}</span></div>
    </div>
    <div class="detail-section">
      <div class="detail-row"><span>Updated</span><span>${timeAgo(r.updated_at)}</span></div>
    </div>
    <div class="detail-disclaimer">⚠️ Structure/ATR-based levels — informational only, not financial advice.</div>
  `;

  document.getElementById("detailOverlay").classList.add("open");
}

function closeDetail(){
  document.getElementById("detailOverlay").classList.remove("open");
}

function fmtPrice(p){
  if (p === null || p === undefined) return "—";
  if (p < 0.01) return p.toFixed(6);
  if (p < 1) return p.toFixed(5);
  if (p < 100) return p.toFixed(4);
  return p.toFixed(2);
}

function fmtCompact(n){
  if (n === null || n === undefined) return "—";
  if (Math.abs(n) >= 1e9) return (n/1e9).toFixed(2)+"B";
  if (Math.abs(n) >= 1e6) return (n/1e6).toFixed(2)+"M";
  if (Math.abs(n) >= 1e3) return (n/1e3).toFixed(1)+"K";
  return n.toFixed(0);
}

function render(rows){
  const container = document.getElementById("rows");
  let filtered = rows;
  if (searchTerm){
    filtered = filtered.filter(r => r.symbol.includes(searchTerm));
  }
  if (!filtered.length){
    container.innerHTML = `<div class="empty">No pairs match yet — scanner is warming up, or try a different filter.</div>`;
    return;
  }
  container.innerHTML = filtered.map(r => {
    const chgClass = (r.pct_change_24h ?? 0) >= 0 ? "chg-pos" : "chg-neg";
    const chgSign = (r.pct_change_24h ?? 0) >= 0 ? "+" : "";
    const longPct = r.long_pct ?? 50;
    const bosTag = r.bos ? `<span class="bos-tag ${r.bos}">BOS ${r.bos === 'bullish' ? '▲' : '▼'}</span>` : "";
    const hasLevels = r.signal !== "NEUTRAL" && r.stop_loss != null && r.take_profit != null;
    const levelsTag = hasLevels
      ? `<div class="num" style="color:var(--muted);margin-top:3px;font-size:10.5px;">SL ${fmtPrice(r.stop_loss)} · TP1 ${fmtPrice(r.take_profit)}${r.take_profit2 != null ? ' · TP2 ' + fmtPrice(r.take_profit2) : ''}</div>`
      : "";
    return `
      <div class="row" data-symbol="${r.symbol}" onclick="openDetail('${r.symbol}')">
        <div class="sym">${r.symbol.replace('USDT','')}
          <span class="sub">${r.symbol}</span>
        </div>
        <div>
          <div class="price">${fmtPrice(r.price)}</div>
          <div class="num ${chgClass}">${chgSign}${(r.pct_change_24h ?? 0).toFixed(2)}%</div>
        </div>
        <div class="num">${r.rsi ?? "—"}</div>
        <div>
          <span class="badge ${r.signal}">${r.signal}</span>
          ${bosTag}
          ${levelsTag}
        </div>
        <div class="col-oi">
          <div class="num">${longPct.toFixed(0)}% / ${(100-longPct).toFixed(0)}%</div>
          <div class="ls-bar"><div class="l" style="width:${longPct}%"></div></div>
          <div class="num" style="color:var(--muted);margin-top:3px;">OI ${fmtCompact(r.open_interest)}</div>
        </div>
      </div>
    `;
  }).join("");
}

async function load(){
  try{
    const params = new URLSearchParams();
    if (currentFilter !== "ALL") params.set("signal", currentFilter);
    params.set("sort", currentSort);
    const res = await fetch(`${API_BASE}/api/scanner?${params}`);
    if (!res.ok){
      const body = await res.text();
      document.getElementById("metaTxt").textContent = `error ${res.status}: ${body.slice(0,120)}`;
      return;
    }
    const data = await res.json();
    lastRows = data.rows || [];
    document.getElementById("metaTxt").textContent =
      `tracking ${data.tracked}/${data.universe} pairs · cycle #${data.meta.cycle_count}`;
    render(lastRows);
  }catch(e){
    document.getElementById("metaTxt").textContent = `fetch failed: ${e.message}`;
  }
}

load();
setInterval(load, 8000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
