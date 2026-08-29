"""
AuraPlay Scanner — single-file build.

Everything (config, Bybit client, indicators, signal engine, Telegram
alerts, FastAPI app, and the dashboard UI) lives in this one file so it's
easy to upload straight from a phone: just this file + requirements.txt.

Bybit USDT perpetuals scanner (filtered to pairs also listed on Binance,
since that's where trades are actually placed): cycles through every
matching pair, computes RSI, EMA trend, SuperTrend, VWAP, OBV, liquidity
sweeps, FVGs, Break-of-Structure (BOS), Long/Short ratio, Open Interest,
and large-trade ("whale") flow — then runs a structure -> BOS -> retracement
-> confirmed-entry pipeline before firing a signal, serves it as a live
dashboard, and pushes Buy/Sell/BOS/trail-stop/TP-hit alerts to Telegram.
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
# ---- Bybit V5 (linear USDT perpetuals) public REST base ----
BYBIT_BASE = "https://api.bybit.com"
BYBIT_INTERVAL_MAP = {"1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
                       "1h": "60", "2h": "120", "4h": "240", "1d": "D"}

# ---- Binance (used ONLY as a symbol filter — "tokens also listed on Binance",
# since that's where you trade — never as a data source anymore) ----
BINANCE_FAPI_BASE = "https://fapi.binance.com"
BINANCE_FILTER_ENABLED = os.getenv("BINANCE_FILTER_ENABLED", "true").lower() == "true"
BINANCE_FILTER_REFRESH_SECONDS = int(os.getenv("BINANCE_FILTER_REFRESH_SECONDS", "86400"))  # once/day — this is just a symbol list, not live data

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

# ---- Entry confirmation pipeline: structure -> BOS (on a CLOSED candle) ->
# retracement into the impulse zone -> a confirming candle. This trades
# fewer, later entries for a materially better entry price than reacting to
# the breakout candle itself.
RETRACEMENT_LOOKBACK_BARS = int(os.getenv("RETRACEMENT_LOOKBACK_BARS", "12"))  # how far back to look for the BOS that started the current move

# ---- Whale / large-trade detection (public trade-tape data — informational
# confluence booster only, never a hard gate) ----
WHALE_TRADE_USD_THRESHOLD = float(os.getenv("WHALE_TRADE_USD_THRESHOLD", "50000"))
WHALE_TRADE_LOOKBACK = int(os.getenv("WHALE_TRADE_LOOKBACK", "50"))  # how many recent trades to scan per symbol

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
# BYBIT V5 CLIENT (public REST — no API key needed)
# ======================================================================

_client: Optional[httpx.AsyncClient] = None
BYBIT_CONCURRENCY = int(os.getenv("BYBIT_CONCURRENCY", "3"))  # max simultaneous in-flight requests, across all symbols
_bybit_semaphore = asyncio.Semaphore(BYBIT_CONCURRENCY)


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(base_url=BYBIT_BASE, timeout=15.0)
    return _client


LAST_RATE_LIMIT: dict = {"status": None, "path": None, "at": None}


async def _get(path: str, params: dict | None = None):
    """Unwraps Bybit's {retCode, retMsg, result} envelope. Returns result dict
    on success, None if the request genuinely failed (caller must distinguish
    that from a legitimately empty result)."""
    client = get_client()
    for attempt in range(3):
        try:
            async with _bybit_semaphore:
                resp = await client.get(path, params=params)
            if resp.status_code in (403, 429):
                LAST_RATE_LIMIT["status"] = resp.status_code
                LAST_RATE_LIMIT["path"] = path
                LAST_RATE_LIMIT["at"] = time.time()
                wait = 2 ** (attempt + 1)
                log.warning(f"Rate limited ({resp.status_code}) on {path}, backing off {wait}s")
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            body = resp.json()
            if body.get("retCode") == 10006:  # Bybit's own "too many visits" code
                LAST_RATE_LIMIT["status"] = 10006
                LAST_RATE_LIMIT["path"] = path
                LAST_RATE_LIMIT["at"] = time.time()
                wait = 2 ** (attempt + 1)
                log.warning(f"Bybit rate limit (retCode 10006) on {path}, backing off {wait}s")
                await asyncio.sleep(wait)
                continue
            if body.get("retCode") != 0:
                log.warning(f"Bybit error on {path}: {body.get('retMsg')}")
                return None
            return body.get("result")
        except httpx.HTTPError as e:
            log.warning(f"Request failed ({path}): {e}, attempt {attempt + 1}/3")
            await asyncio.sleep(1.5 * (attempt + 1))
    return None


async def get_all_perpetual_symbols() -> list[str]:
    """All actively trading USDT linear perpetual symbols on Bybit."""
    result = await _get("/v5/market/instruments-info", {"category": "linear", "status": "Trading", "limit": 1000})
    if not result:
        return []
    symbols = []
    cursor = result.get("nextPageCursor")
    for item in result.get("list", []):
        if item.get("quoteCoin") == QUOTE_ASSET and item.get("contractType") == "LinearPerpetual":
            symbols.append(item["symbol"])
    # paginate if Bybit says there's more (universe is a few hundred symbols, one extra page covers it)
    if cursor:
        result2 = await _get("/v5/market/instruments-info",
                              {"category": "linear", "status": "Trading", "limit": 1000, "cursor": cursor})
        if result2:
            for item in result2.get("list", []):
                if item.get("quoteCoin") == QUOTE_ASSET and item.get("contractType") == "LinearPerpetual":
                    symbols.append(item["symbol"])
    return sorted(set(symbols) - EXCLUDE_SYMBOLS)


async def get_24hr_tickers() -> dict:
    """Bulk ticker stats for ALL linear symbols in one call — includes price
    change, and open interest (openInterestValue), so this single call covers
    what used to be two separate Binance calls."""
    result = await _get("/v5/market/tickers", {"category": "linear"})
    if not result:
        return {}
    return {d["symbol"]: d for d in result.get("list", [])}


async def get_klines(symbol: str, interval: str = None, limit: int = None):
    interval = interval or SIGNAL_INTERVAL
    limit = limit or KLINE_LIMIT
    bybit_interval = BYBIT_INTERVAL_MAP.get(interval, "60")
    result = await _get("/v5/market/kline", {"category": "linear", "symbol": symbol,
                                              "interval": bybit_interval, "limit": limit})
    if result is None:
        return None  # genuinely failed request — retryable
    rows = result.get("list", [])
    # Bybit returns newest-first; every downstream function assumes oldest-first like Binance did
    return list(reversed(rows))


async def get_long_short_account_ratio(symbol: str, period: str = None):
    period = period or LS_RATIO_PERIOD
    bybit_period = BYBIT_INTERVAL_MAP.get(period, "60")
    # Bybit's period param for this endpoint takes the same style strings as klines intervals for common cases
    period_str = {"60": "1h", "240": "4h", "D": "1d"}.get(bybit_period, "1h")
    result = await _get("/v5/market/account-ratio", {"category": "linear", "symbol": symbol,
                                                       "period": period_str, "limit": 1})
    if not result:
        return None
    lst = result.get("list", [])
    return lst[0] if lst else None


async def get_recent_trades(symbol: str, limit: int = None):
    limit = limit or WHALE_TRADE_LOOKBACK
    result = await _get("/v5/market/recent-trade", {"category": "linear", "symbol": symbol, "limit": limit})
    if not result:
        return []
    return result.get("list", [])


# ======================================================================
# BINANCE SYMBOL FILTER (isolated — used ONLY to know which tokens are also
# listed on Binance, since that's where trades are actually placed. Never
# used for price/indicator/signal data. A Binance outage here degrades to
# "no filtering" rather than breaking the scanner.)
# ======================================================================

_binance_client: Optional[httpx.AsyncClient] = None


def _get_binance_client() -> httpx.AsyncClient:
    global _binance_client
    if _binance_client is None:
        _binance_client = httpx.AsyncClient(base_url=BINANCE_FAPI_BASE, timeout=10.0)
    return _binance_client


async def fetch_binance_symbol_set() -> set[str] | None:
    """Returns None on failure (caller should keep using the last-known set,
    or skip filtering entirely if there's no prior set yet) rather than ever
    raising — this must never be able to take the scanner down."""
    try:
        client = _get_binance_client()
        resp = await client.get("/fapi/v1/exchangeInfo")
        resp.raise_for_status()
        data = resp.json()
        return {
            s["symbol"] for s in data.get("symbols", [])
            if s.get("status") == "TRADING" and s.get("quoteAsset") == QUOTE_ASSET
            and s.get("contractType") == "PERPETUAL"
        }
    except Exception as e:
        log.warning(f"Binance symbol-filter fetch failed (non-fatal, keeping prior list): {e}")
        return None

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
    """Did the last CLOSED candle wick through the most recent swing high/low
    and close back inside it? (classic liquidity grab). Evaluated on the
    last closed candle, never the live/forming one — same reasoning as
    break_of_structure. Returns the actual swing price levels too, since
    those are the real invalidation points for a stop-loss — not just a
    boolean."""
    if len(df) < 2:
        return {"swept_low": False, "swept_high": False, "swept_low_level": None, "swept_high_level": None}
    last_closed_idx = len(df) - 2
    last = df.iloc[last_closed_idx]
    mask = pd.Series(True, index=df.index)
    mask.iloc[last_closed_idx + 1:] = False
    recent_highs = df.loc[is_high & mask, "high"]
    recent_lows = df.loc[is_low & mask, "low"]

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
    Evaluated on the LAST CLOSED candle (index -2), never the live/forming
    one (index -1) — a forming candle's wick can break structure and then
    close back inside it a few minutes later, which was firing false signals.
    Returns 'bullish', 'bearish', or None."""
    if len(df) < 3:
        return None
    last_closed = len(df) - 2
    close_now = df["close"].iloc[last_closed]
    close_prev = df["close"].iloc[last_closed - 1]
    mask = pd.Series(True, index=df.index)
    mask.iloc[last_closed + 1:] = False
    recent_highs = df.loc[is_high & mask, "high"]
    recent_lows = df.loc[is_low & mask, "low"]

    bullish = len(recent_highs) > 0 and close_now > recent_highs.iloc[-1] and close_prev <= recent_highs.iloc[-1]
    bearish = len(recent_lows) > 0 and close_now < recent_lows.iloc[-1] and close_prev >= recent_lows.iloc[-1]

    if bullish:
        return "bullish"
    if bearish:
        return "bearish"
    return None


def bos_history(df: pd.DataFrame, is_high: pd.Series, is_low: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Same BOS logic as break_of_structure, but computed for every closed
    candle (not just the last one) so we can find the candle that STARTED
    the current move — the "first BOS" — rather than only ever seeing
    whichever break happens to be most recent. Excludes the live/forming
    candle throughout."""
    n = len(df)
    bull = pd.Series(False, index=df.index)
    bear = pd.Series(False, index=df.index)
    if n < 3:
        return bull, bear
    highs = df["high"]; lows = df["low"]; closes = df["close"]
    last_high_val = None
    last_low_val = None
    for i in range(1, n - 1):  # exclude the live last candle (n-1)
        if is_high.iloc[i]:
            last_high_val = highs.iloc[i]
        if is_low.iloc[i]:
            last_low_val = lows.iloc[i]
        if last_high_val is not None and closes.iloc[i] > last_high_val and closes.iloc[i - 1] <= last_high_val:
            bull.iloc[i] = True
        if last_low_val is not None and closes.iloc[i] < last_low_val and closes.iloc[i - 1] >= last_low_val:
            bear.iloc[i] = True
    return bull, bear


def retracement_entry(df: pd.DataFrame, is_high: pd.Series, is_low: pd.Series,
                       bull_bos: pd.Series, bear_bos: pd.Series, direction: str,
                       lookback: int = None) -> tuple[bool, float | None, float | None]:
    """Structure -> BOS -> retracement -> confirmed entry, all from one candle
    window (no cross-cycle state needed):
      1. Find the FIRST BOS in `direction` within the last `lookback` closed
         candles — the move's actual origin, not just the latest noise.
      2. Define its zone: the bullish/bearish FVG that formed right after it
         (the real imbalance to trade back into), falling back to the broken
         swing level itself (order-block retest) if no FVG formed.
      3. Require price actually traded back into that zone on a LATER candle
         (a genuine pullback, not an instant continuation).
      4. Require the LAST CLOSED candle itself confirms continuation from
         inside the zone (closes back in the trend's direction).
    Returns (confirmed, zone_low, zone_high).
    """
    lookback = lookback or RETRACEMENT_LOOKBACK_BARS
    n = len(df)
    last_closed = n - 2
    if last_closed < 4:
        return False, None, None

    bos_col = bull_bos if direction == "bullish" else bear_bos
    search_start = max(1, last_closed - lookback)

    # FIRST bos in the window, not the most recent — this is the fix for
    # "entry was near the top": find where the move actually started.
    bos_idx = None
    for i in range(search_start, last_closed):
        if bos_col.iloc[i]:
            bos_idx = i
            break
    if bos_idx is None:
        return False, None, None

    zone_low = zone_high = None
    if direction == "bullish":
        for j in range(bos_idx, min(bos_idx + 4, n - 1)):
            if df["bull_fvg"].iloc[j] and j >= 1 and j + 1 < n:
                zone_low = float(df["high"].iloc[j - 1])
                zone_high = float(df["low"].iloc[j + 1])
                break
        if zone_low is None:
            mask = is_high.copy()
            mask.iloc[bos_idx:] = False
            highs_before = df.loc[mask, "high"]
            level = float(highs_before.iloc[-1]) if len(highs_before) else float(df["low"].iloc[bos_idx])
            zone_low, zone_high = level * 0.999, level * 1.005
    else:
        for j in range(bos_idx, min(bos_idx + 4, n - 1)):
            if df["bear_fvg"].iloc[j] and j >= 1 and j + 1 < n:
                zone_high = float(df["low"].iloc[j - 1])
                zone_low = float(df["high"].iloc[j + 1])
                break
        if zone_low is None:
            mask = is_low.copy()
            mask.iloc[bos_idx:] = False
            lows_before = df.loc[mask, "low"]
            level = float(lows_before.iloc[-1]) if len(lows_before) else float(df["high"].iloc[bos_idx])
            zone_high, zone_low = level * 1.001, level * 0.995

    if zone_low is None or zone_high is None or zone_low > zone_high:
        return False, None, None

    touched = False
    for k in range(bos_idx + 1, last_closed + 1):
        if df["low"].iloc[k] <= zone_high and df["high"].iloc[k] >= zone_low:
            touched = True
            break
    if not touched:
        return False, zone_low, zone_high

    last = df.iloc[last_closed]
    if direction == "bullish":
        confirmed = bool(last["low"] <= zone_high and last["close"] > last["open"] and last["close"] >= zone_low)
    else:
        confirmed = bool(last["high"] >= zone_low and last["close"] < last["open"] and last["close"] <= zone_high)
    return confirmed, zone_low, zone_high


def detect_whale_activity(trades: list, price: float) -> dict:
    """Scans recent public trades for unusually large orders (informational
    confluence booster — never a hard gate, since large-trade flow is noisy
    on its own). Returns net bias and total notional of large trades seen."""
    buy_notional = 0.0
    sell_notional = 0.0
    for t in trades:
        try:
            sz = float(t.get("size", 0))
            px = float(t.get("price", price))
            notional = sz * px
            if notional < WHALE_TRADE_USD_THRESHOLD:
                continue
            side = t.get("side", "").lower()
            if side == "buy":
                buy_notional += notional
            elif side == "sell":
                sell_notional += notional
        except (TypeError, ValueError):
            continue
    total = buy_notional + sell_notional
    bias = None
    if total > 0:
        if buy_notional > sell_notional * 1.3:
            bias = "bullish"
        elif sell_notional > buy_notional * 1.3:
            bias = "bearish"
    return {"whale_bias": bias, "whale_notional": total}

# ======================================================================
# SIGNAL ENGINE (combines indicators + L/S ratio + OI into BUY/SELL/BOS)
# ======================================================================


class SymbolSnapshot:
    __slots__ = (
        "symbol", "price", "pct_change_24h", "rsi", "ema_fast", "ema_slow",
        "supertrend_trend", "vwap", "atr", "obv", "open_interest", "long_short_account_ratio",
        "long_pct", "short_pct", "signal", "bos", "stop_loss", "take_profit", "take_profit2",
        "stop_ref_below", "stop_ref_above", "whale_bias", "whale_notional",
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

    last_closed_idx = len(df) - 2
    last_closed = df.iloc[last_closed_idx]
    live_price = float(df["close"].iloc[-1])  # actual current price — used for display and SL/TP arithmetic

    sweep = liquidity_sweep(df, is_high, is_low)
    bull_bos, bear_bos = bos_history(df, is_high, is_low)
    bos = break_of_structure(df, is_high, is_low)

    bull_confirmed, bull_zone_low, bull_zone_high = retracement_entry(df, is_high, is_low, bull_bos, bear_bos, "bullish")
    bear_confirmed, bear_zone_low, bear_zone_high = retracement_entry(df, is_high, is_low, bull_bos, bear_bos, "bearish")

    # Stop reference now prefers the retracement zone boundary itself when a
    # setup just confirmed — that's the precise invalidation point of THIS
    # entry, sharper than the generic "nearest swept level" fallback.
    stop_ref_below = bull_zone_low if bull_confirmed else (sweep["swept_low_level"] or nearest_structure_level(df, is_high, is_low, live_price, "below"))
    stop_ref_above = bear_zone_high if bear_confirmed else (sweep["swept_high_level"] or nearest_structure_level(df, is_high, is_low, live_price, "above"))

    target_above = nearest_structure_level(df, is_high, is_low, live_price, "above")
    target_below = nearest_structure_level(df, is_high, is_low, live_price, "below")

    return {
        "df": df, "is_high": is_high, "is_low": is_low,  # kept for TP2 lookup after TP1 is known — not sent to JSON
        "price": live_price,
        "rsi": round(float(last_closed["rsi"]), 2),
        "ema_fast": float(last_closed["ema_fast"]),
        "ema_slow": float(last_closed["ema_slow"]),
        "supertrend_trend": int(last_closed["st_trend"]),
        "vwap": float(last_closed["vwap"]) if pd.notna(last_closed["vwap"]) else None,
        "atr": float(last_closed["atr"]) if pd.notna(last_closed["atr"]) else None,
        "obv": float(last_closed["obv"]),
        "swept_low": sweep["swept_low"],
        "swept_high": sweep["swept_high"],
        "stop_ref_below": stop_ref_below,
        "stop_ref_above": stop_ref_above,
        "target_above": target_above,
        "target_below": target_below,
        "bull_fvg_recent": bool(df["bull_fvg"].iloc[:last_closed_idx + 1].tail(5).any()),
        "bear_fvg_recent": bool(df["bear_fvg"].iloc[:last_closed_idx + 1].tail(5).any()),
        "bos": bos,
        "bullish_retracement_confirmed": bull_confirmed,
        "bearish_retracement_confirmed": bear_confirmed,
    }


async def build_snapshot(symbol: str, ticker_24h: dict | None) -> SymbolSnapshot | None:
    klines = await get_klines(symbol)
    if klines is None:
        raise RuntimeError("Bybit request failed (rate limited or network error) — will retry next cycle")
    if len(klines) < PIVOT_LOOKBACK * 2 + 5:
        return None  # genuinely too little history from Bybit — not a retryable failure

    computed = await asyncio.to_thread(_compute_indicators, klines)

    # ---- long/short ratio (Bybit) + whale/large-trade scan ----
    ls = await get_long_short_account_ratio(symbol)
    trades = await get_recent_trades(symbol)
    whale = detect_whale_activity(trades, computed["price"])

    snap = SymbolSnapshot()
    snap.symbol = symbol
    snap.price = computed["price"]
    # Bybit's price24hPcnt is a fraction (0.0166 == 1.66%), unlike Binance's percent-number field
    snap.pct_change_24h = float(ticker_24h["price24hPcnt"]) * 100 if ticker_24h and ticker_24h.get("price24hPcnt") else None
    snap.rsi = computed["rsi"]
    snap.ema_fast = computed["ema_fast"]
    snap.ema_slow = computed["ema_slow"]
    snap.supertrend_trend = computed["supertrend_trend"]
    snap.vwap = computed["vwap"]
    snap.obv = computed["obv"]
    # Bybit's bulk ticker already carries open interest — no separate call needed
    oiv = ticker_24h.get("openInterestValue") if ticker_24h else None
    snap.open_interest = float(oiv) if oiv not in (None, "") else None
    snap.long_short_account_ratio = (
        float(ls["buyRatio"]) / float(ls["sellRatio"]) if ls and float(ls.get("sellRatio", 0) or 0) > 0 else None
    )
    snap.long_pct = float(ls["buyRatio"]) * 100 if ls else None
    snap.short_pct = float(ls["sellRatio"]) * 100 if ls else None
    snap.swept_low = computed["swept_low"]
    snap.swept_high = computed["swept_high"]
    snap.bull_fvg_recent = computed["bull_fvg_recent"]
    snap.bear_fvg_recent = computed["bear_fvg_recent"]
    snap.bos = computed["bos"]
    snap.atr = computed["atr"]
    snap.stop_ref_below = computed["stop_ref_below"]
    snap.stop_ref_above = computed["stop_ref_above"]
    snap.whale_bias = whale["whale_bias"]
    snap.whale_notional = whale["whale_notional"]
    snap.updated_at = time.time()

    # ---- confluence signal ----
    # Core trigger is now the full structure -> BOS -> retracement -> confirmed
    # entry pipeline (retracement_entry), which already requires a same-
    # direction BOS as its first step — REQUIRE_BOS_CONFLUENCE is superseded
    # by this and kept only for reference. Trend/VWAP/RSI remain as broader
    # context filters so we don't buy a confirmed dip that's still fighting
    # the higher-level trend.
    bullish_trend = snap.supertrend_trend == 1 and snap.ema_fast > snap.ema_slow
    bearish_trend = snap.supertrend_trend == -1 and snap.ema_fast < snap.ema_slow
    above_vwap = snap.vwap is not None and snap.price > snap.vwap
    below_vwap = snap.vwap is not None and snap.price < snap.vwap

    buy = (
        bullish_trend
        and above_vwap
        and snap.rsi < RSI_OVERBOUGHT
        and computed["bullish_retracement_confirmed"]
    )
    sell = (
        bearish_trend
        and below_vwap
        and snap.rsi > RSI_OVERSOLD
        and computed["bearish_retracement_confirmed"]
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
            if stop_ref is not None and stop_ref >= snap.price:
                stop_ref = None  # wrong side of current price — price moved past the zone since confirmation; not a usable stop
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
            if stop_ref is not None and stop_ref <= snap.price:
                stop_ref = None  # wrong side of current price — same reasoning as the BUY case
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
                         take_profit2: float | None = None, whale_bias: str | None = None,
                         whale_notional: float | None = None) -> str:
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
    if whale_bias:
        agrees = (whale_bias == "bullish" and signal == "BUY") or (whale_bias == "bearish" and signal == "SELL")
        tag = "🐋 Large orders agree with this signal" if agrees else f"🐋 Large orders leaning {whale_bias} (against this signal)"
        lines.append(f"{tag} (~${whale_notional:,.0f} notional)")
    lines.append("💡 Consider taking partial profit at TP1, trailing the rest toward TP2.")
    lines.append("⚠️ ATR/structure-based levels, not financial advice — confirm before trading.")
    return "\n".join(lines)


def format_tp_hit_alert(symbol: str, direction: str, which: str, level: float, price: float) -> str:
    emoji = "🎯"
    label = "TP1" if which == "tp1" else "TP2 (final target)"
    note = "Consider closing the remainder here." if which == "tp2" else "Consider taking partial profit and trailing the rest."
    return (f"{emoji} <b>{label} hit — {symbol}</b>\n"
            f"Direction: {direction}\n"
            f"Target: {_fmt_price(level)}\n"
            f"Price: {_fmt_price(price)}\n"
            f"{note}")


def format_bos_alert(symbol: str, direction: str, price: float) -> str:
    emoji = "🔺" if direction == "bullish" else "🔻"
    return f"{emoji} <b>BOS ({direction.upper()}) — {symbol}</b>\nPrice: {price:.6f}" if price < 1 else \
           f"{emoji} <b>BOS ({direction.upper()}) — {symbol}</b>\nPrice: {price:.4f}"


def format_trail_alert(symbol: str, direction: str, new_stop: float, price: float) -> str:
    return (f"🔧 <b>Trail stop — {symbol}</b>\n"
            f"New stop: {_fmt_price(new_stop)}\n"
            f"Current price: {_fmt_price(price)}\n"
            f"Structure has advanced in your favor — consider moving your stop up to lock in gains.")


def format_close_alert(symbol: str, direction: str, reason: str, price: float, stop: float) -> str:
    if reason == "stopped_out":
        return (f"⛔ <b>Stopped out — {symbol}</b>\n"
                f"Direction: {direction}\n"
                f"Price: {_fmt_price(price)} hit stop {_fmt_price(stop)}\n"
                f"This setup is no longer tracked.")
    else:
        return (f"🚩 <b>Trend reversed — {symbol}</b>\n"
                f"Direction: {direction}\n"
                f"Price: {_fmt_price(price)}\n"
                f"SuperTrend flipped against the {direction} setup — no longer tracked. "
                f"Consider closing or reassessing if you're still in this trade.")

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
FAILED_SYMBOLS: dict[str, str] = {}  # symbol -> last failure reason (insufficient history, request error, etc.)
PROCESS_STARTED_AT = time.time()  # if this resets unexpectedly between checks, the process restarted
UNIVERSE: list[str] = []
SCAN_META = {"last_full_cycle_started": None, "cycle_count": 0, "universe_size": 0}
BINANCE_SYMBOLS: set[str] | None = None  # cached filter set — None until first successful fetch
BINANCE_SYMBOLS_FETCHED_AT: float | None = None


async def refresh_binance_filter():
    global BINANCE_SYMBOLS, BINANCE_SYMBOLS_FETCHED_AT
    if not BINANCE_FILTER_ENABLED:
        return
    if BINANCE_SYMBOLS_FETCHED_AT and (time.time() - BINANCE_SYMBOLS_FETCHED_AT) < BINANCE_FILTER_REFRESH_SECONDS:
        return  # still fresh, this is just a symbol list — no need to hit Binance often
    new_set = await fetch_binance_symbol_set()
    if new_set:
        BINANCE_SYMBOLS = new_set
        BINANCE_SYMBOLS_FETCHED_AT = time.time()
        log.info(f"Binance symbol filter refreshed: {len(new_set)} symbols")
    # on failure, BINANCE_SYMBOLS simply keeps its last-known value (or stays
    # None if we've never successfully fetched it yet) — never raises


async def refresh_universe():
    global UNIVERSE
    await refresh_binance_filter()
    symbols = await get_all_perpetual_symbols()
    if symbols:
        if BINANCE_FILTER_ENABLED and BINANCE_SYMBOLS:
            filtered = sorted(set(symbols) & BINANCE_SYMBOLS)
            log.info(f"Universe refreshed: {len(symbols)} Bybit pairs -> {len(filtered)} also listed on Binance")
            UNIVERSE = filtered
        else:
            log.info(f"Universe refreshed: {len(symbols)} Bybit pairs (no Binance filter applied yet)")
            UNIVERSE = symbols
        SCAN_META["universe_size"] = len(UNIVERSE)


async def process_symbol(symbol: str, ticker_24h: dict | None):
    try:
        snap = await build_snapshot(symbol, ticker_24h)
    except Exception as e:
        log.warning(f"{symbol}: snapshot failed: {e}")
        FAILED_SYMBOLS[symbol] = str(e)[:200]
        return
    if snap is None:
        FAILED_SYMBOLS[symbol] = "insufficient candle history"
        return

    FAILED_SYMBOLS.pop(symbol, None)
    STATE[symbol] = snap.to_dict()

    # ---- fire Telegram alerts only on state change (no spam every cycle) ----
    prev_signal = LAST_SIGNAL.get(symbol, "NEUTRAL")
    if snap.signal != prev_signal and snap.signal in ("BUY", "SELL"):
        text = format_signal_alert(symbol, snap.signal, snap.price, snap.rsi, snap.long_pct,
                                    snap.open_interest, snap.stop_loss, snap.take_profit, snap.take_profit2,
                                    snap.whale_bias, snap.whale_notional)
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
        ACTIVE_TRADES[symbol] = {
            "direction": snap.signal, "trail_stop": snap.stop_loss,
            "tp1": snap.take_profit, "tp2": snap.take_profit2,
            "tp1_hit": False, "tp2_hit": False,
        }
    elif symbol in ACTIVE_TRADES:
        trade = ACTIVE_TRADES[symbol]
        direction = trade["direction"]

        # ---- TP1 / TP2 hit alerts — checked every cycle a trade is tracked,
        # regardless of whether trailing itself has anything to update ----
        tp1_reached = (
            (direction == "BUY" and trade["tp1"] is not None and snap.price >= trade["tp1"])
            or (direction == "SELL" and trade["tp1"] is not None and snap.price <= trade["tp1"])
        )
        if tp1_reached and not trade["tp1_hit"]:
            trade["tp1_hit"] = True
            await send_alert(format_tp_hit_alert(symbol, direction, "tp1", trade["tp1"], snap.price))
        tp2_reached = (
            (direction == "BUY" and trade["tp2"] is not None and snap.price >= trade["tp2"])
            or (direction == "SELL" and trade["tp2"] is not None and snap.price <= trade["tp2"])
        )
        if tp2_reached and not trade["tp2_hit"]:
            trade["tp2_hit"] = True
            await send_alert(format_tp_hit_alert(symbol, direction, "tp2", trade["tp2"], snap.price))
            del ACTIVE_TRADES[symbol]  # final target hit — nothing left to trail
            return

        trend_ok = (
            (direction == "BUY" and snap.supertrend_trend == 1)
            or (direction == "SELL" and snap.supertrend_trend == -1)
        )
        stopped_out = (
            (direction == "BUY" and snap.price <= trade["trail_stop"])
            or (direction == "SELL" and snap.price >= trade["trail_stop"])
        )
        if stopped_out:
            await send_alert(format_close_alert(symbol, direction, "stopped_out", snap.price, trade["trail_stop"]))
            del ACTIVE_TRADES[symbol]
        elif not trend_ok:
            await send_alert(format_close_alert(symbol, direction, "trend_flip", snap.price, trade["trail_stop"]))
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
            "failed_count": len(FAILED_SYMBOLS),
            "rows": rows,
        }
    except Exception as e:
        log.exception("get_scanner failed")
        return {"meta": SCAN_META, "tracked": len(STATE), "universe": len(UNIVERSE),
                "failed_count": len(FAILED_SYMBOLS), "rows": [], "error": str(e)}


@app.get("/api/symbol/{symbol}")
async def get_symbol(symbol: str):
    return STATE.get(symbol.upper(), {"error": "not tracked yet"})


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "tracked": len(STATE),
        "universe": len(UNIVERSE),
        "failed_count": len(FAILED_SYMBOLS),
        "failed_sample": dict(list(FAILED_SYMBOLS.items())[:20]),
        "process_uptime_seconds": round(time.time() - PROCESS_STARTED_AT, 1),
        "last_rate_limit": {
            **LAST_RATE_LIMIT,
            "seconds_ago": round(time.time() - LAST_RATE_LIMIT["at"], 1) if LAST_RATE_LIMIT["at"] else None,
        },
        "binance_filter": {
            "enabled": BINANCE_FILTER_ENABLED,
            "symbol_count": len(BINANCE_SYMBOLS) if BINANCE_SYMBOLS else 0,
            "fetched_seconds_ago": round(time.time() - BINANCE_SYMBOLS_FETCHED_AT, 1) if BINANCE_SYMBOLS_FETCHED_AT else None,
        },
        "scan_meta": SCAN_META,
    }


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

<footer>Data: Bybit USDT Perpetuals (filtered to pairs also listed on Binance) · Signals are informational, not financial advice</footer>

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
      <div class="detail-row"><span>Large orders</span><span>${r.whale_bias ? (r.whale_bias.charAt(0).toUpperCase()+r.whale_bias.slice(1)) + ' (~$' + fmtCompact(r.whale_notional) + ')' : "—"}</span></div>
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
      `tracking ${data.tracked}/${data.universe} pairs${data.failed_count ? ' · ' + data.failed_count + ' failed' : ''} · cycle #${data.meta.cycle_count}`;
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
