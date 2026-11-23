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
import re
import time
import html as html_lib
import textwrap
import zlib
from datetime import datetime, timedelta, timezone
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
                       "1h": "60", "2h": "120", "4h": "240", "1d": "D", "1w": "W"}


def _interval_to_hours(interval: str) -> float:
    """Used to convert a fixed time window (e.g. 8h) into a bar count for
    whichever SIGNAL_INTERVAL is actually configured, instead of assuming 4h."""
    try:
        if interval.endswith("d"):
            return float(interval[:-1]) * 24
        if interval.endswith("h"):
            return float(interval[:-1])
        if interval.endswith("m"):
            return float(interval[:-1]) / 60
    except (ValueError, AttributeError):
        pass
    return 4.0  # fallback matching the historical default

# ---- Binance (OPTIONAL symbol filter — off by default. Was used to
# restrict to "also listed on Binance" back when trades were placed there;
# now that trading has moved to Bybit directly, this isn't needed. Left in
# place, disabled, in case it's ever wanted again — never used as a data
# source either way.) ----
BINANCE_FAPI_BASE = "https://fapi.binance.com"
BINANCE_FILTER_ENABLED = os.getenv("BINANCE_FILTER_ENABLED", "false").lower() == "true"
BINANCE_FILTER_REFRESH_SECONDS = int(os.getenv("BINANCE_FILTER_REFRESH_SECONDS", "86400"))  # once/day — this is just a symbol list, not live data

# ---- Non-crypto exclusion (Bybit's linear-perpetual universe also includes
# "TradFi Perpetuals" — USDT-settled contracts tracking stocks, ETFs, and
# commodities like TSLA, NVDA, QQQ, XAU. These trade on the same order book
# as crypto perpetuals with no distinguishing API field, so this is a
# maintained exclusion list rather than a clean filter — expect to add to it
# as Bybit lists new ones.) ----
NON_CRYPTO_BASE_COINS = {
    # Big tech / semis
    "TSLA", "NVDA", "META", "GOOGL", "GOOG", "MSFT", "AAPL", "INTC", "TSM", "MU", "ORCL",
    "AMZN", "AMD", "NFLX", "AVGO", "QCOM", "CSCO", "IBM", "ADBE", "CRM", "PYPL",
    # Crypto-adjacent equities
    "MSTR", "COIN", "CRCL", "HOOD",
    # ETFs / indices
    "QQQ", "SPY", "EWJ", "EWY", "DIA",
    # Metals / commodities
    "XAU", "XAG", "WTI", "BRENT",
    # Misc TradFi listings observed
    "DJT", "MCD", "XLP",
}

# ---- Telegram ----
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # user or channel/group id

# ---- Narrative alerts / LLM / daily pulse ----
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")          # optional — without it the built-in templates are used
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
AUTO_ALERT_NARRATIVE = os.getenv("AUTO_ALERT_NARRATIVE", "1") == "1"  # send chart + analysis after each BUY/SELL alert
PULSE_HOUR_WAT = int(os.getenv("PULSE_HOUR_WAT", "12"))         # daily Market Pulse hour, Nigeria time (WAT = UTC+1)
WAT = timezone(timedelta(hours=1))

# ---- Universe ----
QUOTE_ASSET = os.getenv("QUOTE_ASSET", "USDT")          # only USDT-margined pairs
ONLY_PERPETUAL = True
EXCLUDE_SYMBOLS = set(os.getenv("EXCLUDE_SYMBOLS", "").split(",")) if os.getenv("EXCLUDE_SYMBOLS") else set()

# ---- Timeframes used for indicator calc (mirrors the 4H/1H setup you trade) ----
SIGNAL_INTERVAL = os.getenv("SIGNAL_INTERVAL", "4h")     # candle interval for signal calc — 4h reduces noise vs 1h across BOS, retracement, everything downstream
KLINE_LIMIT = int(os.getenv("KLINE_LIMIT", "150"))        # candles fetched per symbol
LS_RATIO_PERIOD = os.getenv("LS_RATIO_PERIOD", SIGNAL_INTERVAL)  # period for long/short ratio endpoint — follows SIGNAL_INTERVAL for consistency

# ---- Indicator params ----
EMA_FAST = int(os.getenv("EMA_FAST", "20"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "50"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "6"))            # matches your screenshot (RSI 6)
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "88"))  # RSI(6) pins high during real trend moves — 80 was excluding genuine breakouts
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "12"))
VWAP_BAND_PCT = float(os.getenv("VWAP_BAND_PCT", "0.3"))  # price allowed to sit this far (%) on the "wrong" side of VWAP and still count as aligned — was a strict cross before
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
PIVOT_LOOKBACK = int(os.getenv("PIVOT_LOOKBACK", "3"))    # bars each side for swing high/low — was 5; 3 matches the spec's significant-high/low definition and finds structure more often
FVG_MIN_GAP_PCT = float(os.getenv("FVG_MIN_GAP_PCT", "0.05"))  # % gap to count as a real FVG

# ---- Entry confirmation pipeline: structure -> BOS (on a CLOSED candle) ->
# retracement into the impulse zone -> a confirming candle. This trades
# fewer, later entries for a materially better entry price than reacting to
# the breakout candle itself.
RETRACEMENT_LOOKBACK_BARS = int(os.getenv("RETRACEMENT_LOOKBACK_BARS", "18"))  # how far back to look for the BOS that started the current move
MIN_RETRACEMENT_GAP_BARS = int(os.getenv("MIN_RETRACEMENT_GAP_BARS", "2"))  # minimum candles between the BOS break and the confirming candle — without this, "confirmation" could fire on the very next candle, which isn't a real pullback
MIN_PULLBACK_DEPTH_ATR = float(os.getenv("MIN_PULLBACK_DEPTH_ATR", "0.3"))  # price must actually retrace at least this many ATRs from the post-BOS extreme before we call it "pulled back" — without this, confirmation can fire while the move is still making fresh highs. Was 0.4 — this was the single biggest bottleneck in the funnel (96% attrition from bos_found to retracement_confirmed), loosened modestly.
MIN_CONFIRM_BODY_ATR = float(os.getenv("MIN_CONFIRM_BODY_ATR", "0.15"))  # the confirming candle needs a real body, not an indecisive doji. Was 0.2, loosened alongside the pullback depth above for the same reason.

# ---- Whale / large-trade detection (public trade-tape data — informational
# confluence booster only, never a hard gate) ----
WHALE_TRADE_USD_THRESHOLD = float(os.getenv("WHALE_TRADE_USD_THRESHOLD", "50000"))
WHALE_TRADE_LOOKBACK = int(os.getenv("WHALE_TRADE_LOOKBACK", "50"))  # how many recent trades to scan per symbol

# ---- Triple-touch detection (resistance/support tested 3+ times) ----
TRIPLE_TOUCH_MIN_COUNT = int(os.getenv("TRIPLE_TOUCH_MIN_COUNT", "3"))
TRIPLE_TOUCH_TOLERANCE_PCT = float(os.getenv("TRIPLE_TOUCH_TOLERANCE_PCT", "0.5"))  # how close two touches need to be to count as "the same level"
TRIPLE_TOUCH_LOOKBACK_BARS = int(os.getenv("TRIPLE_TOUCH_LOOKBACK_BARS", "40"))  # candles searched for repeated touches (40 x 4h ≈ 6.5 days)
TRIPLE_TOUCH_MIN_SPAN_BARS = int(os.getenv("TRIPLE_TOUCH_MIN_SPAN_BARS", "10"))  # touches must span at least this many candles — otherwise it's one clustered blob, not 3 distinct tests over time (10 x 4h ≈ 40h)
TRIPLE_TOUCH_MAX_DISTANCE_PCT = float(os.getenv("TRIPLE_TOUCH_MAX_DISTANCE_PCT", "5.0"))  # ignore levels this far from current price — a stale historical level isn't actionable right now

# ---- Post-signal quality gate: hard filters, soft flags, 0-10 scoring ----
# Applied AFTER the core structure->BOS->retracement engine already confirmed
# a BUY/SELL — this is an additional layer, not a replacement.
RSI14_PERIOD = 14
# NOTE on the two RSI14 constants below: these are now the OUTER hard bounds
# only. A tighter BORDERLINE_* zone sits just inside each one — a signal in
# that zone isn't auto-discarded, but it must show materially stronger whale
# + fund-flow confirmation to survive (see evaluate_signal_quality). This
# replaces the old single-cutoff behavior, which discarded some setups at
# 68/32 that were otherwise well-confirmed.
HARD_RSI14_BUY_MAX = float(os.getenv("HARD_RSI14_BUY_MAX", "70"))   # discard BUY outright if RSI(14) is this extended, no exceptions
HARD_RSI14_SELL_MIN = float(os.getenv("HARD_RSI14_SELL_MIN", "30"))  # discard SELL outright if RSI(14) is this extended, no exceptions (mirror of the BUY case)
BORDERLINE_RSI14_BUY_MIN = float(os.getenv("BORDERLINE_RSI14_BUY_MIN", "68"))   # RSI14 in [this, HARD_RSI14_BUY_MAX) requires whale/fund confirmation to pass
BORDERLINE_RSI14_SELL_MAX = float(os.getenv("BORDERLINE_RSI14_SELL_MAX", "32"))  # RSI14 in (HARD_RSI14_SELL_MIN, this] requires whale/fund confirmation to pass
BORDERLINE_WHALE_LS_BUY_MIN = float(os.getenv("BORDERLINE_WHALE_LS_BUY_MIN", "2.0"))   # required whale buy:sell ratio to accept a borderline-RSI BUY
BORDERLINE_WHALE_LS_SELL_MAX = float(os.getenv("BORDERLINE_WHALE_LS_SELL_MAX", "0.6"))  # required whale buy:sell ratio ceiling to accept a borderline-RSI SELL
MIN_TP_GAP_PCT = float(os.getenv("MIN_TP_GAP_PCT", "0.5"))   # TP2 must clear TP1 by at least this % of entry price, else TP2 is dropped (soft — see TP_SEPARATION_* for the hard reject)
MIN_TP1_RR = float(os.getenv("MIN_TP1_RR", "1.5"))           # TP1 reward:risk below this discards the signal entirely
TP_SEPARATION_HARD_MIN_PCT = float(os.getenv("TP_SEPARATION_HARD_MIN_PCT", "1.0"))  # |TP2-TP1|/TP1 below this % = malformed signal, hard reject
TP_SEPARATION_SOFT_MIN_PCT = float(os.getenv("TP_SEPARATION_SOFT_MIN_PCT", "2.0"))  # below this % (but above the hard floor) = weak trailing room, flag + score penalty
OI_DECLINE_FLAG_PCT = float(os.getenv("OI_DECLINE_FLAG_PCT", "15"))     # flag if OI has dropped this much vs 30 days ago
OI_CONTRADICTION_DIFF_PCT = float(os.getenv("OI_CONTRADICTION_DIFF_PCT", "50"))  # if the ticker-derived and OI-history-derived 30d change disagree by more than this many points, trust the OI-history value and flag it
FUNDING_CROWDED_LONG_PCT = float(os.getenv("FUNDING_CROWDED_LONG_PCT", "0.05"))  # funding rate (%) above this = crowded long
FUNDING_SHORT_PRESSURE_PCT = float(os.getenv("FUNDING_SHORT_PRESSURE_PCT", "-0.01"))  # funding rate (%) below this = short pressure
FUNDING_HISTORY_LIMIT = int(os.getenv("FUNDING_HISTORY_LIMIT", "90"))  # periods of funding-rate history fetched to detect a locked/illiquid rate
FUNDING_LOCK_HARD_MAX_UNIQUE = int(os.getenv("FUNDING_LOCK_HARD_MAX_UNIQUE", "2"))  # at most this many distinct funding values across the whole history = illiquid, hard reject
FUNDING_LOCK_SOFT_MAX_UNIQUE = int(os.getenv("FUNDING_LOCK_SOFT_MAX_UNIQUE", "5"))  # at most this many distinct values = low variance, flag + score penalty
LS_CROWDED_PCT = float(os.getenv("LS_CROWDED_PCT", "65"))    # long% above this = crowded
LS_SQUEEZE_PCT = float(os.getenv("LS_SQUEEZE_PCT", "70"))    # long% above this = squeeze risk (supersedes "crowded")
VOLUME_LOW_PCT = float(os.getenv("VOLUME_LOW_PCT", "80"))    # flag if 24h volume is below this % of the 7-day average
RESISTANCE_PROXIMITY_PCT = float(os.getenv("RESISTANCE_PROXIMITY_PCT", "1.5"))  # entry within this % of daily resistance/support gets flagged
WHALE_LS_WEAK_THRESHOLD = float(os.getenv("WHALE_LS_WEAK_THRESHOLD", "1.2"))  # whale buy:sell ratio below this on a BUY = weak conviction (soft flag)
WHALE_HARD_BLOCK_BUY_MIN = float(os.getenv("WHALE_HARD_BLOCK_BUY_MIN", "0.8"))   # whale buy:sell ratio below this on a BUY = hard reject, not just a flag
WHALE_HARD_BLOCK_SELL_MAX = float(os.getenv("WHALE_HARD_BLOCK_SELL_MAX", "1.3"))  # whale buy:sell ratio above this on a SELL = hard reject, not just a flag
PRICE_PAST_ENTRY_PCT = float(os.getenv("PRICE_PAST_ENTRY_PCT", "0.6"))  # current price this far (%) beyond the confirmed retracement-zone entry = stale, reject
# ---- core confirmation gate: ALL of these must hold for the direction in
# question, checked before anything else, or the signal is skipped outright.
# Tighter/stricter than the individual soft checks further down (which stay
# in place as informational flags/scoring on top of signals that clear this). ----
CORE_GATE_WHALE_BUY_MIN = float(os.getenv("CORE_GATE_WHALE_BUY_MIN", "1.32"))     # whale buy:sell ratio must exceed this for a BUY (unless the volume-surge override below applies)
CORE_GATE_WHALE_SELL_MAX = float(os.getenv("CORE_GATE_WHALE_SELL_MAX", "0.76"))   # whale buy:sell ratio must be below this for a SELL (unless the volume-surge override below applies)
CORE_GATE_PRICE_CHG_MIN_PCT = float(os.getenv("CORE_GATE_PRICE_CHG_MIN_PCT", "2.1"))    # min |24h % change| required, both directions
CORE_GATE_PRICE_CHG_MAX_PCT = float(os.getenv("CORE_GATE_PRICE_CHG_MAX_PCT", "14.0"))   # max |24h % change| allowed, both directions — overextension, unconditional reject regardless of fallback
CORE_GATE_8H_CHANGE_MIN_PCT = float(os.getenv("CORE_GATE_8H_CHANGE_MIN_PCT", "2.0"))    # fallback momentum: min |8h % change| required when the 24h primary check misses
CORE_GATE_1H_CHANGE_MIN_PCT = float(os.getenv("CORE_GATE_1H_CHANGE_MIN_PCT", "0.5"))    # fallback momentum: min |1h % change| required alongside the 8h check
ENTRY_BOS_MAX_BARS_AGO = int(os.getenv("ENTRY_BOS_MAX_BARS_AGO", str(RETRACEMENT_LOOKBACK_BARS)))  # BOS older than this many candles is stale. NOTE: retracement_entry() enforces its own minimum gap (MIN_RETRACEMENT_GAP_BARS, currently 3) between the BOS and the confirming candle, and searches no further back than RETRACEMENT_LOOKBACK_BARS — so bars_ago is architecturally always in [MIN_RETRACEMENT_GAP_BARS, RETRACEMENT_LOOKBACK_BARS]. This MUST stay >= RETRACEMENT_LOOKBACK_BARS or it will silently reject nearly everything the retracement engine is designed to confirm (a value of 3 here, for instance, only ever lets bars_ago==3 through).
CORE_GATE_VOLUME_MIN_PCT = float(os.getenv("CORE_GATE_VOLUME_MIN_PCT", "50"))    # 24h volume must exceed this % of the 7d average (was 70 — loosened)
VOLUME_SURGE_MIN_PCT = float(os.getenv("VOLUME_SURGE_MIN_PCT", "150"))  # 24h volume at least this % of the 7d avg counts as a "surge"
VOLUME_SURGE_MOMENTUM_MIN_PCT = float(os.getenv("VOLUME_SURGE_MOMENTUM_MIN_PCT", "3.0"))  # min |24h % change|, same direction as the trade, required alongside a volume surge for it to substitute for a missed whale-ratio check
CORE_GATE_OI_24H_MIN_PCT = float(os.getenv("CORE_GATE_OI_24H_MIN_PCT", "-15"))   # OI 24h change must exceed this (i.e. not have collapsed past it) — missing data does NOT reject, only flags ⚠️ unverified
SIGNAL_SCORE_ALERT_MIN = int(os.getenv("SIGNAL_SCORE_ALERT_MIN", "5"))   # below this, suppress entirely (was 7 — now 5–6 can pass)
SIGNAL_SCORE_GREEN_MIN = int(os.getenv("SIGNAL_SCORE_GREEN_MIN", "8"))   # at/above this, 🟢 tag instead of 🟡

DAILY_CONTEXT_CACHE_SECONDS = int(os.getenv("DAILY_CONTEXT_CACHE_SECONDS", "43200"))  # 12h — OI history, funding history, and daily structure change slowly, no need to refetch every cycle

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


def _is_crypto_instrument(item: dict) -> bool:
    """Filters out Bybit's TradFi perpetuals (stocks/ETFs/commodities) and
    tokenized-equity-style listings, which share the same linear-perpetual
    category as crypto with no clean distinguishing API field."""
    base = (item.get("baseCoin") or "").upper()
    symbol = (item.get("symbol") or "").upper()
    if base in NON_CRYPTO_BASE_COINS:
        return False
    if "STOCK" in base or "STOCK" in symbol:  # catches xStock-style tokenized equities (e.g. PENGSTOCK)
        return False
    return True


async def get_all_perpetual_symbols() -> list[str]:
    """All actively trading USDT linear crypto perpetual symbols on Bybit
    (TradFi/stock perpetuals excluded — see _is_crypto_instrument)."""
    result = await _get("/v5/market/instruments-info", {"category": "linear", "status": "Trading", "limit": 1000})
    if not result:
        return []
    symbols = []
    cursor = result.get("nextPageCursor")
    for item in result.get("list", []):
        if item.get("quoteCoin") == QUOTE_ASSET and item.get("contractType") == "LinearPerpetual" and _is_crypto_instrument(item):
            symbols.append(item["symbol"])
    # paginate if Bybit says there's more (universe is a few hundred symbols, one extra page covers it)
    if cursor:
        result2 = await _get("/v5/market/instruments-info",
                              {"category": "linear", "status": "Trading", "limit": 1000, "cursor": cursor})
        if result2:
            for item in result2.get("list", []):
                if item.get("quoteCoin") == QUOTE_ASSET and item.get("contractType") == "LinearPerpetual" and _is_crypto_instrument(item):
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


async def get_open_interest_history(symbol: str, interval_time: str = "1d", limit: int = 35):
    result = await _get("/v5/market/open-interest", {"category": "linear", "symbol": symbol,
                                                       "intervalTime": interval_time, "limit": limit})
    if not result:
        return []
    return result.get("list", [])


async def get_funding_rate_history(symbol: str, limit: int = None):
    """Used to detect a funding rate that's locked/flat across many periods
    (a sign of an illiquid or thinly-quoted contract) — separate from the
    single current funding_rate already carried on the ticker."""
    limit = limit or FUNDING_HISTORY_LIMIT
    result = await _get("/v5/market/funding/history", {"category": "linear", "symbol": symbol,
                                                         "limit": min(limit, 200)})
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
    """Bybit V5 kline rows -> tidy OHLCV DataFrame.
    Each row: [startTime, open, high, low, close, volume, turnover] (7 fields —
    Bybit's format, notably shorter than Binance's old 12-field one)."""
    cols = ["open_time", "open", "high", "low", "close", "volume", "turnover"]
    df = pd.DataFrame(klines, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"].astype(np.int64), unit="ms")
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
                       lookback: int = None) -> tuple[bool, float | None, float | None, int | None, str]:
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
    Returns (confirmed, zone_low, zone_high, bos_bars_ago, stage) — bos_bars_ago
    tells the caller how many candles back the actual break happened, since
    by design it's never the same candle as the confirmation. stage is a
    funnel-diagnostic label: "no_bos" | "gap_too_small" | "no_zone" |
    "no_touch" | "not_confirmed" | "confirmed"."""
    lookback = lookback or RETRACEMENT_LOOKBACK_BARS
    n = len(df)
    last_closed = n - 2
    if last_closed < 4:
        return False, None, None, None, "no_bos"

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
        return False, None, None, None, "no_bos"

    # Require real separation between the break and the "pullback" — without
    # this, the very next candle could satisfy everything, which isn't a
    # pullback at all, just noise within one candle of the breakout.
    if last_closed - bos_idx < MIN_RETRACEMENT_GAP_BARS:
        return False, None, None, None, "gap_too_small"

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
        return False, None, None, None, "no_zone"

    touched = False
    for k in range(bos_idx + 1, last_closed + 1):
        if df["low"].iloc[k] <= zone_high and df["high"].iloc[k] >= zone_low:
            touched = True
            break
    if not touched:
        return False, zone_low, zone_high, None, "no_touch"

    bars_ago = last_closed - bos_idx
    last = df.iloc[last_closed]
    atr_val = float(df["atr"].iloc[last_closed]) if "atr" in df.columns and pd.notna(df["atr"].iloc[last_closed]) else None

    if direction == "bullish":
        confirmed = bool(last["low"] <= zone_high and last["close"] > last["open"] and last["close"] >= zone_low)
        if confirmed and atr_val:
            # Must have genuinely paused/reversed from the post-BOS extreme —
            # without this, confirmation can fire while price is still
            # making fresh highs, which isn't a pullback at all.
            impulse_high = float(df["high"].iloc[bos_idx + 1:last_closed + 1].max())
            pulled_back = (impulse_high - float(last["close"])) >= atr_val * MIN_PULLBACK_DEPTH_ATR
            confirm_body = abs(float(last["close"]) - float(last["open"]))
            has_real_body = confirm_body >= atr_val * MIN_CONFIRM_BODY_ATR
            confirmed = confirmed and pulled_back and has_real_body
    else:
        confirmed = bool(last["high"] >= zone_low and last["close"] < last["open"] and last["close"] <= zone_high)
        if confirmed and atr_val:
            impulse_low = float(df["low"].iloc[bos_idx + 1:last_closed + 1].min())
            pulled_back = (float(last["close"]) - impulse_low) >= atr_val * MIN_PULLBACK_DEPTH_ATR
            confirm_body = abs(float(last["close"]) - float(last["open"]))
            has_real_body = confirm_body >= atr_val * MIN_CONFIRM_BODY_ATR
            confirmed = confirmed and pulled_back and has_real_body
    stage = "confirmed" if confirmed else "not_confirmed"
    return confirmed, zone_low, zone_high, (bars_ago if confirmed else None), stage


def detect_triple_touch(df: pd.DataFrame, is_high: pd.Series, is_low: pd.Series,
                         direction: str, current_price: float = None, lookback: int = None) -> tuple[float | None, int]:
    """Finds the price level (from confirmed swing highs/lows in the recent
    window) that's been touched the most times within a small tolerance band
    — a classic triple-top (resistance) or triple-bottom (support) setup.
    Rejects clustered "blob" touches that all happened within a short span
    (that's one consolidation, not 3 distinct tests over time) and levels
    too far from current price to be actionable right now.
    Returns (level, touch_count); level is None if nothing qualifies."""
    lookback = lookback or TRIPLE_TOUCH_LOOKBACK_BARS
    n = len(df)
    last_closed = n - 2
    if last_closed < 5:
        return None, 0
    start = max(0, last_closed - lookback)

    mask = (is_high if direction == "resistance" else is_low).copy()
    mask.iloc[:start] = False
    mask.iloc[last_closed + 1:] = False
    col = "high" if direction == "resistance" else "low"
    idx = np.where(mask.values)[0]
    values = df[col].values[idx]
    if len(values) < TRIPLE_TOUCH_MIN_COUNT:
        return None, 0

    best_level, best_count, best_indices = None, 0, None
    for v in values:
        band = abs(v) * TRIPLE_TOUCH_TOLERANCE_PCT / 100
        member_mask = np.abs(values - v) <= band
        count = int(np.sum(member_mask))
        if count > best_count:
            best_count, best_level = count, float(v)
            best_indices = idx[member_mask]

    if best_count < TRIPLE_TOUCH_MIN_COUNT:
        return None, 0

    span = int(best_indices.max() - best_indices.min())
    if span < TRIPLE_TOUCH_MIN_SPAN_BARS:
        return None, 0  # clustered blob, not genuinely tested over time

    if current_price:
        distance_pct = abs(best_level - current_price) / current_price * 100
        if distance_pct > TRIPLE_TOUCH_MAX_DISTANCE_PCT:
            return None, 0  # too far from price to be actionable right now

    return best_level, best_count


def classify_bos_quality(df: pd.DataFrame, is_high: pd.Series, is_low: pd.Series,
                          bos_idx: int, direction: str, whale_bias: str | None,
                          lookback: int = 15) -> bool:
    """Not every structural break is worth an alert. A good one looks like:
    price was declining/ranging beforehand (a genuine reversal or a breakout
    from consolidation — not just another push within an already-established
    trend), it has real room to run before hitting the next opposing level
    (not immediately running into resistance), and large orders aren't
    actively fighting it. This is what separates a real setup from noise."""
    start = max(0, bos_idx - lookback)
    if bos_idx - start < 5:
        return False  # not enough prior context to judge

    atr_val = df["atr"].iloc[bos_idx] if "atr" in df.columns else None
    if atr_val is None or pd.isna(atr_val) or atr_val <= 0:
        return False

    pre_range = df["high"].iloc[start:bos_idx].max() - df["low"].iloc[start:bos_idx].min()
    is_consolidating = pre_range <= atr_val * 2.5  # tight range beforehand = a real base, not a runaway trend

    pre_ema_fast = df["ema_fast"].iloc[start:bos_idx]
    pre_ema_slow = df["ema_slow"].iloc[start:bos_idx]
    break_price = float(df["close"].iloc[bos_idx])

    if direction == "bullish":
        already_uptrending = (pre_ema_fast > pre_ema_slow).mean() >= 0.8
        if already_uptrending and not is_consolidating:
            return False  # continuation noise in an existing trend, not a real reversal/base breakout
        target = nearest_structure_level(df, is_high, is_low, break_price, "above")
        if target is not None and (target - break_price) < atr_val * 1.0:
            return False  # breaking straight into nearby resistance — not enough room to run
        if whale_bias == "bearish":
            return False  # large sell orders actively fighting this breakout
    else:
        already_downtrending = (pre_ema_fast < pre_ema_slow).mean() >= 0.8
        if already_downtrending and not is_consolidating:
            return False
        target = nearest_structure_level(df, is_high, is_low, break_price, "below")
        if target is not None and (break_price - target) < atr_val * 1.0:
            return False
        if whale_bias == "bullish":
            return False

    return True


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
    if sell_notional > 0:
        whale_ls_ratio = buy_notional / sell_notional
    elif buy_notional > 0:
        whale_ls_ratio = 99.0  # effectively "all buys, no sells" — capped so it stays JSON-safe (not literal infinity)
    else:
        whale_ls_ratio = None
    return {"whale_bias": bias, "whale_notional": total, "whale_ls_ratio": whale_ls_ratio}


def detect_order_flow(trades: list, price: float) -> dict:
    """Taker-style order flow from the full recent trade tape (not only whales).
    Bybit recent-trade `side` is the aggressor side on linear perps for this endpoint
    in practice — used as a proxy for who is lifting/hitting the book right now."""
    buy_notional = 0.0
    sell_notional = 0.0
    buy_count = 0
    sell_count = 0
    for t in trades:
        try:
            sz = float(t.get("size", 0))
            px = float(t.get("price", price))
            notional = sz * px
            if notional <= 0:
                continue
            side = (t.get("side") or "").lower()
            if side == "buy":
                buy_notional += notional
                buy_count += 1
            elif side == "sell":
                sell_notional += notional
                sell_count += 1
        except (TypeError, ValueError):
            continue
    total = buy_notional + sell_notional
    if total <= 0:
        return {"buy_pct": None, "sell_pct": None, "bias": None, "buy_notional": 0.0,
                "sell_notional": 0.0, "trade_count": 0}
    buy_pct = buy_notional / total * 100
    sell_pct = sell_notional / total * 100
    bias = None
    if buy_pct >= 58:
        bias = "buyers"
    elif sell_pct >= 58:
        bias = "sellers"
    else:
        bias = "mixed"
    return {
        "buy_pct": buy_pct,
        "sell_pct": sell_pct,
        "bias": bias,
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
        "trade_count": buy_count + sell_count,
    }


def describe_candle_character(df: pd.DataFrame, atr_val: float | None, n: int = 3) -> list[dict]:
    """Character of the last n CLOSED candles (excludes the live forming bar)."""
    out = []
    if df is None or len(df) < 3:
        return out
    last_closed = len(df) - 2
    atr_val = atr_val or float(df["atr"].iloc[last_closed]) if "atr" in df.columns else None
    for i in range(max(0, last_closed - n + 1), last_closed + 1):
        o = float(df["open"].iloc[i])
        h = float(df["high"].iloc[i])
        l = float(df["low"].iloc[i])
        c = float(df["close"].iloc[i])
        body = abs(c - o)
        full = max(h - l, 1e-12)
        upper = h - max(o, c)
        lower = min(o, c) - l
        direction = "bullish" if c > o else "bearish" if c < o else "doji"
        body_atr = (body / atr_val) if atr_val and atr_val > 0 else None
        rejection = None
        if lower >= body * 1.8 and lower / full >= 0.45:
            rejection = "lower_wick"
        elif upper >= body * 1.8 and upper / full >= 0.45:
            rejection = "upper_wick"
        strength = "strong" if body_atr is not None and body_atr >= 0.6 else (
            "moderate" if body_atr is not None and body_atr >= 0.25 else "weak"
        )
        out.append({
            "direction": direction,
            "body_atr": body_atr,
            "strength": strength,
            "rejection": rejection,
            "close": c,
            "open": o,
            "high": h,
            "low": l,
        })
    return out


def _last_swing_levels(df: pd.DataFrame, is_high: pd.Series, is_low: pd.Series) -> dict:
    """Most recent confirmed swing high / low on closed candles only."""
    if df is None or len(df) < 5:
        return {"swing_high": None, "swing_low": None}
    last_closed = len(df) - 2
    mask = pd.Series(False, index=df.index)
    mask.iloc[: last_closed + 1] = True
    highs = df.loc[is_high & mask, "high"]
    lows = df.loc[is_low & mask, "low"]
    return {
        "swing_high": float(highs.iloc[-1]) if len(highs) else None,
        "swing_low": float(lows.iloc[-1]) if len(lows) else None,
    }


def trend_indicator_text(st, ema_f, ema_s) -> str | None:
    """e.g. 'SuperTrend bearish · EMA20 below EMA50 (bearish)' — None if data missing."""
    if st not in (1, -1) or ema_f is None or ema_s is None:
        return None
    st_word = "bullish" if st == 1 else "bearish"
    ema_word = "bullish" if ema_f > ema_s else "bearish"
    rel = "above" if ema_f > ema_s else "below"
    agree = "aligned" if st_word == ema_word else "conflicting"
    return (f"SuperTrend {st_word} · EMA{EMA_FAST} {rel} EMA{EMA_SLOW} ({ema_word}) — {agree}")


def _fp(p):
    if p is None:
        return "—"
    return f"{p:.6f}" if p < 1 else f"{p:.4f}"


def _lab(x) -> str:
    """Protect a literal '4h' label from the timeframe relabel done at the end of the formatter."""
    return str(x).replace("4h", "4§h")


def _fmt_t(ms: int, ctf: str) -> str:
    try:
        d = datetime.fromtimestamp(ms / 1000, WAT)
        return d.strftime("%H:%M WAT") if ctf in ("1m", "3m", "5m", "15m", "30m", "1h") else d.strftime("%d %b %H:%M WAT")
    except Exception:
        return "?"


def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ru = up.ewm(alpha=1 / period, adjust=False).mean()
    rd = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ru / rd.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def detect_rsi_divergence(df, is_high, is_low) -> dict | None:
    """Regular and hidden RSI14 divergence between the last two confirmed swing points."""
    if df is None or is_high is None or is_low is None or len(df) < 40:
        return None
    last_closed = len(df) - 2
    rsi = _rsi_series(df["close"])
    lows = [i for i in np.where(is_low.values)[0] if i <= last_closed]
    highs = [i for i in np.where(is_high.values)[0] if i <= last_closed]
    found = []
    if len(lows) >= 2:
        i1, i2 = lows[-2], lows[-1]
        if last_closed - i2 <= 40 and i2 - i1 >= 3:
            p1, p2 = float(df["low"].iloc[i1]), float(df["low"].iloc[i2])
            r1, r2 = float(rsi.iloc[i1]), float(rsi.iloc[i2])
            if p2 < p1 and r2 > r1 + 2:
                found.append({"kind": "bullish", "p1": p1, "p2": p2, "r1": r1, "r2": r2, "i2": i2})
            elif p2 > p1 and r2 < r1 - 2:
                found.append({"kind": "hidden_bullish", "p1": p1, "p2": p2, "r1": r1, "r2": r2, "i2": i2})
    if len(highs) >= 2:
        i1, i2 = highs[-2], highs[-1]
        if last_closed - i2 <= 40 and i2 - i1 >= 3:
            p1, p2 = float(df["high"].iloc[i1]), float(df["high"].iloc[i2])
            r1, r2 = float(rsi.iloc[i1]), float(rsi.iloc[i2])
            if p2 > p1 and r2 < r1 - 2:
                found.append({"kind": "bearish", "p1": p1, "p2": p2, "r1": r1, "r2": r2, "i2": i2})
            elif p2 < p1 and r2 > r1 + 2:
                found.append({"kind": "hidden_bearish", "p1": p1, "p2": p2, "r1": r1, "r2": r2, "i2": i2})
    if not found:
        return None
    best = max(found, key=lambda d: (d["i2"], not d["kind"].startswith("hidden")))
    best["bars_ago"] = last_closed - best["i2"]
    return best


def compute_fib(df, is_high, is_low, price, atr_val) -> dict | None:
    """Fibonacci retracement of the latest confirmed swing leg."""
    if df is None or is_high is None or is_low is None or price is None or not atr_val:
        return None
    last_closed = len(df) - 2
    hi = [i for i in np.where(is_high.values)[0] if i <= last_closed]
    lo = [i for i in np.where(is_low.values)[0] if i <= last_closed]
    if not hi or not lo:
        return None
    ih, il = hi[-1], lo[-1]
    if last_closed - max(ih, il) > 60:
        return None
    H, Lw = float(df["high"].iloc[ih]), float(df["low"].iloc[il])
    rng = H - Lw
    if rng < 2 * atr_val:
        return None
    up = ih > il                      # latest pivot is the high -> leg ran up
    lv = {r: (H - r * rng if up else Lw + r * rng) for r in (0.382, 0.5, 0.618, 0.705, 0.786)}
    retr = (H - price) / rng if up else (price - Lw) / rng
    ext = H + 0.272 * rng if up else Lw - 0.272 * rng
    return {"up": up, "start": Lw if up else H, "end": H if up else Lw, "levels": lv, "retr": retr, "ext_1272": ext}


def equal_levels(df, is_high, is_low, atr_val, lookback: int = 120):
    """Clusters of near-equal swing highs / lows (resting liquidity)."""
    if df is None or not atr_val or is_high is None or is_low is None:
        return [], []
    last_closed = len(df) - 2
    start = max(0, last_closed - lookback)
    tol = 0.25 * atr_val

    def cluster(mask, col):
        vals = sorted(float(df[col].iloc[i]) for i in np.where(mask.values)[0] if start <= i <= last_closed)
        groups, cur = [], []
        for v in vals:
            if cur and v - cur[-1] <= tol:
                cur.append(v)
            else:
                if len(cur) >= 2:
                    groups.append(cur)
                cur = [v]
        if len(cur) >= 2:
            groups.append(cur)
        return [(sum(g) / len(g), len(g)) for g in groups]

    return cluster(is_high, "high"), cluster(is_low, "low")


def key_levels_from_candles(daily_rows, weekly_rows) -> dict:
    out = {}
    try:
        d = [(float(r[1]), float(r[2]), float(r[3])) for r in (daily_rows or [])]
        if len(d) >= 2:
            out["DO"], out["PDH"], out["PDL"] = d[-1][0], d[-2][1], d[-2][2]
        w = [(float(r[1]), float(r[2]), float(r[3])) for r in (weekly_rows or [])]
        if len(w) >= 2:
            out["WO"], out["PWH"], out["PWL"] = w[-1][0], w[-2][1], w[-2][2]
    except Exception:
        pass
    return out


def compute_depth(computed: dict, read: dict, daily_rows, weekly_rows) -> dict:
    df, ih, il = computed.get("df"), computed.get("is_high"), computed.get("is_low")
    price, atr_v = read.get("price"), read.get("atr")
    out = {"div": None, "fib": None, "ref_levels": {}, "eq_highs": [], "eq_lows": []}
    for key, fn in (("div", lambda: detect_rsi_divergence(df, ih, il)),
                    ("fib", lambda: compute_fib(df, ih, il, price, atr_v))):
        try:
            out[key] = fn()
        except Exception as e:
            log.warning("depth %s failed: %s", key, e)
    out["ref_levels"] = key_levels_from_candles(daily_rows, weekly_rows)
    try:
        out["eq_highs"], out["eq_lows"] = equal_levels(df, ih, il, atr_v)
    except Exception as e:
        log.warning("equal levels failed: %s", e)
    return out


def evaluate_trigger(plan: dict, rows, ignore_before_ms: int = 0) -> dict:
    """Live check on the confirmation timeframe: has a rejection candle printed in the entry zone
    and is it holding? rows = oldest-first Bybit klines (last row is the forming candle)."""
    try:
        c = [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in rows]
    except Exception:
        return {"status": "unknown"}
    if len(c) < 5 or not plan:
        return {"status": "unknown"}
    closed, live = c[:-1], c[-1]
    price = live[4]
    long = plan["side"] == "long"
    zlo, zhi, inv = plan["zone_lo"], plan["zone_hi"], plan["invalidation"]
    tp1 = plan["tps"][0] if plan.get("tps") else None
    lvl = zhi if long else zlo
    res = {"price": price, "level": lvl, "last_closed_time": closed[-1][0]}

    if (long and closed[-1][4] < inv) or ((not long) and closed[-1][4] > inv):
        res["status"] = "invalid"
        return res
    if tp1 is not None and ((long and price >= tp1) or ((not long) and price <= tp1)):
        res["status"] = "missed"
        return res

    cands = []
    for i in range(max(0, len(closed) - 5), len(closed)):
        t, o, h, l, cl = closed[i]
        if t <= ignore_before_ms:
            continue
        rng = h - l
        if rng <= 0:
            continue
        body = abs(cl - o)
        if long:
            ok = inv < l <= zhi and cl > zhi and cl > o and ((min(o, cl) - l) / rng >= 0.25 or body / rng >= 0.5)
        else:
            ok = zlo <= h < inv and cl < zlo and cl < o and ((h - max(o, cl)) / rng >= 0.25 or body / rng >= 0.5)
        if ok:
            cands.append(i)
    if cands:
        def broke_after(r):
            return any((x[4] <= zhi) if long else (x[4] >= zlo) for x in closed[r + 1:])
        holding = [r for r in cands if not broke_after(r)]
        r = holding[0] if holding else cands[-1]          # earliest rejection that is still holding
        res["rej_time"], res["rej_close"] = closed[r][0], closed[r][4]
        if not holding:
            res["status"] = "failed"
        else:
            res["status"] = "confirmed" if closed[r + 1:] else "rejection"
        return res

    if long:
        if price > zhi:
            res["status"], res["dist"] = "waiting", (price - zhi) / price * 100
        elif price >= zlo:
            res["status"] = "in_zone"
        else:
            res["status"], res["dist"] = "through", (zlo - price) / price * 100
    else:
        if price < zlo:
            res["status"], res["dist"] = "waiting", (zlo - price) / price * 100
        elif price <= zhi:
            res["status"] = "in_zone"
        else:
            res["status"], res["dist"] = "through", (price - zhi) / price * 100
    return res


def _trigger_text(tr: dict | None, plan: dict | None, ctf: str) -> str | None:
    if not tr or not plan or tr.get("status") in (None, "unknown"):
        return None
    long = plan["side"] == "long"
    c, L = _lab(ctf), _fp(tr.get("level"))
    bull = "bullish" if long else "bearish"
    ab = "above" if long else "below"
    t = _fmt_t(tr["rej_time"], ctf) if tr.get("rej_time") else ""
    st = tr["status"]
    if st == "waiting":
        return f"price {tr['dist']:.1f}% {ab} the zone, no {c} trigger yet"
    if st == "in_zone":
        return f"price in the zone, watching for a {bull} {c} rejection closing {ab} {L}"
    if st == "through":
        return f"price {tr['dist']:.1f}% through the zone, wait for a {c} close back {ab} {L}"
    if st == "rejection":
        return f"{bull} {c} rejection at {t} (close {_fp(tr['rej_close'])}), waiting for the next candle to hold {ab} {L}"
    if st == "confirmed":
        return f"✅ confirmed: {bull} {c} rejection at {t} is holding {ab} {L}"
    if st == "failed":
        return f"the {t} rejection failed (a {c} candle closed back through {L}), wait for a new one"
    if st == "invalid":
        return f"a {c} candle closed beyond invalidation {_fp(plan['invalidation'])}, idea is off"
    if st == "missed":
        return f"price already hit TP1 ({_fp(plan['tps'][0])}), setup missed"
    return None


def depth_lines(read: dict) -> dict:
    """Compact divergence / key levels / Fibonacci / confluence / live-trigger lines (no leading '- ')."""
    fp = _fp
    out = {}
    tf = read.get("tf") or "4h"
    ctf = _CONFIRM_TF.get(tf, "1h")
    price, atr_v = read.get("price"), read.get("atr")

    dv = read.get("div")
    if dv:
        k, ago = dv["kind"], dv["bars_ago"]
        when = f" ({ago} candles ago)" if ago else ""
        pr = f"{fp(dv['p1'])} → {fp(dv['p2'])}"
        rs = f"{dv['r1']:.0f} → {dv['r2']:.0f}"
        out["div"] = {
            "bullish": f"Divergence: bullish. Price lower low ({pr}), RSI14 higher low ({rs}); selling is fading{when}",
            "bearish": f"Divergence: bearish. Price higher high ({pr}), RSI14 lower high ({rs}); buying is fading{when}",
            "hidden_bullish": f"Divergence: hidden bullish. Price higher low ({pr}), RSI14 lower low ({rs}); supports upside continuation{when}",
            "hidden_bearish": f"Divergence: hidden bearish. Price lower high ({pr}), RSI14 higher high ({rs}); supports downside continuation{when}",
        }[k]
    else:
        out["div"] = "Divergence: none on the last two swings"

    ref = read.get("ref_levels") or {}
    levels = [(n, ref[k]) for k, n in (("PDH", "PDH"), ("PDL", "PDL"), ("PWH", "PWH"), ("PWL", "PWL"),
                                       ("DO", "daily open"), ("WO", "weekly open")) if ref.get(k)]
    levels += [(f"eq highs {n}x", v) for v, n in (read.get("eq_highs") or [])]
    levels += [(f"eq lows {n}x", v) for v, n in (read.get("eq_lows") or [])]
    if levels and price:
        above = sorted([(n, v) for n, v in levels if v > price], key=lambda x: x[1])[:2]
        below = sorted([(n, v) for n, v in levels if v <= price], key=lambda x: -x[1])[:2]
        fmt = lambda n, v: f"{n} {fp(v)} ({(v - price) / price * 100:+.1f}%)"
        parts = []
        if above:
            parts.append("above " + ", ".join(fmt(n, v) for n, v in above))
        if below:
            parts.append("below " + ", ".join(fmt(n, v) for n, v in below))
        flags = [(price > ref[k]) for k in ("DO", "WO") if ref.get(k)]
        if len(flags) == 2:
            parts.append("above daily & weekly open (bullish)" if all(flags) else
                         "below daily & weekly open (bearish)" if not any(flags) else
                         f"{'above' if flags[0] else 'below'} daily open, {'above' if flags[1] else 'below'} weekly open (mixed)")
        out["levels"] = "Levels: " + " · ".join(parts)

    fb = read.get("fib")
    if fb:
        lv, r = fb["levels"], fb["retr"]
        pos = ("leg invalid" if r < 0 else "shallow" if r < 0.382 else "toward the midpoint" if r < 0.5 else
               "near the golden pocket" if r < 0.618 else "in the golden pocket" if r <= 0.786 else
               "deep" if r <= 1 else "past the origin")
        out["fib"] = (f"Fib: leg {'up' if fb['up'] else 'down'} {fp(fb['start'])} → {fp(fb['end'])} · {r * 100:.0f}% retraced ({pos}) · "
                      f"0.5 = {fp(lv[0.5])} · golden pocket {fp(min(lv[0.618], lv[0.786]))}-{fp(max(lv[0.618], lv[0.786]))} · "
                      f"1.272 ext = {fp(fb['ext_1272'])}")

    plans = []
    if read.get("plan"):
        plans.append(("", read["plan"], read.get("trigger")))
    else:
        if read.get("plan_bull"):
            plans.append(("bullish", read["plan_bull"], read.get("trigger_bull")))
        if read.get("plan_bear"):
            plans.append(("bearish", read["plan_bear"], read.get("trigger_bear")))
    conf = []
    tol = 0.25 * atr_v if atr_v else 0
    for label, pl, _ in plans:
        names = [n for n, v in levels if pl["zone_lo"] - tol <= v <= pl["zone_hi"] + tol]
        if fb:
            lo_, hi_ = sorted([fb["levels"][0.618], fb["levels"][0.786]])
            if lo_ <= pl["zone_hi"] and hi_ >= pl["zone_lo"]:
                names.append("golden pocket")
        if names:
            conf.append(f"{label or pl['side']} zone overlaps " + ", ".join(names))
    if conf:
        out["confluence"] = "Confluence: " + "; ".join(conf)
    trig = []
    for label, pl, tr in plans:
        tx = _trigger_text(tr, pl, ctf)
        if tx:
            trig.append((label, tx))
    if trig:
        out["trigger"] = (f"Live trigger ({_lab(ctf)}): {trig[0][1]}" if len(trig) == 1 and not trig[0][0]
                          else f"Live trigger ({_lab(ctf)}): " + "; ".join(f"{l} scenario, {t}" for l, t in trig))
    return out


def cap_first(t: str) -> str:
    return t[:1].upper() + t[1:] if t else t


def split_message(text: str, limit: int = 3800) -> list[str]:
    """Split on line boundaries so long reports are never truncated by Telegram's 4096-char cap."""
    if len(text) <= limit:
        return [text]
    chunks, cur = [], ""
    for ln in text.split("\n"):
        if cur and len(cur) + len(ln) + 1 > limit:
            chunks.append(cur.rstrip())
            cur = ""
        cur += ln + "\n"
    if cur.strip():
        chunks.append(cur.rstrip())
    return chunks


def _fvg_zones(df, price, lookback: int = 80):
    """Nearest unfilled fair-value-gap imbalances: demand (bullish gap below price) and
    supply (bearish gap above price), each as (low, high) or None."""
    try:
        if df is None or len(df) < 8 or price is None or "bull_fvg" not in df.columns:
            return None, None
        last_closed = len(df) - 2
        start = max(1, last_closed - lookback)
        demands, supplies = [], []
        for j in range(start, last_closed):          # j = middle candle; j+1 must be a closed candle
            if df["bull_fvg"].iloc[j]:
                lo = float(df["high"].iloc[j - 1]); hi = float(df["low"].iloc[j + 1])
                later = df["low"].iloc[j + 2: last_closed + 1]
                if hi < price and (later.empty or float(later.min()) > lo):
                    demands.append((lo, hi))
            if df["bear_fvg"].iloc[j]:
                hi = float(df["low"].iloc[j - 1]); lo = float(df["high"].iloc[j + 1])
                later = df["high"].iloc[j + 2: last_closed + 1]
                if lo > price and (later.empty or float(later.max()) < hi):
                    supplies.append((lo, hi))
        demand = max(demands, key=lambda z: z[1]) if demands else None
        supply = min(supplies, key=lambda z: z[0]) if supplies else None
        return demand, supply
    except Exception:
        return None, None


def _trade_plan(df, is_high, is_low, price, atr_val, stance, demand=None, supply=None) -> dict | None:
    """Finora-style plan: pullback entry zone (anchored on an unfilled imbalance when one exists,
    else the nearest swing level), structural take-profits and an invalidation level."""
    if df is None or is_high is None or is_low is None or price is None or not atr_val:
        return None
    if stance not in ("bullish", "bearish"):
        return None
    last_closed = len(df) - 2
    mask = pd.Series(False, index=df.index)
    mask.iloc[: last_closed + 1] = True
    highs = sorted({float(x) for x in df.loc[is_high & mask, "high"]})
    lows = sorted({float(x) for x in df.loc[is_low & mask, "low"]})

    def spaced(levels, n):
        out = []
        for lv in levels:
            if not out or abs(lv - out[-1]) > atr_val * 0.5:
                out.append(lv)
            if len(out) >= n:
                break
        return out

    if stance == "bearish":
        above = [h for h in highs if h > price + 0.25 * atr_val]
        if supply and supply[0] > price + 0.2 * atr_val and supply[0] - price <= 8 * atr_val:
            zone_lo = supply[0]
            zone_hi = min(supply[1], supply[0] + 1.5 * atr_val)
            src = "unfilled supply imbalance"
        else:
            anchor = above[0] if above else price + atr_val
            zone_hi = anchor
            zone_lo = max(anchor - 0.4 * atr_val, price + 0.2 * atr_val)
            src = "swing-high"
        beyond = [h for h in above if h > zone_hi + 0.5 * atr_val]
        inv = (beyond[0] if beyond else zone_hi + 1.5 * atr_val) + 0.1 * atr_val
        tps = spaced(sorted([l for l in lows if l < price - 0.5 * atr_val], reverse=True), 3)
        if not tps:
            tps = [price - atr_val * m for m in (1.5, 3.0, 4.5)]
        return {"side": "short", "zone_lo": zone_lo, "zone_hi": zone_hi, "invalidation": inv, "tps": tps, "zone_src": src}

    below_desc = sorted([l for l in lows if l < price - 0.25 * atr_val], reverse=True)
    if demand and demand[1] < price - 0.2 * atr_val and price - demand[1] <= 8 * atr_val:
        zone_hi = demand[1]
        zone_lo = max(demand[0], demand[1] - 1.5 * atr_val)
        src = "unfilled demand imbalance"
    else:
        anchor = below_desc[0] if below_desc else price - atr_val
        zone_lo = anchor
        zone_hi = min(anchor + 0.4 * atr_val, price - 0.2 * atr_val)
        src = "swing-low"
    beyond = [l for l in below_desc if l < zone_lo - 0.5 * atr_val]
    inv = (beyond[0] if beyond else zone_lo - 1.5 * atr_val) - 0.1 * atr_val
    tps = spaced([h for h in highs if h > price + 0.5 * atr_val], 3)
    if not tps:
        tps = [price + atr_val * m for m in (1.5, 3.0, 4.5)]
    return {"side": "long", "zone_lo": zone_lo, "zone_hi": zone_hi, "invalidation": inv, "tps": tps, "zone_src": src}


def build_structure_read(computed: dict, flow: dict, whale: dict,
                         long_pct: float | None, funding_rate: float | None,
                         pct_change_24h: float | None,
                         daily_bias: str | None = None,
                         extra: dict | None = None) -> dict:
    """Assemble a Finora-style structure + order-flow read from existing engine data."""
    df = computed.get("df")
    price = computed.get("price")
    atr_val = computed.get("atr")
    bos = computed.get("bos")
    st = computed.get("supertrend_trend")
    ema_f = computed.get("ema_fast")
    ema_s = computed.get("ema_slow")
    vwap_v = computed.get("vwap")
    candles = describe_candle_character(df, atr_val, n=3)
    swings = _last_swing_levels(df, computed.get("is_high"), computed.get("is_low")) if df is not None else {}

    bullish_trend = st == 1 or (ema_f is not None and ema_s is not None and ema_f > ema_s)
    bearish_trend = st == -1 or (ema_f is not None and ema_s is not None and ema_f < ema_s)
    if bullish_trend and not bearish_trend:
        structure_4h = "bullish"
    elif bearish_trend and not bullish_trend:
        structure_4h = "bearish"
    else:
        structure_4h = "mixed"

    last_c = candles[-1] if candles else None
    active_move = None
    if last_c and last_c["direction"] == "bullish" and structure_4h == "bearish":
        active_move = "rebound_in_bearish_structure"
    elif last_c and last_c["direction"] == "bearish" and structure_4h == "bullish":
        active_move = "pullback_in_bullish_structure"
    elif last_c and last_c["direction"] == structure_4h:
        active_move = "with_structure"

    stance = structure_4h
    # Soften stance when order flow fights structure
    if structure_4h == "bullish" and flow.get("bias") == "sellers":
        stance_note = "Structure bullish but taker flow is sell-dominant — caution on chasing longs."
    elif structure_4h == "bearish" and flow.get("bias") == "buyers":
        stance_note = "Structure bearish but taker flow is buy-dominant — rebound may extend before continuation."
    else:
        stance_note = None

    demand, supply = _fvg_zones(df, price)
    vol_ratio = None
    try:
        lc = len(df) - 2
        base = float(df["volume"].iloc[max(0, lc - 20):lc].mean())
        if base > 0:
            vol_ratio = float(df["volume"].iloc[lc]) / base
    except Exception:
        pass
    range_pos = None
    if swings.get("swing_low") and swings.get("swing_high") and swings["swing_high"] > swings["swing_low"] and price:
        range_pos = (price - swings["swing_low"]) / (swings["swing_high"] - swings["swing_low"])
    is_h, is_l = computed.get("is_high"), computed.get("is_low")
    read = {
        "price": price,
        "structure_4h": structure_4h,
        "daily_bias": daily_bias,
        "bos": bos,
        "swing_high": swings.get("swing_high"),
        "swing_low": swings.get("swing_low"),
        "support": computed.get("support_level"),
        "support_touches": computed.get("support_touches"),
        "resistance": computed.get("resistance_level"),
        "resistance_touches": computed.get("resistance_touches"),
        "candles": candles,
        "active_move": active_move,
        "flow": flow,
        "whale": whale,
        "long_pct": long_pct,
        "funding_rate": funding_rate,
        "pct_change_24h": pct_change_24h,
        "vwap": vwap_v,
        "atr": atr_val,
        "stance": stance,
        "stance_note": stance_note,
        "bull_confirmed": computed.get("bullish_retracement_confirmed"),
        "bear_confirmed": computed.get("bearish_retracement_confirmed"),
        "bull_entry_ref": computed.get("bull_entry_ref"),
        "bear_entry_ref": computed.get("bear_entry_ref"),
        "plan": _trade_plan(df, is_h, is_l, price, atr_val, structure_4h, demand, supply),
        "plan_bull": _trade_plan(df, is_h, is_l, price, atr_val, "bullish", demand, supply) if structure_4h == "mixed" else None,
        "plan_bear": _trade_plan(df, is_h, is_l, price, atr_val, "bearish", demand, supply) if structure_4h == "mixed" else None,
        "supertrend": st,
        "ema_fast": ema_f,
        "ema_slow": ema_s,
        "rsi6": computed.get("rsi"),
        "rsi14": computed.get("rsi14"),
        "swept_low": computed.get("swept_low"),
        "swept_high": computed.get("swept_high"),
        "demand_zone": demand,
        "supply_zone": supply,
        "vol_ratio": vol_ratio,
        "range_pos": range_pos,
    }
    if extra:
        read.update(extra)
    return read


def format_structure_analysis(symbol: str, read: dict) -> str:
    """Compact Finora-style report: short data lines for the facts, a little prose for stance and plan.
    Normally fits in a single Telegram message."""
    seed = zlib.crc32(symbol.encode())
    ctr = [0]

    def pick(opts):
        ctr[0] += 1
        return opts[(seed + ctr[0]) % len(opts)]

    fp = _fp

    def jn(items):
        items = list(items)
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

    price = read.get("price")
    s4 = read.get("structure_4h") or "mixed"
    sl_, sh_ = read.get("swing_low"), read.get("swing_high")
    plan = read.get("plan")
    move = read.get("active_move")
    ema_f, ema_s = read.get("ema_fast"), read.get("ema_slow")
    chg = read.get("pct_change_24h")
    tf = read.get("tf") or "4h"
    ctf = _CONFIRM_TF.get(tf, "1h")
    c_s = "4§h" if ctf == "4h" else ctf
    dl = depth_lines(read)
    lines = [f"📊 <b>{symbol}.P</b> · 4h", ""]

    # ---- stance ----
    head = f"Price {fp(price)}" + (f" ({chg:+.2f}% 24h)" if chg is not None else "")
    if read.get("htf") is not None:
        htf = list(read["htf"])
    else:
        htf = [(lbl, read.get(key)) for lbl, key in (("daily", "daily_bias"), ("weekly", "weekly_bias")) if read.get(key)]
    if read.get("btc_bias"):
        htf.append(("BTC 4§h" if tf == "4h" else f"BTC {tf}", read["btc_bias"]))
    counter = False
    if s4 in ("bearish", "bullish"):
        head += f". Stance: <b>{s4}</b>, 4h structure pointing {'down' if s4 == 'bearish' else 'up'}"
        ali = [l for l, b in htf if b == s4]
        opp = [l for l, b in htf if b in ("bullish", "bearish") and b != s4]
        neu = [l for l, b in htf if b == "mixed"]
        if ali and not opp:
            head += f", with {jn(ali)} agreeing"
        elif ali and opp:
            head += f", aligned with {jn(ali)} but against {jn(opp)}"
        elif opp:
            counter = True
            head += f", counter-trend against {jn(opp)}"
        elif neu:
            head += f", {jn(neu)} mixed"
    else:
        head += ". Stance: <b>mixed</b>, no clear 4h structure"
        if htf:
            head += " (" + ", ".join(f"{l} {b}" for l, b in htf) + ")"
    if move == "rebound_in_bearish_structure":
        head += ". A rebound is active but not a confirmed flip"
    elif move == "pullback_in_bullish_structure":
        head += ". A pullback is underway inside the uptrend"
    lines.append("- " + head + ".")

    # ---- trend ----
    ti = trend_indicator_text(read.get("supertrend"), ema_f, ema_s)
    if ti:
        pos = ""
        if price is not None and ema_f is not None and ema_s is not None:
            pos = (" · price above both EMAs" if price > max(ema_f, ema_s) else
                   " · price below both EMAs" if price < min(ema_f, ema_s) else " · price between the EMAs")
        lines.append("- Trend: " + ti.replace(" — ", " · ") + pos)

    # ---- momentum ----
    mo = []
    r14, r6 = read.get("rsi14"), read.get("rsi6")
    if r14 is not None:
        lab = ("overbought" if r14 >= 70 else "oversold" if r14 <= 30 else "bullish lean" if r14 >= 55 else
               "bearish lean" if r14 <= 45 else "neutral")
        mo.append(f"RSI14 {r14:.1f} ({lab})" + (f", RSI6 {r6:.1f}" if r6 is not None else ""))
    vr = read.get("vol_ratio")
    if vr is not None:
        mo.append(f"volume {vr:.1f}x avg" + (" (conviction)" if vr >= 1.5 else " (thin)" if vr <= 0.6 else ""))
    vw = read.get("vwap")
    if vw and price:
        mo.append(f"{'above' if price > vw else 'below'} VWAP {fp(vw)}")
    if mo:
        lines.append("- Momentum: " + " · ".join(mo))
    if dl.get("div"):
        lines.append("- " + dl["div"])

    # ---- candles ----
    candles = read.get("candles") or []
    if candles:
        seq = " → ".join(fp(c["close"]) for c in candles)
        dirs = [c["direction"] for c in candles]
        nb, ns = dirs.count("bullish"), dirs.count("bearish")
        tail = "all bullish" if nb == len(dirs) else "all bearish" if ns == len(dirs) else f"{nb} bullish / {ns} bearish"
        last = candles[-1]
        rej = (" · lower-wick rejection" if last.get("rejection") == "lower_wick" else
               " · upper-wick rejection" if last.get("rejection") == "upper_wick" else "")
        lines.append(f"- Last {len(candles)} candles: {seq} ({tail}){rej}")

    # ---- flow / positioning ----
    fl = []
    flow = read.get("flow") or {}
    fb = flow.get("bias")
    if fb == "sellers":
        fl.append(f"sellers {flow.get('sell_pct', 0):.0f}% of taker flow" + (" (confirms)" if s4 == "bearish" else " (caution)" if s4 == "bullish" else ""))
    elif fb == "buyers":
        fl.append(f"buyers {flow.get('buy_pct', 0):.0f}% of taker flow" + (" (confirms)" if s4 == "bullish" else " (caution on shorts)" if s4 == "bearish" else ""))
    elif fb == "mixed":
        fl.append("taker flow mixed")
    whale = read.get("whale") or {}
    if whale.get("whale_bias"):
        fl.append(f"large prints {whale['whale_bias']}")
    lp = read.get("long_pct")
    if lp is not None:
        tag = ""
        if (s4 == "bearish" and lp >= 60) or (s4 == "bullish" and lp >= 60) or lp >= 70:
            tag = " (crowded long)"
        elif lp <= 40 and s4 in ("bearish", "bullish") or lp <= 30:
            tag = " (crowded short, squeeze risk)"
        fl.append(f"{lp:.0f}% accounts long{tag}")
    fr = read.get("funding_rate")
    if fr is not None:
        pf = fr * 100 if abs(fr) < 1 else fr
        fl.append(f"funding {pf:.4f}%" + (" (elevated)" if abs(pf) >= 0.03 else ""))
    oi = read.get("oi_chg")
    if oi is not None and chg is not None and abs(oi) >= 2:
        meaning = ("shorts pressing" if oi > 0 and chg < 0 else "new longs" if oi > 0 else
                   "short covering" if chg > 0 else "longs closing")
        fl.append(f"OI {oi:+.1f}% ({meaning})")
    if fl:
        lines.append("- Flow: " + " · ".join(fl))

    # ---- range, zones, liquidity ----
    z = []
    rp = read.get("range_pos")
    if rp is not None:
        if rp > 1:
            z.append(f"above the swing-high {fp(sh_)}")
        elif rp < 0:
            z.append(f"below the swing-low {fp(sl_)}")
        else:
            z.append(f"{'upper' if rp >= 0.7 else 'lower' if rp <= 0.3 else 'mid'} {rp * 100:.0f}% of {fp(sl_)}-{fp(sh_)}"
                     + (" (premium)" if rp >= 0.7 else " (discount)" if rp <= 0.3 else ""))
    if read.get("demand_zone"):
        z.append(f"demand {fp(read['demand_zone'][0])}-{fp(read['demand_zone'][1])}")
    if read.get("supply_zone"):
        z.append(f"supply {fp(read['supply_zone'][0])}-{fp(read['supply_zone'][1])}")
    if read.get("resistance"):
        z.append(f"resistance {fp(read['resistance'])} ({read.get('resistance_touches') or 3}x)")
    if read.get("support"):
        z.append(f"support {fp(read['support'])} ({read.get('support_touches') or 3}x)")
    if read.get("swept_low") and sl_:
        z.append(f"swept {fp(sl_)} swing low and reclaimed")
    if read.get("swept_high") and sh_:
        z.append(f"swept {fp(sh_)} swing high and rejected")
    if z:
        lines.append("- Range & zones: " + " · ".join(z))
    if dl.get("levels"):
        lines.append("- " + dl["levels"])
    if dl.get("fib"):
        lines.append("- " + dl["fib"])

    # ---- plan ----
    def rr_text(pl):
        try:
            mid = (pl["zone_lo"] + pl["zone_hi"]) / 2
            risk = abs(pl["invalidation"] - mid)
            return f"{abs(pl['tps'][0] - mid) / risk:.1f}" if risk > 0 and pl["tps"] else None
        except Exception:
            return None

    if plan:
        short = plan["side"] == "short"
        zone = f"{fp(plan['zone_lo'])}-{fp(plan['zone_hi'])}"
        src = plan.get("zone_src")
        zone_s = f"{zone} ({src})" if src and "imbalance" in src else zone
        conf = read.get("bear_confirmed") if short else read.get("bull_confirmed")
        ref = read.get("bear_entry_ref") if short else read.get("bull_entry_ref")
        if short:
            entry = pick([
                f"Plan: short a pullback into {zone_s} once a {c_s} bearish rejection candle holds below it; if TP1 hits first, the setup is missed",
                f"Plan: wait for a retest of {zone_s} rejected on the {c_s} and holding below; no retest, no trade",
                f"Plan: short a rally into {zone_s} after a {c_s} bearish rejection that holds below; if price drops to TP1 untested, skip it",
            ])
        else:
            entry = pick([
                f"Plan: buy a dip into {zone_s} once a {c_s} bullish rejection candle holds above it; if TP1 hits first, the setup is missed",
                f"Plan: wait for a retest of {zone_s} that holds on the {c_s}; no retest, no trade",
                f"Plan: buy a dip into {zone_s} after a {c_s} bullish rejection that holds above; if price runs to TP1 untested, skip it",
            ])
        if conf:
            entry += f". A {'bearish' if short else 'bullish'} retracement entry already confirmed" + (f" near {fp(ref)}" if ref else "")
        lines.append("- " + entry + ".")
        if dl.get("confluence"):
            lines.append("- " + dl["confluence"])
        if dl.get("trigger"):
            lines.append("- " + dl["trigger"])
        if plan["tps"]:
            rr = rr_text(plan)
            tp = "Targets: " + " → ".join(fp(t) for t in plan["tps"]) + (f" (≈1:{rr} to TP1)" if rr else "")
            liq = sl_ if short else sh_
            if liq:
                tp += f". The {fp(liq)} swing {'low' if short else 'high'} is liquidity; a sweep and {'reclaim' if short else 'rejection'} warns against holding {'shorts' if short else 'longs'}"
            lines.append("- " + tp + ".")
        inv = fp(plan["invalidation"])
        tail = ". Counter-trend: size smaller, take profit early" if counter else ""
        lines.append(f"- Invalid: 4h close {'above' if short else 'below'} {inv}{tail}.")
    else:
        pb, pr = read.get("plan_bull"), read.get("plan_bear")
        if pb or pr:
            lines.append("- Structure is mixed, so let one side prove itself:")
            for label, pl, up_ in (("Bullish", pb, True), ("Bearish", pr, False)):
                if not pl:
                    continue
                rr = rr_text(pl)
                brk = None
                if up_ and sh_ and price and price < sh_:
                    brk = f"a {c_s} close above {fp(sh_)}"
                if (not up_) and sl_ and price and price > sl_:
                    brk = f"a {c_s} close below {fp(sl_)}"
                lines.append(f"  {label}: {'dip' if up_ else 'rally'} into {fp(pl['zone_lo'])}-{fp(pl['zone_hi'])}"
                             + (f" or {brk} that holds" if brk else "")
                             + f" → {' → '.join(fp(t) for t in pl['tps'])}" + (f" (≈1:{rr})" if rr else "")
                             + f" · invalid 4h close {'below' if up_ else 'above'} {fp(pl['invalidation'])}")
            if dl.get("confluence"):
                lines.append("- " + dl["confluence"])
            if dl.get("trigger"):
                lines.append("- " + dl["trigger"])
        elif sl_ and sh_:
            lines.append(f"- No clean setup. A 4h close above {fp(sh_)} turns it bullish, below {fp(sl_)} bearish; stay patient.")
        else:
            lines.append("- No clean setup. Wait for a clearer 4h BOS plus candle confirmation.")

    lines.append("")
    lines.append("📝 Not investment advice — an educational report.")
    out = "\n".join(lines)
    if tf != "4h":
        out = re.sub(r"(?<!\d)4h", tf, out)        # relabel the analysed timeframe (keeps '24h' intact)
    return out.replace("§", "")


def render_chart_png(symbol: str, df: pd.DataFrame, read: dict, tf: str = None) -> bytes | None:
    """Candlestick chart with the plan zone, targets and invalidation marked.
    Returns PNG bytes, or None if matplotlib isn't installed / rendering fails."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from io import BytesIO
    except ImportError:
        log.warning("matplotlib not installed — sending analysis without chart image")
        return None
    try:
        d = df.tail(110).reset_index(drop=True)
        n = len(d)
        up, down = "#26a69a", "#ef5350"
        fig, ax = plt.subplots(figsize=(10, 5.6), dpi=110)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        x = np.arange(n)
        cols = [up if c >= o else down for o, c in zip(d["open"], d["close"])]
        ax.vlines(x, d["low"], d["high"], colors=cols, linewidth=0.9, zorder=2)
        body_lo = np.minimum(d["open"], d["close"])
        body_h = np.maximum((d["open"] - d["close"]).abs(), d["close"].abs() * 1e-5)
        ax.bar(x, body_h, bottom=body_lo, width=0.7, color=cols, zorder=3)

        if "ema_fast" in d.columns and "ema_slow" in d.columns:
            ax.plot(x, d["ema_fast"], color="#1e88e5", linewidth=1.0, label=f"EMA{EMA_FAST}", zorder=4)
            ax.plot(x, d["ema_slow"], color="#8e24aa", linewidth=1.0, label=f"EMA{EMA_SLOW}", zorder=4)
        if "st" in d.columns and "st_trend" in d.columns:
            st_up = d["st"].where(d["st_trend"] == 1)
            st_dn = d["st"].where(d["st_trend"] == -1)
            ax.plot(x, st_up, color=up, linewidth=1.4, label="SuperTrend", zorder=4)
            ax.plot(x, st_dn, color=down, linewidth=1.4, zorder=4)
        ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=3)

        right = n + 10
        plan = read.get("plan")
        scenario = not plan
        plans = [plan] if plan else [pp for pp in (read.get("plan_bull"), read.get("plan_bear")) if pp]
        for pl_ in plans:
            zc = down if pl_["side"] == "short" else up
            tag = ("Short zone" if pl_["side"] == "short" else "Long zone") if scenario else "Entry zone"
            ax.axhspan(pl_["zone_lo"], pl_["zone_hi"], xmin=0, xmax=1, color=zc, alpha=0.18, zorder=1)
            ax.text(right - 0.5, pl_["zone_hi"], tag, ha="right", va="bottom", fontsize=8, color=zc)
            ax.axhline(pl_["invalidation"], color="#fb8c00", linestyle=":", linewidth=1.1, zorder=1)
            ax.text(right - 0.5, pl_["invalidation"], "Invalidation", ha="right", va="bottom", fontsize=8, color="#fb8c00")
            for i, t in enumerate(pl_["tps"][:1] if scenario else pl_["tps"], 1):
                ax.axhline(t, color="#5c6bc0", linestyle="--", linewidth=0.9, zorder=1)
                ax.text(right - 0.5, t, f"TP{i}", ha="right", va="bottom", fontsize=8, color="#5c6bc0")
        for key, label in (("swing_high", "Swing high"), ("swing_low", "Swing low")):
            v = read.get(key)
            if v:
                ax.axhline(v, color="#9e9e9e", linewidth=0.8, zorder=1)
        price = read.get("price")
        if price:
            ax.axhline(price, color="#212121", linewidth=0.7, linestyle="-", zorder=1)

        ax.set_xlim(-1, right)
        lo = float(d["low"].min())
        hi = float(d["high"].max())
        for pl_ in plans:
            lo = min([lo, pl_["invalidation"]] + list(pl_["tps"]) + [pl_["zone_lo"]])
            hi = max([hi, pl_["invalidation"]] + list(pl_["tps"]) + [pl_["zone_hi"]])
        pad = (hi - lo) * 0.04
        ax.set_ylim(lo - pad, hi + pad)
        step = max(1, n // 8)
        ticks = list(range(0, n, step))
        ax.set_xticks(ticks)
        _fmt = "%d %b" if (tf or SIGNAL_INTERVAL) in ("1d", "1w") else "%d/%m %H:%M"
        ax.set_xticklabels([d["open_time"].iloc[i].strftime(_fmt) for i in ticks], fontsize=8)
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, color="#eeeeee", linewidth=0.6, zorder=0)
        for sp in ("top", "left"):
            ax.spines[sp].set_visible(False)
        ax.set_title(f"{symbol}.P · {tf or SIGNAL_INTERVAL} · Bybit", loc="left", fontsize=10)
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()
    except Exception as e:
        log.warning("chart render failed for %s: %s", symbol, e)
        try:
            plt.close("all")
        except Exception:
            pass
        return None


async def _daily_structure_bias(symbol: str) -> str | None:
    """Lightweight daily bias from Supertrend/EMA on 1D candles."""
    try:
        dkl = await get_klines(symbol, interval="1d", limit=80)
        if not dkl or len(dkl) < 30:
            return None
        dcomp = await asyncio.to_thread(_compute_indicators, dkl)
        st = dcomp.get("supertrend_trend")
        ef, es = dcomp.get("ema_fast"), dcomp.get("ema_slow")
        if st == 1 or (ef is not None and es is not None and ef > es):
            return "bullish"
        if st == -1 or (ef is not None and es is not None and ef < es):
            return "bearish"
        return "mixed"
    except Exception as e:
        log.warning(f"{symbol}: daily bias failed: {e}")
        return None


async def _tf_bias(symbol: str, interval: str, limit: int, min_len: int) -> str | None:
    """Strict bias on another timeframe: bullish/bearish only when SuperTrend and EMA20/50 agree."""
    try:
        kl = await get_klines(symbol, interval=interval, limit=limit)
        if not kl or len(kl) < min_len:
            return None
        comp = await asyncio.to_thread(_compute_indicators, kl)
        st, ef, es = comp.get("supertrend_trend"), comp.get("ema_fast"), comp.get("ema_slow")
        if ef is None or es is None:
            return None
        if st == 1 and ef > es:
            return "bullish"
        if st == -1 and ef < es:
            return "bearish"
        return "mixed"
    except Exception as e:
        log.warning(f"{symbol}: {interval} bias failed: {e}")
        return None


_BTC_BIAS_CACHE: dict = {}
_HTF_MAP = {"1m": ["5m", "15m"], "3m": ["15m", "1h"], "5m": ["15m", "1h"], "15m": ["1h", "4h"],
            "30m": ["1h", "4h"], "1h": ["4h", "1d"], "2h": ["4h", "1d"], "4h": ["1d", "1w"],
            "1d": ["1w"], "1w": []}
# entry-confirmation timeframe: one step below the analysed timeframe, so price can't run
# far before the trigger prints (4h analysis -> 1h rejection candle that holds the zone)
_CONFIRM_TF = {"1m": "1m", "3m": "1m", "5m": "1m", "15m": "5m", "30m": "15m",
               "1h": "15m", "2h": "30m", "4h": "1h", "1d": "4h", "1w": "1d"}
_HTF_LABEL = {"1d": "daily", "1w": "weekly", "4h": "4§h"}   # § is stripped later (protects it from the timeframe relabel)


async def _btc_bias(tf: str = None) -> str | None:
    tf = tf or SIGNAL_INTERVAL
    ts, val = _BTC_BIAS_CACHE.get(tf, (0.0, None))
    if time.time() - ts < 600:
        return val
    val = await _tf_bias("BTCUSDT", tf, KLINE_LIMIT, 60)
    _BTC_BIAS_CACHE[tf] = (time.time(), val)
    return val


LLM_ANALYSIS_SYSTEM = (
    "You write concise crypto trade-analysis reports for Telegram, in the style of a professional analyst. "
    "Use ONLY the facts provided. Copy every price and level exactly as written — never invent, round or add "
    "levels, indicators, news or numbers. Output 4 to 6 bullet lines, each starting with '- ', plain text only "
    "(no markdown, no headers, no emojis). Bullet 1: stance and 4h structure (and daily context if given). "
    "Then: trend indicators, latest candle and order flow. Then, only if a trade plan is provided: the entry idea "
    "(conditional on a rejection candle on the stated confirmation timeframe that holds the zone), take profits in order, and the invalidation level. If no plan is provided, "
    "say to wait for a clearer 4h BOS and candle confirmation. Never promise outcomes."
)
LLM_PULSE_SYSTEM = (
    "You write the 'Today's read' section of a daily crypto Market Pulse for Telegram. Use ONLY the facts provided; "
    "never invent numbers, news or levels. Output 3 to 4 short bullet lines, each starting with '- ', plain text only "
    "(no markdown, no emojis). Explain what the positioning, sentiment and trend breadth mean for traders, in plain words."
)
_DEC_RE = re.compile(r"\d+\.\d+")


async def llm_write(system: str, facts: str, max_tokens: int = 700) -> str | None:
    """Ask Claude to write text from computed facts. Returns None if no key / any failure
    (callers then fall back to the built-in templates)."""
    if not ANTHROPIC_API_KEY:
        return None
    try:
        client = _get_tg_client()
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": ANTHROPIC_MODEL, "max_tokens": max_tokens, "system": system,
                  "messages": [{"role": "user", "content": facts}]},
        )
        if resp.status_code != 200:
            log.warning("LLM request failed: %s %s", resp.status_code, resp.text[:200])
            return None
        data = resp.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
        return text or None
    except Exception as e:
        log.warning("LLM error: %s", e)
        return None


def llm_numbers_ok(text: str, facts: str) -> bool:
    """Guard against invented figures: every decimal number the LLM wrote must appear in the facts."""
    return all(n in facts for n in _DEC_RE.findall(text))


def clean_bullets(text: str) -> str:
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if ln.startswith(("-", "•", "·", "*")):
            out.append("- " + html_lib.escape(ln.lstrip("-•·* ").strip()))
    return "\n".join(out)


def read_facts_text(symbol: str, read: dict) -> str:
    def fp(p):
        if p is None:
            return "n/a"
        return f"{p:.6f}" if p < 1 else f"{p:.4f}"
    L = [f"Symbol: {symbol} perpetual, timeframe {read.get('tf') or '4h'}",
         f"Entry confirmation timeframe: {_CONFIRM_TF.get(read.get('tf') or '4h', '1h')} (invalidation is judged on the {read.get('tf') or '4h'} close)",
         f"Price: {fp(read.get('price'))}",
         f"24h change: {read.get('pct_change_24h'):+.2f}%" if read.get("pct_change_24h") is not None else "24h change: n/a",
         f"4h structure: {read.get('structure_4h')}",
         f"Daily bias: {read.get('daily_bias') or 'n/a'}"]
    ti = trend_indicator_text(read.get("supertrend"), read.get("ema_fast"), read.get("ema_slow"))
    if ti:
        L.append(f"Trend indicators: {ti}")
    L.append(f"Last swing-low: {fp(read.get('swing_low'))}; last swing-high: {fp(read.get('swing_high'))}")
    if read.get("support"):
        L.append(f"Triple-tested support: {fp(read['support'])} ({read.get('support_touches')}x)")
    if read.get("resistance"):
        L.append(f"Triple-tested resistance: {fp(read['resistance'])} ({read.get('resistance_touches')}x)")
    c = (read.get("candles") or [None])[-1]
    if c:
        L.append(f"Latest closed 4h candle: {c['direction']} ({c['strength']}), close {fp(c['close'])}"
                 + (f", {c['rejection']} rejection" if c.get("rejection") else ""))
    if read.get("active_move"):
        L.append(f"Active move: {read['active_move']}")
    fl = read.get("flow") or {}
    if fl.get("bias"):
        L.append(f"Taker flow: {fl['bias']} (buy {fl.get('buy_pct', 0):.0f}% / sell {fl.get('sell_pct', 0):.0f}%)")
    wh = read.get("whale") or {}
    if wh.get("whale_bias"):
        L.append(f"Large prints lean: {wh['whale_bias']}")
    if read.get("long_pct") is not None:
        L.append(f"Accounts long/short: {read['long_pct']:.0f}% / {100 - read['long_pct']:.0f}%")
    if read.get("funding_rate") is not None:
        fr = read["funding_rate"]
        L.append(f"Funding: {fr * 100:.4f}%" if abs(fr) < 1 else f"Funding: {fr:.4f}%")
    if read.get("stance_note"):
        L.append(f"Note: {read['stance_note']}")
    L.append("Higher timeframes: " + (", ".join(f"{l.replace('§', '')} {b}" for l, b in (read.get("htf") or [])) or "n/a")
             + f", BTC same-timeframe {read.get('btc_bias') or 'n/a'}")
    if read.get("rsi14") is not None:
        L.append(f"RSI14: {read['rsi14']:.1f}; RSI6: {read.get('rsi6', 0):.1f}")
    if read.get("vol_ratio") is not None:
        L.append(f"Last candle volume vs 20-bar average: {read['vol_ratio']:.1f}x")
    if read.get("oi_chg") is not None:
        L.append(f"Open interest 24h change: {read['oi_chg']:+.1f}%")
    if read.get("demand_zone"):
        L.append(f"Unfilled demand imbalance: {fp(read['demand_zone'][0])}-{fp(read['demand_zone'][1])}")
    if read.get("supply_zone"):
        L.append(f"Unfilled supply imbalance: {fp(read['supply_zone'][0])}-{fp(read['supply_zone'][1])}")
    if read.get("swept_low"):
        L.append("Last closed candle swept the swing low and closed back above it")
    if read.get("swept_high"):
        L.append("Last closed candle swept the swing high and closed back below it")
    for key, label in (("plan_bull", "Bullish scenario"), ("plan_bear", "Bearish scenario")):
        sp = read.get(key)
        if sp:
            L.append(f"{label}: entry zone {fp(sp['zone_lo'])}-{fp(sp['zone_hi'])}; targets {', '.join(fp(t) for t in sp['tps'])}; invalidation {fp(sp['invalidation'])}")
    pl = read.get("plan")
    if pl:
        L.append(f"Trade plan ({pl['side']}): entry zone {fp(pl['zone_lo'])}-{fp(pl['zone_hi'])}; "
                 f"take profits in order: {', '.join(fp(t) for t in pl['tps'])}; "
                 f"invalidation: 4h close {'above' if pl['side'] == 'short' else 'below'} {fp(pl['invalidation'])}")
    else:
        L.append("Trade plan: none (structure mixed)")
    return "\n".join(L)


def normalize_symbol(raw: str) -> str:
    s = (raw or "").strip().upper().replace("/", "").replace("-", "")
    if not s:
        return s
    if not s.endswith("USDT"):
        s = s + "USDT"
    return s


async def _analyze_core(symbol: str, tf: str = "4h") -> tuple[str, bytes | None, dict | None]:
    """Full on-demand structure + order-flow analysis on any supported timeframe.
    Returns (text, chart_png_or_None)."""
    symbol = normalize_symbol(symbol)
    if not symbol:
        return "Usage: /analyze SYMBOL [timeframe]  (e.g. /analyze PHAUSDT 1h)", None, None
    if tf not in SUPPORTED_TFS:
        tf = "4h"

    klines = await get_klines(symbol, interval=tf)
    if klines is None:
        return f"{symbol}: Bybit request failed (rate limit or network). Try again shortly.", None, None
    if len(klines) < PIVOT_LOOKBACK * 2 + 5:
        return f"{symbol}: couldn't find enough candle data for this pair on Bybit.", None, None

    computed = await asyncio.to_thread(_compute_indicators, klines)
    trades = await get_recent_trades(symbol)
    whale = detect_whale_activity(trades, computed["price"])
    flow = detect_order_flow(trades, computed["price"])
    ls = await get_long_short_account_ratio(symbol, period=tf)
    long_pct = float(ls["buyRatio"]) * 100 if ls else None

    tickers = await get_24hr_tickers()
    t = tickers.get(symbol) if tickers else None
    pct_24h = float(t["price24hPcnt"]) * 100 if t and t.get("price24hPcnt") is not None else None

    funding_rate = None
    try:
        hist = await get_funding_rate_history(symbol, limit=3)
        if hist:
            # most recent funding
            best = None
            for rec in hist:
                try:
                    ts = float(rec.get("fundingRateTimestamp") or rec.get("timestamp") or 0)
                    rate = float(rec.get("fundingRate"))
                    if best is None or ts > best[0]:
                        best = (ts, rate)
                except (TypeError, ValueError):
                    continue
            if best:
                funding_rate = best[1]
    except Exception:
        pass

    async def _oi_change():
        try:
            return find_oi_24h_change_pct(await get_open_interest_history(symbol, "1d", 5))
        except Exception:
            return None

    htf_tfs = _HTF_MAP.get(tf, [])

    async def _htf(h):
        if tf == "4h" and h == "1d":
            return await _daily_structure_bias(symbol)       # unchanged behaviour for the default 4h view
        return await _tf_bias(symbol, h, 80 if h != "1w" else 60, 30 if h != "1w" else 15)

    results = await asyncio.gather(
        _btc_bias(tf) if symbol != "BTCUSDT" else asyncio.sleep(0, result=None),
        _oi_change(),
        *[_htf(h) for h in htf_tfs],
    )
    btc_bias, oi_chg, htf_vals = results[0], results[1], list(results[2:])
    htf = [(_HTF_LABEL.get(h, h), b) for h, b in zip(htf_tfs, htf_vals) if b]
    daily_bias = next((b for h, b in zip(htf_tfs, htf_vals) if h == "1d"), None)
    weekly_bias = next((b for h, b in zip(htf_tfs, htf_vals) if h == "1w"), None)
    read = build_structure_read(computed, flow, whale, long_pct, funding_rate, pct_24h, daily_bias,
                                extra={"weekly_bias": weekly_bias, "btc_bias": btc_bias, "oi_chg": oi_chg,
                                       "htf": htf, "tf": tf})
    ctf = _CONFIRM_TF.get(tf, "1h")
    try:
        daily_rows, weekly_rows, trig_rows = await asyncio.gather(
            get_klines(symbol, interval="1d", limit=10),
            get_klines(symbol, interval="1w", limit=4),
            get_klines(symbol, interval=ctf, limit=30),
        )
    except Exception as e:
        log.warning("%s: depth context fetch failed: %s", symbol, e)
        daily_rows = weekly_rows = trig_rows = None
    read.update(compute_depth(computed, read, daily_rows, weekly_rows))
    if trig_rows:
        for tk, pk in (("trigger", "plan"), ("trigger_bull", "plan_bull"), ("trigger_bear", "plan_bear")):
            if read.get(pk):
                read[tk] = evaluate_trigger(read[pk], trig_rows)
    text = format_structure_analysis(symbol, read)
    if ANTHROPIC_API_KEY:
        facts = read_facts_text(symbol, read)
        out = await llm_write(LLM_ANALYSIS_SYSTEM, facts)
        bullets = clean_bullets(out) if out else ""
        if bullets and llm_numbers_ok(out, facts):
            dl_ = depth_lines(read)
            extra = "\n".join("- " + html_lib.escape(dl_[k]) for k in ("div", "levels", "fib", "confluence", "trigger") if dl_.get(k))
            text = (f"📊 <b>{symbol}.P</b> · {tf}\n\n{bullets}" + ("\n" + extra if extra else "")
                    + "\n\n📝 Not investment advice — an educational report.").replace("§", "")
        elif out:
            log.info("%s: LLM text failed number check — using template", symbol)
    png = await asyncio.to_thread(render_chart_png, symbol, computed["df"], read, tf)
    return text, png, read


async def _analyze_full(symbol: str, tf: str = "4h") -> tuple[str, bytes | None]:
    text, png, _ = await _analyze_core(symbol, tf)
    return text, png


async def analyze_symbol(symbol: str) -> str:
    """Text-only analysis (used by the /api/analyze endpoint)."""
    text, _ = await _analyze_full(symbol)
    return text

# ======================================================================
# SIGNAL ENGINE (combines indicators + L/S ratio + OI into BUY/SELL/BOS)
# ======================================================================



class SymbolSnapshot:
    __slots__ = (
        "symbol", "price", "entry_price", "pct_change_24h", "pct_change_8h", "pct_change_1h",
        "momentum_basis", "rsi", "ema_fast", "ema_slow",
        "supertrend_trend", "vwap", "atr", "obv", "open_interest", "open_interest_contracts",
        "long_short_account_ratio",
        "long_pct", "short_pct", "signal", "bos", "stop_loss", "take_profit", "take_profit2",
        "stop_ref_below", "stop_ref_above", "whale_bias", "whale_notional",
        "entry_bos_direction", "entry_bos_bars_ago",
        "signal_score", "signal_flags", "signal_tag",
        "resistance_level", "resistance_touches", "support_level", "support_touches",
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
    df["rsi14"] = rsi(df["close"], RSI14_PERIOD)
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

    bull_confirmed, bull_zone_low, bull_zone_high, bull_bars_ago, bull_stage = retracement_entry(df, is_high, is_low, bull_bos, bear_bos, "bullish")
    bear_confirmed, bear_zone_low, bear_zone_high, bear_bars_ago, bear_stage = retracement_entry(df, is_high, is_low, bull_bos, bear_bos, "bearish")

    # Stop reference now prefers the retracement zone boundary itself when a
    # setup just confirmed — that's the precise invalidation point of THIS
    # entry, sharper than the generic "nearest swept level" fallback.
    stop_ref_below = bull_zone_low if bull_confirmed else (sweep["swept_low_level"] or nearest_structure_level(df, is_high, is_low, live_price, "below"))
    stop_ref_above = bear_zone_high if bear_confirmed else (sweep["swept_high_level"] or nearest_structure_level(df, is_high, is_low, live_price, "above"))

    target_above = nearest_structure_level(df, is_high, is_low, live_price, "above")
    target_below = nearest_structure_level(df, is_high, is_low, live_price, "below")

    resistance_level, resistance_touches = detect_triple_touch(df, is_high, is_low, "resistance", live_price)
    support_level, support_touches = detect_triple_touch(df, is_high, is_low, "support", live_price)

    # 8h change for the momentum fallback — computed in bars rather than a
    # hardcoded "2 candles back" so it stays correct if SIGNAL_INTERVAL is
    # ever changed away from the 4h default.
    bars_for_8h = max(1, round(8 / _interval_to_hours(SIGNAL_INTERVAL)))
    pct_change_8h = None
    ref_idx = last_closed_idx - bars_for_8h
    if ref_idx >= 0:
        close_8h_ago = float(df["close"].iloc[ref_idx])
        if close_8h_ago:
            pct_change_8h = (live_price - close_8h_ago) / close_8h_ago * 100

    return {
        "df": df, "is_high": is_high, "is_low": is_low,  # kept for TP2 lookup after TP1 is known — not sent to JSON
        "price": live_price,
        "pct_change_8h": pct_change_8h,
        "rsi": round(float(last_closed["rsi"]), 2),
        "rsi14": round(float(last_closed["rsi14"]), 2),
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
        # confirmed retracement-zone boundary closest to the trade direction —
        # the "textbook" entry the pipeline actually confirmed on, used to
        # detect when live price has since run past it (see apply_entry_gap_check)
        "bull_entry_ref": bull_zone_high if bull_confirmed else None,
        "bear_entry_ref": bear_zone_low if bear_confirmed else None,
        "bull_stage": bull_stage, "bear_stage": bear_stage,  # funnel diagnostics — how far each direction got
        "bull_fvg_recent": bool(df["bull_fvg"].iloc[:last_closed_idx + 1].tail(5).any()),
        "bear_fvg_recent": bool(df["bear_fvg"].iloc[:last_closed_idx + 1].tail(5).any()),
        "bos": bos,
        "bullish_retracement_confirmed": bull_confirmed,
        "bearish_retracement_confirmed": bear_confirmed,
        "bull_bars_ago": bull_bars_ago,
        "bear_bars_ago": bear_bars_ago,
        "resistance_level": resistance_level,
        "resistance_touches": resistance_touches,
        "support_level": support_level,
        "support_touches": support_touches,
    }


def find_oi_n_days_ago(history: list, days: int = 30) -> float | None:
    """history is Bybit's open-interest-history list; finds the record
    closest to `days` ago rather than assuming a fixed index, since the
    list's ordering isn't guaranteed and gaps can occur."""
    if not history:
        return None
    target_ms = time.time() * 1000 - days * 86400 * 1000
    best, best_diff = None, None
    for rec in history:
        try:
            ts = float(rec["timestamp"])
            diff = abs(ts - target_ms)
            if best_diff is None or diff < best_diff:
                best_diff, best = diff, float(rec["openInterest"])
        except (KeyError, ValueError, TypeError):
            continue
    return best


def find_oi_latest(history: list) -> float | None:
    """Most recent OI reading directly from the open-interest-history
    endpoint — kept as an independent live reference distinct from the
    ticker's OI field, specifically so the two can be cross-checked for
    contradictions (see OI_CONTRADICTION_DIFF_PCT). Both this and
    oi_30d_ago come from the SAME endpoint, so they're always in the same
    unit (contracts) — never mixed with the ticker's USD-denominated
    openInterestValue, which was the root cause of the old bad OI% flags."""
    if not history:
        return None
    best_ts, best_val = None, None
    for rec in history:
        try:
            ts = float(rec["timestamp"])
            if best_ts is None or ts > best_ts:
                best_ts, best_val = ts, float(rec["openInterest"])
        except (KeyError, ValueError, TypeError):
            continue
    return best_val


def find_oi_24h_change_pct(history: list) -> float | None:
    """24h OI change computed from the two most recent points in the same
    daily-interval open-interest-history call already used for oi_30d_ago —
    no extra API call needed. Same unit (contracts) throughout, so this is
    never contaminated by the ticker's USD-denominated openInterestValue."""
    records = []
    for rec in history:
        try:
            records.append((float(rec["timestamp"]), float(rec["openInterest"])))
        except (KeyError, ValueError, TypeError):
            continue
    if len(records) < 2:
        return None
    records.sort(key=lambda r: r[0])
    prev_oi = records[-2][1]
    latest_oi = records[-1][1]
    if prev_oi == 0:
        return None
    return (latest_oi - prev_oi) / prev_oi * 100


async def get_daily_context(symbol: str) -> dict:
    """OI-30-days-ago, 7-day average volume, nearest daily resistance/
    support, and funding-rate history — all change slowly, so this is
    cached for hours rather than refetched every scan cycle. A failure here
    degrades to missing fields (the quality gate just skips those specific
    flags/points), never a crash."""
    cached = DAILY_CONTEXT_CACHE.get(symbol)
    if cached and (time.time() - cached["fetched_at"]) < DAILY_CONTEXT_CACHE_SECONDS:
        return cached

    result = {"fetched_at": time.time(), "oi_30d_ago": None, "oi_latest_from_history": None,
              "oi_24h_change_pct": None,
              "avg_volume_7d": None, "daily_resistance": None, "daily_support": None,
              "funding_rate_history": []}
    try:
        oi_history = await get_open_interest_history(symbol)
        result["oi_30d_ago"] = find_oi_n_days_ago(oi_history, 30)
        result["oi_latest_from_history"] = find_oi_latest(oi_history)
        result["oi_24h_change_pct"] = find_oi_24h_change_pct(oi_history)
    except Exception as e:
        log.warning(f"{symbol}: OI history fetch failed (non-fatal): {e}")

    try:
        funding_hist = await get_funding_rate_history(symbol)
        rates = []
        for rec in funding_hist:
            try:
                rates.append(float(rec["fundingRate"]))
            except (KeyError, ValueError, TypeError):
                continue
        result["funding_rate_history"] = rates
    except Exception as e:
        log.warning(f"{symbol}: funding rate history fetch failed (non-fatal): {e}")

    try:
        daily_klines = await get_klines(symbol, interval="1d", limit=35)
        if daily_klines and len(daily_klines) >= 10:
            closed_rows = daily_klines[:-1]  # oldest-first list; last row is the still-forming day
            turnovers = [float(row[6]) for row in closed_rows[-7:]]
            if turnovers:
                result["avg_volume_7d"] = sum(turnovers) / len(turnovers)
            ddf = klines_to_df(daily_klines)
            d_is_high, d_is_low = pivot_highs_lows(ddf, PIVOT_LOOKBACK)
            current_price = float(ddf["close"].iloc[-1])
            result["daily_resistance"] = nearest_structure_level(ddf, d_is_high, d_is_low, current_price, "above")
            result["daily_support"] = nearest_structure_level(ddf, d_is_high, d_is_low, current_price, "below")
    except Exception as e:
        log.warning(f"{symbol}: daily klines context fetch failed (non-fatal): {e}")

    DAILY_CONTEXT_CACHE[symbol] = result
    return result


def _reject(reason: str, score: int = 0, flags: list[str] | None = None) -> dict:
    """Uniform shape for every hard-reject exit path in evaluate_signal_quality —
    `reason` always lands in flags so it's visible on the dashboard/logs even
    though the signal itself gets suppressed before it ever reaches Telegram."""
    return {"passed": False, "flags": (flags or []) + [reason], "score": score,
            "drop_tp2": False, "discard": True, "momentum_basis": None}


def evaluate_signal_quality(snap, rsi14: float, funding_rate: float | None,
                             oi_30d_ago: float | None, oi_latest_from_history: float | None,
                             oi_24h_change_pct: float | None,
                             avg_volume_7d: float | None, volume_24h: float | None,
                             daily_resistance: float | None, daily_support: float | None,
                             whale_ls_ratio: float | None, funding_rate_history: list | None = None,
                             pct_change_8h: float | None = None, pct_change_1h: float | None = None) -> dict:
    """Applied AFTER the core structure->BOS->retracement engine already
    confirmed a BUY/SELL — hard filters can discard the signal or trim TP2,
    soft flags annotate without suppressing, and the 0-10 score makes the
    final call on whether it's actually worth sending."""
    direction = snap.signal
    flags: list[str] = []
    score = 0
    drop_tp2 = False
    momentum_basis = None
    funding_rate_history = funding_rate_history or []

    def _pct(v):
        return f"{v:.1f}%" if v is not None else "unknown"

    def _ratio(v):
        return f"{v:.2f}:1" if v is not None else "unknown"

    # ---- fund side: majority of the last 3 funding-rate periods, not just
    # the latest single value — a lot less noisy on a rate that flips sign
    # from one period to the next. Falls back to the single latest rate's
    # sign if there isn't enough history yet. ----
    recent_funding = funding_rate_history[-3:] if funding_rate_history else []
    if len(recent_funding) >= 3:
        pos = sum(1 for r in recent_funding if r > 0)
        neg = sum(1 for r in recent_funding if r < 0)
        fund_side = "Bullish" if pos >= 2 else "Bearish" if neg >= 2 else None
    else:
        fund_side = "Bullish" if (funding_rate is not None and funding_rate > 0) else \
                    "Bearish" if (funding_rate is not None and funding_rate < 0) else None

    vol_ratio_pct = (volume_24h / avg_volume_7d * 100) if (avg_volume_7d and avg_volume_7d > 0
                                                            and volume_24h is not None) else None

    # ---- BOS freshness — structure that broke too many candles ago
    # shouldn't be traded off as if it just happened ----
    if snap.entry_bos_bars_ago is None or snap.entry_bos_bars_ago > ENTRY_BOS_MAX_BARS_AGO:
        return _reject("Structure stale — BOS too old")

    # ---- CORE CONFIRMATION GATE — every one of these must hold for the
    # signal's direction, checked before anything else below. Missing data
    # fails closed (treated as a failure, not skipped) since this gate is
    # meant to be a hard go/no-go, not best-effort — EXCEPT open interest,
    # which fails OPEN (flagged unverified, not rejected) since OI history
    # is the one input most likely to be briefly unavailable without that
    # meaning anything is actually wrong with the setup. Entry staleness is
    # NOT re-checked here — it's already enforced upstream in
    # apply_entry_gap_check against the confirmed retracement zone, before
    # this function is even called. ----
    if direction == "BUY":
        whale_ok = whale_ls_ratio is not None and whale_ls_ratio > CORE_GATE_WHALE_BUY_MIN
        if not whale_ok:
            # secondary trigger: a genuine volume surge + confirmed same-direction
            # momentum can substitute for whale confirmation — catches setups
            # where large-trade flow hasn't caught up yet but the broader market
            # already has (retail/algo-driven surges, momentum before whales pile in)
            surge_ok = vol_ratio_pct is not None and vol_ratio_pct >= VOLUME_SURGE_MIN_PCT and \
                       snap.pct_change_24h is not None and snap.pct_change_24h >= VOLUME_SURGE_MOMENTUM_MIN_PCT
            if not surge_ok:
                return _reject(f"Whale ratio {_ratio(whale_ls_ratio)} does not exceed required "
                                f">{CORE_GATE_WHALE_BUY_MIN}:1 for long, and no volume-surge override "
                                f"(need >={VOLUME_SURGE_MIN_PCT}% of 7d avg volume + >={VOLUME_SURGE_MOMENTUM_MIN_PCT}% 24h move)")
            flags.append(f"Volume surge override — whale ratio {_ratio(whale_ls_ratio)} bypassed "
                         f"via {_pct(vol_ratio_pct)} volume + {_pct(snap.pct_change_24h)} 24h move")
            score -= 1
        # Fund side hard check removed — user will verify funding manually
        if fund_side and fund_side != "Bullish":
            flags.append(f"⚠️ Fund side is {fund_side} (not Bullish) — verify manually")
        elif fund_side is None:
            flags.append("⚠️ Fund side unknown — verify funding manually")
        if snap.pct_change_24h is not None and snap.pct_change_24h > CORE_GATE_PRICE_CHG_MAX_PCT:
            return _reject(f"24h change {_pct(snap.pct_change_24h)} — overextended, late entry risk", flags=flags)
        primary_ok = snap.pct_change_24h is not None and \
            CORE_GATE_PRICE_CHG_MIN_PCT <= snap.pct_change_24h <= CORE_GATE_PRICE_CHG_MAX_PCT
        if primary_ok:
            momentum_basis = "PRIMARY"
        else:
            fallback_ok = pct_change_8h is not None and pct_change_8h > CORE_GATE_8H_CHANGE_MIN_PCT and \
                          pct_change_1h is not None and pct_change_1h > CORE_GATE_1H_CHANGE_MIN_PCT
            if not fallback_ok:
                return _reject("No momentum confirmed (24h primary and 8h/1h fallback both missed)", flags=flags)
            momentum_basis = "FALLBACK"
        if vol_ratio_pct is None or vol_ratio_pct <= CORE_GATE_VOLUME_MIN_PCT:
            return _reject(f"Volume {_pct(vol_ratio_pct)} of 7d avg, required >{CORE_GATE_VOLUME_MIN_PCT}%", flags=flags)
        if oi_24h_change_pct is not None and oi_24h_change_pct <= CORE_GATE_OI_24H_MIN_PCT:
            return _reject(f"OI 24h change {_pct(oi_24h_change_pct)}, required >{CORE_GATE_OI_24H_MIN_PCT}%", flags=flags)
        elif oi_24h_change_pct is None:
            flags.append("⚠️ OI UNVERIFIED — 24h OI change could not be confirmed")
    elif direction == "SELL":
        whale_ok = whale_ls_ratio is not None and whale_ls_ratio < CORE_GATE_WHALE_SELL_MAX
        if not whale_ok:
            surge_ok = vol_ratio_pct is not None and vol_ratio_pct >= VOLUME_SURGE_MIN_PCT and \
                       snap.pct_change_24h is not None and snap.pct_change_24h <= -VOLUME_SURGE_MOMENTUM_MIN_PCT
            if not surge_ok:
                return _reject(f"Whale ratio {_ratio(whale_ls_ratio)} does not fall below required "
                                f"<{CORE_GATE_WHALE_SELL_MAX}:1 for short, and no volume-surge override "
                                f"(need >={VOLUME_SURGE_MIN_PCT}% of 7d avg volume + <=-{VOLUME_SURGE_MOMENTUM_MIN_PCT}% 24h move)")
            flags.append(f"Volume surge override — whale ratio {_ratio(whale_ls_ratio)} bypassed "
                         f"via {_pct(vol_ratio_pct)} volume + {_pct(snap.pct_change_24h)} 24h move")
            score -= 1
        # Fund side hard check removed — user will verify funding manually
        if fund_side and fund_side != "Bearish":
            flags.append(f"⚠️ Fund side is {fund_side} (not Bearish) — verify manually")
        elif fund_side is None:
            flags.append("⚠️ Fund side unknown — verify funding manually")
        if snap.pct_change_24h is not None and snap.pct_change_24h < -CORE_GATE_PRICE_CHG_MAX_PCT:
            return _reject(f"24h change {_pct(snap.pct_change_24h)} — overextended, late entry risk", flags=flags)
        primary_ok = snap.pct_change_24h is not None and \
            -CORE_GATE_PRICE_CHG_MAX_PCT <= snap.pct_change_24h <= -CORE_GATE_PRICE_CHG_MIN_PCT
        if primary_ok:
            momentum_basis = "PRIMARY"
        else:
            fallback_ok = pct_change_8h is not None and pct_change_8h < -CORE_GATE_8H_CHANGE_MIN_PCT and \
                          pct_change_1h is not None and pct_change_1h < -CORE_GATE_1H_CHANGE_MIN_PCT
            if not fallback_ok:
                return _reject("No momentum confirmed (24h primary and 8h/1h fallback both missed)", flags=flags)
            momentum_basis = "FALLBACK"
        if vol_ratio_pct is None or vol_ratio_pct <= CORE_GATE_VOLUME_MIN_PCT:
            return _reject(f"Volume {_pct(vol_ratio_pct)} of 7d avg, required >{CORE_GATE_VOLUME_MIN_PCT}%", flags=flags)
        if oi_24h_change_pct is not None and oi_24h_change_pct <= CORE_GATE_OI_24H_MIN_PCT:
            return _reject(f"OI 24h change {_pct(oi_24h_change_pct)}, required >{CORE_GATE_OI_24H_MIN_PCT}%", flags=flags)
        elif oi_24h_change_pct is None:
            flags.append("⚠️ OI UNVERIFIED — 24h OI change could not be confirmed")

    # ---- outer hard RSI bounds (unconditional reject) ----
    if direction == "BUY" and rsi14 >= HARD_RSI14_BUY_MAX:
        return _reject(f"RSI14 {rsi14:.1f} at/above hard ceiling {HARD_RSI14_BUY_MAX}", flags=flags)
    if direction == "SELL" and rsi14 <= HARD_RSI14_SELL_MIN:
        return _reject(f"RSI14 {rsi14:.1f} at/below hard floor {HARD_RSI14_SELL_MIN}", flags=flags)

    # ---- whale flow hard block — a signal against a decisively opposite
    # whale flow gets rejected outright rather than merely penalized, since
    # a soft flag previously let e.g. a 0.5:1-short whale ratio pass on a
    # BUY signal ----
    if direction == "BUY" and whale_ls_ratio is not None and whale_ls_ratio < WHALE_HARD_BLOCK_BUY_MIN:
        return _reject(f"Whale flow too short for long ({whale_ls_ratio:.2f}:1)", flags=flags)
    if direction == "SELL" and whale_ls_ratio is not None and whale_ls_ratio > WHALE_HARD_BLOCK_SELL_MAX:
        return _reject(f"Whale flow too long for short ({whale_ls_ratio:.2f}:1)", flags=flags)

    # ---- fund flow + whale both against direction (soft flag only — fund side
    # is no longer a hard gate; user verifies funding manually) ----
    if direction == "BUY" and fund_side == "Bearish" and whale_ls_ratio is not None and whale_ls_ratio < 1.0:
        flags.append("⚠️ Fund flow and whale flow both bearish — verify manually")
    if direction == "SELL" and fund_side == "Bullish" and whale_ls_ratio is not None and whale_ls_ratio > 1.2:
        flags.append("⚠️ Fund flow and whale flow both bullish — verify manually")

    # ---- borderline RSI zone: needs stronger whale confirmation only
    # (fund-side requirement removed) ----
    if direction == "BUY" and BORDERLINE_RSI14_BUY_MIN <= rsi14 < HARD_RSI14_BUY_MAX:
        ok = whale_ls_ratio is not None and whale_ls_ratio >= BORDERLINE_WHALE_LS_BUY_MIN
        if not ok:
            return _reject(f"Borderline RSI14 ({rsi14:.1f}) requires whale_ratio >= "
                            f"{BORDERLINE_WHALE_LS_BUY_MIN}", flags=flags)
    if direction == "SELL" and HARD_RSI14_SELL_MIN < rsi14 <= BORDERLINE_RSI14_SELL_MAX:
        ok = whale_ls_ratio is not None and whale_ls_ratio <= BORDERLINE_WHALE_LS_SELL_MAX
        if not ok:
            return _reject(f"Borderline RSI14 ({rsi14:.1f}) requires whale_ratio <= "
                            f"{BORDERLINE_WHALE_LS_SELL_MAX}", flags=flags)


    # ---- funding-rate lock detection — a rate that hasn't actually moved
    # across many periods signals an illiquid/thinly-quoted contract, not a
    # genuine funding signal ----
    if len(funding_rate_history) >= 10:
        unique_values = {round(v, 8) for v in funding_rate_history}
        if len(unique_values) <= FUNDING_LOCK_HARD_MAX_UNIQUE:
            return _reject("Funding rate locked — illiquid environment", flags=flags)
        elif len(unique_values) <= FUNDING_LOCK_SOFT_MAX_UNIQUE:
            flags.append("Funding rate low variance — monitor for squeeze")
            score -= 1

    # ---- TP1/TP2 separation — enforced up front, before scoring, since a
    # malformed (near-identical) target pair shouldn't earn any points at all ----
    if snap.take_profit is not None and snap.take_profit2 is not None and snap.take_profit != 0:
        tp_separation_pct = abs(snap.take_profit2 - snap.take_profit) / abs(snap.take_profit) * 100
        if tp_separation_pct < TP_SEPARATION_HARD_MIN_PCT:
            return _reject("TP1 and TP2 too close — malformed signal", flags=flags)
        elif tp_separation_pct < TP_SEPARATION_SOFT_MIN_PCT:
            flags.append("TP1/TP2 separation weak — limited trailing potential")
            score -= 0.5
    # kept as a secondary trim (not a reject) for the case where TP2 clears
    # the hard floor above but is still tight relative to price rather than to TP1
    if snap.take_profit is not None and snap.take_profit2 is not None and snap.price:
        gap_pct = abs(snap.take_profit2 - snap.take_profit) / snap.price * 100
        if gap_pct <= MIN_TP_GAP_PCT:
            drop_tp2 = True

    entry = snap.entry_price if snap.entry_price else snap.price
    tp1_rr = None
    if snap.stop_loss is not None and snap.take_profit is not None and entry:
        risk = abs(entry - snap.stop_loss)
        reward = abs(snap.take_profit - entry)
        if risk > 0:
            tp1_rr = reward / risk
    if tp1_rr is None or tp1_rr < MIN_TP1_RR:
        return _reject("TP1 reward:risk below minimum", flags=flags)

    if 45 <= rsi14 <= 62:
        score += 2
    elif 63 <= rsi14 <= 67:
        score += 1

    # ---- OI change: contracts-only comparison. oi_30d_ago and
    # oi_latest_from_history both come from the same open-interest-history
    # endpoint (units: contracts) — snap.open_interest is deliberately NOT
    # used here, since that's the ticker's USD-denominated openInterestValue
    # and mixing the two units is what produced the old "declining 96%"
    # false flags on OI that was actually rising live. oi_latest_from_history
    # is an independent live cross-check against a ticker-derived figure. ----
    if oi_30d_ago and oi_30d_ago > 0:
        live_oi_change_pct = None
        if oi_latest_from_history is not None:
            live_oi_change_pct = (oi_latest_from_history - oi_30d_ago) / oi_30d_ago * 100

        oi_change_pct = live_oi_change_pct
        if snap.open_interest_contracts is not None:
            signal_oi_change_pct = (snap.open_interest_contracts - oi_30d_ago) / oi_30d_ago * 100
            if live_oi_change_pct is not None:
                if abs(signal_oi_change_pct - live_oi_change_pct) > OI_CONTRADICTION_DIFF_PCT:
                    oi_change_pct = live_oi_change_pct
                    flags.append("OI source corrected — live data used")
                elif (signal_oi_change_pct > 0) != (live_oi_change_pct > 0):
                    oi_change_pct = live_oi_change_pct
                    score -= 1
                    flags.append("OI data conflict detected — live value applied")
                else:
                    oi_change_pct = signal_oi_change_pct
            else:
                oi_change_pct = signal_oi_change_pct

        if oi_change_pct is not None:
            if oi_change_pct <= -OI_DECLINE_FLAG_PCT:
                flags.append(f"OI declining {abs(oi_change_pct):.0f}% vs 30d ago")
            if oi_change_pct > 5:
                score += 2
            elif abs(oi_change_pct) < 5:
                score += 1

    if funding_rate is not None:
        funding_pct = funding_rate * 100
        if funding_pct > FUNDING_CROWDED_LONG_PCT:
            flags.append(f"crowded long funding ({funding_pct:.3f}%)")
        elif funding_pct < FUNDING_SHORT_PRESSURE_PCT:
            flags.append(f"short pressure funding ({funding_pct:.3f}%)")

    if snap.long_pct is not None:
        if snap.long_pct > LS_SQUEEZE_PCT:
            flags.append(f"squeeze risk ({snap.long_pct:.0f}% long)")
        elif snap.long_pct > LS_CROWDED_PCT:
            flags.append(f"crowded longs ({snap.long_pct:.0f}% long)")
        if 52 <= snap.long_pct <= 65:
            score += 1

    if avg_volume_7d and avg_volume_7d > 0 and volume_24h is not None:
        vol_ratio_pct = volume_24h / avg_volume_7d * 100
        if vol_ratio_pct < VOLUME_LOW_PCT:
            flags.append(f"low volume ({vol_ratio_pct:.0f}% of 7d avg)")
        if volume_24h >= avg_volume_7d:
            score += 1

    near_resistance = False
    if direction == "BUY" and daily_resistance and snap.price:
        near_resistance = abs(daily_resistance - snap.price) / snap.price * 100 <= RESISTANCE_PROXIMITY_PCT
    elif direction == "SELL" and daily_support and snap.price:
        near_resistance = abs(daily_support - snap.price) / snap.price * 100 <= RESISTANCE_PROXIMITY_PCT
    if near_resistance:
        flags.append("entry at resistance" if direction == "BUY" else "entry at support")
    else:
        score += 1

    if direction == "BUY" and whale_ls_ratio is not None:
        if whale_ls_ratio < WHALE_LS_WEAK_THRESHOLD:
            flags.append(f"weak whale conviction ({whale_ls_ratio:.2f}:1)")
        if whale_ls_ratio >= 1.4:
            score += 2
        elif whale_ls_ratio >= 1.2:
            score += 1

    if tp1_rr >= 2.0:
        score += 1

    score = max(0, round(score))
    return {"passed": score >= SIGNAL_SCORE_ALERT_MIN, "flags": flags, "score": score,
            "drop_tp2": drop_tp2, "discard": False, "momentum_basis": momentum_basis}


def apply_entry_gap_check(snap: SymbolSnapshot, computed: dict) -> str | None:
    """Entry-staleness guard (spec Step 5, kept deliberately simple — no
    adjustment attempt, just a straight skip). Compares live price against
    the confirmed retracement-zone boundary (the actual level the pipeline
    confirmed on), not against itself — snap.price and "entry" used to be
    treated as identical, which is how signals were dispatching 1-2% past a
    usable entry. If price has drifted too far from that confirmed level,
    the signal is simply too stale to trade off — no attempt to rescue it
    by hunting for a nearby pullback level, since a moved entry with a
    reworked stop/TP is arguably a different trade than the one that was
    actually confirmed."""
    direction = snap.signal
    if direction not in ("BUY", "SELL") or not snap.price:
        return None

    if direction == "BUY":
        entry_ref = computed.get("bull_entry_ref")
    else:
        entry_ref = computed.get("bear_entry_ref")
    if not entry_ref or entry_ref <= 0:
        return None  # no confirmed zone boundary available — nothing to compare against, don't fabricate a reject

    gap_pct = abs(snap.price - entry_ref) / entry_ref * 100
    if gap_pct > PRICE_PAST_ENTRY_PCT:
        return "Entry stale — price moved too far"
    return None


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
    # Bybit's bulk ticker already carries open interest — no separate call needed.
    # openInterestValue is USD-denominated (display only); openInterest is in
    # contracts, the SAME unit as the open-interest-history endpoint used for
    # the 30d-ago comparison — keeping these separate is what fixes the old
    # unit-mismatch bug (comparing a USD value against a contract count
    # produced nonsense % changes, e.g. "OI down 96%" while live OI was up).
    oiv = ticker_24h.get("openInterestValue") if ticker_24h else None
    snap.open_interest = float(oiv) if oiv not in (None, "") else None
    oic = ticker_24h.get("openInterest") if ticker_24h else None
    snap.open_interest_contracts = float(oic) if oic not in (None, "") else None
    snap.long_short_account_ratio = (
        float(ls["buyRatio"]) / float(ls["sellRatio"]) if ls and float(ls.get("sellRatio", 0) or 0) > 0 else None
    )
    snap.long_pct = float(ls["buyRatio"]) * 100 if ls else None
    snap.short_pct = float(ls["sellRatio"]) * 100 if ls else None
    snap.swept_low = computed["swept_low"]
    snap.swept_high = computed["swept_high"]
    snap.bull_fvg_recent = computed["bull_fvg_recent"]
    snap.bear_fvg_recent = computed["bear_fvg_recent"]
    if computed["bos"] is not None:
        bos_idx = len(computed["df"]) - 2
        quality = classify_bos_quality(computed["df"], computed["is_high"], computed["is_low"],
                                        bos_idx, computed["bos"], whale["whale_bias"])
        snap.bos = computed["bos"] if quality else None
    else:
        snap.bos = None
    snap.atr = computed["atr"]
    snap.stop_ref_below = computed["stop_ref_below"]
    snap.resistance_level = computed["resistance_level"]
    snap.resistance_touches = computed["resistance_touches"]
    snap.support_level = computed["support_level"]
    snap.support_touches = computed["support_touches"]
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
    # Trend alignment loosened from AND to OR — either Supertrend OR the EMA
    # cross agreeing is enough, doesn't need both simultaneously.
    bullish_trend = snap.supertrend_trend == 1 or snap.ema_fast > snap.ema_slow
    bearish_trend = snap.supertrend_trend == -1 or snap.ema_fast < snap.ema_slow
    # VWAP softened from a strict cross to a 0.3% band around it — price
    # doesn't need to have fully cleared VWAP, just not be meaningfully on
    # the wrong side of it.
    above_vwap = snap.vwap is not None and snap.price >= snap.vwap * (1 - VWAP_BAND_PCT / 100)
    below_vwap = snap.vwap is not None and snap.price <= snap.vwap * (1 + VWAP_BAND_PCT / 100)

    FUNNEL["scanned"] += 1
    if bullish_trend:
        FUNNEL["bull_trend_ok"] += 1
    if bearish_trend:
        FUNNEL["bear_trend_ok"] += 1
    if above_vwap:
        FUNNEL["bull_vwap_ok"] += 1
    if below_vwap:
        FUNNEL["bear_vwap_ok"] += 1
    if computed["bull_stage"] != "no_bos":
        FUNNEL["bull_bos_found"] += 1
    if computed["bear_stage"] != "no_bos":
        FUNNEL["bear_bos_found"] += 1
    if computed["bullish_retracement_confirmed"]:
        FUNNEL["bull_retracement_confirmed"] += 1
    if computed["bearish_retracement_confirmed"]:
        FUNNEL["bear_retracement_confirmed"] += 1

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
        snap.entry_bos_direction = "bullish"
        snap.entry_bos_bars_ago = computed["bull_bars_ago"]
        FUNNEL["signal_candidate"] += 1
    elif sell:
        snap.signal = "SELL"
        snap.entry_bos_direction = "bearish"
        snap.entry_bos_bars_ago = computed["bear_bars_ago"]
        FUNNEL["signal_candidate"] += 1
    else:
        snap.signal = "NEUTRAL"
        snap.entry_bos_direction = None
        snap.entry_bos_bars_ago = None
    snap.entry_price = snap.price  # default: no adjustment. May be moved by apply_entry_gap_check below.
    snap.pct_change_8h = computed["pct_change_8h"]
    snap.pct_change_1h = None
    snap.momentum_basis = None

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

    # ---- post-signal quality gate: hard filters, soft flags, 0-10 score ----
    snap.signal_score = 0
    snap.signal_flags = []
    snap.signal_tag = None

    if snap.signal in ("BUY", "SELL"):
        # ---- entry-staleness check runs first: rejects outright if price
        # has drifted too far from the confirmed retracement-zone level ----
        gap_reject_reason = apply_entry_gap_check(snap, computed)
        if gap_reject_reason:
            snap.signal_flags = [gap_reject_reason]
            snap.signal = "NEUTRAL"
            snap.entry_bos_direction = None
            snap.entry_bos_bars_ago = None
            return snap
        FUNNEL["entry_gap_ok"] += 1

        funding_rate = None
        volume_24h = None
        if ticker_24h:
            try:
                if ticker_24h.get("fundingRate") not in (None, ""):
                    funding_rate = float(ticker_24h["fundingRate"])
            except (TypeError, ValueError):
                pass
            try:
                if ticker_24h.get("turnover24h") not in (None, ""):
                    volume_24h = float(ticker_24h["turnover24h"])
            except (TypeError, ValueError):
                pass

        daily_ctx = await get_daily_context(symbol)

        # 1h close-to-close change — only needed for the momentum fallback,
        # so this fetch only fires for symbols that already cleared the
        # BOS/retracement engine (a small subset of the full scan universe),
        # not on every symbol every cycle.
        try:
            klines_1h = await get_klines(symbol, interval="1h", limit=10)
            if klines_1h and len(klines_1h) >= 3:
                df_1h = klines_to_df(klines_1h)
                last_closed_1h = len(df_1h) - 2
                if last_closed_1h >= 1:
                    close_now_1h = float(df_1h["close"].iloc[last_closed_1h])
                    close_prev_1h = float(df_1h["close"].iloc[last_closed_1h - 1])
                    if close_prev_1h:
                        snap.pct_change_1h = (close_now_1h - close_prev_1h) / close_prev_1h * 100
        except Exception as e:
            log.warning(f"{symbol}: 1h kline fetch for momentum fallback failed (non-fatal): {e}")

        quality = evaluate_signal_quality(
            snap, computed["rsi14"], funding_rate,
            daily_ctx["oi_30d_ago"], daily_ctx["oi_latest_from_history"], daily_ctx["oi_24h_change_pct"],
            daily_ctx["avg_volume_7d"], volume_24h,
            daily_ctx["daily_resistance"], daily_ctx["daily_support"], whale["whale_ls_ratio"],
            daily_ctx["funding_rate_history"], snap.pct_change_8h, snap.pct_change_1h,
        )
        snap.signal_score = quality["score"]
        snap.signal_flags = quality["flags"]
        snap.momentum_basis = quality.get("momentum_basis")
        if quality["drop_tp2"]:
            snap.take_profit2 = None
        if quality["discard"] or not quality["passed"]:
            snap.signal = "NEUTRAL"
            snap.entry_bos_direction = None
            snap.entry_bos_bars_ago = None
            if quality["discard"]:
                # a discard's reason is always the last flag appended by
                # _reject() — see evaluate_signal_quality
                reason = quality["flags"][-1] if quality["flags"] else "unknown reject"
            else:
                reason = f"score {quality['score']} below alert threshold {SIGNAL_SCORE_ALERT_MIN}"
            _record_core_gate_reject(reason)
        else:
            snap.signal_tag = "🟢" if quality["score"] >= SIGNAL_SCORE_GREEN_MIN else "🟡"
            FUNNEL["core_gate_passed"] += 1

    return snap

# ======================================================================
# TELEGRAM ALERTS + COMMAND BOT (/analyze)
# ======================================================================

_tg_client: httpx.AsyncClient | None = None


def _get_tg_client():
    global _tg_client
    if _tg_client is None:
        _tg_client = httpx.AsyncClient(timeout=30.0)
    return _tg_client


async def send_alert(text: str, chat_id: str | None = None):
    """Send a message. Defaults to TELEGRAM_CHAT_ID (scanner alerts).
    Pass chat_id to reply to a command user."""
    target = chat_id or TELEGRAM_CHAT_ID
    if not TELEGRAM_BOT_TOKEN or not target:
        log.warning("Telegram not configured — skipping alert: %s", text[:120])
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    # Telegram hard limit ~4096 chars
    if len(text) > 4000:
        text = text[:3990] + "\n…"
    payload = {
        "chat_id": target,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        client = _get_tg_client()
        resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            log.warning("Telegram send failed: %s %s", resp.status_code, resp.text[:200])
    except httpx.HTTPError as e:
        log.warning("Telegram send error: %s", e)


async def send_photo(png: bytes, chat_id: str | None = None, caption: str | None = None):
    target = chat_id or TELEGRAM_CHAT_ID
    if not TELEGRAM_BOT_TOKEN or not target or not png:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    data = {"chat_id": target}
    if caption:
        data["caption"] = caption
    try:
        client = _get_tg_client()
        resp = await client.post(url, data=data, files={"photo": ("chart.png", png, "image/png")})
        if resp.status_code != 200:
            log.warning("Telegram photo failed: %s %s", resp.status_code, resp.text[:200])
    except httpx.HTTPError as e:
        log.warning("Telegram photo error: %s", e)


SUPPORTED_TFS = ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w")
_SYMBOL_TOKEN_RE = re.compile(r"^[A-Z0-9]{2,15}$")
_TF_RE = re.compile(r"^(\d+)(M|MIN|MINS|MINUTE|MINUTES|H|HR|HRS|HOUR|HOURS|D|DAY|DAYS|W|WK|WEEK|WEEKS)$")
_TF_WORDS = {"DAILY": "1d", "DAY": "1d", "D": "1d", "WEEKLY": "1w", "WEEK": "1w", "W": "1w",
             "HOURLY": "1h", "H": "1h"}


def parse_tf(text: str) -> str | None:
    """'1h', '1hr', '15m', '15min', '1 hour', 'daily', 'D' -> '1h' / '15m' / '1d'.
    Returns 'BAD:<text>' if it looks like a timeframe but isn't supported, None if it isn't one."""
    t = (text or "").strip().upper().replace(" ", "")
    if not t:
        return None
    if t in _TF_WORDS:
        return _TF_WORDS[t]
    m = _TF_RE.match(t)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    kind = "m" if unit.startswith("M") else "h" if unit.startswith("H") else "d" if unit.startswith("D") else "w"
    tf = f"{n}{kind}"
    return tf if tf in SUPPORTED_TFS else f"BAD:{tf}"


def parse_symbol_tf(text: str) -> tuple[str, str] | None:
    """'ENA' / 'pha 4hr' / 'BTC 1h' / '15m btc' / 'ENAUSDT.P 30 min' -> (symbol_token, tf).
    Default timeframe is 4h. Returns None if the message isn't a symbol request."""
    toks = (text or "").strip().upper().split()
    if not 1 <= len(toks) <= 4:
        return None
    # timeframe first ("1h btc")
    if len(toks) >= 2 and parse_tf(toks[0]) and not _TF_RE.match(toks[1]):
        toks = toks[1:] + toks[:1]
    first = toks[0]
    if first.endswith(".P"):
        first = first[:-2]
    if not _SYMBOL_TOKEN_RE.match(first):
        return None
    rest = "".join(toks[1:])
    if not rest:
        return first, "4h"
    tf = parse_tf(rest)
    if tf is None:
        return None
    return first, tf


async def run_analysis(chat_id: str, raw_symbol: str, tf: str = "4h"):
    if tf.startswith("BAD:"):
        await send_alert(f"{tf[4:]} isn't supported. Use one of: " + ", ".join(SUPPORTED_TFS) +
                         ".\nExample: <code>BTC 1h</code>", chat_id=chat_id)
        return
    symbol = normalize_symbol(raw_symbol)
    await send_alert(f"Analyzing <b>{symbol}</b> ({tf})…", chat_id=chat_id)
    try:
        text, png, read = await _analyze_core(raw_symbol, tf)
        if read:
            n = register_watches(chat_id, symbol, tf, read)
            if n:
                text += (f"\n\n👁 Watching this setup — I will message you here when the {_CONFIRM_TF.get(tf, '1h')} "
                         f"rejection candle prints and holds. /watches to manage.")
        if png:
            await send_photo(png, chat_id=chat_id)
        for chunk in split_message(text):
            await send_alert(chunk, chat_id=chat_id)
    except Exception as e:
        log.exception("analysis failed")
        await send_alert(f"Analysis failed: {e}", chat_id=chat_id)


def format_signals_list() -> str:
    rows = [r for r in STATE.values() if r.get("signal") in ("BUY", "SELL")]
    if not rows:
        return (f"🎯 <b>Signals</b>\n\nNo pairs currently meet the setup criteria.\n"
                f"Scanned so far: {len(STATE)} of {len(UNIVERSE)} pairs.")
    rows.sort(key=lambda r: r.get("signal_score") or 0, reverse=True)
    lines = [f"🎯 <b>Signals</b> — {len(rows)} pair(s) meeting the setup criteria", ""]
    for r in rows[:15]:
        emoji = "🟢" if r["signal"] == "BUY" else "🔴"
        entry = r.get("entry_price") or r.get("price")
        line = f"{emoji} <b>{r['symbol']}</b> — {r['signal']} · Score {r.get('signal_score', 0)}/10\n"
        line += f"Entry {_fmt_price(entry)}"
        if r.get("stop_loss") is not None:
            line += f" · SL {_fmt_price(r['stop_loss'])}"
        if r.get("take_profit") is not None:
            line += f" · TP1 {_fmt_price(r['take_profit'])}"
        if r.get("take_profit2") is not None:
            line += f" · TP2 {_fmt_price(r['take_profit2'])}"
        ti = trend_indicator_text(r.get("supertrend_trend"), r.get("ema_fast"), r.get("ema_slow"))
        if ti:
            line += f"\n{ti}"
        lines.append(line)
        lines.append("")
    if len(rows) > 15:
        lines.append(f"…and {len(rows) - 15} more.")
    lines.append("<i>Not financial advice.</i>")
    return "\n".join(lines)


WATCHES: dict = {}
MAX_WATCHES_PER_CHAT = 10
WATCH_TTL_SEC = 24 * 3600


def register_watches(chat_id: str, symbol: str, tf: str, read: dict) -> int:
    """Watch each plan from this analysis for a confirmed trigger; alerts go to chat_id."""
    ctf = _CONFIRM_TF.get(tf, "1h")
    items = []
    if read.get("plan"):
        items.append((read["plan"], read.get("trigger")))
    else:
        for pk, tk in (("plan_bull", "trigger_bull"), ("plan_bear", "trigger_bear")):
            if read.get(pk):
                items.append((read[pk], read.get(tk)))
    n = 0
    for plan, tr in items:
        st = (tr or {}).get("status")
        if st not in ("waiting", "in_zone", "through", "rejection"):
            continue
        key = (str(chat_id), symbol, tf, plan["side"])
        WATCHES[key] = {"chat_id": str(chat_id), "symbol": symbol, "tf": tf, "ctf": ctf, "plan": dict(plan),
                        "stage": 1 if st == "rejection" else 0, "rej_time": (tr or {}).get("rej_time"),
                        "ignore_before": 0, "created": time.time()}
        n += 1
    mine = sorted([k for k in WATCHES if k[0] == str(chat_id)], key=lambda k: WATCHES[k]["created"])
    for k in mine[:-MAX_WATCHES_PER_CHAT]:
        WATCHES.pop(k, None)
    return n


def _watch_msg(kind: str, w: dict, tr: dict) -> str:
    pl, ctf = w["plan"], w["ctf"]
    long = pl["side"] == "long"
    side = "LONG" if long else "SHORT"
    ab = "above" if long else "below"
    zone = f"{_fp(pl['zone_lo'])}-{_fp(pl['zone_hi'])}"
    head = f"<b>{w['symbol']}</b> {side} ({w['tf']} setup)"
    lvl = _fp(tr.get("level"))
    if kind == "rejection":
        return (f"🟡 {head}\n{ctf} {'bullish' if long else 'bearish'} rejection candle printed in the zone {zone} "
                f"at {_fmt_t(tr['rej_time'], ctf)} (close {_fp(tr['rej_close'])}).\n"
                f"Waiting for the next {ctf} candle to hold {ab} {lvl}.")
    if kind == "confirmed":
        tps = ", ".join(_fp(t) for t in pl.get("tps", []))
        return (f"✅ {head} — <b>trigger confirmed</b>\n{ctf} rejection at {_fmt_t(tr['rej_time'], ctf)} is holding {ab} {lvl}.\n"
                f"Entry zone {zone} · price {_fp(tr['price'])}\nTargets: {tps}\n"
                f"Invalidation: {w['tf']} close {'below' if long else 'above'} {_fp(pl['invalidation'])}\n"
                f"<i>Not financial advice.</i>")
    if kind == "failed":
        return (f"❌ {head}\nThe {_fmt_t(tr['rej_time'], ctf)} rejection failed — a {ctf} candle closed back through {lvl}. "
                f"Still watching for a fresh rejection.")
    if kind == "invalid":
        return f"🚫 {head}\nA {ctf} candle closed beyond the invalidation level {_fp(pl['invalidation'])}. Watch removed."
    if kind == "missed":
        return f"⏭ {head}\nPrice reached the first target ({_fp(pl['tps'][0])}) before a trigger. Setup missed — watch removed."
    return ""


async def _check_watches():
    if not WATCHES:
        return
    now = time.time()
    groups: dict = {}
    for key, w in list(WATCHES.items()):
        if now - w["created"] > WATCH_TTL_SEC:
            WATCHES.pop(key, None)
            await send_alert(f"⌛ Watch expired: <b>{w['symbol']}</b> {w['plan']['side'].upper()} ({w['tf']} setup) — no trigger in 24h.",
                             chat_id=w["chat_id"])
            continue
        groups.setdefault((w["symbol"], w["ctf"]), []).append((key, w))
    for (symbol, ctf), items in groups.items():
        rows = await get_klines(symbol, interval=ctf, limit=30)
        if not rows:
            continue
        for key, w in items:
            tr = evaluate_trigger(w["plan"], rows, w["ignore_before"])
            st = tr.get("status")
            if st in ("invalid", "missed"):
                await send_alert(_watch_msg(st, w, tr), chat_id=w["chat_id"])
                WATCHES.pop(key, None)
            elif st == "failed":
                await send_alert(_watch_msg("failed", w, tr), chat_id=w["chat_id"])
                w["ignore_before"], w["stage"], w["rej_time"] = tr["last_closed_time"], 0, None
            elif st == "confirmed":
                await send_alert(_watch_msg("confirmed", w, tr), chat_id=w["chat_id"])
                WATCHES.pop(key, None)
            elif st == "rejection":
                if w["rej_time"] != tr["rej_time"]:
                    await send_alert(_watch_msg("rejection", w, tr), chat_id=w["chat_id"])
                    w["stage"], w["rej_time"] = 1, tr["rej_time"]
            elif st in ("waiting", "in_zone", "through"):
                w["stage"] = 0


async def trigger_watch_loop():
    """Every minute: check each watched setup on its confirmation timeframe and DM the requester."""
    while True:
        await asyncio.sleep(60)
        try:
            await _check_watches()
        except Exception as e:
            log.warning("watch loop error: %s", e)


def format_watches(chat_id: str) -> str:
    mine = [w for k, w in WATCHES.items() if k[0] == str(chat_id)]
    if not mine:
        return "👁 No active watches.\nAnalyze a pair (e.g. <code>BTC 1h</code>) and I will watch its setup for you."
    lines = ["👁 <b>Active watches</b>", ""]
    for w in sorted(mine, key=lambda x: x["created"]):
        pl = w["plan"]
        lines.append(f"{w['symbol']} {pl['side'].upper()} · {w['tf']} setup · zone {_fp(pl['zone_lo'])}-{_fp(pl['zone_hi'])} · "
                     f"trigger {w['ctf']} · {'rejection printed' if w['stage'] else 'waiting'}")
    lines += ["", "Remove with <code>/unwatch SYMBOL</code> or <code>/unwatch all</code>."]
    return "\n".join(lines)


async def send_signal_narrative(symbol: str):
    """Chart + written analysis, posted right after a BUY/SELL alert."""
    try:
        text, png = await _analyze_full(symbol)
        if png:
            await send_photo(png)
        for chunk in split_message(text):
            await send_alert(chunk)
    except Exception as e:
        log.warning("%s: signal narrative failed: %s", symbol, e)


def _breadth_class(r: dict) -> str:
    st, ef, es = r.get("supertrend_trend"), r.get("ema_fast"), r.get("ema_slow")
    if ef is None or es is None:
        return "mixed"
    if st == 1 and ef > es:
        return "bullish"
    if st == -1 and ef < es:
        return "bearish"
    return "mixed"


def _scan_stats() -> dict:
    rows = list(STATE.values())
    counts = {"bullish": 0, "bearish": 0, "mixed": 0}
    for r in rows:
        counts[_breadth_class(r)] += 1
    sigs = [r for r in rows if r.get("signal") in ("BUY", "SELL")]
    sigs.sort(key=lambda r: r.get("signal_score") or 0, reverse=True)
    movers = [r for r in rows if r.get("pct_change_24h") is not None]
    movers.sort(key=lambda r: r["pct_change_24h"], reverse=True)
    return {"rows": rows, "counts": counts, "sigs": sigs, "movers": movers}


def format_scan_summary() -> str:
    st = _scan_stats()
    if not st["rows"]:
        return "🔍 <b>Scan summary</b>\n\nThe scanner is still warming up — no pairs analysed yet."
    c = st["counts"]
    buys = sum(1 for r in st["sigs"] if r["signal"] == "BUY")
    sells = len(st["sigs"]) - buys
    lines = ["🔍 <b>Scan summary</b>", "",
             f"Scanned {len(st['rows'])} of {len(UNIVERSE)} pairs (4h)",
             f"Trend (SuperTrend + EMA{EMA_FAST}/{EMA_SLOW}): 🟢 {c['bullish']} bullish · 🔴 {c['bearish']} bearish · ⚪ {c['mixed']} mixed",
             f"Signals: {buys} BUY · {sells} SELL"]
    for r in st["sigs"][:3]:
        e = "🟢" if r["signal"] == "BUY" else "🔴"
        lines.append(f"{e} {r['symbol']} {r['signal']} · score {r.get('signal_score', 0)}/10")
    mv = st["movers"]
    if len(mv) >= 6:
        lines.append("")
        lines.append("Top gainers 24h: " + " · ".join(f"{r['symbol']} {r['pct_change_24h']:+.1f}%" for r in mv[:3]))
        lines.append("Top losers 24h: " + " · ".join(f"{r['symbol']} {r['pct_change_24h']:+.1f}%" for r in reversed(mv[-3:])))
    lines.append("")
    lines.append("<i>Informational only — not financial advice.</i>")
    return "\n".join(lines)


async def fetch_fear_greed() -> tuple[int, str] | None:
    try:
        client = _get_tg_client()
        r = await client.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        if r.status_code != 200:
            return None
        d = r.json()["data"][0]
        return int(d["value"]), str(d["value_classification"])
    except Exception as e:
        log.warning("fear&greed fetch failed: %s", e)
        return None


def render_pulse_png(date_str: str, headline: str, btc_price: float | None,
                     btc_chg: float | None, closes: list[float]) -> bytes | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from io import BytesIO
    except ImportError:
        return None
    try:
        fig = plt.figure(figsize=(10, 5.4), dpi=110)
        fig.patch.set_facecolor("#0b0b0d")
        fig.text(0.06, 0.92, f"Market Pulse · {date_str}", color="#bdbdbd", fontsize=11, va="top")
        fig.text(0.06, 0.84, "\n".join(textwrap.wrap(headline, 26)[:3]), color="white",
                 fontsize=27, fontweight="bold", va="top", linespacing=1.15)
        if btc_price is not None:
            up = (btc_chg or 0) >= 0
            col = "#26a69a" if up else "#ef5350"
            arrow = "▲" if up else "▼"
            chg_s = f"  {arrow} {abs(btc_chg):.1f}%  24h" if btc_chg is not None else ""
            fig.text(0.06, 0.40, f"BTC ${btc_price:,.0f}", color="white", fontsize=13, fontweight="bold", va="top")
            fig.text(0.30, 0.40, chg_s, color=col, fontsize=12, va="top")
        if len(closes) >= 3:
            ax = fig.add_axes([0.06, 0.05, 0.88, 0.28])
            ax.set_facecolor("#0b0b0d")
            ax.plot(range(len(closes)), closes, color="#ff7a1a", linewidth=2)
            ax.fill_between(range(len(closes)), closes, min(closes), color="#ff7a1a", alpha=0.12)
            ax.scatter([len(closes) - 1], [closes[-1]], color="#ff7a1a", s=40, zorder=5)
            ax.set_ylim(min(closes) * 0.998, max(closes) * 1.002)
            ax.axis("off")
        buf = BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        plt.close(fig)
        return buf.getvalue()
    except Exception as e:
        log.warning("pulse chart failed: %s", e)
        return None


async def build_market_pulse() -> tuple[bytes | None, str]:
    now = datetime.now(WAT)
    date_str = f"{now.strftime('%b')} {now.day}"
    tickers = await get_24hr_tickers()
    btc = tickers.get("BTCUSDT") if tickers else None
    btc_price = btc_chg = None
    try:
        if btc:
            btc_price = float(btc["lastPrice"])
            btc_chg = float(btc["price24hPcnt"]) * 100
    except (KeyError, TypeError, ValueError):
        pass
    ls = await get_long_short_account_ratio("BTCUSDT")
    long_pct = float(ls["buyRatio"]) * 100 if ls and ls.get("buyRatio") is not None else None
    fg = await fetch_fear_greed()
    kl = await get_klines("BTCUSDT", "1h", 24)
    closes = [float(r[4]) for r in kl] if kl else []

    st = _scan_stats()
    alts = [r["pct_change_24h"] for r in st["rows"] if r["symbol"] != "BTCUSDT" and r.get("pct_change_24h") is not None]
    alt_rel = "in line"
    if alts and btc_chg is not None:
        diff = float(np.median(alts)) - btc_chg
        alt_rel = "outperforming" if diff > 1 else ("lagging" if diff < -1 else "in line")

    # subtitle
    sub = f"{fg[1]}. " if fg else ""
    if btc_price is not None and btc_chg is not None:
        sub += f"BTC {'down' if btc_chg < 0 else 'up'} {abs(btc_chg):.1f}% at ${btc_price:,.0f}, alts {alt_rel}."
    else:
        sub += "BTC data unavailable right now."

    # template read bullets
    bullets = []
    crowd_headline = None
    if long_pct is not None and btc_chg is not None:
        if long_pct >= 60 and btc_chg < 0:
            bullets.append(f"The crowd is {long_pct:.0f}% long into a falling market — crowded longs can speed up a drop if support breaks.")
            crowd_headline = f"Crowd is {long_pct:.0f}% long into a falling market"
        elif long_pct <= 40 and btc_chg > 0:
            bullets.append(f"The crowd is {100 - long_pct:.0f}% short into a rising market — a squeeze higher is possible if resistance breaks.")
            crowd_headline = f"Crowd is {100 - long_pct:.0f}% short into a rising market"
        else:
            bullets.append(f"Positioning is fairly balanced ({long_pct:.0f}% long / {100 - long_pct:.0f}% short accounts).")
    if fg:
        v = fg[0]
        if v <= 25:
            bullets.append(f"Extreme fear ({v}/100) — the crowd is scared; sharp bounces often start from here, but trends can stay down.")
        elif v <= 45:
            bullets.append(f"Fear ({v}/100) — sentiment is cautious; rallies may face selling until confidence returns.")
        elif v <= 55:
            bullets.append(f"Neutral sentiment ({v}/100) — no strong crowd bias either way.")
        elif v <= 75:
            bullets.append(f"Greed ({v}/100) — momentum is healthy, but late longs are vulnerable to pullbacks.")
        else:
            bullets.append(f"Extreme greed ({v}/100) — the crowd is euphoric; tops often form when everyone is already in.")
    c = st["counts"]
    if st["rows"]:
        bullets.append(f"Of {len(st['rows'])} scanned pairs, {c['bullish']} are bullish and {c['bearish']} bearish on the 4h trend (SuperTrend + EMA{EMA_FAST}/{EMA_SLOW}).")

    # optional LLM rewrite (guarded)
    if ANTHROPIC_API_KEY and bullets:
        facts = "\n".join(filter(None, [
            sub,
            f"Fear & Greed: {fg[0]}/100 ({fg[1]})" if fg else "",
            f"BTC accounts long/short: {long_pct:.0f}% / {100 - long_pct:.0f}%" if long_pct is not None else "",
            f"Scanned pairs: {len(st['rows'])}; bullish {c['bullish']}, bearish {c['bearish']}, mixed {c['mixed']}",
            f"Active signals: {len(st['sigs'])}",
        ]))
        out = await llm_write(LLM_PULSE_SYSTEM, facts, max_tokens=400)
        b = clean_bullets(out) if out else ""
        if b and llm_numbers_ok(out, facts):
            bullets = [ln[2:] for ln in b.splitlines()]

    lines = [f"📊 <b>Market Pulse · {date_str}</b>", f"<i>{html_lib.escape(sub)}</i>", "", "🧠 <b>Today's read</b>"]
    lines += [f"· {b}" for b in bullets]
    if st["sigs"]:
        lines += ["", "🎯 <b>Signals</b>"]
        for r in st["sigs"][:3]:
            e = "🟢" if r["signal"] == "BUY" else "🔴"
            lines.append(f"{e} {r['symbol']} {r['signal']} · score {r.get('signal_score', 0)}/10")
    mv = st["movers"]
    if len(mv) >= 6:
        lines += ["", "📈 <b>Movers (24h)</b>",
                  "Up: " + " · ".join(f"{r['symbol']} {r['pct_change_24h']:+.1f}%" for r in mv[:3]),
                  "Down: " + " · ".join(f"{r['symbol']} {r['pct_change_24h']:+.1f}%" for r in reversed(mv[-3:]))]
    lines += ["", "<i>Informational only — not financial advice.</i>"]

    headline = crowd_headline
    if not headline:
        if fg and fg[0] <= 25:
            headline = "Extreme fear grips the market"
        elif fg and fg[0] >= 75:
            headline = "Extreme greed is building"
        elif btc_chg is not None:
            headline = f"BTC {'slips' if btc_chg < 0 else 'climbs'} {abs(btc_chg):.1f}% in 24h"
        else:
            headline = "Daily market pulse"
    png = await asyncio.to_thread(render_pulse_png, date_str, headline, btc_price, btc_chg, closes)
    return png, "\n".join(lines)


async def send_market_pulse(chat_id: str | None = None):
    png, text = await build_market_pulse()
    if png:
        await send_photo(png, chat_id=chat_id)
    await send_alert(text, chat_id=chat_id)


async def daily_pulse_loop():
    """Sends the Market Pulse every day at PULSE_HOUR_WAT:00 Nigeria time (UTC+1, no DST)."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Daily pulse disabled — TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set")
        return
    while True:
        now = datetime.now(WAT)
        target = now.replace(hour=PULSE_HOUR_WAT, minute=0, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        await asyncio.sleep((target - now).total_seconds())
        try:
            await send_market_pulse()
        except Exception as e:
            log.warning("daily pulse failed: %s", e)
        await asyncio.sleep(61)


async def _tg_api(method: str, params: dict | None = None):
    if not TELEGRAM_BOT_TOKEN:
        return None
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    try:
        client = _get_tg_client()
        resp = await client.get(url, params=params or {})
        if resp.status_code != 200:
            log.warning("Telegram API %s failed: %s", method, resp.text[:200])
            return None
        body = resp.json()
        if not body.get("ok"):
            return None
        return body.get("result")
    except httpx.HTTPError as e:
        log.warning("Telegram API error %s: %s", method, e)
        return None


async def handle_telegram_command(chat_id: str, text: str):
    text = (text or "").strip()
    if not text:
        return
    lower = text.lower()
    if lower.startswith("/start") or lower.startswith("/help"):
        await send_alert(
            "<b>AuraPlay Bot</b>\n\n"
            "/analyze SYMBOL — 4H structure + order-flow read\n"
            "Example: <code>/analyze PHAUSDT</code> or <code>/analyze PHA</code>\n"
            "Or just send a symbol, e.g. <code>ENA</code> — add a timeframe for others, e.g. <code>BTC 1h</code>, <code>SOL 15m</code>, <code>ETH 1d</code>\n"
            "/signals — pairs currently meeting the setup criteria\n"
            "/summary — scan summary across all pairs\n"
            "/watches — setups I am watching for you (alerts you when the trigger holds)\n"
            "/pulse — daily Market Pulse (also sent automatically at midday WAT)\n\n"
            "Scanner alerts still post automatically to the configured chat.",
            chat_id=chat_id,
        )
        return
    if lower.startswith("/analyze"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            await send_alert("Usage: /analyze SYMBOL\nExample: /analyze PHAUSDT", chat_id=chat_id)
            return
        parsed = parse_symbol_tf(parts[1].strip())
        if not parsed:
            await send_alert("Usage: /analyze SYMBOL [timeframe]\nExample: /analyze PHAUSDT 1h", chat_id=chat_id)
            return
        await run_analysis(chat_id, parsed[0], parsed[1])
        return
    if lower.startswith("/signals"):
        await send_alert(format_signals_list(), chat_id=chat_id)
        return
    if lower.startswith("/watches"):
        await send_alert(format_watches(chat_id), chat_id=chat_id)
        return
    if lower.startswith("/unwatch"):
        arg = (text.split(maxsplit=1)[1].strip().upper() if len(text.split(maxsplit=1)) > 1 else "")
        if not arg:
            await send_alert("Usage: /unwatch SYMBOL  or  /unwatch all", chat_id=chat_id)
            return
        target = None if arg == "ALL" else normalize_symbol(arg)
        keys = [k for k in WATCHES if k[0] == str(chat_id) and (target is None or k[1] == target)]
        for k in keys:
            WATCHES.pop(k, None)
        await send_alert(f"Removed {len(keys)} watch(es).", chat_id=chat_id)
        return
    if lower.startswith("/summary"):
        await send_alert(format_scan_summary(), chat_id=chat_id)
        return
    if lower.startswith("/pulse"):
        try:
            await send_market_pulse(chat_id=chat_id)
        except Exception as e:
            log.exception("pulse failed")
            await send_alert(f"Pulse failed: {e}", chat_id=chat_id)
        return


_HANDLER_TASKS: set = set()


async def _handle_update(chat_id: str, text: str):
    if text.startswith("/"):
        await handle_telegram_command(chat_id, text)
    else:
        parsed = parse_symbol_tf(text)
        if parsed:
            await run_analysis(chat_id, parsed[0], parsed[1])


async def _handle_update_safe(chat_id: str, text: str):
    """Each message runs in its own task with a timeout, so one slow request
    (e.g. Bybit busy) can never freeze the polling loop for everyone."""
    try:
        await asyncio.wait_for(_handle_update(chat_id, text), timeout=150)
    except asyncio.TimeoutError:
        log.warning("Telegram handler timed out for: %s", text[:60])
        await send_alert("That is taking too long — Bybit is busy right now. Please try again in a minute.", chat_id=chat_id)
    except Exception as e:
        log.exception("Telegram handler failed")
        try:
            await send_alert(f"Something went wrong: {e}", chat_id=chat_id)
        except Exception:
            pass


async def telegram_bot_loop():
    """Long-poll getUpdates for /analyze and /help. Runs alongside the scanner."""
    if not TELEGRAM_BOT_TOKEN:
        log.warning("TELEGRAM_BOT_TOKEN not set — command bot disabled")
        return
    log.info("Telegram command bot starting (long-poll)")
    offset = 0
    while True:
        try:
            updates = await _tg_api("getUpdates", {
                "offset": offset,
                "timeout": 25,
                "allowed_updates": '["message"]',
            })
            if not updates:
                await asyncio.sleep(1)
                continue
            for upd in updates:
                offset = max(offset, int(upd.get("update_id", 0)) + 1)
                msg = upd.get("message") or {}
                chat = msg.get("chat") or {}
                chat_id = str(chat.get("id", ""))
                text = msg.get("text") or ""
                if chat_id and text:
                    log.info("Telegram message from %s: %s", chat_id, text[:60])
                    task = asyncio.create_task(_handle_update_safe(chat_id, text))
                    _HANDLER_TASKS.add(task)
                    task.add_done_callback(_HANDLER_TASKS.discard)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("telegram_bot_loop error: %s", e)
            await asyncio.sleep(3)


def _fmt_price(p: float) -> str:
    return f"{p:.6f}" if p < 1 else f"{p:.4f}"


def _fmt_pct_opt(v: float | None) -> str:
    return f"{v:+.2f}%" if v is not None else "unknown"


def format_signal_alert(symbol: str, signal: str, price: float, rsi: float,
                         long_pct: float | None, oi: float | None,
                         stop_loss: float | None = None, take_profit: float | None = None,
                         take_profit2: float | None = None, whale_bias: str | None = None,
                         whale_notional: float | None = None, tag: str | None = None,
                         score: int = 0, flags: list[str] | None = None,
                         entry_price: float | None = None, pct_change_24h: float | None = None,
                         pct_change_8h: float | None = None, pct_change_1h: float | None = None,
                         momentum_basis: str | None = None,
                         supertrend_trend: int | None = None, ema_fast: float | None = None,
                         ema_slow: float | None = None) -> str:
    emoji = tag or ("🟢" if signal == "BUY" else "🔴")
    entry = entry_price if entry_price else price
    lines = [f"{emoji} <b>{signal} — {symbol}</b>"]
    lines.append(f"Entry: {_fmt_price(entry)}")
    lines.append(f"RSI: {rsi}")
    ti = trend_indicator_text(supertrend_trend, ema_fast, ema_slow)
    if ti:
        lines.append(f"Trend: {ti}")
    if stop_loss is not None and take_profit is not None:
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        rr = f"{reward / risk:.2f}" if risk > 0 else "—"
        lines.append(f"Stop Loss: {_fmt_price(stop_loss)}")
        lines.append(f"TP1: {_fmt_price(take_profit)} (1:{rr})")
        if take_profit2 is not None:
            reward2 = abs(take_profit2 - entry)
            rr2 = f"{reward2 / risk:.2f}" if risk > 0 else "—"
            lines.append(f"TP2: {_fmt_price(take_profit2)} (1:{rr2})")
    if momentum_basis == "PRIMARY" and pct_change_24h is not None:
        lines.append(f"Momentum: {pct_change_24h:+.1f}% 24h [PRIMARY]")
    elif momentum_basis == "FALLBACK":
        lines.append(f"Momentum: {_fmt_pct_opt(pct_change_8h)} 8h / {_fmt_pct_opt(pct_change_1h)} 1h [FALLBACK]")
    if long_pct is not None:
        lines.append(f"Long/Short: {long_pct:.1f}% / {100 - long_pct:.1f}%")
    if oi is not None:
        lines.append(f"Open Interest: {oi:,.0f}")
    if whale_bias:
        agrees = (whale_bias == "bullish" and signal == "BUY") or (whale_bias == "bearish" and signal == "SELL")
        confirmation = "confirms" if agrees else "conflicts with"
        lines.append(f"🐋 <b>WHALE ALERT</b>: large {whale_bias} orders detected (~${whale_notional:,.0f}) — {confirmation} this signal")
    lines.append("💡 Consider taking partial profit at TP1, trailing the rest toward TP2.")
    lines.append("⚠️ ATR/structure-based levels, not financial advice — confirm before trading.")
    lines.append(f"Score: {score}/10")
    lines.append(f"⚠️ Flags: {', '.join(flags) if flags else 'None'}")
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


def format_triple_touch_alert(symbol: str, kind: str, level: float, touches: int, price: float) -> str:
    emoji = "🧱" if kind == "resistance" else "🛡️"
    return (f"{emoji} <b>TRIPLE {kind.upper()} — {symbol}</b>\n"
            f"Level: {_fmt_price(level)} tested {touches}x on 4h candles\n"
            f"Current price: {_fmt_price(price)}\n"
            f"Watch for a breakout or rejection here.")


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
LAST_TRIPLE_TOUCH: dict[str, dict] = {}  # symbol -> {"resistance": level_or_None, "support": level_or_None} already alerted
# symbol -> {"direction": "BUY"/"SELL", "trail_stop": float} — tracks an
# open setup so we can suggest tightening the stop as structure advances in
# your favor. Cleared once the signal itself flips away (trend invalidated).
ACTIVE_TRADES: dict[str, dict] = {}
FAILED_SYMBOLS: dict[str, str] = {}  # symbol -> last failure reason (insufficient history, request error, etc.)
PROCESS_STARTED_AT = time.time()  # if this resets unexpectedly between checks, the process restarted
UNIVERSE: list[str] = []
SCAN_META = {"last_full_cycle_started": None, "cycle_count": 0, "universe_size": 0}

# Per-cycle funnel counters — reset at the start of each full universe pass,
# logged at the end. Lets us see WHERE candidates actually drop off instead
# of guessing at thresholds blind. Safe to increment from concurrent
# coroutines within a batch: asyncio is single-threaded, and a bare `+= 1`
# doesn't yield mid-statement, so no lock is needed.
# Per-cycle funnel counters — reset at the start of each full universe pass,
# logged at the end. Lets us see WHERE candidates actually drop off instead
# of guessing at thresholds blind. Safe to increment from concurrent
# coroutines within a batch: asyncio is single-threaded, and a bare `+= 1`
# doesn't yield mid-statement, so no lock is needed.
# NOTE: trend/VWAP are tracked per-direction (bull_/bear_) rather than as a
# combined bull-OR-bear metric — Supertrend is always +1 or -1, so a combined
# "trend_ok" metric is a tautology (always ~100%, tells you nothing). Same
# issue with combining the VWAP soft-bands. Per-direction numbers are the
# ones that actually mean something.
FUNNEL = {"scanned": 0,
          "bull_trend_ok": 0, "bear_trend_ok": 0,
          "bull_vwap_ok": 0, "bear_vwap_ok": 0,
          "bull_bos_found": 0, "bear_bos_found": 0,
          "bull_retracement_confirmed": 0, "bear_retracement_confirmed": 0,
          "signal_candidate": 0, "entry_gap_ok": 0,
          "core_gate_passed": 0, "signal_fired": 0}
FUNNEL_REJECT_REASONS: dict[str, int] = {}  # core-gate reject reason -> count this cycle


def _reset_funnel():
    for k in FUNNEL:
        FUNNEL[k] = 0
    FUNNEL_REJECT_REASONS.clear()


def _record_core_gate_reject(reason: str):
    FUNNEL_REJECT_REASONS[reason] = FUNNEL_REJECT_REASONS.get(reason, 0) + 1


def _log_funnel(cycle_count: int):
    log.info(
        f"[FUNNEL cycle {cycle_count}] scanned={FUNNEL['scanned']} "
        f"-> bull_trend_ok={FUNNEL['bull_trend_ok']} bear_trend_ok={FUNNEL['bear_trend_ok']} "
        f"-> bull_vwap_ok={FUNNEL['bull_vwap_ok']} bear_vwap_ok={FUNNEL['bear_vwap_ok']} "
        f"-> bull_bos_found={FUNNEL['bull_bos_found']} bear_bos_found={FUNNEL['bear_bos_found']} "
        f"-> bull_retracement_confirmed={FUNNEL['bull_retracement_confirmed']} "
        f"bear_retracement_confirmed={FUNNEL['bear_retracement_confirmed']} "
        f"-> signal_candidate={FUNNEL['signal_candidate']} -> entry_gap_ok={FUNNEL['entry_gap_ok']} "
        f"-> core_gate_passed={FUNNEL['core_gate_passed']} -> signal_fired={FUNNEL['signal_fired']}"
    )
    if FUNNEL_REJECT_REASONS:
        top = sorted(FUNNEL_REJECT_REASONS.items(), key=lambda kv: -kv[1])
        log.info(f"[FUNNEL cycle {cycle_count}] core-gate reject reasons: {top}")
BINANCE_SYMBOLS: set[str] | None = None  # cached filter set — None until first successful fetch
BINANCE_SYMBOLS_FETCHED_AT: float | None = None
DAILY_CONTEXT_CACHE: dict[str, dict] = {}  # symbol -> {"fetched_at", "oi_30d_ago", "oi_latest_from_history", "oi_24h_change_pct", "avg_volume_7d", "daily_resistance", "daily_support", "funding_rate_history"} — refreshed rarely, these change slowly


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
                                    snap.whale_bias, snap.whale_notional, snap.signal_tag,
                                    snap.signal_score, snap.signal_flags, snap.entry_price,
                                    snap.pct_change_24h, snap.pct_change_8h, snap.pct_change_1h,
                                    snap.momentum_basis, snap.supertrend_trend,
                                    snap.ema_fast, snap.ema_slow)
        await send_alert(text)
        FUNNEL["signal_fired"] += 1
        if AUTO_ALERT_NARRATIVE:
            await send_signal_narrative(symbol)
    LAST_SIGNAL[symbol] = snap.signal

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
            _log_funnel(SCAN_META["cycle_count"])
            _reset_funnel()
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
    asyncio.create_task(telegram_bot_loop())
    asyncio.create_task(daily_pulse_loop())
    asyncio.create_task(trigger_watch_loop())


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


@app.get("/api/analyze/{symbol}")
async def api_analyze(symbol: str):
    """On-demand structure + order-flow analysis (same engine as /analyze on Telegram)."""
    try:
        text = await analyze_symbol(symbol)
        return {"symbol": normalize_symbol(symbol), "analysis": text}
    except Exception as e:
        log.exception("api_analyze failed")
        return {"symbol": symbol, "error": str(e)}


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "tracked": len(STATE),
        "universe": len(UNIVERSE),
        "signal_interval": SIGNAL_INTERVAL,
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
      <div class="detail-row"><span>BOS</span><span>${r.entry_bos_direction ? (r.entry_bos_direction.toUpperCase() + ' · ' + r.entry_bos_bars_ago + ' candles ago') : (r.bos ? r.bos.toUpperCase() : "—")}</span></div>
      <div class="detail-row"><span>Large orders</span><span>${r.whale_bias ? (r.whale_bias.charAt(0).toUpperCase()+r.whale_bias.slice(1)) + ' (~$' + fmtCompact(r.whale_notional) + ')' : "—"}</span></div>
      <div class="detail-row"><span>Resistance</span><span>${r.resistance_level != null ? fmtPrice(r.resistance_level) + ' (' + r.resistance_touches + 'x)' : "—"}</span></div>
      <div class="detail-row"><span>Support</span><span>${r.support_level != null ? fmtPrice(r.support_level) + ' (' + r.support_touches + 'x)' : "—"}</span></div>
    </div>
    ${hasLevels ? `<div class="detail-section">
      <div class="detail-row"><span>Score</span><span>${r.signal_tag || ''} ${r.signal_score}/10</span></div>
      <div class="detail-row"><span>Flags</span><span>${(r.signal_flags && r.signal_flags.length) ? r.signal_flags.join(', ') : 'None'}</span></div>
    </div>` : ''}
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
