# AuraPlay Scanner

A Binance USDT-M Futures scanner: cycles through every perpetual pair, computes
RSI, EMA trend, SuperTrend, VWAP, OBV, liquidity sweeps, FVGs, Break-of-Structure
(BOS), Long/Short account ratio, and Open Interest — then serves it as a live
dashboard and pushes Buy/Sell/BOS alerts to Telegram.

## What it does

- **Backend** (`/backend`, FastAPI + Python): polls Binance's public Futures
  REST API, cycles through the whole tradable USDT-perp universe in small
  batches (so it stays under Binance's rate limits), computes indicators with
  pandas, and holds the latest snapshot per symbol in memory.
- **Frontend** (`/frontend/index.html`): a single-page dashboard served by the
  same backend. Scanner table, filter by Buy/Sell/Neutral, sort by mover/RSI/OI,
  search by symbol. Polls the backend every 8s — no build step, just static HTML.
- **Telegram**: whenever a symbol's signal flips to BUY or SELL, or a new BOS
  fires, it pushes a message to your bot/chat. Alerts only fire on state
  *change*, not every scan cycle, so you won't get spammed.

## 1. Telegram setup

1. Message **@BotFather** on Telegram → `/newbot` → grab the token.
2. Send your new bot any message (or add it to a group/channel).
3. Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser —
   find `"chat":{"id": ...}` in the response. That's your chat ID.

## 2. Deploy on Render (free) + keep it awake

Render's free web service tier is genuinely $0/month, no card required — the
only catch is it sleeps after 15 minutes with no incoming requests (then takes
30-50s to wake back up). We work around that with a free uptime pinger.

**A. Deploy**

1. Push this folder to a new GitHub repo (works fine from the GitHub mobile
   app too: create repo → upload these files, including `render.yaml`).
2. Go to [render.com](https://render.com) → **New → Blueprint** → connect the
   repo. Render reads `render.yaml` at the repo root and sets everything up
   automatically (root dir `backend`, build/start commands, free plan).
   - If you'd rather set it up manually instead of via Blueprint: **New → Web
     Service** → pick the repo → Root Directory: `backend` → Build Command:
     `pip install -r requirements.txt` → Start Command:
     `uvicorn main:app --host 0.0.0.0 --port $PORT` → Instance Type: **Free**.
3. Under **Environment**, add:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
   - (optional tuning — see `backend/config.py`: `RSI_PERIOD`,
     `SYMBOLS_PER_BATCH`, `BATCH_INTERVAL_SECONDS`, etc.)
4. Deploy. Once live, open the Render-provided `.onrender.com` URL — the
   dashboard loads at `/`.

**B. Stop it from sleeping (UptimeRobot, also free)**

1. Sign up at [uptimerobot.com](https://uptimerobot.com) (free plan).
2. **Add New Monitor** → HTTP(s) → paste your Render URL +
   `/api/health` (e.g. `https://auraplay-scanner.onrender.com/api/health`).
3. Set the check interval to **every 5 minutes** (free plan's minimum).
4. Save. UptimeRobot's pings count as real traffic, so Render never sees 15
   idle minutes and the scan loop + Telegram alerts keep running continuously.

That's the whole cost: $0. The tradeoff versus Railway is Render's free
instance is a bit lower-powered (512MB RAM/shared CPU) — fine for this
scanner, since it's mostly I/O-bound (waiting on Binance's API), not
CPU-heavy.

**If you outgrow it:** Railway's ~$1/month credit is the natural next step
(same code, same `Procfile` already included) since a lightweight single
service like this rarely exceeds that credit.

The frontend lives inside the backend's `frontend/` folder and is served
directly, so there's nothing separate to host — one Railway service does both.

## 3. Tuning the signal logic

`backend/signals.py` has the confluence rules in one place, with the reasoning
commented above the function. Right now:

- **BUY**: SuperTrend bullish + EMA20 > EMA50 + price above VWAP + RSI not
  overbought + (liquidity sweep of a recent low OR a bullish FVG in the last
  5 candles).
- **SELL**: the mirror.
- **BOS**: reported independently whenever price closes through the last
  confirmed swing high/low.

This is a starting framework, not a line-for-line port of your AuraPlay
Scalper Pro Pine Script (that has liquidity sweep + FVG + SuperTrend + VWAP +
EMA confluence too, but the exact weighting/thresholds live in the Pine code).
Once you've watched it run for a bit, tell me what to loosen/tighten — RSI
thresholds, FVG minimum gap size, pivot lookback, whether BOS should gate the
buy/sell signal instead of firing separately — and I'll adjust `signals.py`
and `config.py` to match.

## 4. Rate limits & scan speed

Binance's futures REST limit is 2400 request-weight/minute. With ~300 USDT
perpetual pairs and ~4 calls per symbol per cycle, scanning everyone in one
shot would blow through that. Instead the scanner processes
`SYMBOLS_PER_BATCH` (default 15) symbols every `BATCH_INTERVAL_SECONDS`
(default 20s) — a full pass across ~300 pairs takes roughly 6-7 minutes.
Lower `BATCH_INTERVAL_SECONDS` or raise `SYMBOLS_PER_BATCH` if you want faster
full-universe coverage (just don't push it high enough to get rate-limited —
watch the Railway logs for 429/418 warnings).

## Local testing

```
cd backend
pip install -r requirements.txt
python main.py
```
Then open `http://localhost:8000`.
