# AuraPlay Scanner (single-file build)

Everything — config, Binance client, indicators, signal engine, Telegram
alerts, the FastAPI app, and the dashboard UI — lives in **one file**,
`main.py`. This build exists purely to make uploading from a phone painless:
no folders to get right, no risk of files landing in the wrong place.

Only **4 files total**, all sitting flat at the repo root:
```
main.py
requirements.txt
render.yaml
Procfile
```

## 1. Get these files onto GitHub

1. GitHub app → **+** → **New repository** → name it e.g. `aura-scanner` → Create.
2. **Add file → Upload files** → select all 4 files above → commit.
   No folders, no paths to type — they all go straight into the repo root.

## 2. Create your Telegram bot

1. Telegram → **@BotFather** → `/newbot` → follow the prompts → you get a
   token like `123456789:AAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`. Save it.
2. Send your bot any message (so it has a chat to reply in), or add it to a
   group/channel with post rights if you want alerts there instead.
3. In a browser: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` — find
   `"chat":{"id": 123456789, ...}`. That number is your chat ID.

## 3. Deploy on Render (free)

1. [render.com](https://render.com) → sign up (GitHub login is easiest).
2. **New → Blueprint** → select your `aura-scanner` repo → Render reads
   `render.yaml` and configures the service automatically → **Apply**.
   - Manual alternative: **New → Web Service** → pick the repo → Build
     Command: `pip install -r requirements.txt` → Start Command:
     `uvicorn main:app --host 0.0.0.0 --port $PORT` → Instance Type: **Free**.
3. When prompted, paste in `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` from
   step 2.
4. Deploy — first build takes a few minutes (installing pandas etc).
5. Open `https://<your-service-name>.onrender.com` — the dashboard loads at
   `/`. It'll show "tracking 0/X pairs" for the first minute or two while the
   first scan cycle runs.

## 4. Keep it awake (UptimeRobot, free)

Render's free tier sleeps after 15 minutes with no requests. Fix:

1. [uptimerobot.com](https://uptimerobot.com) → free signup.
2. **Add New Monitor** → HTTP(s) → paste
   `https://<your-service-name>.onrender.com/api/health` → interval: 5 min → Save.

Now Render always sees recent traffic, so the scan loop + Telegram alerts
never pause.

## What it does

- Cycles through **every USDT-M perpetual pair on Binance Futures** in small
  batches (~15 every 20s) — respects Binance's rate limits, full universe
  pass takes roughly 6-7 minutes.
- Computes RSI(6), EMA20/50 trend, SuperTrend, VWAP, OBV, liquidity sweeps,
  Fair Value Gaps, Break-of-Structure, plus Binance's own Long/Short account
  ratio and Open Interest per pair.
- Confluence engine turns that into BUY / SELL / NEUTRAL, with BOS reported
  as an independent structural event.
- Telegram gets pushed a message only when a symbol's signal **changes**
  (not every cycle) — same for new BOS events. No spam.
- Dashboard: dark scanner table, filter by signal, sort by movers/RSI/OI,
  search by symbol, polls every 8s.

## Tuning the signal logic

Search for `SIGNAL ENGINE` inside `main.py` — the confluence rules (and the
reasoning behind them) are all in one place, same as before. Also see the
`CONFIG` section at the top of the file for every tunable threshold
(`RSI_PERIOD`, `SYMBOLS_PER_BATCH`, `BATCH_INTERVAL_SECONDS`, etc.) — all
overridable as environment variables on Render without touching code.

This is a starting framework, not a line-for-line port of your AuraPlay
Scalper Pro Pine Script. Once you've watched it run, tell me what to
loosen/tighten and I'll adjust `main.py` directly.

## Local testing

```
pip install -r requirements.txt
python main.py
```
Then open `http://localhost:8000`.

## If you outgrow the free tier

Railway's ~$1/month credit is the natural next step — same `main.py`, same
`Procfile` already included, nothing to change.
