# Poly Trading Engine v3 - Deployment Status

## ✅ System Operational

**Deployed:** Local development instance  
**Status:** Fully functional  
**Date:** January 2025  
**URL:** http://localhost:8000  

---

## Current Configuration

### Server
- **Host:** 0.0.0.0
- **Port:** 8000
- **Process:** Running in background (PID varies)
- **Database:** SQLite (poly_engine.db)
- **Logs:** logs/ directory

### Price Feed
- **Primary:** Binance WebSocket (blocked/timing out in current region)
- **Active Fallback:** REST API polling
  - CoinGecko (primary fallback)
  - CoinPaprika (secondary fallback)
- **Update Interval:** ~10 seconds
- **Current Price:** $60,332 BTC/USDT
- **Feed Status:** ✅ Connected via REST fallback

### Paper Trading Account
- **Starting Capital:** $1,000
- **Current Equity:** $1,000
- **Realized PnL:** $0
- **Open Positions:** 0
- **Total Trades:** 0

---

## Architecture

### Components

1. **ws_server.py** - Main application server
   - FastAPI + WebSocket server
   - Paper trading engine
   - Account management
   - Price feed orchestration (primary + fallback)
   - Dashboard WebSocket bridge

2. **btc_prob_engine/** - BTC probability engine
   - Data layer (feed.py) - Binance WebSocket feeds
   - Feature layer (engineer.py) - Technical indicators
   - Model layer (probability.py) - Directional probability
   - Risk layer (risk/engine.py) - Position sizing & risk

3. **src/database/** - Persistence layer
   - Account state
   - Position history
   - Trade records
   - Equity snapshots
   - Daily statistics

4. **poly-trading-dashboard.html** - Frontend dashboard
   - Real-time price & PnL display
   - Position management
   - Trade history
   - Analytics & charts

---

## Fallback Feed Architecture

### Problem
Binance WebSocket connections timing out (common in certain regions/networks)

### Solution
Automatic fallback hierarchy:

```
1. Binance WebSocket (wss://stream.binance.com:9443)
   ├─ Retry 3 times with exponential backoff
   └─ If failed → switch to REST fallback

2. REST Fallback (polling every 10s)
   ├─ CoinGecko API (primary)
   ├─ CoinPaprika API (secondary)
   └─ Periodically retry Binance WebSocket (every 30 polls)

3. Auto-promote back to Binance when available
```

### Configuration
- `FALLBACK_POLL_INTERVAL=10` - REST polling interval (seconds)
- `FALLBACK_BINANCE_RETRY_EVERY=30` - Binance retry frequency (polls)
- `BINANCE_PROXY_URL` - Optional relay URL (disabled in .env)

---

## Files Modified

### .env
```bash
# BINANCE_PROXY_URL=ws://localhost:9001
# Relay disabled - using direct Binance connection or fallback REST feeds
```

**Reason:** Disabled relay dependency to use direct Binance connection with automatic REST fallback.

**Note:** `.env` is gitignored (not committed to repository)

---

## Running the System

### Start Server
```bash
python ws_server.py
```

### Access Dashboard
```bash
http://localhost:8000
```

### Health Check
```bash
curl http://localhost:8000/health
```

### Stop Server
Find process and kill:
```bash
# Windows
netstat -ano | findstr :8000
wmic process where "ProcessId=<PID>" delete

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

---

## Expected Warnings (Normal Behavior)

### Binance WebSocket Timeouts
```
WARNING | ws_server | binance feed lost: timed out during opening handshake
```
**Status:** Normal in regions where Binance is restricted/slow  
**Action:** System automatically switches to REST fallback

### CoinGecko Rate Limiting
```
WARN | ws_server | fallback coingecko error: HTTP Error 429: Too Many Requests
```
**Status:** Normal when polling too frequently  
**Action:** System automatically switches to CoinPaprika

### BTC Probability Engine Feed Warnings
```
WARN | feed | feed disconnected: timed out during opening handshake
```
**Status:** Normal when auto-trading engine is started  
**Action:** Engine continues with available data from main price feed

---

## Auto-Trading Engine

### Status
- **Engine:** Stopped (start via dashboard)
- **Strategy:** BTC directional probability
- **Signal Threshold:** 0.60 (60% confidence)
- **Position Sizing:** Kelly Criterion (max 20% per trade)
- **Risk Management:**
  - Take Profit: 1.5%
  - Stop Loss: 1.0%
  - Max Hold Time: 60 minutes
  - Max Open Positions: 3

### To Start
1. Open dashboard at http://localhost:8000
2. Click "Start Engine" button
3. Monitor logs and positions

---

## Data Persistence

### Database: poly_engine.db
- Accounts
- Sessions
- Positions (open & closed)
- Trades
- Equity snapshots
- Daily statistics
- System logs

### Logs
- `logs/poly_engine.log` - Full debug logs
- `logs/error.log` - Warnings & errors only

---

## Next Steps

### Recommended Actions
1. ✅ System is operational - proceed with testing
2. Test manual paper trades via dashboard
3. Start auto-trading engine and monitor performance
4. Review logs for any unexpected errors
5. Adjust risk parameters in .env if needed

### Optional Enhancements
1. Deploy to cloud VPS for better Binance connectivity
2. Add additional price feed sources
3. Implement webhook notifications (Discord/Telegram)
4. Add multi-symbol support
5. Build historical backtest analysis

---

## Support

For issues or questions, check:
- Application logs in `logs/` directory
- Health endpoint: http://localhost:8000/health
- Analytics endpoint: http://localhost:8000/analytics
- Engine state: http://localhost:8000/engine-state

---

**Last Updated:** January 2025  
**System Version:** Poly Trading Engine v3  
**Deployment:** Local development (production-ready architecture)
