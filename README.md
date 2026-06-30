# Poly Trading Engine v3

**Live Paper Trading Platform with BTC Probability-Based Auto-Trading**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Quick Start

### Windows (One-Click)

```bash
# Clone the repository
git clone https://github.com/ryofficeics-pixel/poly-trading-engine.git
cd poly-trading-engine

# Install dependencies
pip install -r requirements.txt

# Launch (auto-opens browser)
start_server.bat
```

### Manual Start

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from src.database.db import init_db; init_db()"

# Start server
python ws_server.py

# Open browser
http://localhost:8000
```

---

## ✨ Features

### 📊 Core Trading Engine

- **Paper Trading** — Simulate trades with $1,000 starting capital (configurable)
- **Auto-Trading** — ML-powered probability model generates signals automatically
- **Risk Management** — Kelly fraction sizing, volatility gates, drawdown protection
- **Position Management** — Automatic exits (TP: 1.5%, SL: 1.0%, max hold: 60 min)
- **Real-Time Monitoring** — WebSocket dashboard with live equity curve

### 🌍 REST-Only Mode (Unique Feature)

**Problem:** Binance WebSocket blocked in your region?  
**Solution:** Works 100% with REST APIs (CoinGecko + CoinPaprika)

- ✅ No VPN required
- ✅ Real-time candle synthesis from 5-second price ticks
- ✅ Full auto-trading pipeline functional
- ✅ Automatic failover between data sources

### 🧠 BTC Probability Model

**Features Extracted:**
- RSI (1m, 5m, 1h timeframes)
- GARCH volatility (regime detection)
- Bollinger Bands %B
- ATR (volatility-adjusted)
- Hurst exponent (mean reversion vs trending)
- Order book imbalance (simulated)
- Buy/sell ratio (simulated)

**Output:**
- Long probability (0-1)
- Confidence score (0-1)
- Signal: LONG, SHORT, or FLAT
- Recommended position size (% of equity)

**Risk Engine:**
- Kelly fraction position sizing (capped at 20%)
- Volatility regime gates (blocks trades in EXTREME volatility)
- Exposure limits (max 50% of equity in positions)
- Drawdown protection (15% daily loss triggers kill switch)

---

## 💻 Dashboard

**Access:** `http://localhost:8000`

**Features:**
- Live BTC price ticker
- Real-time equity curve chart
- Open positions table with unrealized PnL
- Recent trades log
- Engine state (probability, confidence, risk score)
- Manual trade controls
- Engine start/stop buttons

---

## ⚙️ Configuration

**Edit `.env` file:**

```env
# REST-only mode (no Binance WebSocket)
SKIP_BINANCE_WS=true
FALLBACK_POLL_INTERVAL=5

# Trading parameters
DEFAULT_CAPITAL=1000.0
MAX_POSITION_PCT=10.0
SIGNAL_THRESHOLD=0.60
TAKE_PROFIT_PCT=1.5
STOP_LOSS_PCT=1.0
MAX_HOLD_MINUTES=60
MAX_OPEN_POSITIONS=3
```

**Key Settings:**

| Parameter | Description | Default |
|-----------|-------------|--------|
| `SKIP_BINANCE_WS` | Force REST-only mode | `true` |
| `FALLBACK_POLL_INTERVAL` | Price update frequency (seconds) | `5` |
| `DEFAULT_CAPITAL` | Starting paper balance | `$1,000` |
| `MAX_POSITION_PCT` | Max position size (% of equity) | `10%` |
| `SIGNAL_THRESHOLD` | Min confidence to trade | `60%` |
| `TAKE_PROFIT_PCT` | Auto-exit profit target | `1.5%` |
| `STOP_LOSS_PCT` | Auto-exit loss limit | `1.0%` |
| `MAX_HOLD_MINUTES` | Max time to hold position | `60 min` |
| `MAX_OPEN_POSITIONS` | Concurrent position limit | `3` |

---

## 📈 How It Works

### Trading Pipeline

```
1. Price Tick (every 5 seconds)
   CoinGecko/CoinPaprika REST API
   ↓
   
2. Candle Synthesis
   Build 1m/5m/1h OHLCV candles from ticks
   ↓
   
3. Feature Extraction (on closed 1m candle)
   RSI, GARCH vol, Bollinger %B, ATR, Hurst, etc.
   ↓
   
4. Probability Model
   Predict long/short direction with confidence
   ↓
   
5. Risk Check
   Kelly sizing + volatility gates + exposure limits
   ↓
   
6. Signal Emission (if approved)
   Direction: UP or DOWN
   Size: % of equity
   ↓
   
7. Paper Fill
   Execute at current price
   Track in SQLite
   ↓
   
8. Exit Management (every 2 seconds)
   Check TP/SL/time/signal flip
   Close if criteria met
```

---

## 📊 Performance

### Latency

- Price tick processing: <1ms
- Feature extraction: ~2ms
- Probability inference: ~1ms
- **Total pipeline: ~5ms per closed candle**

### Throughput

- Price updates: 12/minute (5s interval)
- Trading signals: 0-5/hour (model-dependent)
- Max concurrent positions: 3

### Resources

- CPU: 2-5% (during trading)
- RAM: ~150 MB
- Disk: <1 MB/hour (SQLite)
- Network: ~50 KB/minute (REST polling)

---

## 🔧 API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/health` | GET | System status |
| `/analytics` | GET | Trade analytics |
| `/equity-curve` | GET | Historical equity |
| `/trades?limit=50` | GET | Recent trades |
| `/daily-stats?days=30` | GET | Daily PnL |
| `/engine-state` | GET | Model state |

### WebSocket (`/ws`)

**Client → Server:**
```json
{"cmd": "start"}  // Start engine
{"cmd": "stop"}   // Stop engine
{"cmd": "manual_buy", "direction": "UP", "size_pct": 5}
{"cmd": "manual_sell", "position_id": "p123"}
```

**Server → Client:**
```json
{"type": "price", "price": 60000, "equity": 1050.25}
{"type": "positions", "data": [...]}
{"type": "log", "entry": {"ts": "15:30:45", "level": "INFO", "msg": "..."}}
{"type": "engine_state", "data": {"long_prob": 0.65, "confidence": 0.72, ...}}
```

---

## 🛡️ Security & Risk

### Paper Trading Safeguards

✅ **No real funds** — All trades simulated  
✅ **Kill switch** — Manual override to halt trading  
✅ **Position limits** — Max 3 concurrent, 10% equity each  
✅ **Drawdown protection** — 15% daily loss auto-stops engine  
✅ **Exposure caps** — Max 50% equity in open positions

### API Key Safety

✅ **No API keys required** — Uses public REST endpoints  
✅ **No credentials stored** — CoinGecko/CoinPaprika unauthenticated  
✅ **`.env` gitignored** — Config not committed to repo

---

## 📚 Documentation

- **[HANDOFF.md](HANDOFF.md)** — Complete system architecture, deployment guide, troubleshooting
- **[PROJECT_REPORT.md](PROJECT_REPORT.md)** — Implementation journey, metrics, lessons learned
- **[DEPLOYMENT.md](DEPLOYMENT.md)** — Production deployment guide

---

## 🐛 Troubleshooting

### Engine won't start

**Symptom:** Click START, no response  
**Fix:** Reload dashboard page, check WebSocket connection

### No candles closing

**Symptom:** Engine started, but no activity  
**Fix:** Check `/health` — `feed_connected` should be `true`. Wait 60 seconds for first candle.

### Rate limit errors (429)

**Symptom:** CoinGecko 429 warnings in logs  
**Fix:** System auto-falls back to CoinPaprika. To reduce frequency, increase `FALLBACK_POLL_INTERVAL` to 10-15s.

### Server won't start (port in use)

**Symptom:** `error while attempting to bind on address`  
**Fix:** 
```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill it
taskkill /F /PID <pid>
```

---

## 🚀 Deployment

### Local (Development)

```bash
python ws_server.py
```

### Production (VPS/Cloud)

**Recommended Platforms:**
- Railway.app (free tier)
- Render.com (free tier)
- DigitalOcean ($5/month)
- AWS EC2 t2.micro (free tier)

**Environment Variables:**
```
SKIP_BINANCE_WS=true
FALLBACK_POLL_INTERVAL=5
PORT=8000
```

**Process Manager:**
```bash
# Using PM2
pm2 start ws_server.py --interpreter python3 --name poly-engine

# Using systemd (see DEPLOYMENT.md)
```

---

## 📄 Project Structure

```
poly-trading-engine/
├── ws_server.py                    # Main server + orchestration
├── candle_synthesizer.py           # REST tick → OHLCV converter
├── btc_prob_engine/
│   ├── engine.py                   # Pipeline orchestrator
│   ├── data/feed.py                # Binance feed (unused in REST mode)
│   ├── features/engineer.py        # Feature extraction
│   ├── models/probability.py       # Prediction model
│   └── risk/engine.py              # Risk + sizing
├── src/
│   ├── portfolio/account.py        # Paper account
│   └── database/db.py              # SQLite persistence
├── poly-trading-dashboard.html     # Frontend UI
├── start_server.bat                # Windows startup script
├── .env                            # Configuration
└── requirements.txt                # Dependencies
```

---

## 🔮 Future Roadmap

### Short-term
- [ ] Binance proxy/relay for VPN users
- [ ] CoinGecko Pro API support
- [ ] Admin panel (capital reset, manual trades)
- [ ] Telegram/Discord notifications

### Medium-term
- [ ] Multi-symbol (ETH, SOL, BNB)
- [ ] Live account integration
- [ ] Backtesting framework
- [ ] Advanced risk controls

### Long-term
- [ ] Multi-exchange support (Bybit, OKX)
- [ ] Portfolio-level risk management
- [ ] ML model retraining pipeline
- [ ] Cloud-hosted SaaS

---

## 📜 License

MIT License - see LICENSE file for details

---

## 👏 Acknowledgments

**Architecture inspired by:**
- [ccxt/ccxt](https://github.com/ccxt/ccxt) — Multi-exchange API
- [bmoscon/cryptofeed](https://github.com/bmoscon/cryptofeed) — Callback pattern
- [nkaz001/hftbacktest](https://github.com/nkaz001/hftbacktest) — Tick-level data handling

---

## 📧 Support

**Issues:** [GitHub Issues](https://github.com/ryofficeics-pixel/poly-trading-engine/issues)  
**Discussions:** [GitHub Discussions](https://github.com/ryofficeics-pixel/poly-trading-engine/discussions)

---

## ⚠️ Disclaimer

This is a **paper trading platform** for educational and research purposes. No real funds are at risk. Past performance does not guarantee future results. Always conduct your own research before trading with real money.

---

**Built with ❤️ by OpenAgentic AI Assistant**
