# 🤖 AI Stock Trading RL Agent

An intelligent stock trading agent built with Reinforcement Learning (PPO algorithm) that learns to make Buy, Sell, and Hold decisions based on real market data and technical indicators.

---

## 🚀 Live Demo

Deployed on Google Cloud Run — Asia South (Mumbai)

---

## 📌 Project Overview

This project uses **Proximal Policy Optimization (PPO)** — a state-of-the-art RL algorithm — to train an AI agent that simulates stock trading. The agent observes market conditions (price, volume, RSI, MACD, Bollinger Bands) and learns the best trading strategy to maximize portfolio returns.

```
Market Data → Technical Indicators → RL Agent (PPO) → Trading Decision → Portfolio
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| RL Framework | Stable-Baselines3 (PPO) |
| Environment | Custom Gymnasium Env |
| Data Source | yfinance (Yahoo Finance) |
| Indicators | ta library (RSI, MACD, BB) |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML + Chart.js (Dark Theme) |
| Container | Docker (python:3.11-slim) |
| Deploy | Google Cloud Run |

---

## 📁 Project Structure

```
rl-trading-agent/
├── environment.py      → Custom Gymnasium trading environment
├── agent.py            → PPO Agent (Stable-Baselines3)
├── train.py            → Training script
├── indicators.py       → RSI, MACD, Bollinger Bands
├── main.py             → FastAPI backend (port 8080)
├── Dockerfile          → Cloud Run container
├── requirements.txt    → Python dependencies
├── cloudbuild.yaml     → Google Cloud Build config
├── .env.example        → Environment variables template
└── frontend/
    └── index.html      → Dark theme dashboard
```

---

## ⚙️ How It Works

### 1. Environment
- **State:** Price, Volume, RSI, MACD, Bollinger Bands, Portfolio Value, Shares Held
- **Actions:** 0 = Hold, 1 = Buy, 2 = Sell
- **Reward:** Portfolio profit/loss percentage
- **Episode Length:** 252 trading days (1 year)
- **Initial Capital:** $10,000

### 2. Agent
- Algorithm: **PPO (Proximal Policy Optimization)**
- Policy: MlpPolicy (Neural Network)
- Training Steps: 100,000 timesteps

### 3. API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Server health check |
| GET | /train | Start agent training |
| GET | /status | Training progress |
| GET | /predict | Get BUY/SELL/HOLD action |
| GET | /portfolio | Current portfolio stats |
| GET | /history | All trade history |
| POST | /reset | Reset environment |

---

## 🖥️ Run Locally

### Prerequisites
- Python 3.11+
- pip

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/jyoti-codessss/rl-trading-agent.git
cd rl-trading-agent

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
pip install ta

# 4. Run the server
python main.py

# 5. Open dashboard
# http://localhost:8080/
# http://localhost:8080/docs   (API docs)
```

---

## 🐳 Docker

```bash
# Build image
docker build -t rl-trading-agent .

# Run container
docker run -p 8080:8080 rl-trading-agent
```

---

## ☁️ Deploy to Google Cloud Run

```bash
# Set project
gcloud config set project future-union-495107-p0

# Build and deploy
gcloud builds submit --tag gcr.io/future-union-495107-p0/rl-trading-agent
gcloud run deploy rl-trading-agent \
  --image gcr.io/future-union-495107-p0/rl-trading-agent \
  --platform managed \
  --region asia-south1 \
  --memory 2Gi \
  --port 8080 \
  --allow-unauthenticated
```

---

## 📊 Dashboard Features

- **Real-time Portfolio Stats** — Total value, return %, shares, cash
- **BUY / SELL / HOLD Indicator** — Color coded signals
- **Price Chart** — AAPL stock price visualization
- **Portfolio Value Chart** — Live portfolio growth tracking
- **Trade History Table** — All past decisions with timestamp
- **Train Agent Button** — Start training with progress bar
- **Auto-refresh** — Updates every 5 seconds

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| STOCK_TICKER | AAPL | Stock symbol to trade |
| INITIAL_CAPITAL | 10000 | Starting capital ($) |
| TRAINING_TIMESTEPS | 100000 | PPO training steps |
| MODEL_PATH | models/trading_agent | Model save location |
| PORT | 8080 | Server port |

---

## 🚀 Future Enhancements

- Add more stock tickers (GOOGL, TSLA, MSFT)
- Live trading via Alpaca API
- Gemini AI market sentiment analysis
- Mobile responsive PWA
- Email alerts on trade signals
- Backtesting with historical data

---

## 👩‍💻 Built With

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [yfinance](https://pypi.org/project/yfinance/)
- [Google Cloud Run](https://cloud.google.com/run)

---

## 📄 License

MIT License — feel free to use and modify!

---

*Built with ❤️ using Gemini CLI + Google Cloud*
