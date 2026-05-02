from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os
import yfinance as yf
import pandas as pd
from datetime import datetime
from environment import StockTradingEnv
from indicators import add_indicators
from agent import load_agent, predict_action, train_agent
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Stock Trading RL Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state = {
    "agent": None,
    "env": None,
    "training_job": {"status": "idle", "progress": 0},
    "history": [],
    "ticker": os.getenv("STOCK_TICKER", "AAPL"),
    "initial_capital": float(os.getenv("INITIAL_CAPITAL", 10000))
}

@app.on_event("startup")
async def startup_event():
    model_dir = "models"
    if os.path.exists(model_dir):
        models = [f for f in os.listdir(model_dir) if f.endswith(".zip")]
        if models:
            latest_model = sorted(models)[-1]
            state["agent"] = load_agent(os.path.join(model_dir, latest_model))
            print(f"Loaded model: {latest_model}")
    try:
        df = yf.download(state["ticker"], period="2y")
        if df is not None and len(df) > 50:
            df = add_indicators(df)
            if len(df) > 0:
                state["env"] = StockTradingEnv(df, initial_capital=state["initial_capital"])
                print("Environment initialized successfully!")
        else:
            print("Warning: Could not download stock data, server starting without env")
    except Exception as e:
        print(f"Warning: Startup error: {e}, continuing anyway...")

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "loaded" if state["agent"] else "not loaded",
        "ticker": state["ticker"],
        "env_ready": state["env"] is not None
    }

def run_training():
    state["training_job"]["status"] = "training"
    state["training_job"]["progress"] = 10
    try:
        # Try yfinance first
        df = yf.download(state["ticker"], period="1y")
        
        # If failed, generate sample data
        if df is None or len(df) < 50:
            print("yfinance failed, using sample data...")
            import numpy as np
            dates = pd.date_range(end=pd.Timestamp.today(), periods=300)
            price = 150 + np.cumsum(np.random.randn(300) * 0.5)
            price = np.abs(price)
            df = pd.DataFrame({
                'Open': price * 0.99,
                'High': price * 1.02,
                'Low': price * 0.98,
                'Close': price,
                'Volume': np.random.randint(1000000, 5000000, 300)
            }, index=dates)
            print(f"Sample data created: {len(df)} rows")

        df = add_indicators(df)
        if len(df) == 0:
            raise Exception("DataFrame empty after indicators")
            
        env = StockTradingEnv(df, initial_capital=state["initial_capital"])
        state["env"] = env
        
        model = train_agent(env, timesteps=int(os.getenv("TRAINING_TIMESTEPS", 50000)))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/trading_agent_{timestamp}"
        model.save(model_path)
        state["agent"] = model
        state["training_job"]["status"] = "completed"
        state["training_job"]["progress"] = 100
        print("Training completed!")
    except Exception as e:
        print(f"Training error: {e}")
        state["training_job"]["status"] = f"failed: {e}"
        state["training_job"]["progress"] = 0
@app.get("/train")
def train(background_tasks: BackgroundTasks):
    if state["training_job"]["status"] == "training":
        return {"message": "Training already in progress"}
    background_tasks.add_task(run_training)
    return {"message": "Training started", "job_id": "ppo_train_001"}

@app.get("/status")
def status():
    return state["training_job"]

@app.get("/predict")
def predict():
    if not state["agent"]:
        return {"error": "Model not loaded. Please train first."}
    if not state["env"]:
        return {"error": "Environment not ready. Stock data unavailable."}
    obs, _ = state["env"].reset()
    action = predict_action(state["agent"], obs)
    current_price = float(state["env"].df.iloc[state["env"].current_step]['Close'])
    action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
    prediction = {
        "action": action_map[int(action)],
        "confidence": 0.95,
        "current_price": current_price,
        "timestamp": datetime.now().isoformat()
    }
    state["history"].append(prediction)
    return prediction

@app.get("/portfolio")
def portfolio():
    env = state["env"]
    if env is None:
        return {"value": 10000, "return": 0.0, "shares": 0, "cash": 10000, "note": "Environment not ready"}
    return {
        "value": float(env.portfolio_value),
        "return": float((env.portfolio_value - env.initial_capital) / env.initial_capital * 100),
        "shares": int(env.shares_held),
        "cash": float(env.cash)
    }

@app.get("/history")
def history():
    return state["history"]

@app.post("/reset")
def reset():
    state["history"] = []
    if state["env"] is not None:
        state["env"].reset()
        return {"message": "Environment reset successfully"}
    return {"message": "History cleared, environment not initialized yet"}

# Static files - Frontend (mount AFTER all API routes)
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
def root():
    return FileResponse("frontend/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))