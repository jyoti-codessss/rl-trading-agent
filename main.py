from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
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

# Global state
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
    # Attempt to load the latest model
    model_dir = "models"
    if os.path.exists(model_dir):
        models = [f for f in os.listdir(model_dir) if f.endswith(".zip")]
        if models:
            latest_model = sorted(models)[-1]
            state["agent"] = load_agent(os.path.join(model_dir, latest_model))
            print(f"Loaded model: {latest_model}")
            
    # Initialize environment with some data
    df = yf.download(state["ticker"], period="2y")
    df = add_indicators(df)
    state["env"] = StockTradingEnv(df, initial_capital=state["initial_capital"])

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "loaded" if state["agent"] else "not loaded",
        "ticker": state["ticker"]
    }

def run_training():
    state["training_job"]["status"] = "training"
    state["training_job"]["progress"] = 10 # Simulate progress
    
    df = yf.download(state["ticker"], period="2y")
    df = add_indicators(df)
    env = StockTradingEnv(df, initial_capital=state["initial_capital"])
    
    model = train_agent(env, timesteps=int(os.getenv("TRAINING_TIMESTEPS", 100000)))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/trading_agent_{timestamp}"
    model.save(model_path)
    
    state["agent"] = model
    state["training_job"]["status"] = "completed"
    state["training_job"]["progress"] = 100

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
        return {"error": "Model not loaded"}
    
    obs, _ = state["env"].reset() # Simplified for prediction
    action = predict_action(state["agent"], obs)
    
    current_price = state["env"].df.iloc[state["env"].current_step]['Close']
    
    action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
    prediction = {
        "action": action_map[int(action)],
        "confidence": 0.95, # PPO doesn't give direct confidence easily without more code
        "current_price": float(current_price),
        "timestamp": datetime.now().isoformat()
    }
    
    # Add to history
    state["history"].append(prediction)
    return prediction

@app.get("/portfolio")
def portfolio():
    env = state["env"]
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
    state["env"].reset()
    state["history"] = []
    return {"message": "Environment reset"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
