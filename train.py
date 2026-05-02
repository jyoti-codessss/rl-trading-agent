import yfinance as yf
from environment import StockTradingEnv
from indicators import add_indicators
from agent import train_agent
import datetime
import os

def main():
    ticker = "AAPL"
    print(f"Downloading data for {ticker}...")
    
    # Download 2 years of data
    df = yf.download(ticker, period="2y")
    
    # Calculate indicators
    df = add_indicators(df)
    
    # Create environment
    env = StockTradingEnv(df)
    
    print("Starting training...")
    model = train_agent(env, timesteps=100000)
    
    # Save model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/trading_agent_{timestamp}"
    os.makedirs("models", exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
