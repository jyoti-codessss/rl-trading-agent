import ta
import pandas as pd

def add_indicators(df):
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.copy()
    
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(close=df['Close'])
    df['macd'] = macd.macd()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    
    # Volume MA
    df['volume_ma'] = df['Volume'].rolling(window=20).mean()
    
    df.dropna(inplace=True)
    return df