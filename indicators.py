import pandas_ta as ta
import pandas as pd

def add_indicators(df):
    """
    Calculates technical indicators using pandas-ta.
    """
    # RSI (14 period)
    df['rsi'] = ta.rsi(df['Close'], length=14)
    
    # MACD (12, 26, 9)
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']
    
    # Bollinger Bands (20 period)
    bbands = ta.bbands(df['Close'], length=20, std=2)
    df['bb_upper'] = bbands['BBU_20_2.0']
    df['bb_lower'] = bbands['BBL_20_2.0']
    df['bb_mid'] = bbands['BBM_20_2.0']
    
    # Volume moving average
    df['volume_sma'] = ta.sma(df['Volume'], length=20)
    
    # Fill NaN values
    df.fillna(0, inplace=True)
    
    return df
