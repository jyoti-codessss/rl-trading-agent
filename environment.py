import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_capital=10000):
        super(StockTradingEnv, self).__init__()
        
        self.df = df
        self.initial_capital = initial_capital
        self.current_step = 0
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [price, volume, rsi, macd, bb_upper, bb_lower, portfolio_value, shares_held]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.shares_held = 0
        self.current_step = 0
        
        return self._next_observation(), {}

    def _next_observation(self):
        row = self.df.iloc[self.current_step]
        obs = np.array([
            row['Close'],
            row['Volume'],
            row['rsi'],
            row['macd'],
            row['bb_upper'],
            row['bb_lower'],
            self.portfolio_value,
            self.shares_held
        ], dtype=np.float32)
        return obs

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        
        # Execute action
        if action == 1: # Buy
            shares_to_buy = self.cash // current_price
            self.shares_held += shares_to_buy
            self.cash -= shares_to_buy * current_price
        elif action == 2: # Sell
            self.cash += self.shares_held * current_price
            self.shares_held = 0
            
        self.current_step += 1
        
        # Calculate new portfolio value
        self.portfolio_value = self.cash + self.shares_held * current_price
        
        # Reward: Portfolio profit/loss percentage
        reward = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        # Done if end of data or 252 days
        terminated = self.current_step >= len(self.df) - 1 or self.current_step >= 252
        truncated = False
        
        obs = self._next_observation()
        
        return obs, reward, terminated, truncated, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value:.2f}")
