from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

def train_agent(env, timesteps=100000):
    """
    Trains a PPO agent on the provided environment.
    """
    # Wrap environment
    vec_env = DummyVecEnv([lambda: env])
    
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1, 
        learning_rate=0.0003, 
        n_steps=2048, 
        batch_size=64
    )
    
    model.learn(total_timesteps=timesteps)
    return model

def load_agent(model_path):
    """
    Loads a trained PPO agent from a file.
    """
    return PPO.load(model_path)

def predict_action(agent, observation):
    """
    Predicts the next action based on the observation.
    """
    action, _states = agent.predict(observation, deterministic=True)
    return action
