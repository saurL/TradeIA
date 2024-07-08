import gym
from stable_baselines3 import PPO
from CustomEnv import StockTradingEnv
import pandas as pd
from stable_baselines3.common.logger import configure
from config import INDICATORS

train = pd.read_csv('data.csv')
train = train.set_index('date_index')
train.index.names = ['']
# Créer l'environnement personnalisé
stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}

env = StockTradingEnv(df = train, **env_kwargs)
env_train, _ = env.get_sb_env()
# Initialiser le modèle PPO avec l'environnement personnalisé
model = PPO('MlpPolicy', env_train, verbose=1)

# Entraîner le modèle
model.learn(total_timesteps=10000)

# Sauvegarder le modèle
model.save("ppo_custom_env")


# Évaluer le modèle
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()
env.close()
