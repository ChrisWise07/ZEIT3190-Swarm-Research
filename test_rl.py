from stable_baselines3 import PPO
import os
from basic_tiled_environment import TiledEnvForNavigation
import time

current_time = int(time.time())

models_dir = f"models/{current_time}/"
logdir = f"logs/{current_time}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = TiledEnvForNavigation()
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
iters = 0
while True:
    iters += 1
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name=f"PPO{current_time}",
    )
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
