import argparse
import wandb
import os
from stable_baselines3 import DQN
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from helper_files import (
    MODELS_DIRECTORY,
    LOGS_DIRECTORY,
)


def train_skill_with_environment(args: argparse.Namespace) -> None:
    if args.offline:
        os.environ["WANDB_API_KEY"] = "dcda94c92ef247d730cd5188cd920b5a5002aa1d"
        os.environ["WANDB_MODE"] = "offline"

    config = vars(args)

    config.update(
        {
            "policy_type": "MlpPolicy",
            "env_name": f"{args.training_environment.__name__}",
        }
    )

    wandb.tensorboard.patch(root_logdir=f"{LOGS_DIRECTORY}")

    run = wandb.init(
        name=f"{args.experiment_name}",
        project="swarm_research",
        config=config,
        sync_tensorboard=True,
    )

    env = DummyVecEnv([lambda: Monitor(args.training_environment(**config))])

    model = DQN(
        policy=config["policy_type"],
        env=env,
        verbose=1,
        tensorboard_log=f"{LOGS_DIRECTORY}/{args.experiment_name}_{run.id}",
    )

    model.learn(
        total_timesteps=(config["max_num_steps"] * args.num_episodes),
        callback=WandbCallback(
            gradient_save_freq=config["max_num_steps"],
            model_save_freq=config["max_num_steps"],
            model_save_path=f"{MODELS_DIRECTORY}/{args.experiment_name}_{run.id}",
            verbose=2,
        ),
    )

    run.finish()
