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
    TRAINED_MODELS_DIRECTORY,
)


def train_skill_with_environment(args: argparse.Namespace) -> None:
    config = vars(args)

    config.update(
        {
            "policy_type": "MlpPolicy",
            "env_name": config["training_environment"].__name__,
        }
    )

    if config["offline"]:
        os.environ["WANDB_API_KEY"] = "dcda94c92ef247d730cd5188cd920b5a5002aa1d"
        os.environ["WANDB_MODE"] = "offline"

    wandb.tensorboard.patch(root_logdir=f"{LOGS_DIRECTORY}")

    run = wandb.init(
        name=config["experiment_name"],
        project="swarm_research",
        config=config,
        sync_tensorboard=True,
    )

    env = DummyVecEnv([lambda: Monitor(config["training_environment"](**config))])

    tensorboard_log_path = f"{LOGS_DIRECTORY}/{config['experiment_name']}_{run.id}"

    model = DQN(
        policy=config["policy_type"],
        env=env,
        verbose=config["verbose"],
        tensorboard_log=tensorboard_log_path,
    )

    if config["previous_model"]:
        model.set_parameters(f"{TRAINED_MODELS_DIRECTORY}/{config['previous_model']}")

    model.learn(
        total_timesteps=(config["max_num_steps"] * config["num_episodes"]),
        callback=WandbCallback(
            gradient_save_freq=config["max_num_steps"],
            model_save_freq=config["max_num_steps"],
            model_save_path=f"{MODELS_DIRECTORY}/{config['experiment_name']}_{run.id}",
            verbose=config["verbose"],
        ),
    )

    run.finish()
