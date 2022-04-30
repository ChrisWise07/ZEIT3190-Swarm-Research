import argparse
import os

from helper_files import (
    MODELS_DIRECTORY,
    LOGS_DIRECTORY,
    TRAINED_MODELS_DIRECTORY,
)


def train_skill_with_environment(args: argparse.Namespace) -> None:
    import wandb
    from stable_baselines3 import PPO
    from wandb.integration.sb3 import WandbCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

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

    env = config["training_environment"](**config)

    tensorboard_log_path = f"{LOGS_DIRECTORY}/{config['experiment_name']}_{run.id}"

    model = PPO(
        policy=config["policy_type"],
        env=env,
        verbose=config["verbose"],
        tensorboard_log=tensorboard_log_path,
    )

    env.set_model(model)

    if config["previous_model"]:
        model.set_parameters(f"{TRAINED_MODELS_DIRECTORY}/{config['previous_model']}")

    model.learn(
        total_timesteps=(config["max_num_of_steps"] * config["num_episodes"]),
        callback=WandbCallback(
            gradient_save_freq=config["max_num_of_steps"],
            model_save_freq=config["max_num_of_steps"],
            model_save_path=f"{MODELS_DIRECTORY}/{config['experiment_name']}_{run.id}",
            verbose=config["verbose"],
        ),
    )

    run.finish()
