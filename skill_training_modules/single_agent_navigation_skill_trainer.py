import argparse
import wandb
from stable_baselines3 import DQN
from wandb.integration.sb3 import WandbCallback
from .skill_training_environments import SingleAgentNavigationTrainer
from .training_environment_utils import return_vectorised_monitored_environment
from helper_files import (
    MODELS_DIRECTORY,
    LOGS_DIRECTORY,
)


def train_navigation_skill_with_single_agent(args: argparse.Namespace) -> None:
    config = vars(args)

    config.update(
        {
            "policy_type": "MlpPolicy",
            "env_name": "SingleAgentNavigationTrainer",
        }
    )

    wandb.tensorboard.patch(root_logdir=f"{LOGS_DIRECTORY}")

    run = wandb.init(
        name=f"{args.experiment_name}",
        project="swarm_research",
        config=config,
        sync_tensorboard=True,
    )

    env = return_vectorised_monitored_environment(
        base_environment=SingleAgentNavigationTrainer(
            max_steps_num=args.max_num_steps,
            width=args.tiled_environment_width,
            height=args.tiled_environment_height,
        )
    )

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
