import argparse
import os


def test_skill_with_environment(args: argparse.Namespace) -> None:
    import wandb

    config = vars(args)

    config.update(
        {
            "env_name": config["testing_environment"].__name__,
        }
    )

    if config["offline"]:
        os.environ["WANDB_API_KEY"] = "dcda94c92ef247d730cd5188cd920b5a5002aa1d"
        os.environ["WANDB_MODE"] = "offline"

    run = wandb.init(
        name=config["experiment_name"],
        project="swarm_research_evaluation_data",
        config=config,
    )

    env = config["testing_environment"](**config)

    for _ in range(config["num_episodes"]):
        env.reset()
        for step_num in range(int(config["max_num_of_steps"])):
            env.step(step_num)

    run.finish()
