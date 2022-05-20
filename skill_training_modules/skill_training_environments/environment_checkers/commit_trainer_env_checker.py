from abstract_env_checker import generic_env_check

generic_env_check(
    "commit_to_opinion",
    {
        "max_num_of_steps": 30,
        "width": 5,
        "height": 5,
        "num_of_swarm_agents": 5,
        "model": None,
        "random_agent_per_step": False,
    },
)
