from abstract_env_checker import generic_env_check

generic_env_check(
    "dynamic_opinion_weighting",
    {
        "max_num_of_steps": 10,
        "width": 5,
        "height": 5,
        "num_of_swarm_agents": 5,
        "communication_range": 1,
        "model": None,
        "random_agent_per_step": False,
        "collective_opinion_weighting": 0.6,
    },
)
