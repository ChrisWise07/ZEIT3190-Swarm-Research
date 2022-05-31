from abstract_env_checker import generic_env_check

generic_env_check(
    "dynamic_opinion_weighting",
    {
        "max_num_of_steps": 10,
        "width": 5,
        "height": 5,
        "num_of_swarm_agents": 4,
        "random_agent_per_step": False,
        "max_new_opinion_weighting": 0.1,
        "num_of_malicious_agents": 1,
        "sensing_noise": 0.1,
        "communication_noise": 0.1,
        "model": None,
    },
)
