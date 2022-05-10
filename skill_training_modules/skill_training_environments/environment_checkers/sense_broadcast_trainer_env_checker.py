from abstract_env_checker import generic_env_check

generic_env_check(
    "sense_broadcast",
    {
        "max_num_of_steps": 10,
        "width": 5,
        "height": 5,
        "num_of_swarm_agents": 5,
        "communication_range": 1,
        "model": "sense_broadcast_model_lesson_weighting",
    },
)
