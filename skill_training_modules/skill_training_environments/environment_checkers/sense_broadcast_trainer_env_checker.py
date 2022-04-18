from ctypes import get_errno
from abstract_env_checker import generic_env_check

generic_env_check(
    "sense_broadcast",
    {
        "max_num_steps": 10,
        "width": 5,
        "height": 5,
        "num_of_swarm_agents": 5,
        "communication_range": 1,
    },
)
