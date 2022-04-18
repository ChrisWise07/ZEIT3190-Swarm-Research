from skill_testing_modules import test_random_walk_with_single_agent
from skill_training_modules import (
    SingleAgentNavigationTrainer,
    MultiAgentNavigationTrainer,
    SenseBroadcastTrainer,
)

experiment_modules_map = {
    "test_random_walk_with_single_agent": test_random_walk_with_single_agent,
    "single_agent_navigation_trainer": SingleAgentNavigationTrainer,
    "multi_agent_navigation_trainer": MultiAgentNavigationTrainer,
    "sense_broadcast_trainer": SenseBroadcastTrainer,
}
