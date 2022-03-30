from skill_testing_modules import test_random_walk_with_single_agent
from skill_training_modules.single_agent_navigation_skill_trainer import (
    train_navigation_skill_with_single_agent,
)

experiment_modules_map = {
    "test_random_walk_with_single_agent": test_random_walk_with_single_agent,
    "train_navigation_skill_with_single_agent": train_navigation_skill_with_single_agent,
}
