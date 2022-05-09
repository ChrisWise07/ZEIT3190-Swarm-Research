from skill_testing_modules import (
    CellsPerMinuteEvaluator,
    CellsPerMinuteEvaluatorFreeRegion,
    CorrectSenseBroadcastEvaluator,
    RandomActionCorrectSenseBroadcastEvaluator,
)
from skill_training_modules import (
    SingleAgentNavigationTrainer,
    MultiAgentNavigationTrainer,
    SenseBroadcastTrainer,
    CommitToOpinionTrainer,
)

experiment_modules_map = {
    "single_agent_navigation_trainer": SingleAgentNavigationTrainer,
    "multi_agent_navigation_trainer": MultiAgentNavigationTrainer,
    "sense_broadcast_trainer": SenseBroadcastTrainer,
    "commit_to_opinion_trainer": CommitToOpinionTrainer,
    "cells_per_minute_evaluator": CellsPerMinuteEvaluator,
    "cells_per_minute_evaluator_free_region": CellsPerMinuteEvaluatorFreeRegion,
    "correct_sense_broadcast_evaluator": CorrectSenseBroadcastEvaluator,
    "random_action_correct_sense_broadcast_evaluator": RandomActionCorrectSenseBroadcastEvaluator,
}
