from skill_testing_modules import (
    CellsPerMinuteEvaluator,
    CellsPerMinuteEvaluatorFreeRegion,
    CorrectSenseBroadcastEvaluator,
    RandomActionCorrectSenseBroadcastEvaluator,
    CommitToOpinionEvaluator,
)
from skill_training_modules import (
    SingleAgentNavigationTrainer,
    MultiAgentNavigationTrainer,
    SenseBroadcastTrainer,
    CommitToOpinionTrainerSafe,
    CommitToOpinionTrainer,
    DynamicOpinionWeightingTrainer,
)

experiment_modules_map = {
    "single_agent_navigation_trainer": SingleAgentNavigationTrainer,
    "multi_agent_navigation_trainer": MultiAgentNavigationTrainer,
    "sense_broadcast_trainer": SenseBroadcastTrainer,
    "commit_to_opinion_trainer_safe": CommitToOpinionTrainerSafe,
    "commit_to_opinion_trainer": CommitToOpinionTrainer,
    "dynamic_opinion_weight_trainer": DynamicOpinionWeightingTrainer,
    "cells_per_minute_evaluator": CellsPerMinuteEvaluator,
    "cells_per_minute_evaluator_free_region": CellsPerMinuteEvaluatorFreeRegion,
    "correct_sense_broadcast_evaluator": CorrectSenseBroadcastEvaluator,
    "random_action_correct_sense_broadcast_evaluator": RandomActionCorrectSenseBroadcastEvaluator,
    "commit_to_opinion_evaluator": CommitToOpinionEvaluator,
}
