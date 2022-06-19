#!/bin/bash
export EVAL_MODEL_NAME="test_eval"
export TESTING_EVIRONMENT="final_evaluator"
export MAX_NEW_OPINION_WEIGHTING=0.1 #0.1, 0.2, 0.4
export OPINION_WEIGHTING_METHOD="list_of_weights" #equation_based, list_of_weights
export NUM_OF_MALICIOUS_AGENTS=0 #0, 1
export SENSING_NOISE=0.0 #0.0, 0.1
export COMMUNICATION_NOISE=0.0 #0.0, 0.1
export COMMITMENT_THRESHOLD=0.05 #0.025, 0.05, 0.1

export HEIGHT=38 #38, 150
export WIDTH=38 #38, 150
export NUM_OF_SWARM_AGENTS=25 #25, 100
export NUM_EPISODES=10
export MAX_NUM_OF_STEPS=7500 #7500, 30000

for testing_environment in "nonclustered_tile_grid" "clustered_inital_observation_useful_tile_grid" "clustered_inital_observation_not_useful_tile_grid" 
do
    for ratio in 0.52 0.62 0.72
    do
        python main.py --experiment_name="${testing_environment}_${ratio}_${EVAL_MODEL_NAME}_${OPINION_WEIGHTING_METHOD}_${MAX_NEW_OPINION_WEIGHTING}_num_malicious_agents_${NUM_OF_MALICIOUS_AGENTS}_${COMMITMENT_THRESHOLD}" --environment_type_name=$testing_environment --ratio_of_white_to_black_tiles=$ratio --testing_environment=$TESTING_EVIRONMENT --height=$HEIGHT --width=$WIDTH --num_of_swarm_agents=$NUM_OF_SWARM_AGENTS --num_episodes=$NUM_EPISODES --max_num_of_steps=$MAX_NUM_OF_STEPS  --max_new_opinion_weighting=$MAX_NEW_OPINION_WEIGHTING --opinion_weighting_method=$OPINION_WEIGHTING_METHOD --num_of_malicious_agents=$NUM_OF_MALICIOUS_AGENTS --sensing_noise=$SENSING_NOISE --communication_noise=$COMMUNICATION_NOISE --commitment_threshold=$COMMITMENT_THRESHOLD --eval_model_name=$EVAL_MODEL_NAME
    done
done