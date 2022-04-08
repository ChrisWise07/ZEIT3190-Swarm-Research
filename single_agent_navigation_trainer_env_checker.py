from stable_baselines3.common.env_checker import check_env
from skill_training_modules import SingleAgentNavigationTrainer

env = SingleAgentNavigationTrainer(max_num_steps=40, width=5, height=5)
check_env(env)

episodes = 2

iterations_per_episode = 5

for episode in range(episodes):
    print("--- Reset ---")
    obv = env.reset()
    print("first observation", obv)

    for iteration in range(iterations_per_episode):
        random_action = env.action_space.sample()
        print("action", random_action)
        obs, reward, done, info = env.step(random_action)
        print("observation", obs)
        print("reward", reward)
