from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


def return_vectorised_monitored_environment(base_environment: any):
    """
    This function returns a monitored and vectorised environment
    """
    return DummyVecEnv([lambda: Monitor(base_environment)])
