#! python3

''' This file is intended to register all environment and algorithms 
    into openai baselines system. So that, the experiment can be performed 
    and compared with other reinforcement learning algorithm.
'''

from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym.envs.registration import register
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly

from .env.logisticEnv import LogisticEnv
from .env.robustLinearReg import RobustLinearEnv

register(
    id='Logistic-v0',
    entry_point='learn2optimize.env.logisticEnv:LogisticEnv',
    max_episode_steps=1000,
)

register(
    id='RobustLinear-v0',
    entry_point='learn2optimize.env.robustLinearReg:RobustLinearEnv',
    max_episode_steps=1000,
)

