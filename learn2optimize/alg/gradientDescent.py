#! python3
''' The algorithm that returns a model only performs gradient descent.
    It will not perform any learning operation.
'''

from baselines.common.input import observation_placeholder
from baselines.common.policies import build_policy

import numpy as np

class GradientDescent():
    ''' A policy that act as gradient descent for only Logistic environment and its numpy definiton.
    '''
    def __init__(self, ob_space, ac_space):
        ''' 'ob_space', 'ac_space': both of them are 1-d numpy array
        '''
        self.observation_space = ob_space
        self.action_space = ac_space
        self.param_dim = 4
        
        self.reset()
        
    def reset(self):
        ''' While the environment is reset, please invoke this method to clear optimizer history information.0
        '''
        self.time_step_cnt = 0

    def step(self, observation):
        ''' observation should be an np array, whose last (n) items are the gradients.
                (n) is the dimension of the action space.
        '''
        self.time_step_cnt += 1
        return observation[-4:] / (self.time_step_cnt)
        

def learn(*,
        network,
        env,
        **network_kwargs,
        ):

    # setup runable policy
    policy = build_policy(env, network, value_network='copy', **network_kwargs)
    ob_space = env.observation_space
    ac_space = env.action_space

    # initialize the gradient descent policy directly
        

