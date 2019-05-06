#! python3

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from gym import Env, error, spaces
from baselines import logger

# suddenly found that there are a lot of code can be re-used.
from .logisticEnv import LogisticEnv

class RobustLinearEnv(LogisticEnv):
    ''' Create a Robust Linear Regression problem for reinforcemenet learning 
        agent to optimize.
        To pack the computation, each x_i is a column vector, which forms a big
        matrix together. And y becomes a row vector (or a matrix)
    '''
    def __init__(self, sess= None):
        ''' 
        '''
        Env.__init__(self)
        # a dictionary specifying configurations
        # It will not change later.
        self.configs = {
            # the number of problems (each set of (X,Y) data is a different problem). (different objective functions)
            "problem_num": 120,
            # the number of data for each problem in each type.
            "num_data_each": 25,
            # the number of data types for one problem.
            "num_data_type": 4,
            # the maximum number of times the agent is allowed to optimize this problem.
            "max_opt_times": 1000,
            # the dimension for input data 'x' and also the dimension of weight.
            "x_dim": 3,
            # 'c' is the constant specified in the paper.
            "c": 1,
            # In the paper's common settings
            "horizon": 25,
        }

        logger.log("Robust Linear Regression environment initializing...")

        # generate data sampling operations
        self.distributions = [self._rand_Gaussian_Dist() for _ in range(self.configs["num_data_type"])]
        self.sample_data_ops = [dist.sample(self.configs["num_data_each"]) for dist in self.distributions]
        self.perturbation_dist = tfd.Normal(loc=0, scale=1)
        self.perturbation_ops = self.perturbation_dist.sample(self.configs["num_data_each"])

        # assign tf session
        if sess != None:
            self.sess = sess
        else:
            self.sess = tf.Session()

        # setup the formula
        self.vars = {}
        with tf.variable_scope("regression-objective"):
            # not initialized storing value, which will be numpy array when running.
            self.vars["w_val"] = None
            self.vars["b_val"] = None
            # equation placeholder
            self.vars["w"] = tf.placeholder(tf.float32, shape= (self.configs["x_dim"],), name= "w")
            self.vars["b"] = tf.placeholder(tf.float32, shape= (1,), name= "b")
            self.vars["x"] = tf.placeholder(tf.float32, shape= (self.configs["x_dim"], None), name= "x")
            self.vars["y"] = tf.placeholder(tf.float32, shape= (1, None), name= "y")

            w_reshape = tf.reshape(self.vars["w"], [1, self.configs["x_dim"]])
            # based on the equation, "dom" is the term in the parenthesis to be squared.
            self.vars["dom"] = self.vars["y"] - (tf.matmul(w_reshape, self.vars["x"]) - self.vars["b"])
            self.vars["squared"] = tf.math.square(self.vars["dom"])
            self.vars["each"] = tf.divide(self.vars["squared"], (self.vars["squared"] + self.configs["c"]*self.configs["c"]))
            self.vars["loss"] = tf.reduce_mean(self.vars["each"], axis= 1)
            # gradients for the controller information
            self.vars["gradients"] = tf.gradients(self.vars["loss"], [self.vars["w"], self.vars["b"]])

        # initializing data, storing function is not implemented
        self.all_data = [self._generate_data() for _ in range(self.configs["problem_num"])]
        self.data_ind = 0 # using index to retrive data from all collection

        # reset the environment for starting
        self.reset()        

    def _generate_data(self):
        ''' The method runs the operation to sample the data (x), which you have 
            to setup ahead.
            And you will get a pair of (X, Y) from each sampe_data_ops
            NOTE: possible mis-understanding. This function generates data for one
                objective function, where data drawn from 4 different Gaussian will
                be along the same vector. (I cannot confirm whether this is the same
                as the paper described)
        '''
        targetW = np.random.rand(1, self.configs["x_dim"])
        targetb = np.random.rand(1)
        X = []
        Y = []
        for op in self.sample_data_ops:
            X_data = self.sess.run(op).transpose()
            Y_data = np.matmul(targetW, X_data) # this should be a row vector
            Y_data += targetb + self.sess.run(self.perturbation_ops).transpose()

            X.append(X_data)
            Y.append(Y_data)

        X = np.concatenate(X, axis= 1)
        Y = np.concatenate(Y, axis= 1)
        return (X, Y)

    def _rand_Gaussian_Dist(self):
        ''' The method generate a multivariate gaussian distribution with 
            random mean and covariance.
            It returns a distribution that can be sampled
        '''
        mean = np.random.rand(self.configs["x_dim"])
        covar = np.identity(self.configs["x_dim"])

        # generate the distribution which is not an operation
        return tfd.MultivariateNormalFullCovariance(
                loc= mean,
                covariance_matrix= covar
            )        

