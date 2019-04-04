#! python3

import numpy as np
import pickle
import os

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from gym import Env, error, spaces

class LogisticEnv(Env):
    ''' Trying to learn to optimize logistic regression problem

          T
     +---+   +-------------------+
     |   |   |                   |
     |   |   |                   |
     |   |   |                   |
     |   |   |                   |
     |   |   |                   |       +----------------------+
     | w |   |         X         |   +   |    b(broadcasted)    |
     |   |   |                   |       +----------------------+
     |   |   |                   |
     |   |   |                   |
     |   |   |                   |
     |   |   |                   |
     +---+   +-------------------+

        where vectors are still stored as a 1-D array
    '''
    def __init__(self, sess= None, data_path= None, weights_path= None):
        ''' Both paths are file path, not directory path. (they has to be pkl file stored previously)
        '''
        super(LogisticEnv, self).__init__()
        # a dictionary specifying configurations
        # Don't put object in here
        self.configs = {
            # each type of eample will be drawn half of the total sample amount.
            "num_data_total": 100,
            # the maximum number of times the agent is allowed to optimize this problem.
            "max_opt_times": 40,
            # assuming the value is x, the number of parameters in w and d will be x + 1
            "x_dim": 3,
            "lambda": tf.constant(0.0005), # for l-2 regularization according to the paper
            # the H mentioned in the paper which is the length of the trajectory of hisotical information.
            "horizon": 25,
            "data_path": data_path, # load and save (X,Y) data
            "weights_path": weights_path, # load and save learnt weights
        }

        print("Logistic regression environment initializing...")
        
        # data generator of 2 multivariate Gaussians
        # (zero-th one for y == 0, another for y == 1)
        self.distributions = [self._rand_Gaussian_Dist(), self._rand_Gaussian_Dist()]

        # assign tf session
        if sess != None:
            self.sess = sess
        else:
            self.sess = tf.Session()

        # setup formula
        self.vars = {}
        with tf.variable_scope("discriminator") as scope:
            self.vars["w"] = tf.get_variable(name= "w", shape= (self.configs["x_dim"],))
            self.vars["b"] = tf.get_variable(name= "b", shape= (1,))
            # where 'None' indicates the number of data generated
            self.vars["x"] = tf.placeholder(tf.float32, shape= (self.configs["x_dim"], None), name= "x")
            self.vars["y"] = tf.placeholder(tf.float32, shape= (1, None), name= "y") # the true value

            w_reshape = tf.reshape(self.vars["w"], [1, self.configs["x_dim"]])
            # to deal with precision problem, sigmoid and log are put together in loss function part
            self.vars["weighted"] = (tf.matmul(w_reshape, self.vars["x"]) + self.vars["b"])

        with tf.variable_scope("loss_func") as scope:
            self.vars["losses"] = (self.vars["y"] * tf.math.log_sigmoid(self.vars["weighted"]) + \
                    (1 - self.vars["y"]) * tf.math.log_sigmoid(-self.vars["weighted"]))
                    # (1 - sigmoid(x)) == sigmoid(-x)
            self.vars["loss"] = -tf.reduce_mean(self.vars["losses"], 1) + self.configs["lambda"] / 2 * tf.norm(self.vars["w"])
            self.vars["gradients"] = tf.gradients(self.vars["loss"], [self.vars["w"], self.vars["b"]])

        # if nowhere to load the data, sample them
        if self.configs["data_path"] is None or not os.path.isfile(self.configs["data_path"]):
            # both of which will be concatencate into a matrix with (n) columns
            X, Y = self._generate_data(self.configs["num_data_total"])
            self.data = (X, Y)
            print("(X,Y) data generated.")
            # check again whether to store the data
            if isinstance(self.configs["data_path"], str):
                pickle.dump(self.data)
                print("(X,Y) data stored at: %s" % self.configs["data_path"])
        else:
            with open(self.configs["data_path"]) as f:
                self.data = pickle.load(f)
                print("(X,Y) data loaded at: %s" % self.configs["data_path"])

        # reset/initialize the environment
        self.reset()

        print("Logistic regression initialization done.")
            
    def _generate_data(self, n):
        ''' Generate 'n' number of data (with input and output) in total.
            Each type of data is evenly generated in terms of amount.
            return: (X, Y)
                X: a (d by n) np array
                Y: a (1 by n) np array
        '''
        X = []
        Y = []
        for i, dist in enumerate(self.distributions):
            num = n // len(self.distributions)
            X_data = self.sess.run(dist.sample(num)) # calculate the data
            X.append(X_data.transpose()) # transpose the array, so that each column is a vector
            Y.append(np.array([[i for _ in range(num)]]))
        # concatencate each array into matrix.
        X = np.concatenate(X, axis= 1)
        Y = np.concatenate(Y, axis= 1)
        return (X, Y)
        
    def _rand_Gaussian_Dist(self):
        ''' The method generate a multivariate gaussian distribution with 
            random mean and covariance.
            It returns a distribution that can be sampled
        '''
        mean = np.random.rand(self.configs["x_dim"])
        covar = np.random.rand(self.configs["x_dim"], self.configs["x_dim"])
        covar = np.dot(covar, covar.transpose()) #ensure that the random generated matrix is positive-semidefinite

        # generate the distself.vars["r"]ibution
        return tfd.MultivariateNormalFullCovariance(
                loc= mean,
                covariance_matrix= covar
            )

    @property
    def observation_space(self):
        ''' calculate the total dimension of the state space
        '''

        # (assuming the current iterate mentioned is weight + b)
        parm_dim = self.configs["x_dim"] + 1
        # previous 'horizon' number of objective values
        obj_vals_changes = self.configs["horizon"] * 1
        # previous 'horizon' number of gradient of all weights and bias
        # the gradients in each iteration are packed together
        grads = self.configs["horizon"] * (self.configs["x_dim"] + 1)

        total_dim = parm_dim + obj_vals_changes + grads
        low = -1 * np.ones(total_dim) # change limit for testing
        high = 1 * np.ones(total_dim)
        return spaces.Box(low= low, high= high, dtype= np.float32)

    @property
    def action_space(self):
        ''' Action is a vector that added to all weights and bias to 
            tune the parameters.
            (weight, bias)
        '''
        dim = self.configs["x_dim"] + 1
        low = -1 * np.ones(dim) # change limit for testing
        high = 1 * np.ones(dim)
        return spaces.Box(low= low, high= high, dtype= np.float32)

    def reset(self):
        ''' Randomly initializing problem parameters and clear history trajectories.
        '''

        # initialize the weights
        # TODO: provide method to load and store weights at certain frequencies.
        self.sess.run([self.vars["w"].initializer, self.vars["b"].initializer])

        # according to the paper setting, add history objective value and history gradient as state space
        # This is the only place to construct a trajectory dictionary and add fields
        self.trajectory = {}
        self.trajectory["obj_vals_changes"] = [np.zeros(1) for _ in range(self.configs["horizon"])]
        self.trajectory["gradient_history"] = [np.zeros((self.configs["x_dim"]+1)) for _ in range(self.configs["horizon"])]
        self.trajectory["optimize_times"] = 0

        # calculate the gradient of 'w' and 'b' w.r.t the total loss
        feed_dict = {self.vars["x"]: self.data[0], self.vars["y"]: self.data[1]}
        w, b, self.trajectory["curr_grads"], self.trajectory["curr_loss"] = self.sess.run([
                self.vars["w"],
                self.vars["b"],
                self.vars["gradients"],
                self.vars["loss"]], feed_dict= feed_dict)
        w_grad = self.trajectory["curr_grads"][0] # just for notice
        b_grad = self.trajectory["curr_grads"][1]

        # pack the observation an concatencate into numpy array
        to_cat = [w, b] + self.trajectory["obj_vals_changes"] + self.trajectory["gradient_history"]
        observation = np.concatenate(to_cat, axis= 0)

        return observation

    def step(self, action):
        ''' Take in the parameter changes (w, b), where 'b' changes should be the last one.
            action: it has to be a nparray with 1-dimension
        '''
        self.sess.run([
            self.vars["w"].assign_add(action[:-1]), \
            self.vars["b"].assign_add([action[-1]]) \
        ])

        # get new state
        feed_dict = {self.vars["x"]: self.data[0], self.vars["y"]: self.data[1]}
        w, b, gradients, loss = self.sess.run([
                self.vars["w"],
                self.vars["b"],
                self.vars["gradients"],
                self.vars["loss"]], feed_dict= feed_dict)
        # update history
        self.trajectory["obj_vals_changes"].append(loss - self.trajectory["curr_loss"])
        self.trajectory["obj_vals_changes"].pop(0)
        self.trajectory["curr_loss"] = loss
        self.trajectory["gradient_history"].append(np.concatenate(gradients))
        self.trajectory["gradient_history"].pop(0)
        self.trajectory["curr_grads"] = gradients

        # pack the observation and concatencate into numpy array
        to_cat = [w, b] + self.trajectory["obj_vals_changes"] + self.trajectory["gradient_history"]
        observation = np.concatenate(to_cat, axis= 0)

        # calculate the reward
        reward = -loss

        # determine if the environment is done
        done = (self.trajectory["optimize_times"] >= self.configs["max_opt_times"])
        if done: # TODO
            self.reset()

        # pack some info
        info = {}

        # pack as Step object and return
        # reward returning reward[0] because openai API need a numpy number, not an array
        return observation, reward[0], done, info
