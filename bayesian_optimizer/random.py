"""Random Bayesian Optimization Run"""
from copy import copy
import numpy as np
from bayesian_optimizer.gp_bucb import GP_BUCB


class Random_Batch(GP_BUCB):
    """ Implements random batch optimization"""

    def init_bo(self, batch_size, X_train, y_train):
        """Initialize Bayesian Optimization"""

        self.X = np.zeros((1, self.dim))
        self.y = np.zeros((1, 1))
        self.X_var = np.zeros((1, self.dim))
        self.y_var = np.zeros((1, 1))
        self.init = 1

        np.random.seed(self.parameters['Seed'])
        parameter_names = ['Lower_Bounds', 'Upper_Bounds']
        self.opt_parameters = copy(
            {k: self.parameters[k] for k in parameter_names})
        self.m = self.build_model(copy(self.parameters), self.X, self.y)
        self.batch_size = batch_size

    def build_model(self, model_parameters, X, y):
        """Build Model"""

        return None

    def optimize_function(self):
        """Optimize aquisition function"""

        dim = self.dim
        lower_bounds = np.array(self.opt_parameters['Lower_Bounds'])
        upper_bounds = np.array(self.opt_parameters['Upper_Bounds'])
        x = (np.random.rand(dim) *
             (upper_bounds - lower_bounds) + lower_bounds)
        x = np.reshape(x, (-1, dim))

        return x, 0

    def update_model_parameters(self):
        print 'do nothing'

    def get_parameters_history(self):

        return [0]


class Random_Batch_Discrete(GP_BUCB):
    """ Implements random batch optimization"""

    def init_bo(self, batch_size, X_train, y_train):
        """Initialize Bayesian Optimization"""

        self.X = np.zeros((1, self.dim))
        self.y = np.zeros((1, 1))
        self.X_var = np.zeros((1, self.dim))
        self.y_var = np.zeros((1, 1))
        self.init = 1

        np.random.seed(self.parameters['Seed'])
        parameter_names = ['Lower_Bounds', 'Upper_Bounds']
        self.opt_parameters = copy(
            {k: self.parameters[k] for k in parameter_names})
        self.m = self.build_model(copy(self.parameters), self.X, self.y)
        self.batch_size = batch_size

    def build_model(self, model_parameters, X, y):
        """Build Model"""

        return None

    def optimize_function_discrete(self):
        """Optimize aquisition function"""

        rows = self.discrete_set.shape[0]
        idx = np.random.randint(rows, size=1)
        x = np.reshape(self.discrete_set[idx, :], (-1, self.dim))

        return x, 0

    def update_model_parameters(self):
        print 'do nothing'

