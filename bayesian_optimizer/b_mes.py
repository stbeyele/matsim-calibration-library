import sys
import numpy as np
from scipy.stats import norm
from bayesian_optimizer.gp_bucb import GP_BUCB
from sampler.gumble_sampler import Gumble_Sampler
#  from bayesian_optimizer.gp_bucb_dim import BO_Batch_Dimension


class B_MES(GP_BUCB):
    '''Implements the MES algorithm'''

    def __init__(self, parameters):
        """Set specific input and output dimensions and init add. variable"""

        super(self.__class__, self).__init__(parameters)
        self.max_sampler = None
        self.parameters['Fidelities'] = 0
        self.maxes = None

    def init_bo(self, batch_size, X_train, y_train):
        super(self.__class__, self).init_bo(batch_size, X_train, y_train)

        self.max_sampler = Gumble_Sampler(self.parameters,
                                          self.opt_parameters,
                                          self.dim)

    def aquisition_function(self, x, grad):
        """calculate aquisition function"""

        x = np.reshape(x, (1, -1))

        N = self.maxes.shape[1]
        K = self.maxes.shape[0]
        acq_value = 0.0

        mu, w = self.get_mean(x)
        var, w = self.get_variance(x)

        gamma = (self.maxes - mu[0, :]) / np.sqrt(var[0, :])

        p_term = norm.pdf(gamma)
        c_term = norm.cdf(gamma)
        np.place(c_term, c_term == 0.0, 1.0)
        acq_value = acq_value + np.sum(w * np.sum(
            gamma * p_term / (2.0 * c_term) - np.log(c_term), 0))

        acq_value = acq_value / (float(K) * float(N))

        return acq_value

    def get_variance(self, x):
        """Get Variance of GP"""

        _, var, w = self.m.predict_samples(self.X_var, self.y_var, x)

        return var, w

    def get_mean(self, x):
        """Get mean of GP"""

        y, _, w = self.m.predict_samples(self.X, self.y, x)

        return y, w


    def init_optimization(self):
        """Initialize optimization parameters for aqu fct and optimization"""

        start, maxes = self.max_sampler.get_maxes_start_values(
            self.gp_model, self.X, self.y, self.X_var, self.y_var, 1.0)
        self.maxes = np.transpose(maxes)
        return start
