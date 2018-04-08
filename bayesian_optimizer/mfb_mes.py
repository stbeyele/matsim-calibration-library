import sys
import time
from copy import copy, deepcopy
import itertools
import numpy as np
from scipy.stats import norm
from utilities.runner_utilities import convert_fidelity_matrix
from bayesian_optimizer.gp_bucb import GP_BUCB
from bayesian_optimizer.kernels import Delta_Kernel
from utilities.sampler_utilities import set_parameters_mes_fidelity
from sampler.gumble_sampler import Gumble_Sampler


class MFB_MES(GP_BUCB):

    def __init__(self, parameters):
        """Set specific input and output dimensions and init add. variable"""
        super(self.__class__, self).__init__(parameters)
        self.output_dim = 1
        self.input_dim = self.dim + self.parameters['Fidelities'] - 1
        self.fidelity_choice = None
        self.y_full_fidelity = None
        self.set_params = set_parameters_mes_fidelity
        self.kernel_number = self.parameters['Fidelities']
        self.kernel_dim = self.dim
        self.maxes = None
        self.max_sampler = None
        self.acq_fct_logger = np.zeros((1, self.parameters['Fidelities']))
        self.acq_std = 0
        self.fidelity_exploration = self.parameters['Fidelity_Exploration']
        self.acq_std_dev_logger = np.zeros((1, self.parameters['Fidelities']))

    def init_bo(self, batch_size, X_train, y_train):
        super(self.__class__, self).init_bo(batch_size, X_train, y_train)

        self.max_sampler = Gumble_Sampler(self.parameters,
                                          self.opt_parameters,
                                          self.dim)

    def prepare_kernel_parameters(self, model_parameters, idx):
        """Prepare kernel parameters to build kernels with base class"""

        kernel_parameters = {}
        kernel_keys = ['Variance', 'Lengthscale', 'Kernel']
        for key in kernel_keys:
            kernel_parameters[key] = model_parameters[key][idx]

        kernel_parameters['ARD'] = model_parameters['ARD']
        kernel_parameters['Dimension'] = model_parameters['Dimension']
        kernel_parameters['Active_Dims'] = range(self.dim)
        kernel_parameters['Kernel_Transform'] = \
            model_parameters['Kernel_Transform']
        kernel_parameters['Sampler'] = model_parameters['Sampler']

        return kernel_parameters

    def build_kernel(self, kernel_parameters):
        """Build kernel"""

        kernel_parameters_0 = self.prepare_kernel_parameters(kernel_parameters,
                                                             0)
        kernel_tot = super(self.__class__, self).build_kernel(
            kernel_parameters_0)

        for idx in range(1, len(kernel_parameters['Kernel'])):
            kernel_parameters_idx = self.prepare_kernel_parameters(
                kernel_parameters, idx)
            kernel = super(self.__class__, self).build_kernel(
                kernel_parameters_idx)
            kernel_tot = kernel_tot + Delta_Kernel(
                self.dim + (idx - 1)) * kernel

        return kernel_tot

    def get_kernel_parameters(self, model, parameters):
        """Get trained kernel parameters"""

        kernels = model.kern.sorted_params

        for idx, kernel in enumerate(kernels):
            if not hasattr(kernel, 'lengthscales'):
                kernel = kernel.sorted_params[1]
            kernel_parameters = {}
            kernel_parameters['Optimized_Variance'] = \
                kernel.variance.value.tolist()
            kernel_parameters['Optimized_Lengthscales'] = \
                kernel.lengthscales.value.tolist()
            parameters[kernel.name + '_' + str(idx + 1)] = kernel_parameters

        return parameters

    def aquisition_function(self, x, grad):
        """Aquisition function of Bayesian optimization"""

        x = np.reshape(x, (-1, self.dim))
        x = np.hstack((x, convert_fidelity_matrix(
            self.parameters['Fidelities'], self.fidelity_choice, 1,
            np.ones((self.parameters['Fidelities'])))))

        mu, w = self.get_mean(x)
        var, w = self.get_variance(x)

        gamma = (self.maxes - mu[0, :]) / np.sqrt(var[0, :])

        p_term = norm.pdf(gamma)
        c_term = norm.cdf(gamma)
        np.place(c_term, c_term == 0.0, 1.0)

        acq_value = np.mean(
            gamma * p_term / (2.0 * c_term) - np.log(c_term), 0)
        self.acq_std = np.std(acq_value)
        acq_value = np.mean(w * acq_value)

        return acq_value

    def init_optimization(self):
        """Initialize optimization parameters for aqu fct and optimization"""

        start, maxes = self.max_sampler.get_maxes_start_values(
            self.m, self.X, self.y, self.X_var, self.y_var,
            self.fidelity_choice)
        self.maxes = np.transpose(maxes)
        return start

    def get_variance(self, x):
        """Get Variance of GP"""

        _, var, w = self.m.predict_samples(self.X_var, self.y_var, x)

        return var, w

    def get_mean(self, x):
        """Get mean of GP"""

        y, _, w = self.m.predict_samples(self.X, self.y, x)

        return y, w

    def optimize_function(self):
        """optimize aquisition function"""

        xvals = np.zeros(self.parameters['Fidelities'])
        x = np.zeros((self.parameters['Fidelities'], self.dim))
        costs = np.zeros(self.parameters['Fidelities'])
        xvar_std = np.zeros(self.parameters['Fidelities'])

        for idx, cost in enumerate(self.parameters['Cost']):
            self.fidelity_choice = idx
            if self.discrete_mode:
                x[idx, :], xval =\
                    super(self.__class__, self).optimize_function_discrete()
            else:
                x[idx, :], xval =\
                    super(self.__class__, self).optimize_function()

            costs[idx] = float(cost)
            xvals[idx] = xval
            xvar_std[idx] = self.acq_std

        print costs
        print xvals
        print xvar_std
        if self.fidelity_exploration == 'Scaled':
            scaling_factor = np.max(xvals) / xvals
        else:
            scaling_factor = np.ones(self.parameters['Fidelities']) *\
                self.fidelity_exploration
        print scaling_factor
        xvals = (xvals + scaling_factor * xvar_std) / costs

        index = np.argmax(xvals)
        print '----------------first values-------------------------'
        print xvals
        x = x[index, :]
        xval = xvals[index]
        self.acq_fct_logger = np.vstack((self.acq_fct_logger, xvals))
        self.acq_std_dev_logger = np.vstack((self.acq_std_dev_logger,
                                             xvar_std))

        x = np.reshape(x, (-1, self.dim))
        x = np.hstack((x, convert_fidelity_matrix(
            self.parameters['Fidelities'], index, 1,
            np.ones((self.parameters['Fidelities'])))))

        return x, xval

    def optimize_function_discrete(self):
        """optimize aquisition function"""

        return self.optimize_function()

    def optimize_model(self, x, y, kernel_list):
        """Train GP model"""

        combination_list = []
        kernel_combinations = []
        for idx in range(self.parameters['Fidelities']):
            combination_list.append(kernel_list)
        for combination in itertools.product(*combination_list):
            kernel_combinations.append(combination)

        self.parameters['Lengthscale'] = None
        for kernel_parameter in ['Variance', 'Lengthscale']:
            self.parameters[kernel_parameter] = \
                self.parameters['Fidelities'] * \
                [self.parameters[kernel_parameter]]

        max_kernel, results = super(self.__class__, self).optimize_model(
            x, y, kernel_combinations)

        return max_kernel, results

    def full_fidelity_loss(self, X_new):
        """Get y values for full fidelity"""

        X_new = X_new[:, 0:self.dim]
        Fidelity_Matrix = np.ones((X_new.shape[0],
                                   self.parameters['Fidelities'] - 1))
        X_new = np.hstack((X_new, Fidelity_Matrix))
        y_full_fidelity, _ = self.gp_model.predict(self.X, self.y, X_new)
        y_full_fidelity = np.reshape(y_full_fidelity, ((-1, 1)))

        return y_full_fidelity

    def update_model(self, X_new, y_new):
        """Update GP with new function values"""

        super(self.__class__, self).update_model(
            X_new, y_new)

        self.y_full_fidelity = np.vstack(
            (self.y_full_fidelity, self.full_fidelity_loss(X_new)))

    def update_data(self, X_new, y_new):
        self.X = np.vstack((self.X, X_new))
        self.X_var = deepcopy(self.X)
        self.y_var = np.zeros((self.X_var.shape[0], self.output_dim))
        self.y = np.vstack((self.y, y_new))
        self.update_model_parameters()

    def recheck_acq_fct(self, X_new):

        for batch_number in range(self.batch_size):
            acq_val = np.zeros((self.parameters['Fidelities']))
            for fid, cost in enumerate(self.parameters['Cost']):
                self.fidelity_choice = fid
                self.init_optimization()

                acq_val[fid] = self.aquisition_function(X_new[batch_number, :self.opt_dim], None) / float(cost)
            print '--------------optimized values---------------'
            print acq_val
            self.acq_fct_logger = np.vstack((self.acq_fct_logger, acq_val))
            index = np.argmax(acq_val)
            print index

            if index == (self.parameters['Fidelities'] - 1):
                print 'yes-------------'
                fidelity = convert_fidelity_matrix(self.parameters['Fidelities'], index, 1, [1.0, 1.0, 1.0])[0, :]
                x = np.hstack((X_new[batch_number, :self.opt_dim], fidelity))
                x = np.reshape(x, ((1, -1)))
                self.X_var = np.vstack((self.X_var, x))
                self.y_var = np.zeros((self.X_var.shape[0], self.output_dim))

        return copy(self.X_var[(-1 * self.batch_size):, :])

    def bo_history(self):
        """Return the Bayesian Optimization History"""

        return copy(self.X[self.train_size:, :]),\
            copy(self.y[self.train_size:, :]), \
            copy(self.y_full_fidelity[1:, :])

    def get_acq_fct_history(self):
        """Returns the acquisiton function histroy"""

        return copy(self.acq_fct_logger[1:, :])

    def get_acq_stats(self):
        """Returns the acquisiton function histroy"""

        return copy(self.acq_std_dev_logger[1:, :])
